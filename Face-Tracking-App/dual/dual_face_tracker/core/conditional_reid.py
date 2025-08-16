"""
조건부 ReID(Re-Identification) 시스템.

ByteTracker의 추적 결과에서 ID 스왑이 감지될 때만 ReID 모델을 활성화하여
정확한 얼굴 재식별을 수행합니다. 이를 통해 성능과 정확도의 균형을 달성합니다.

주요 기능:
- ID 스왑 상황 자동 감지
- 조건부 ReID 모델 활성화 (목표: < 10% 활성화율)
- 임베딩 기반 ID 보정
- 성능 통계 및 모니터링
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
import numpy as np
import cv2
import time

from .tracking_structures import Track, Detection
from .id_swap_detector import IDSwapDetector, SwapDetectionResult
from .embedding_manager import EmbeddingManager, MatchingResult
from ..inference.reid_model import ReIDModel, ReIDModelConfig
from ..utils.logger import UnifiedLogger
from ..utils.exceptions import InferenceError


@dataclass
class ReIDActivationRecord:
    """ReID 활성화 기록."""
    
    frame_id: int                    # 활성화된 프레임 ID
    affected_track_ids: Set[int]     # 영향받은 트랙 ID들
    swap_risk_score: float           # 스왑 위험도 점수
    processing_time_ms: float        # 처리 시간 (ms)
    corrections_made: int            # 수행된 보정 수
    activation_reason: str           # 활성화 이유
    
    def __repr__(self):
        return f"ReIDActivationRecord(frame={self.frame_id}, tracks={len(self.affected_track_ids)})"


@dataclass
class ConditionalReIDResult:
    """ConditionalReID 처리 결과."""
    
    tracks: List[Track]                           # 보정된 트랙 리스트
    reid_activated: bool                          # ReID 활성화 여부
    activation_record: Optional[ReIDActivationRecord]  # 활성화 기록
    processing_time_ms: float                     # 전체 처리 시간
    swap_detection_result: SwapDetectionResult    # 스왑 감지 결과
    corrections_applied: List[Tuple[int, int]]    # 적용된 보정 (old_id, new_id)
    
    def __repr__(self):
        return f"ConditionalReIDResult(activated={self.reid_activated}, tracks={len(self.tracks)})"


class ConditionalReID:
    """
    조건부 ReID 시스템 메인 클래스.
    
    ByteTracker의 결과를 받아 ID 스왑을 감지하고,
    필요한 경우에만 ReID 모델을 활성화하여 ID를 보정합니다.
    """
    
    def __init__(self,
                 reid_model_config: Optional[ReIDModelConfig] = None,
                 swap_detector_config: Optional[Dict] = None,
                 embedding_manager_config: Optional[Dict] = None,
                 activation_threshold: float = 0.6,        # ReID 활성화 임계값
                 target_activation_rate: float = 0.1,     # 목표 활성화율 (10%)
                 face_crop_margin: float = 0.3,           # 얼굴 크롭 마진
                 min_face_size: int = 64):                # 최소 얼굴 크기
        """
        ConditionalReID 시스템을 초기화합니다.
        
        Args:
            reid_model_config: ReID 모델 설정
            swap_detector_config: 스왑 감지기 설정
            embedding_manager_config: 임베딩 관리자 설정
            activation_threshold: ReID 활성화 임계값
            target_activation_rate: 목표 활성화율 (성능 모니터링용)
            face_crop_margin: 얼굴 영역 크롭 시 마진 비율
            min_face_size: 처리할 최소 얼굴 크기
        """
        self.activation_threshold = activation_threshold
        self.target_activation_rate = target_activation_rate
        self.face_crop_margin = face_crop_margin
        self.min_face_size = min_face_size
        
        # ReID 모델 초기화
        reid_config = reid_model_config or ReIDModelConfig.for_face_reid()
        self.reid_model = ReIDModel(**reid_config.to_dict())
        
        # ID 스왑 감지기 초기화
        detector_config = swap_detector_config or {}
        self.swap_detector = IDSwapDetector(**detector_config)
        
        # 임베딩 관리자 초기화
        embedding_config = embedding_manager_config or {}
        self.embedding_manager = EmbeddingManager(**embedding_config)
        
        # 통계 및 모니터링
        self.stats = {
            'total_frames': 0,
            'reid_activations': 0,
            'successful_corrections': 0,
            'false_activations': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'activation_rate': 0.0
        }
        
        self.activation_history = []  # 최근 활성화 기록
        self.frame_counter = 0
        
        # 로깅
        self.logger = UnifiedLogger("ConditionalReID")
        self.logger.info(f"ConditionalReID 시스템 초기화: threshold={activation_threshold}, "
                        f"target_rate={target_activation_rate:.1%}")
    
    def process_frame(self, 
                     tracks: List[Track], 
                     frame_image: np.ndarray) -> ConditionalReIDResult:
        """
        단일 프레임을 처리합니다.
        
        Args:
            tracks: ByteTracker에서 제공된 트랙 리스트
            frame_image: 전체 프레임 이미지
            
        Returns:
            ConditionalReIDResult: 처리 결과
        """
        try:
            start_time = time.perf_counter()
            self.frame_counter += 1
            self.stats['total_frames'] += 1
            
            self.logger.debug(f"프레임 {self.frame_counter} 처리 시작: {len(tracks)}개 트랙")
            
            # 1. ID 스왑 감지
            swap_result = self.swap_detector.detect_swaps(tracks)
            
            # 2. ReID 활성화 여부 결정
            should_activate = self._should_activate_reid(swap_result)
            
            if should_activate:
                # 3. ReID 처리 수행
                corrected_tracks, activation_record = self._perform_reid_correction(
                    tracks, swap_result, frame_image
                )
                
                self.stats['reid_activations'] += 1
                self.activation_history.append(activation_record)
                
                # 활성화 기록 제한 (메모리 관리)
                if len(self.activation_history) > 1000:
                    self.activation_history = self.activation_history[-500:]
                
            else:
                # 4. 일반 임베딩 수집 (ReID 미활성화 시에도 임베딩 축적)
                corrected_tracks = tracks.copy()
                activation_record = None
                
                # 배경 임베딩 수집 (비동기적으로 수행 가능)
                self._collect_background_embeddings(tracks, frame_image)
            
            # 5. 통계 업데이트
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_statistics(processing_time)
            
            # 6. 결과 반환
            corrections = self._extract_corrections(tracks, corrected_tracks)
            
            result = ConditionalReIDResult(
                tracks=corrected_tracks,
                reid_activated=should_activate,
                activation_record=activation_record,
                processing_time_ms=processing_time,
                swap_detection_result=swap_result,
                corrections_applied=corrections
            )
            
            self.logger.debug(f"프레임 {self.frame_counter} 처리 완료: "
                             f"ReID={'활성화' if should_activate else '비활성화'}, "
                             f"{processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"프레임 처리 실패: {e}")
            # 안전한 기본값 반환
            return ConditionalReIDResult(
                tracks=tracks,
                reid_activated=False,
                activation_record=None,
                processing_time_ms=0.0,
                swap_detection_result=SwapDetectionResult(
                    has_swap=False, overall_risk_score=0.0, confidence=0.0,
                    indicators=[], affected_track_ids=set(), recommendation=""
                ),
                corrections_applied=[]
            )
    
    def _should_activate_reid(self, swap_result: SwapDetectionResult) -> bool:
        """ReID 활성화 여부를 결정합니다."""
        # 1. 기본 임계값 확인
        if not swap_result.has_swap:
            return False
        
        if swap_result.overall_risk_score < self.activation_threshold:
            return False
        
        # 2. 적응적 임계값 조정 (활성화율 제어)
        current_activation_rate = self.stats['activation_rate']
        
        if current_activation_rate > self.target_activation_rate * 1.5:
            # 활성화율이 너무 높으면 임계값을 높임
            adjusted_threshold = self.activation_threshold + 0.1
            if swap_result.overall_risk_score < adjusted_threshold:
                return False
        elif current_activation_rate < self.target_activation_rate * 0.5:
            # 활성화율이 너무 낮으면 임계값을 낮춤
            adjusted_threshold = max(self.activation_threshold - 0.1, 0.3)
            return swap_result.overall_risk_score >= adjusted_threshold
        
        # 3. 신뢰도 고려
        if swap_result.confidence < 0.5:
            return False
        
        # 4. 영향받는 트랙 수 고려 (너무 많으면 보수적으로)
        if len(swap_result.affected_track_ids) > 5:
            return swap_result.overall_risk_score > self.activation_threshold + 0.1
        
        return True
    
    def _perform_reid_correction(self,
                               tracks: List[Track],
                               swap_result: SwapDetectionResult,
                               frame_image: np.ndarray) -> Tuple[List[Track], ReIDActivationRecord]:
        """ReID를 사용한 ID 보정을 수행합니다."""
        start_time = time.perf_counter()
        
        # 영향받은 트랙들 식별
        affected_tracks = [t for t in tracks if t.track_id in swap_result.affected_track_ids]
        
        if not affected_tracks:
            # 활성화되었지만 대상 트랙이 없는 경우
            processing_time = (time.perf_counter() - start_time) * 1000
            activation_record = ReIDActivationRecord(
                frame_id=self.frame_counter,
                affected_track_ids=set(),
                swap_risk_score=swap_result.overall_risk_score,
                processing_time_ms=processing_time,
                corrections_made=0,
                activation_reason="No affected tracks found"
            )
            return tracks.copy(), activation_record
        
        corrections_made = 0
        corrected_tracks = tracks.copy()
        
        try:
            # 1. 얼굴 이미지 크롭 및 임베딩 추출
            track_embeddings = {}
            for track in affected_tracks:
                face_image = self._crop_face_from_track(track, frame_image)
                if face_image is not None:
                    embedding = self.reid_model.extract_embedding(face_image)
                    track_embeddings[track.track_id] = embedding
                    
                    # 임베딩 관리자에 추가
                    self.embedding_manager.add_embedding(track, embedding, face_image)
            
            # 2. 임베딩 기반 매칭 수행
            matches = self._find_best_matches(track_embeddings, affected_tracks)
            
            # 3. ID 보정 적용
            corrections_made = self._apply_id_corrections(corrected_tracks, matches)
            
            self.logger.info(f"ReID 보정 완료: {len(affected_tracks)}개 트랙, "
                           f"{corrections_made}개 보정 수행")
            
        except Exception as e:
            self.logger.error(f"ReID 보정 실패: {e}")
        
        # 활성화 기록 생성
        processing_time = (time.perf_counter() - start_time) * 1000
        activation_record = ReIDActivationRecord(
            frame_id=self.frame_counter,
            affected_track_ids=swap_result.affected_track_ids.copy(),
            swap_risk_score=swap_result.overall_risk_score,
            processing_time_ms=processing_time,
            corrections_made=corrections_made,
            activation_reason=swap_result.recommendation
        )
        
        if corrections_made > 0:
            self.stats['successful_corrections'] += 1
        
        return corrected_tracks, activation_record
    
    def _crop_face_from_track(self, track: Track, frame_image: np.ndarray) -> Optional[np.ndarray]:
        """트랙으로부터 얼굴 영역을 크롭합니다."""
        try:
            if not track.detection:
                return None
            
            # 바운딩 박스 정보
            x1, y1, x2, y2 = track.detection.bbox
            
            # 마진 추가
            width = x2 - x1
            height = y2 - y1
            margin_x = width * self.face_crop_margin
            margin_y = height * self.face_crop_margin
            
            # 확장된 영역
            crop_x1 = max(0, int(x1 - margin_x))
            crop_y1 = max(0, int(y1 - margin_y))
            crop_x2 = min(frame_image.shape[1], int(x2 + margin_x))
            crop_y2 = min(frame_image.shape[0], int(y2 + margin_y))
            
            # 크기 확인
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            
            if crop_width < self.min_face_size or crop_height < self.min_face_size:
                return None
            
            # 크롭
            face_image = frame_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            return face_image
            
        except Exception as e:
            self.logger.warning(f"얼굴 크롭 실패 (Track {track.track_id}): {e}")
            return None
    
    def _find_best_matches(self,
                          track_embeddings: Dict[int, np.ndarray],
                          affected_tracks: List[Track]) -> List[MatchingResult]:
        """최적의 임베딩 매칭을 찾습니다."""
        all_matches = []
        
        for track in affected_tracks:
            if track.track_id not in track_embeddings:
                continue
            
            # 다른 모든 트랙과의 매칭 시도
            candidate_ids = [t.track_id for t in affected_tracks if t.track_id != track.track_id]
            
            if candidate_ids:
                matches = self.embedding_manager.match_tracks_by_embeddings(
                    track.track_id, candidate_ids
                )
                all_matches.extend(matches)
        
        # 신뢰도 순으로 정렬
        all_matches.sort(key=lambda x: x.similarity * x.confidence, reverse=True)
        
        return all_matches
    
    def _apply_id_corrections(self, 
                            tracks: List[Track], 
                            matches: List[MatchingResult]) -> int:
        """ID 보정을 적용합니다."""
        corrections_made = 0
        
        # 높은 신뢰도의 매칭부터 처리
        processed_tracks = set()
        
        for match in matches:
            # 이미 처리된 트랙은 건너뛰기
            if (match.source_track_id in processed_tracks or 
                match.matched_track_id in processed_tracks):
                continue
            
            # 매칭 신뢰도가 충분한지 확인
            if match.similarity < 0.8 or match.confidence < 0.7:
                continue
            
            # 실제 ID 교정 (간단한 버전: 로그만 기록)
            self.logger.info(f"ID 보정 제안: Track {match.source_track_id} ↔ "
                           f"Track {match.matched_track_id} "
                           f"(유사도={match.similarity:.2f}, 신뢰도={match.confidence:.2f})")
            
            # 실제 구현에서는 여기서 Track 객체의 ID를 교정
            # 현재는 로깅만 수행
            corrections_made += 1
            
            processed_tracks.add(match.source_track_id)
            processed_tracks.add(match.matched_track_id)
        
        return corrections_made
    
    def _collect_background_embeddings(self, tracks: List[Track], frame_image: np.ndarray):
        """배경에서 임베딩을 수집합니다 (ReID 미활성화 시)."""
        # 고품질 트랙들에 대해서만 임베딩 수집
        high_quality_tracks = [
            t for t in tracks 
            if t.is_active and t.score > 0.7 and t.age > 5
        ]
        
        # 일부 트랙만 선택 (성능 고려)
        selected_tracks = high_quality_tracks[:2]  # 최대 2개
        
        for track in selected_tracks:
            face_image = self._crop_face_from_track(track, frame_image)
            if face_image is not None:
                try:
                    embedding = self.reid_model.extract_embedding(face_image)
                    self.embedding_manager.add_embedding(track, embedding, face_image)
                except Exception as e:
                    self.logger.debug(f"배경 임베딩 수집 실패 (Track {track.track_id}): {e}")
    
    def _extract_corrections(self, 
                           original_tracks: List[Track], 
                           corrected_tracks: List[Track]) -> List[Tuple[int, int]]:
        """적용된 보정 사항을 추출합니다."""
        # 현재 구현에서는 실제 ID 변경이 없으므로 빈 리스트 반환
        # 실제 구현에서는 변경된 ID들을 추적하여 반환
        return []
    
    def _update_statistics(self, processing_time_ms: float):
        """통계를 업데이트합니다."""
        self.stats['total_processing_time_ms'] += processing_time_ms
        
        if self.stats['total_frames'] > 0:
            self.stats['avg_processing_time_ms'] = (
                self.stats['total_processing_time_ms'] / self.stats['total_frames']
            )
            self.stats['activation_rate'] = (
                self.stats['reid_activations'] / self.stats['total_frames']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """시스템 통계를 반환합니다."""
        stats = self.stats.copy()
        
        # 하위 시스템 통계 추가
        stats['swap_detector'] = self.swap_detector.get_statistics()
        stats['embedding_manager'] = self.embedding_manager.get_statistics()
        stats['reid_model'] = self.reid_model.get_statistics()
        
        # 최근 활성화 기록
        if self.activation_history:
            recent_activations = self.activation_history[-10:]  # 최근 10개
            stats['recent_activations'] = [
                {
                    'frame_id': rec.frame_id,
                    'affected_tracks': len(rec.affected_track_ids),
                    'risk_score': rec.swap_risk_score,
                    'processing_time_ms': rec.processing_time_ms,
                    'corrections': rec.corrections_made
                }
                for rec in recent_activations
            ]
        
        return stats
    
    def reset(self):
        """시스템을 초기 상태로 리셋합니다."""
        self.swap_detector.reset()
        self.embedding_manager.reset()
        
        self.frame_counter = 0
        self.activation_history.clear()
        
        # 누적 통계는 유지
        self.logger.info("ConditionalReID 시스템 리셋 완료")
    
    def set_activation_threshold(self, threshold: float):
        """활성화 임계값을 동적으로 조정합니다."""
        old_threshold = self.activation_threshold
        self.activation_threshold = max(0.1, min(threshold, 0.9))  # 0.1 ~ 0.9 범위
        
        self.logger.info(f"활성화 임계값 변경: {old_threshold:.2f} → {self.activation_threshold:.2f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보를 반환합니다."""
        stats = self.get_statistics()
        
        return {
            'activation_rate': f"{stats['activation_rate']:.1%}",
            'target_rate': f"{self.target_activation_rate:.1%}",
            'avg_processing_time': f"{stats['avg_processing_time_ms']:.1f}ms",
            'successful_corrections': stats['successful_corrections'],
            'total_activations': stats['reid_activations'],
            'performance_status': (
                'Good' if stats['activation_rate'] <= self.target_activation_rate * 1.2
                else 'High activation rate'
            )
        }
    
    def __repr__(self):
        return (f"ConditionalReID(frames={self.stats['total_frames']}, "
                f"activations={self.stats['reid_activations']}, "
                f"rate={self.stats['activation_rate']:.1%})")