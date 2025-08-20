"""
ID 스왑 감지 시스템.

ByteTracker의 추적 결과에서 ID 스왑이 발생했을 가능성을 감지합니다.
3가지 주요 지표를 사용하여 스왑 위험도를 계산합니다:
1. 위치 급변 감지 (Position Jump Detection)
2. 크기 급변 감지 (Size Change Detection) 
3. 교차 상황 감지 (Crossing Detection)
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque, defaultdict
import numpy as np
import math

from .tracking_structures import Track, TrackState
from ..utils.logger import UnifiedLogger


@dataclass
class SwapIndicator:
    """ID 스왑 지표를 나타내는 데이터 클래스."""
    
    indicator_type: str        # 지표 타입 ("position", "size", "crossing")
    risk_score: float         # 위험도 점수 (0.0 ~ 1.0)
    confidence: float         # 신뢰도 (0.0 ~ 1.0)
    track_ids: List[int]      # 관련된 트랙 ID들
    details: Dict             # 추가 상세 정보
    
    def __repr__(self):
        return f"SwapIndicator({self.indicator_type}, risk={self.risk_score:.2f}, conf={self.confidence:.2f})"


@dataclass
class SwapDetectionResult:
    """ID 스왑 감지 결과."""
    
    has_swap: bool                      # 스왑 감지 여부
    overall_risk_score: float          # 종합 위험도 점수 (0.0 ~ 1.0)
    confidence: float                   # 종합 신뢰도 (0.0 ~ 1.0)
    indicators: List[SwapIndicator]     # 개별 지표들
    affected_track_ids: Set[int]        # 영향받은 트랙 ID들
    recommendation: str                 # 권장 조치
    
    def __repr__(self):
        return f"SwapDetectionResult(swap={self.has_swap}, risk={self.overall_risk_score:.2f})"


class IDSwapDetector:
    """
    ID 스왑 감지기 클래스.
    
    다중 지표를 사용하여 ID 스왑 상황을 감지하고
    ConditionalReID 활성화 여부를 결정합니다.
    """
    
    def __init__(self,
                 position_threshold: float = 100.0,      # 픽셀 단위 위치 변화 임계값
                 size_change_threshold: float = 0.5,     # 크기 변화 비율 임계값 (50%)
                 crossing_threshold: float = 20.0,       # 교차 감지 임계값 (픽셀)
                 history_length: int = 10,               # 히스토리 길이 (프레임)
                 overall_threshold: float = 0.6,         # 종합 위험도 임계값
                 min_track_age: int = 3):                # 최소 트랙 나이 (프레임)
        """
        ID 스왑 감지기를 초기화합니다.
        
        Args:
            position_threshold: 위치 급변 감지 임계값 (픽셀)
            size_change_threshold: 크기 변화 감지 임계값 (비율)
            crossing_threshold: 교차 감지 임계값 (픽셀)
            history_length: 추적 히스토리 길이 (프레임)
            overall_threshold: 종합 위험도 임계값
            min_track_age: 스왑 감지 대상 최소 트랙 나이
        """
        self.position_threshold = position_threshold
        self.size_change_threshold = size_change_threshold
        self.crossing_threshold = crossing_threshold
        self.history_length = history_length
        self.overall_threshold = overall_threshold
        self.min_track_age = min_track_age
        
        # 트랙 히스토리 관리
        self.track_histories = defaultdict(lambda: deque(maxlen=history_length))
        self.previous_tracks = {}
        
        # 통계
        self.stats = {
            'total_detections': 0,
            'swap_detections': 0,
            'false_positives': 0,
            'position_triggers': 0,
            'size_triggers': 0,
            'crossing_triggers': 0
        }
        
        self.logger = UnifiedLogger("IDSwapDetector")
        self.logger.info(f"ID 스왑 감지기 초기화: pos_th={position_threshold}, "
                        f"size_th={size_change_threshold:.1f}, cross_th={crossing_threshold}")
    
    def detect_swaps(self, current_tracks: List[Track]) -> SwapDetectionResult:
        """
        현재 트랙들에서 ID 스왑을 감지합니다.
        
        Args:
            current_tracks: 현재 프레임의 트랙 리스트
            
        Returns:
            SwapDetectionResult: 스왑 감지 결과
        """
        try:
            self.stats['total_detections'] += 1
            
            # 트랙 히스토리 업데이트
            self._update_track_histories(current_tracks)
            
            # 각 지표별 스왑 감지
            indicators = []
            
            # 1. 위치 급변 감지
            position_indicators = self._detect_position_jumps(current_tracks)
            indicators.extend(position_indicators)
            
            # 2. 크기 급변 감지
            size_indicators = self._detect_size_changes(current_tracks)
            indicators.extend(size_indicators)
            
            # 3. 교차 상황 감지
            crossing_indicators = self._detect_crossings(current_tracks)
            indicators.extend(crossing_indicators)
            
            # 종합 위험도 계산
            result = self._calculate_overall_risk(indicators)
            
            # 이전 트랙 정보 업데이트
            self._update_previous_tracks(current_tracks)
            
            # 통계 업데이트
            if result.has_swap:
                self.stats['swap_detections'] += 1
            
            self.logger.debug(f"스왑 감지 완료: {len(indicators)}개 지표, "
                             f"위험도={result.overall_risk_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"ID 스왑 감지 실패: {e}")
            # 실패 시 안전한 기본값 반환
            return SwapDetectionResult(
                has_swap=False,
                overall_risk_score=0.0,
                confidence=0.0,
                indicators=[],
                affected_track_ids=set(),
                recommendation="Detection failed - using safe defaults"
            )
    
    def _update_track_histories(self, current_tracks: List[Track]):
        """트랙 히스토리를 업데이트합니다."""
        for track in current_tracks:
            if track.is_active and track.age >= self.min_track_age:
                # 트랙 위치 및 크기 정보 저장
                track_info = {
                    'center': track.center_point,
                    'bbox': track.tlbr,
                    'area': track.detection.area if track.detection else 0,
                    'frame_id': track.frame_id,
                    'confidence': track.score
                }
                self.track_histories[track.track_id].append(track_info)
    
    def _detect_position_jumps(self, current_tracks: List[Track]) -> List[SwapIndicator]:
        """위치 급변을 감지합니다."""
        indicators = []
        
        for track in current_tracks:
            if (track.track_id not in self.previous_tracks or 
                track.age < self.min_track_age):
                continue
                
            prev_track = self.previous_tracks[track.track_id]
            
            # 현재와 이전 위치 비교
            curr_center = track.center_point
            prev_center = prev_track.get('center', curr_center)
            
            # 유클리드 거리 계산
            distance = math.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            
            # 임계값 초과 시 지표 생성
            if distance > self.position_threshold:
                risk_score = min(distance / (self.position_threshold * 2), 1.0)
                confidence = self._calculate_position_confidence(track, distance)
                
                indicator = SwapIndicator(
                    indicator_type="position",
                    risk_score=risk_score,
                    confidence=confidence,
                    track_ids=[track.track_id],
                    details={
                        'distance': distance,
                        'threshold': self.position_threshold,
                        'prev_center': prev_center,
                        'curr_center': curr_center
                    }
                )
                indicators.append(indicator)
                self.stats['position_triggers'] += 1
                
                self.logger.debug(f"위치 급변 감지: Track {track.track_id}, "
                                 f"거리={distance:.1f}px > {self.position_threshold}px")
        
        return indicators
    
    def _detect_size_changes(self, current_tracks: List[Track]) -> List[SwapIndicator]:
        """크기 급변을 감지합니다."""
        indicators = []
        
        for track in current_tracks:
            if (track.track_id not in self.previous_tracks or 
                track.age < self.min_track_age or
                not track.detection):
                continue
                
            prev_track = self.previous_tracks[track.track_id]
            
            # 현재와 이전 면적 비교
            curr_area = track.detection.area
            prev_area = prev_track.get('area', curr_area)
            
            if prev_area <= 0 or curr_area <= 0:
                continue
            
            # 크기 변화 비율 계산
            size_ratio = abs(curr_area - prev_area) / max(curr_area, prev_area)
            
            # 임계값 초과 시 지표 생성
            if size_ratio > self.size_change_threshold:
                risk_score = min(size_ratio / (self.size_change_threshold * 1.5), 1.0)
                confidence = self._calculate_size_confidence(track, size_ratio)
                
                indicator = SwapIndicator(
                    indicator_type="size",
                    risk_score=risk_score,
                    confidence=confidence,
                    track_ids=[track.track_id],
                    details={
                        'size_ratio': size_ratio,
                        'threshold': self.size_change_threshold,
                        'prev_area': prev_area,
                        'curr_area': curr_area
                    }
                )
                indicators.append(indicator)
                self.stats['size_triggers'] += 1
                
                self.logger.debug(f"크기 급변 감지: Track {track.track_id}, "
                                 f"변화율={size_ratio:.2f} > {self.size_change_threshold:.2f}")
        
        return indicators
    
    def _detect_crossings(self, current_tracks: List[Track]) -> List[SwapIndicator]:
        """트랙 교차 상황을 감지합니다."""
        indicators = []
        
        # 활성 트랙들만 대상
        active_tracks = [t for t in current_tracks if t.is_active and t.age >= self.min_track_age]
        
        # 모든 트랙 쌍에 대해 교차 검사
        for i, track1 in enumerate(active_tracks):
            for track2 in active_tracks[i+1:]:
                
                # 이전 프레임과 현재 프레임에서의 위치 관계 확인
                crossing_score = self._calculate_crossing_score(track1, track2)
                
                if crossing_score > 0.5:  # 교차 가능성이 있는 경우
                    risk_score = min(crossing_score, 1.0)
                    confidence = self._calculate_crossing_confidence(track1, track2)
                    
                    indicator = SwapIndicator(
                        indicator_type="crossing",
                        risk_score=risk_score,
                        confidence=confidence,
                        track_ids=[track1.track_id, track2.track_id],
                        details={
                            'crossing_score': crossing_score,
                            'track1_center': track1.center_point,
                            'track2_center': track2.center_point,
                            'distance': self._calculate_distance(track1.center_point, track2.center_point)
                        }
                    )
                    indicators.append(indicator)
                    self.stats['crossing_triggers'] += 1
                    
                    self.logger.debug(f"교차 감지: Track {track1.track_id} & {track2.track_id}, "
                                     f"교차점수={crossing_score:.2f}")
        
        return indicators
    
    def _calculate_crossing_score(self, track1: Track, track2: Track) -> float:
        """두 트랙 간의 교차 점수를 계산합니다."""
        # 현재 거리
        curr_distance = self._calculate_distance(track1.center_point, track2.center_point)
        
        # 너무 멀리 떨어져 있으면 교차 가능성 없음
        if curr_distance > self.crossing_threshold * 3:
            return 0.0
        
        # 이전 프레임 정보가 있는지 확인
        if (track1.track_id not in self.previous_tracks or 
            track2.track_id not in self.previous_tracks):
            return 0.0
        
        # 이전 위치 정보
        prev1 = self.previous_tracks[track1.track_id].get('center', track1.center_point)
        prev2 = self.previous_tracks[track2.track_id].get('center', track2.center_point)
        prev_distance = self._calculate_distance(prev1, prev2)
        
        # 거리 변화가 거의 없으면 교차 가능성 낮음
        if abs(curr_distance - prev_distance) < 5.0:
            return 0.0
        
        # 선분 교차 검사 (간단한 버전)
        crossing_detected = self._check_line_intersection(
            prev1, track1.center_point,  # track1의 이동 경로
            prev2, track2.center_point   # track2의 이동 경로
        )
        
        if crossing_detected:
            # 교차 시 거리 기반으로 점수 계산
            base_score = 0.8
            distance_factor = max(0, 1.0 - curr_distance / (self.crossing_threshold * 2))
            return base_score + (0.2 * distance_factor)
        
        # 가까워지는 경우에 대한 처리
        if curr_distance < self.crossing_threshold and curr_distance < prev_distance:
            return 0.3 + (0.3 * (1.0 - curr_distance / self.crossing_threshold))
        
        return 0.0
    
    def _check_line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float],
                               p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """두 선분의 교차 여부를 확인합니다."""
        # 벡터 외적을 이용한 교차 검사
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """두 점 사이의 유클리드 거리를 계산합니다."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_position_confidence(self, track: Track, distance: float) -> float:
        """위치 급변 지표의 신뢰도를 계산합니다."""
        # 트랙의 나이가 많을수록 신뢰도 높음
        age_factor = min(track.age / 10.0, 1.0)
        
        # detection confidence가 높을수록 신뢰도 높음
        conf_factor = track.score
        
        # 거리가 극단적일수록 신뢰도 높음
        distance_factor = min(distance / (self.position_threshold * 3), 1.0)
        
        return (age_factor * 0.3 + conf_factor * 0.4 + distance_factor * 0.3)
    
    def _calculate_size_confidence(self, track: Track, size_ratio: float) -> float:
        """크기 급변 지표의 신뢰도를 계산합니다."""
        # 트랙의 나이와 confidence 기반
        age_factor = min(track.age / 10.0, 1.0)
        conf_factor = track.score
        
        # 크기 변화가 극단적일수록 신뢰도 높음
        size_factor = min(size_ratio / (self.size_change_threshold * 2), 1.0)
        
        return (age_factor * 0.3 + conf_factor * 0.3 + size_factor * 0.4)
    
    def _calculate_crossing_confidence(self, track1: Track, track2: Track) -> float:
        """교차 지표의 신뢰도를 계산합니다."""
        # 두 트랙의 평균 나이와 confidence
        avg_age = (track1.age + track2.age) / 2.0
        avg_conf = (track1.score + track2.score) / 2.0
        
        age_factor = min(avg_age / 10.0, 1.0)
        conf_factor = avg_conf
        
        return (age_factor * 0.5 + conf_factor * 0.5)
    
    def _calculate_overall_risk(self, indicators: List[SwapIndicator]) -> SwapDetectionResult:
        """전체 위험도를 계산하여 최종 결과를 생성합니다."""
        if not indicators:
            return SwapDetectionResult(
                has_swap=False,
                overall_risk_score=0.0,
                confidence=1.0,
                indicators=[],
                affected_track_ids=set(),
                recommendation="No swap indicators detected"
            )
        
        # 가중 평균으로 종합 위험도 계산
        total_weighted_risk = 0.0
        total_weight = 0.0
        
        # 지표 타입별 가중치
        weights = {
            'position': 0.4,
            'size': 0.3, 
            'crossing': 0.3
        }
        
        affected_tracks = set()
        
        for indicator in indicators:
            weight = weights.get(indicator.indicator_type, 0.2)
            weighted_risk = indicator.risk_score * indicator.confidence * weight
            
            total_weighted_risk += weighted_risk
            total_weight += weight
            
            affected_tracks.update(indicator.track_ids)
        
        # 정규화
        if total_weight > 0:
            overall_risk = total_weighted_risk / total_weight
        else:
            overall_risk = 0.0
        
        # 종합 신뢰도 계산
        overall_confidence = np.mean([ind.confidence for ind in indicators])
        
        # 스왑 여부 결정
        has_swap = overall_risk > self.overall_threshold
        
        # 권장 조치 결정
        recommendation = self._generate_recommendation(overall_risk, indicators)
        
        return SwapDetectionResult(
            has_swap=has_swap,
            overall_risk_score=overall_risk,
            confidence=overall_confidence,
            indicators=indicators,
            affected_track_ids=affected_tracks,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, risk_score: float, indicators: List[SwapIndicator]) -> str:
        """위험도와 지표를 바탕으로 권장 조치를 생성합니다."""
        if risk_score < 0.3:
            return "Continue normal tracking"
        elif risk_score < 0.6:
            return "Monitor closely - potential swap"
        elif risk_score < 0.8:
            return "Activate ReID - likely swap detected"
        else:
            return "Immediate ReID activation - high swap confidence"
    
    def _update_previous_tracks(self, current_tracks: List[Track]):
        """이전 트랙 정보를 업데이트합니다."""
        self.previous_tracks.clear()
        
        for track in current_tracks:
            if track.is_active:
                self.previous_tracks[track.track_id] = {
                    'center': track.center_point,
                    'bbox': track.tlbr,
                    'area': track.detection.area if track.detection else 0,
                    'frame_id': track.frame_id,
                    'confidence': track.score
                }
    
    def reset(self):
        """감지기를 초기 상태로 리셋합니다."""
        self.track_histories.clear()
        self.previous_tracks.clear()
        
        # 통계는 유지 (전체 세션 통계)
        self.logger.info("ID 스왑 감지기 리셋 완료")
    
    def get_statistics(self) -> Dict:
        """감지기 통계를 반환합니다."""
        stats = self.stats.copy()
        
        if stats['total_detections'] > 0:
            stats['swap_rate'] = stats['swap_detections'] / stats['total_detections']
        else:
            stats['swap_rate'] = 0.0
            
        return stats
    
    def set_thresholds(self, 
                      position_threshold: Optional[float] = None,
                      size_change_threshold: Optional[float] = None,
                      crossing_threshold: Optional[float] = None,
                      overall_threshold: Optional[float] = None):
        """임계값들을 동적으로 조정합니다."""
        if position_threshold is not None:
            self.position_threshold = position_threshold
        if size_change_threshold is not None:
            self.size_change_threshold = size_change_threshold
        if crossing_threshold is not None:
            self.crossing_threshold = crossing_threshold
        if overall_threshold is not None:
            self.overall_threshold = overall_threshold
            
        self.logger.info(f"임계값 업데이트: pos={self.position_threshold}, "
                        f"size={self.size_change_threshold:.2f}, "
                        f"cross={self.crossing_threshold}, "
                        f"overall={self.overall_threshold:.2f}")
    
    def __repr__(self):
        return (f"IDSwapDetector(detections={self.stats['total_detections']}, "
                f"swaps={self.stats['swap_detections']}, "
                f"rate={self.stats.get('swap_rate', 0):.2f})")