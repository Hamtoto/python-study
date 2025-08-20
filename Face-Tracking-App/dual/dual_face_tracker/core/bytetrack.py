"""
ByteTrack 다중 객체 추적 알고리즘 구현.

ByteTrack은 2단계 association을 통해 높은 정확도의 다중 객체 추적을 제공합니다:
1. High-confidence detection과 track 매칭
2. Low-confidence detection으로 잃어버린 track 복구

Reference: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from .tracking_structures import Detection, Track, TrackState, separate_detections_by_confidence
from .matching import MatchingEngine, create_combined_cost_matrix
from ..utils.logger import UnifiedLogger
from ..utils.exceptions import StreamProcessingError


class ByteTracker:
    """
    ByteTrack 다중 객체 추적기.
    
    고성능 실시간 다중 객체 추적을 위한 2단계 association 알고리즘을 구현합니다.
    """
    
    def __init__(self, 
                 high_threshold: float = 0.6,
                 low_threshold: float = 0.1, 
                 iou_threshold: float = 0.3,
                 max_lost_frames: int = 30,
                 min_hits: int = 3,
                 device: str = "cuda"):
        """
        ByteTracker를 초기화합니다.
        
        Args:
            high_threshold: High confidence detection 임계값
            low_threshold: Low confidence detection 임계값 
            iou_threshold: IoU 매칭 임계값
            max_lost_frames: 트랙 제거까지의 최대 lost 프레임 수
            min_hits: 트랙 활성화까지 필요한 최소 히트 수
            device: 계산 디바이스 ("cuda" 또는 "cpu")
        """
        # 알고리즘 파라미터
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        
        # 추적 상태
        self.frame_id = 0
        self.tracks = []  # 모든 트랙 리스트
        
        # 매칭 엔진
        self.matching_engine = MatchingEngine(device=device)
        
        # 로깅
        self.logger = UnifiedLogger("bytetrack")
        
        # 통계
        self.stats = {
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_tracks_removed': 0,
            'active_tracks': 0,
            'lost_tracks': 0
        }
        
        self.logger.success(f"ByteTracker 초기화 완료 - high_th={high_threshold}, "
                           f"low_th={low_threshold}, iou_th={iou_threshold}")
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        새로운 프레임의 detection으로 트랙을 업데이트합니다.
        
        Args:
            detections: 현재 프레임의 detection 리스트
            
        Returns:
            List[Track]: 업데이트된 활성 트랙 리스트
        """
        try:
            self.frame_id += 1
            self.stats['total_detections'] += len(detections)
            
            self.logger.debug(f"프레임 {self.frame_id}: {len(detections)}개 detection 처리 시작")
            
            # 1. Detection을 confidence에 따라 분리
            high_conf_dets, low_conf_dets = separate_detections_by_confidence(
                detections, self.high_threshold
            )
            
            self.logger.debug(f"Detection 분리: high={len(high_conf_dets)}, low={len(low_conf_dets)}")
            
            # 2. 모든 트랙 예측 단계 수행
            self._predict_tracks()
            
            # 3. 활성 트랙과 잃어버린 트랙 분리
            active_tracks = [t for t in self.tracks if t.is_active]
            lost_tracks = [t for t in self.tracks if t.is_lost]
            
            self.logger.debug(f"트랙 상태: active={len(active_tracks)}, lost={len(lost_tracks)}")
            
            # 4. 첫 번째 단계: High confidence detection과 active track 매칭
            matches_1, unmatched_dets_1, unmatched_tracks_1 = self._first_association(
                high_conf_dets, active_tracks
            )
            
            # 5. 매칭된 트랙 업데이트
            for det_idx, track_idx in matches_1:
                active_tracks[track_idx].update(high_conf_dets[det_idx], self.frame_id)
            
            # 6. 두 번째 단계: Low confidence detection과 lost track 매칭
            matches_2, unmatched_dets_2, unmatched_tracks_2 = self._second_association(
                low_conf_dets, lost_tracks, unmatched_tracks_1
            )
            
            # 7. 복구된 트랙 업데이트
            for det_idx, track_idx in matches_2:
                if track_idx < len(lost_tracks):
                    # lost_tracks에서 복구
                    lost_tracks[track_idx].re_activate(low_conf_dets[det_idx], self.frame_id)
                else:
                    # unmatched active tracks에서 복구 
                    track_idx_in_unmatched = track_idx - len(lost_tracks)
                    if track_idx_in_unmatched < len(unmatched_tracks_1):
                        orig_track_idx = unmatched_tracks_1[track_idx_in_unmatched]
                        active_tracks[orig_track_idx].update(low_conf_dets[det_idx], self.frame_id)
            
            # 8. 새로운 트랙 생성 (unmatched high confidence detections)
            self._create_new_tracks(unmatched_dets_1, high_conf_dets)
            
            # 9. 트랙 생명주기 관리
            self._manage_track_lifecycle()
            
            # 10. 활성 트랙 반환
            active_tracks = [t for t in self.tracks if t.is_active and t.is_activated]
            
            # 통계 업데이트
            self._update_statistics()
            
            self.logger.debug(f"프레임 {self.frame_id} 처리 완료: {len(active_tracks)}개 활성 트랙")
            
            return active_tracks
            
        except Exception as e:
            self.logger.error(f"ByteTracker 업데이트 실패: {e}")
            raise StreamProcessingError(f"ByteTracker update failed: {e}")
    
    def _predict_tracks(self):
        """모든 트랙의 예측 단계를 수행합니다."""
        for track in self.tracks:
            if not track.is_removed:
                track.predict()
    
    def _first_association(self, 
                          high_conf_dets: List[Detection], 
                          active_tracks: List[Track]) -> Tuple[List[Tuple[int, int]], 
                                                              List[int], 
                                                              List[int]]:
        """
        첫 번째 단계: High confidence detection과 active track 매칭.
        
        Args:
            high_conf_dets: High confidence detection 리스트
            active_tracks: 활성 트랙 리스트
            
        Returns:
            Tuple: (matches, unmatched_detections, unmatched_tracks)
        """
        if not high_conf_dets or not active_tracks:
            return [], list(range(len(high_conf_dets))), list(range(len(active_tracks)))
        
        # IoU 기반 매칭
        matches, unmatched_dets, unmatched_tracks = self.matching_engine.match_detections_tracks(
            high_conf_dets, active_tracks, self.iou_threshold
        )
        
        self.logger.debug(f"1차 매칭: {len(matches)}개 매칭, "
                         f"{len(unmatched_dets)}개 미매칭 detection")
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _second_association(self, 
                           low_conf_dets: List[Detection],
                           lost_tracks: List[Track],
                           unmatched_active_tracks_idx: List[int]) -> Tuple[List[Tuple[int, int]], 
                                                                          List[int], 
                                                                          List[int]]:
        """
        두 번째 단계: Low confidence detection과 lost track + unmatched active track 매칭.
        
        Args:
            low_conf_dets: Low confidence detection 리스트
            lost_tracks: 잃어버린 트랙 리스트
            unmatched_active_tracks_idx: 1차에서 매칭되지 않은 active track 인덱스
            
        Returns:
            Tuple: (matches, unmatched_detections, unmatched_tracks)
        """
        # 매칭 대상 트랙 결합 (lost tracks + unmatched active tracks)
        combined_tracks = lost_tracks.copy()
        
        # unmatched active tracks 추가 
        active_tracks = [t for t in self.tracks if t.is_active]
        for track_idx in unmatched_active_tracks_idx:
            if track_idx < len(active_tracks):
                combined_tracks.append(active_tracks[track_idx])
        
        if not low_conf_dets or not combined_tracks:
            return [], list(range(len(low_conf_dets))), list(range(len(combined_tracks)))
        
        # 낮은 임계값으로 매칭 (더 관대한 매칭)
        lower_iou_threshold = max(0.1, self.iou_threshold - 0.1)
        matches, unmatched_dets, unmatched_tracks = self.matching_engine.match_detections_tracks(
            low_conf_dets, combined_tracks, lower_iou_threshold
        )
        
        self.logger.debug(f"2차 매칭: {len(matches)}개 매칭, "
                         f"{len(unmatched_dets)}개 미매칭 detection")
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _create_new_tracks(self, unmatched_det_indices: List[int], detections: List[Detection]):
        """
        매칭되지 않은 high confidence detection으로 새로운 트랙을 생성합니다.
        
        Args:
            unmatched_det_indices: 매칭되지 않은 detection 인덱스 리스트
            detections: Detection 리스트
        """
        for det_idx in unmatched_det_indices:
            if det_idx < len(detections):
                detection = detections[det_idx]
                
                # 새 트랙 생성
                new_track = Track(detection, self.frame_id)
                
                # confidence가 충분히 높으면 즉시 활성화
                if detection.confidence >= self.high_threshold:
                    new_track.activate(self.frame_id)
                
                self.tracks.append(new_track)
                self.stats['total_tracks_created'] += 1
                
                self.logger.debug(f"새 트랙 생성: ID={new_track.track_id}, conf={detection.confidence:.2f}")
    
    def _manage_track_lifecycle(self):
        """트랙의 생명주기를 관리합니다."""
        tracks_to_remove = []
        
        for i, track in enumerate(self.tracks):
            # 활성화 조건 확인
            if track.state == TrackState.NEW and track.hits >= self.min_hits:
                track.activate(self.frame_id)
                self.logger.debug(f"트랙 활성화: ID={track.track_id}")
            
            # 제거 조건 확인
            if track.time_since_update >= self.max_lost_frames:
                track.mark_removed()
                tracks_to_remove.append(i)
                self.stats['total_tracks_removed'] += 1
                self.logger.debug(f"트랙 제거: ID={track.track_id}")
            
            # Lost 상태로 전환 조건
            elif track.time_since_update > 5 and track.is_active:
                track.mark_lost()
        
        # 제거된 트랙들을 리스트에서 삭제 (뒤에서부터)
        for i in reversed(tracks_to_remove):
            del self.tracks[i]
    
    def _update_statistics(self):
        """통계 정보를 업데이트합니다."""
        self.stats['active_tracks'] = sum(1 for t in self.tracks if t.is_active and t.is_activated)
        self.stats['lost_tracks'] = sum(1 for t in self.tracks if t.is_lost)
    
    def get_active_tracks(self) -> List[Track]:
        """
        현재 활성 트랙 리스트를 반환합니다.
        
        Returns:
            List[Track]: 활성 트랙 리스트
        """
        return [t for t in self.tracks if t.is_active and t.is_activated]
    
    def get_all_tracks(self) -> List[Track]:
        """
        모든 트랙 리스트를 반환합니다.
        
        Returns:
            List[Track]: 모든 트랙 리스트
        """
        return self.tracks.copy()
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        특정 ID의 트랙을 찾아 반환합니다.
        
        Args:
            track_id: 찾을 트랙 ID
            
        Returns:
            Optional[Track]: 찾은 트랙 또는 None
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_statistics(self) -> Dict:
        """
        추적 통계를 반환합니다.
        
        Returns:
            Dict: 통계 정보 딕셔너리
        """
        stats = self.stats.copy()
        stats['current_frame'] = self.frame_id
        stats['total_tracks'] = len(self.tracks)
        return stats
    
    def reset(self):
        """트래커를 초기 상태로 리셋합니다."""
        self.frame_id = 0
        self.tracks.clear()
        Track._next_id = 1  # 트랙 ID 카운터 리셋
        
        # 통계 리셋
        self.stats = {
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_tracks_removed': 0,
            'active_tracks': 0,
            'lost_tracks': 0
        }
        
        self.logger.info("ByteTracker 리셋 완료")
    
    def set_thresholds(self, high_threshold: float, low_threshold: float, iou_threshold: float):
        """
        임계값들을 동적으로 조정합니다.
        
        Args:
            high_threshold: High confidence 임계값
            low_threshold: Low confidence 임계값
            iou_threshold: IoU 임계값
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        
        self.logger.info(f"임계값 업데이트: high={high_threshold}, "
                        f"low={low_threshold}, iou={iou_threshold}")
    
    def __len__(self) -> int:
        """현재 활성 트랙의 수를 반환합니다."""
        return self.stats['active_tracks']
    
    def __repr__(self):
        return (f"ByteTracker(frame={self.frame_id}, "
                f"active_tracks={self.stats['active_tracks']}, "
                f"total_tracks={len(self.tracks)})")


class ByteTrackConfig:
    """ByteTracker 설정을 관리하는 클래스."""
    
    def __init__(self):
        # 기본 설정값들
        self.high_threshold = 0.6      # High confidence detection 임계값
        self.low_threshold = 0.1       # Low confidence detection 임계값
        self.iou_threshold = 0.3       # IoU 매칭 임계값
        self.max_lost_frames = 30      # 트랙 제거까지 최대 lost 프레임
        self.min_hits = 3              # 트랙 활성화까지 최소 히트 수
        self.device = "cuda"           # 계산 디바이스
    
    @classmethod
    def for_face_tracking(cls):
        """얼굴 추적에 최적화된 설정을 반환합니다."""
        config = cls()
        config.high_threshold = 0.7    # 얼굴은 더 높은 임계값 사용
        config.low_threshold = 0.3     # 낮은 임계값도 상향 조정
        config.iou_threshold = 0.4     # 얼굴 크기 변화를 고려해 높게 설정
        config.max_lost_frames = 15    # 얼굴은 빨리 사라질 수 있으므로 낮게
        config.min_hits = 2            # 빠른 활성화
        return config
    
    @classmethod 
    def for_high_fps(cls):
        """고속 처리를 위한 설정을 반환합니다."""
        config = cls()
        config.high_threshold = 0.8    # 높은 임계값으로 확실한 detection만
        config.low_threshold = 0.4     # 복구 임계값도 높게
        config.max_lost_frames = 10    # 빠른 제거
        config.min_hits = 1            # 즉시 활성화
        return config
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 반환합니다."""
        return {
            'high_threshold': self.high_threshold,
            'low_threshold': self.low_threshold, 
            'iou_threshold': self.iou_threshold,
            'max_lost_frames': self.max_lost_frames,
            'min_hits': self.min_hits,
            'device': self.device
        }