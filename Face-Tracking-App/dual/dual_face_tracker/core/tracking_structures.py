"""
ByteTrack 추적 시스템을 위한 데이터 구조.

Track, TrackState 등 다중 객체 추적에 필요한 
기본 데이터 구조들을 정의합니다.
"""

from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque


class TrackState(Enum):
    """트랙 상태 열거형."""
    NEW = 1      # 새로 생성된 트랙 (아직 확정되지 않음)
    TRACKED = 2  # 활성 추적 중인 트랙
    LOST = 3     # 일시적으로 놓친 트랙 (복구 가능)
    REMOVED = 4  # 제거된 트랙 (더 이상 사용 안 함)


@dataclass
class Detection:
    """
    Face Detection 결과를 나타내는 데이터 클래스.
    
    FaceDetector의 출력과 호환되도록 설계되었습니다.
    """
    # 바운딩 박스 좌표 (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]
    
    # Detection confidence score (0.0 ~ 1.0)
    confidence: float
    
    # 클래스 ID (YOLO person class = 0)
    class_id: int = 0
    
    # 추가 속성들
    area: Optional[float] = None
    center: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        """Detection 생성 후 추가 속성 계산."""
        x1, y1, x2, y2 = self.bbox
        
        # 바운딩 박스 면적 계산
        if self.area is None:
            self.area = (x2 - x1) * (y2 - y1)
            
        # 바운딩 박스 중심점 계산
        if self.center is None:
            self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def x1(self) -> float:
        return self.bbox[0]
    
    @property
    def y1(self) -> float:
        return self.bbox[1]
    
    @property
    def x2(self) -> float:
        return self.bbox[2]
    
    @property
    def y2(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Track:
    """
    개별 객체의 추적 정보를 관리하는 클래스.
    
    ByteTrack 알고리즘에서 각 객체의 상태, 이력, 예측 위치 등을
    관리합니다.
    """
    
    # 전역 트랙 ID 카운터
    _next_id = 1
    
    def __init__(self, detection: Detection, frame_id: int):
        """
        새로운 트랙을 초기화합니다.
        
        Args:
            detection: 초기 detection 정보
            frame_id: 트랙이 생성된 프레임 ID
        """
        # 고유 트랙 ID 할당
        self.track_id = Track._next_id
        Track._next_id += 1
        
        # 트랙 상태 초기화
        self.state = TrackState.NEW
        
        # 위치 정보
        self.mean = np.array([
            detection.center[0],  # center_x
            detection.center[1],  # center_y
            detection.area,       # area
            detection.width / detection.height  # aspect ratio
        ], dtype=np.float32)
        
        # 현재 detection
        self.detection = detection
        
        # 시간 정보
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        
        # 이력 관리 (최대 30프레임)
        self.history = deque(maxlen=30)
        self.history.append(detection)
        
        # 트랙 통계
        self.hits = 1          # 성공적인 매칭 횟수
        self.hit_streak = 1    # 연속 매칭 횟수
        self.age = 1           # 트랙 생성 후 경과 프레임
        
        # ByteTrack 관련 파라미터
        self.is_activated = False  # 트랙 활성화 여부
        self.score = detection.confidence
    
    def update(self, detection: Detection, frame_id: int):
        """
        새로운 detection으로 트랙을 업데이트합니다.
        
        Args:
            detection: 새로운 detection 정보
            frame_id: 현재 프레임 ID
        """
        self.detection = detection
        self.frame_id = frame_id
        self.time_since_update = 0
        
        # 이력에 추가
        self.history.append(detection)
        
        # 상태 업데이트
        if self.state == TrackState.LOST:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.NEW:
            self.state = TrackState.TRACKED
            self.is_activated = True
        
        # 통계 업데이트
        self.hits += 1
        self.hit_streak += 1
        self.score = detection.confidence
        
        # 위치 정보 업데이트 (단순한 EMA 사용)
        alpha = 0.7  # 새로운 관측값의 가중치
        new_mean = np.array([
            detection.center[0],
            detection.center[1], 
            detection.area,
            detection.width / detection.height
        ], dtype=np.float32)
        
        self.mean = alpha * new_mean + (1 - alpha) * self.mean
    
    def predict(self):
        """
        다음 프레임에서의 위치를 예측합니다.
        
        단순한 등속 운동 모델을 사용합니다.
        """
        self.age += 1
        self.time_since_update += 1
        
        # 연속 매칭이 끊어진 경우
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        # 너무 오래 업데이트되지 않은 경우 LOST 상태로 변경
        if self.time_since_update > 5:  # 5프레임
            self.state = TrackState.LOST
            
        # 완전히 제거해야 하는 경우
        if self.time_since_update > 30:  # 30프레임 (1초)
            self.state = TrackState.REMOVED
    
    def activate(self, frame_id: int):
        """트랙을 활성화합니다."""
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
    
    def re_activate(self, detection: Detection, frame_id: int):
        """
        잃어버린 트랙을 다시 활성화합니다.
        
        Args:
            detection: 복구된 detection
            frame_id: 현재 프레임 ID
        """
        self.update(detection, frame_id)
        self.state = TrackState.TRACKED
        self.is_activated = True
    
    def mark_lost(self):
        """트랙을 LOST 상태로 표시합니다."""
        self.state = TrackState.LOST
    
    def mark_removed(self):
        """트랙을 REMOVED 상태로 표시합니다."""
        self.state = TrackState.REMOVED
    
    @property 
    def is_active(self) -> bool:
        """트랙이 활성 상태인지 확인합니다."""
        return self.state == TrackState.TRACKED
    
    @property
    def is_lost(self) -> bool:
        """트랙이 잃어버린 상태인지 확인합니다."""
        return self.state == TrackState.LOST
        
    @property
    def is_removed(self) -> bool:
        """트랙이 제거된 상태인지 확인합니다."""
        return self.state == TrackState.REMOVED
    
    @property
    def tlbr(self) -> np.ndarray:
        """
        현재 예측 위치를 (top, left, bottom, right) 형태로 반환합니다.
        
        Returns:
            np.ndarray: [x1, y1, x2, y2] 바운딩 박스
        """
        if self.detection:
            return np.array([
                self.detection.x1,
                self.detection.y1, 
                self.detection.x2,
                self.detection.y2
            ])
        else:
            # Detection이 없는 경우 mean으로부터 추정
            cx, cy, area, aspect = self.mean
            w = np.sqrt(area * aspect)
            h = area / w
            return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    @property 
    def center_point(self) -> Tuple[float, float]:
        """트랙의 중심점을 반환합니다."""
        return (self.mean[0], self.mean[1])
    
    def __repr__(self):
        return f"Track(id={self.track_id}, state={self.state.name}, score={self.score:.2f})"


def convert_detection_from_face_detector(face_det_result) -> List[Detection]:
    """
    FaceDetector의 출력을 Detection 객체 리스트로 변환합니다.
    
    Args:
        face_det_result: FaceDetector.detect() 메서드의 반환값
        
    Returns:
        List[Detection]: 변환된 Detection 객체 리스트
    """
    detections = []
    
    for det in face_det_result:
        # FaceDetector의 출력 형태에 맞게 변환
        # 실제 FaceDetector 출력 구조에 따라 조정 필요
        if hasattr(det, 'bbox') and hasattr(det, 'confidence'):
            detection = Detection(
                bbox=tuple(det.bbox),
                confidence=det.confidence,
                class_id=0  # person class
            )
            detections.append(detection)
    
    return detections


# 유틸리티 함수들
def filter_detections_by_confidence(detections: List[Detection], 
                                  min_confidence: float = 0.3) -> List[Detection]:
    """
    Confidence threshold로 detection을 필터링합니다.
    
    Args:
        detections: Detection 리스트
        min_confidence: 최소 confidence 값
        
    Returns:
        List[Detection]: 필터링된 Detection 리스트
    """
    return [det for det in detections if det.confidence >= min_confidence]


def separate_detections_by_confidence(detections: List[Detection], 
                                    high_threshold: float = 0.6) -> Tuple[List[Detection], List[Detection]]:
    """
    Detection을 confidence에 따라 high/low로 분리합니다.
    
    ByteTrack 알고리즘의 2단계 매칭을 위한 함수입니다.
    
    Args:
        detections: Detection 리스트
        high_threshold: High confidence 임계값
        
    Returns:
        Tuple[List[Detection], List[Detection]]: (high_conf_dets, low_conf_dets)
    """
    high_conf_dets = []
    low_conf_dets = []
    
    for det in detections:
        if det.confidence >= high_threshold:
            high_conf_dets.append(det)
        else:
            low_conf_dets.append(det)
            
    return high_conf_dets, low_conf_dets