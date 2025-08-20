"""
ByteTrack을 위한 매칭 알고리즘 구현.

IoU(Intersection over Union) 계산과 Hungarian 알고리즘을 사용한
최적 할당을 통해 Detection과 Track을 매칭합니다.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from scipy.optimize import linear_sum_assignment

from .tracking_structures import Detection, Track
from ..utils.logger import UnifiedLogger


class MatchingEngine:
    """
    Detection과 Track 간의 매칭을 수행하는 엔진 클래스.
    
    GPU 가속 IoU 계산과 Hungarian 알고리즘을 사용하여
    최적의 매칭을 찾습니다.
    """
    
    def __init__(self, device: str = "cuda", use_gpu: bool = True):
        """
        매칭 엔진을 초기화합니다.
        
        Args:
            device: 계산에 사용할 디바이스 ("cuda" 또는 "cpu")
            use_gpu: GPU 가속 사용 여부
        """
        self.device = device if use_gpu and torch.cuda.is_available() else "cpu"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = UnifiedLogger("matching_engine")
        
        self.logger.debug(f"매칭 엔진 초기화: device={self.device}, use_gpu={self.use_gpu}")
    
    def calculate_iou_matrix(self, 
                           detections: List[Detection], 
                           tracks: List[Track]) -> np.ndarray:
        """
        Detection과 Track 간의 IoU 행렬을 계산합니다.
        
        Args:
            detections: Detection 리스트
            tracks: Track 리스트
            
        Returns:
            np.ndarray: IoU 행렬 (detections x tracks)
        """
        if not detections or not tracks:
            return np.empty((len(detections), len(tracks)), dtype=np.float32)
        
        if self.use_gpu:
            return self._calculate_iou_matrix_gpu(detections, tracks)
        else:
            return self._calculate_iou_matrix_cpu(detections, tracks)
    
    def _calculate_iou_matrix_gpu(self, 
                                detections: List[Detection], 
                                tracks: List[Track]) -> np.ndarray:
        """
        GPU를 사용한 배치 IoU 계산.
        
        Args:
            detections: Detection 리스트
            tracks: Track 리스트
            
        Returns:
            np.ndarray: IoU 행렬
        """
        # Detection 바운딩 박스를 tensor로 변환
        det_boxes = torch.tensor([
            [det.x1, det.y1, det.x2, det.y2] for det in detections
        ], device=self.device, dtype=torch.float32)
        
        # Track 바운딩 박스를 tensor로 변환
        track_boxes = torch.tensor([
            track.tlbr for track in tracks
        ], device=self.device, dtype=torch.float32)
        
        # 배치 IoU 계산
        iou_matrix = self._batch_iou_gpu(det_boxes, track_boxes)
        
        return iou_matrix.cpu().numpy()
    
    def _calculate_iou_matrix_cpu(self, 
                                detections: List[Detection], 
                                tracks: List[Track]) -> np.ndarray:
        """
        CPU를 사용한 IoU 계산.
        
        Args:
            detections: Detection 리스트
            tracks: Track 리스트
            
        Returns:
            np.ndarray: IoU 행렬
        """
        n_dets = len(detections)
        n_tracks = len(tracks)
        iou_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)
        
        for i, det in enumerate(detections):
            det_box = np.array([det.x1, det.y1, det.x2, det.y2])
            for j, track in enumerate(tracks):
                track_box = track.tlbr
                iou_matrix[i, j] = self._calculate_single_iou(det_box, track_box)
        
        return iou_matrix
    
    def _batch_iou_gpu(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        GPU에서 배치 IoU 계산.
        
        Args:
            boxes1: 첫 번째 박스 집합 (N, 4) - [x1, y1, x2, y2]
            boxes2: 두 번째 박스 집합 (M, 4) - [x1, y1, x2, y2]
            
        Returns:
            torch.Tensor: IoU 행렬 (N, M)
        """
        # 면적 계산
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
        
        # 교집합 계산을 위한 차원 확장
        boxes1_expanded = boxes1.unsqueeze(1)  # (N, 1, 4)
        boxes2_expanded = boxes2.unsqueeze(0)  # (1, M, 4)
        
        # 교집합 영역의 좌표 계산
        inter_x1 = torch.max(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
        inter_y1 = torch.max(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
        inter_x2 = torch.min(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
        inter_y2 = torch.min(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
        
        # 교집합 면적 계산
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height  # (N, M)
        
        # 합집합 면적 계산
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area  # (N, M)
        
        # IoU 계산 (0으로 나누기 방지)
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou
    
    def _calculate_single_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        두 바운딩 박스 간의 IoU를 계산합니다.
        
        Args:
            box1: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
            box2: 두 번째 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            float: IoU 값 (0.0 ~ 1.0)
        """
        # 교집합 영역 계산
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 교집합이 없는 경우
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        # 교집합 면적
        intersection = (x2 - x1) * (y2 - y1)
        
        # 각 박스의 면적
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 합집합 면적
        union = area1 + area2 - intersection
        
        # IoU 계산 (0으로 나누기 방지)
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def hungarian_matching(self, 
                          cost_matrix: np.ndarray, 
                          max_distance: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hungarian 알고리즘을 사용하여 최적 할당을 수행합니다.
        
        Args:
            cost_matrix: 비용 행렬 (거리 행렬, IoU의 경우 1-IoU 사용)
            max_distance: 최대 허용 거리 (이보다 큰 거리는 매칭하지 않음)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - matched_indices: 매칭된 (detection_idx, track_idx) 쌍
                - unmatched_detections: 매칭되지 않은 detection 인덱스
                - unmatched_tracks: 매칭되지 않은 track 인덱스
        """
        if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
            # 빈 행렬인 경우
            unmatched_dets = np.arange(cost_matrix.shape[0])
            unmatched_tracks = np.arange(cost_matrix.shape[1])
            return np.empty((0, 2), dtype=int), unmatched_dets, unmatched_tracks
        
        # Hungarian 알고리즘 실행
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 매칭 결과 분석
        matched_indices = []
        unmatched_detections = []
        unmatched_tracks = []
        
        # 모든 detection 인덱스
        all_det_indices = set(range(cost_matrix.shape[0]))
        # 모든 track 인덱스
        all_track_indices = set(range(cost_matrix.shape[1]))
        
        # 매칭된 쌍들 검사
        matched_det_indices = set()
        matched_track_indices = set()
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= max_distance:
                # 유효한 매칭
                matched_indices.append([row, col])
                matched_det_indices.add(row)
                matched_track_indices.add(col)
        
        # 매칭되지 않은 detection과 track 찾기
        unmatched_detections = list(all_det_indices - matched_det_indices)
        unmatched_tracks = list(all_track_indices - matched_track_indices)
        
        return (np.array(matched_indices, dtype=int), 
                np.array(unmatched_detections, dtype=int),
                np.array(unmatched_tracks, dtype=int))
    
    def match_detections_tracks(self, 
                               detections: List[Detection], 
                               tracks: List[Track],
                               iou_threshold: float = 0.3) -> Tuple[List[Tuple[int, int]], 
                                                                   List[int], 
                                                                   List[int]]:
        """
        Detection과 Track을 매칭합니다.
        
        Args:
            detections: Detection 리스트
            tracks: Track 리스트  
            iou_threshold: IoU 임계값 (이보다 높으면 매칭)
            
        Returns:
            Tuple[List[Tuple[int, int]], List[int], List[int]]:
                - matches: 매칭된 (detection_idx, track_idx) 쌍 리스트
                - unmatched_detections: 매칭되지 않은 detection 인덱스 리스트
                - unmatched_tracks: 매칭되지 않은 track 인덱스 리스트
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # IoU 행렬 계산
        iou_matrix = self.calculate_iou_matrix(detections, tracks)
        
        # 비용 행렬로 변환 (1 - IoU, 낮을수록 좋음)
        cost_matrix = 1.0 - iou_matrix
        
        # Hungarian 매칭
        matched_indices, unmatched_dets, unmatched_tracks = self.hungarian_matching(
            cost_matrix, max_distance=1.0 - iou_threshold
        )
        
        # 결과를 리스트 형태로 변환
        matches = [(int(det_idx), int(track_idx)) for det_idx, track_idx in matched_indices]
        unmatched_detections = unmatched_dets.tolist()
        unmatched_tracks = unmatched_tracks.tolist()
        
        # 매칭 결과 로깅
        self.logger.debug(f"매칭 결과: {len(matches)}개 매칭, "
                         f"{len(unmatched_detections)}개 미매칭 detection, "
                         f"{len(unmatched_tracks)}개 미매칭 track")
        
        return matches, unmatched_detections, unmatched_tracks


def calculate_center_distance(detection: Detection, track: Track) -> float:
    """
    Detection과 Track 간의 중심점 거리를 계산합니다.
    
    Args:
        detection: Detection 객체
        track: Track 객체
        
    Returns:
        float: 중심점 간의 유클리드 거리
    """
    det_center = detection.center
    track_center = track.center_point
    
    dx = det_center[0] - track_center[0]
    dy = det_center[1] - track_center[1]
    
    return np.sqrt(dx*dx + dy*dy)


def calculate_size_similarity(detection: Detection, track: Track) -> float:
    """
    Detection과 Track 간의 크기 유사도를 계산합니다.
    
    Args:
        detection: Detection 객체
        track: Track 객체
        
    Returns:
        float: 크기 유사도 (0.0 ~ 1.0, 높을수록 유사)
    """
    det_area = detection.area
    track_area = track.mean[2]  # area는 mean의 3번째 요소
    
    if det_area == 0 or track_area == 0:
        return 0.0
    
    # 작은 면적 / 큰 면적으로 유사도 계산
    similarity = min(det_area, track_area) / max(det_area, track_area)
    
    return similarity


def create_combined_cost_matrix(detections: List[Detection], 
                              tracks: List[Track],
                              matching_engine: MatchingEngine,
                              iou_weight: float = 0.7,
                              distance_weight: float = 0.2,  
                              size_weight: float = 0.1) -> np.ndarray:
    """
    여러 메트릭을 조합한 비용 행렬을 생성합니다.
    
    Args:
        detections: Detection 리스트
        tracks: Track 리스트
        matching_engine: 매칭 엔진 인스턴스
        iou_weight: IoU 가중치
        distance_weight: 거리 가중치
        size_weight: 크기 가중치
        
    Returns:
        np.ndarray: 결합된 비용 행렬
    """
    if not detections or not tracks:
        return np.empty((len(detections), len(tracks)), dtype=np.float32)
    
    n_dets = len(detections)
    n_tracks = len(tracks)
    
    # IoU 행렬 계산
    iou_matrix = matching_engine.calculate_iou_matrix(detections, tracks)
    iou_cost = 1.0 - iou_matrix  # IoU를 비용으로 변환
    
    # 거리 행렬 계산
    distance_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)
    size_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)
    
    for i, det in enumerate(detections):
        for j, track in enumerate(tracks):
            # 중심점 거리 (정규화)
            distance = calculate_center_distance(det, track)
            # 이미지 크기 대비 정규화 (가정: 1920x1080)
            normalized_distance = distance / 1000.0  
            distance_matrix[i, j] = min(normalized_distance, 1.0)
            
            # 크기 유사도를 비용으로 변환
            size_similarity = calculate_size_similarity(det, track)
            size_matrix[i, j] = 1.0 - size_similarity
    
    # 가중 결합
    combined_cost = (iou_weight * iou_cost + 
                    distance_weight * distance_matrix + 
                    size_weight * size_matrix)
    
    return combined_cost