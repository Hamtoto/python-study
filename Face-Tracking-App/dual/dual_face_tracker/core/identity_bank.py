"""
Identity Bank System for Dual Face Tracking
Phase 2: Identity-based tracking with prototype management

A/B 슬롯 임베딩 뱅크 관리 시스템
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import torch.nn.functional as F


class IdentityBank:
    """A/B 슬롯별 임베딩 뱅크 관리
    
    Features:
    - 각 슬롯(A/B)별로 임베딩 뱅크 유지
    - L2 정규화 + 코사인 거리 계산
    - 중앙값 기반 프로토타입 (노이즈 강건성)
    - EMA 기반 업데이트 옵션
    """
    
    def __init__(self, max_samples: int = 64, device='cuda'):
        """
        Args:
            max_samples: 각 슬롯당 최대 임베딩 샘플 수
            device: 텐서 연산에 사용할 디바이스 ('cuda' or 'cpu')
        """
        self.max_samples = max_samples
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # A/B 슬롯별 임베딩 뱅크
        self.bank = {
            'A': deque(maxlen=max_samples),
            'B': deque(maxlen=max_samples)
        }
        
        # 통계 정보
        self.stats = {
            'A': {'updates': 0, 'last_proto_time': None},
            'B': {'updates': 0, 'last_proto_time': None}
        }
    
    @staticmethod
    def _l2norm(x):
        """L2 정규화 (벡터 정규화)"""
        if isinstance(x, torch.Tensor):
            norm = torch.norm(x) + 1e-8
            return x / norm
        else:
            norm = np.linalg.norm(x) + 1e-8
            return x / norm
    
    def update(self, key: str, emb):
        """임베딩 뱅크 업데이트
        
        Args:
            key: 'A' or 'B'
            emb: 임베딩 벡터 (torch.Tensor or np.ndarray)
        """
        if key not in self.bank:
            raise ValueError(f"Invalid key: {key}. Must be 'A' or 'B'")
        
        # L2 정규화
        normalized_emb = self._l2norm(emb)
        
        # 텐서인 경우 올바른 디바이스로 이동
        if isinstance(normalized_emb, torch.Tensor):
            normalized_emb = normalized_emb.to(self.device)
        
        # 뱅크에 추가
        self.bank[key].append(normalized_emb)
        
        # 통계 업데이트
        self.stats[key]['updates'] += 1
        import time
        self.stats[key]['last_proto_time'] = time.time()
    
    def proto(self, key: str):
        """프로토타입 임베딩 계산 (중앙값 기반)
        
        Args:
            key: 'A' or 'B'
            
        Returns:
            중앙값 기반 프로토타입 임베딩 (정규화됨) 또는 None
        """
        if key not in self.bank:
            return None
            
        if len(self.bank[key]) == 0:
            return None
        elif len(self.bank[key]) == 1:
            return self.bank[key][0]
        else:
            # 중앙값 계산 (노이즈 강건)
            embeddings_list = list(self.bank[key])
            
            if isinstance(embeddings_list[0], torch.Tensor):
                # PyTorch 텐서
                # 모든 임베딩을 올바른 디바이스로 이동
                embeddings_on_device = [emb.to(self.device) for emb in embeddings_list]
                embeddings = torch.stack(embeddings_on_device)
                prototype = torch.median(embeddings, dim=0)[0]
            else:
                # NumPy 배열
                embeddings = np.array(embeddings_list)
                prototype = np.median(embeddings, axis=0)
            
            # 프로토타입을 올바른 디바이스에서 정규화
            normalized_prototype = self._l2norm(prototype)
            if isinstance(normalized_prototype, torch.Tensor):
                normalized_prototype = normalized_prototype.to(self.device)
            
            return normalized_prototype
    
    def dist(self, key: str, emb, use_cosine: bool = True):
        """프로토타입과 임베딩 간 거리 계산
        
        Args:
            key: 'A' or 'B'
            emb: 비교할 임베딩
            use_cosine: True면 코사인 거리, False면 유클리드 거리
            
        Returns:
            거리 값 (0에 가까울수록 유사)
        """
        prototype = self.proto(key)
        if prototype is None:
            return 1.0  # 최대 거리
        
        # 입력 임베딩 정규화
        emb_norm = self._l2norm(emb)
        
        # 디바이스 일치 확인 및 조정
        if isinstance(prototype, torch.Tensor) and isinstance(emb_norm, torch.Tensor):
            if prototype.device != emb_norm.device:
                # 두 텐서를 같은 디바이스로 이동 (self.device 사용)
                prototype = prototype.to(self.device)
                emb_norm = emb_norm.to(self.device)
        
        if use_cosine:
            # 코사인 거리 (1 - cosine_similarity)
            if isinstance(prototype, torch.Tensor) and isinstance(emb_norm, torch.Tensor):
                cosine_sim = F.cosine_similarity(
                    prototype.unsqueeze(0), 
                    emb_norm.unsqueeze(0)
                ).item()
            else:
                # NumPy 배열
                cosine_sim = float(np.dot(prototype, emb_norm))
            
            return 1.0 - cosine_sim  # 거리로 변환 (0이 가장 가까움)
        else:
            # 유클리드 거리
            if isinstance(prototype, torch.Tensor) and isinstance(emb_norm, torch.Tensor):
                distance = torch.dist(prototype, emb_norm).item()
            else:
                distance = float(np.linalg.norm(prototype - emb_norm))
            
            return distance
    
    def similarity(self, key: str, emb):
        """프로토타입과 임베딩 간 코사인 유사도 계산
        
        Args:
            key: 'A' or 'B' 
            emb: 비교할 임베딩
            
        Returns:
            유사도 (0~1, 1에 가까울수록 유사)
        """
        distance = self.dist(key, emb, use_cosine=True)
        return 1.0 - distance  # 거리를 유사도로 변환
    
    def get_best_match(self, candidates: List[Any], embeddings: List[Any]) -> Tuple[str, float, int]:
        """여러 후보 중 가장 적합한 슬롯 찾기
        
        Args:
            candidates: 후보 얼굴 리스트
            embeddings: 해당 임베딩 리스트
            
        Returns:
            (best_slot, best_similarity, best_index)
        """
        best_slot = None
        best_similarity = 0.0
        best_index = -1
        
        for i, emb in enumerate(embeddings):
            for slot in ['A', 'B']:
                sim = self.similarity(slot, emb)
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_slot = slot
                    best_index = i
        
        return best_slot, best_similarity, best_index
    
    def is_valid_match(self, key: str, emb, threshold: float = 0.35) -> bool:
        """임계값 기반 유효한 매칭 검사
        
        Args:
            key: 'A' or 'B'
            emb: 테스트할 임베딩
            threshold: 유사도 임계값 (기본 0.35)
            
        Returns:
            임계값 이상이면 True
        """
        sim = self.similarity(key, emb)
        return sim >= threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """뱅크 통계 정보 반환"""
        return {
            'bank_sizes': {
                'A': len(self.bank['A']),
                'B': len(self.bank['B'])
            },
            'total_updates': {
                'A': self.stats['A']['updates'],
                'B': self.stats['B']['updates'] 
            },
            'max_samples': self.max_samples,
            'has_prototypes': {
                'A': len(self.bank['A']) > 0,
                'B': len(self.bank['B']) > 0
            }
        }
    
    def reset_slot(self, key: str):
        """특정 슬롯 리셋"""
        if key in self.bank:
            self.bank[key].clear()
            self.stats[key]['updates'] = 0
            self.stats[key]['last_proto_time'] = None
    
    def clear_all(self):
        """모든 슬롯 리셋"""
        self.reset_slot('A')
        self.reset_slot('B')


class HungarianMatcher:
    """2×N 헝가리언 매칭 (A/B 슬롯만)"""
    
    def __init__(self, identity_bank: IdentityBank):
        self.identity_bank = identity_bank
    
    def build_cost_matrix(self, detections: List[Any], embeddings: List[Any], 
                         predicted_boxes: Dict[str, Any] = None,
                         prev_boxes: Dict[str, Any] = None,
                         weights: Dict[str, float] = None) -> np.ndarray:
        """비용 행렬 구성
        
        Args:
            detections: 검출된 얼굴 리스트
            embeddings: 해당 임베딩 리스트  
            predicted_boxes: {'A': bbox, 'B': bbox} 예측 박스
            prev_boxes: {'A': bbox, 'B': bbox} 이전 박스
            weights: {'iou': 0.45, 'emb': 0.45, 'motion': 0.10} 가중치
            
        Returns:
            (2, N) 비용 행렬 (A/B × 검출수)
        """
        N = len(detections)
        if N == 0:
            return np.zeros((2, 0))
        
        # 기본 가중치
        if weights is None:
            weights = {'iou': 0.45, 'emb': 0.45, 'motion': 0.10}
        
        # 2×N 비용 행렬 초기화
        cost_matrix = np.zeros((2, N), dtype=np.float32)
        
        for j in range(N):
            detection = detections[j]
            embedding = embeddings[j] if j < len(embeddings) else None
            
            # A/B 각각에 대해 비용 계산
            for i, slot in enumerate(['A', 'B']):
                total_cost = 0.0
                
                # 1. 임베딩 비용 (코사인 거리)
                if embedding is not None:
                    emb_distance = self.identity_bank.dist(slot, embedding)
                    total_cost += emb_distance * weights['emb']
                
                # 2. IoU 비용 (위치 연속성)
                if predicted_boxes and slot in predicted_boxes:
                    iou_cost = 1.0 - self._calculate_iou(detection, predicted_boxes[slot])
                    total_cost += iou_cost * weights['iou']
                elif prev_boxes and slot in prev_boxes:
                    iou_cost = 1.0 - self._calculate_iou(detection, prev_boxes[slot])
                    total_cost += iou_cost * weights['iou']
                
                # 3. 모션 비용 (향후 확장)
                # motion_cost = 0.0  # 추후 구현
                # total_cost += motion_cost * weights['motion']
                
                cost_matrix[i, j] = total_cost
        
        # 임계값 기반 패널티 (프로토타입과 너무 먼 경우)
        threshold = 0.45
        for j in range(N):
            if j < len(embeddings):
                for i, slot in enumerate(['A', 'B']):
                    if self.identity_bank.dist(slot, embeddings[j]) > threshold:
                        cost_matrix[i, j] += 10.0  # 큰 패널티
        
        return cost_matrix
    
    def _calculate_iou(self, det, box):
        """IoU 계산 (간단한 구현)"""
        if hasattr(det, 'bbox'):
            x1, y1, x2, y2 = det.bbox
        else:
            return 0.0
        
        if isinstance(box, (list, tuple)) and len(box) == 4:
            bx1, by1, bx2, by2 = box
        else:
            return 0.0
        
        # 교집합 계산
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)  
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 합집합 계산
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (bx2 - bx1) * (by2 - by1)
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def hungarian_assign(self, cost_matrix: np.ndarray) -> Dict[str, int]:
        """2×N 헝가리언 할당
        
        Args:
            cost_matrix: (2, N) 비용 행렬
            
        Returns:
            {'A': idx_a, 'B': idx_b} 할당 결과 (-1이면 할당 없음)
        """
        if cost_matrix.shape[1] == 0:
            return {'A': -1, 'B': -1}
        
        try:
            from scipy.optimize import linear_sum_assignment
            
            # 헝가리언 알고리즘 적용
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            result = {'A': -1, 'B': -1}
            
            for row_idx, col_idx in zip(row_indices, col_indices):
                slot = 'A' if row_idx == 0 else 'B'
                result[slot] = col_idx
            
            return result
            
        except ImportError:
            # scipy 없는 경우 단순 탐욕 매칭
            return self._greedy_assign(cost_matrix)
    
    def _greedy_assign(self, cost_matrix: np.ndarray) -> Dict[str, int]:
        """탐욕적 할당 (scipy 없는 경우)"""
        result = {'A': -1, 'B': -1}
        used_cols = set()
        
        # A 먼저 할당
        if cost_matrix.shape[1] > 0:
            best_col = np.argmin(cost_matrix[0, :])
            result['A'] = best_col
            used_cols.add(best_col)
        
        # B 할당 (A와 다른 컬럼)
        if cost_matrix.shape[1] > 1:
            available_cols = [i for i in range(cost_matrix.shape[1]) if i not in used_cols]
            if available_cols:
                costs_b = [cost_matrix[1, i] for i in available_cols]
                best_idx = np.argmin(costs_b)
                result['B'] = available_cols[best_idx]
        
        return result