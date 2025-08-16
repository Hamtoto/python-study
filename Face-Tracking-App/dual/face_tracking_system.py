#!/usr/bin/env python3
"""
Phase 5: Dual-Face Tracking System
완전한 얼굴 검출, 추적, 크롭 파이프라인

기능:
- 2명의 얼굴 검출 및 ID 할당
- 실시간 얼굴 추적 (OpenCV CSRT)
- 얼굴 중심 기준 크롭 (2.5배 마진)
- 1920x1080 스플릿 스크린 출력

Author: Dual-Face System v5.0
Date: 2025.08.16
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "src"))

# GPU 설정 및 딥러닝 모델
import torch
import torch.nn.functional as F
from torchvision import transforms
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"🖥️ GPU 설정: {torch.cuda.get_device_name(0)}")

# 프로젝트 모델 import (conditional)
try:
    from src.face_tracker.core.models import ModelManager
    from src.face_tracker.core.embeddings import SmartEmbeddingManager
    from src.face_tracker.utils.similarity import (
        find_matching_id_with_best_fallback_enhanced,
        calculate_face_similarity
    )
    MODEL_MANAGER_AVAILABLE = True
    print("✅ ModelManager + SmartEmbeddingManager + 고급 유사도 함수 임포트 성공")
except ImportError as e:
    print(f"⚠️ 메인 프로젝트 모듈 임포트 실패: {e}")
    MODEL_MANAGER_AVAILABLE = False


class FaceDetection:
    """얼굴 검출 결과 (임베딩 지원)"""
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.confidence = confidence
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.area = self.width * self.height
        
        # 얼굴 인식용 임베딩 (FaceNet)
        self.embedding = None  # torch.Tensor
        self.p1_score = 0.0    # Person1과의 하이브리드 점수
        self.p2_score = 0.0    # Person2와의 하이브리드 점수
        
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)


class FaceTracker:
    """검출 기반 얼굴 트래커 (OpenCV 4.13 호환)"""
    
    def __init__(self, person_id: str, smoothing_alpha: float = 0.3):
        self.person_id = person_id
        self.smoothing_alpha = smoothing_alpha
        
        # 검출 기반 상태
        self.face_detection: Optional[FaceDetection] = None
        self.face_center: Optional[Tuple[float, float]] = None
        self.smooth_center: Optional[Tuple[float, float]] = None
        self.last_good_bbox: Optional[Tuple[int, int, int, int]] = None
        self.tracking_confidence = 0.0
        
        # 검출 히스토리 (위치 기반 추적)
        self.detection_history: List[Tuple[float, float]] = []  # (center_x, center_y) 히스토리
        self.max_history = 30  # 최근 30개 위치 기억 (10 → 30으로 증가)
        self.position_threshold = 50  # 같은 사람으로 인식할 거리 임계값 (더 엄격하게)
        
        # 고정 크롭 크기 설정 (들쭉날쭉 문제 해결)
        self.fixed_crop_size = None  # 동적으로 계산됨
        self.crop_size_history = []  # 크롭 크기 히스토리 (스무딩용)
        self.max_crop_size_history = 10
        self.crop_size_smoothing_alpha = 0.2  # 크롭 크기 스무딩 계수
        
        # 통계
        self.detection_success_count = 0
        self.detection_fail_count = 0
        self.total_frames = 0
        
        # 동적 재할당 시스템 (신뢰도 기반)
        self.confidence_history = []  # 최근 신뢰도 히스토리
        self.max_confidence_history = 20
        self.low_confidence_threshold = 0.6  # 낮은 신뢰도 임계값
        self.reassignment_trigger_count = 10  # 재할당 검토를 위한 낮은 신뢰도 연속 횟수
        
    def update_detection(self, detection: Optional[FaceDetection]) -> bool:
        """검출 기반 업데이트 (트래커 없음)"""
        self.total_frames += 1
        
        if detection is None:
            self.detection_fail_count += 1
            return False
            
        # 검출 성공
        self.face_detection = detection
        self.face_center = detection.center
        self.last_good_bbox = detection.bbox
        self.tracking_confidence = detection.confidence
        
        # 신뢰도 히스토리 업데이트 (재할당 검토용)
        self.confidence_history.append(detection.confidence)
        if len(self.confidence_history) > self.max_confidence_history:
            self.confidence_history.pop(0)
        
        # 히스토리에 위치 추가
        self.detection_history.append(self.face_center)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # 스무딩 적용
        if self.smooth_center is None:
            self.smooth_center = self.face_center
        else:
            self.smooth_center = self._apply_smoothing(self.face_center)
        
        self.detection_success_count += 1
        return True
    
    def get_distance_to_detection(self, detection: FaceDetection) -> float:
        """검출된 얼굴과의 거리 계산 (위치 기반 매칭용, 히스토리 강화)"""
        if not self.detection_history:
            return float('inf')
        
        # 최근 위치들과의 가중 평균 거리 계산 (더 많은 히스토리 사용)
        det_center = detection.center
        distances = []
        weights = []
        
        # 최근 5개 위치 사용 (3 → 5로 증가), 최근일수록 높은 가중치
        recent_history = self.detection_history[-5:]
        for i, hist_center in enumerate(recent_history):
            dist = ((det_center[0] - hist_center[0]) ** 2 + 
                   (det_center[1] - hist_center[1]) ** 2) ** 0.5
            distances.append(dist)
            # 최근일수록 높은 가중치: 0.1, 0.15, 0.2, 0.25, 0.3
            weight = 0.1 + (i * 0.05)
            weights.append(weight)
        
        if distances:
            # 가중 평균 계산
            weighted_sum = sum(d * w for d, w in zip(distances, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        else:
            return float('inf')
    
    def _get_fixed_crop_size(self, frame: np.ndarray, detected_face_size: Optional[int] = None) -> int:
        """프레임 크기 기반 고정 크롭 크기 계산 (들쭉날쭉 방지)"""
        h, w = frame.shape[:2]
        
        # 기본 고정 크기: 프레임 높이의 40%
        base_crop_size = int(h * 0.4)
        
        # 최소/최대 크기 제한
        min_crop_size = int(h * 0.25)  # 최소 25%
        max_crop_size = int(h * 0.6)   # 최대 60%
        
        if detected_face_size is not None:
            # 얼굴이 검출된 경우: 얼굴 크기 기반으로 적응
            adapted_size = max(detected_face_size * 3, base_crop_size)
            adapted_size = max(min_crop_size, min(max_crop_size, adapted_size))
            
            # 크롭 크기 히스토리에 추가
            self.crop_size_history.append(adapted_size)
            if len(self.crop_size_history) > self.max_crop_size_history:
                self.crop_size_history.pop(0)
            
            # 크롭 크기 스무딩
            if self.fixed_crop_size is None:
                self.fixed_crop_size = adapted_size
            else:
                alpha = self.crop_size_smoothing_alpha
                self.fixed_crop_size = int(alpha * adapted_size + (1 - alpha) * self.fixed_crop_size)
        else:
            # 얼굴이 없는 경우: 기존 크기 유지 또는 기본값
            if self.fixed_crop_size is None:
                self.fixed_crop_size = base_crop_size
        
        # 범위 제한
        self.fixed_crop_size = max(min_crop_size, min(max_crop_size, self.fixed_crop_size))
        
        return self.fixed_crop_size
    
    def get_crop_region(self, frame: np.ndarray, margin_factor: float = 2.5) -> np.ndarray:
        """고정 크기 크롭 영역 반환 (들쭉날쭉 문제 해결)"""
        h, w = frame.shape[:2]
        
        if self.smooth_center is None or self.last_good_bbox is None:
            # 얼굴이 없으면 기본 영역 반환 (좌우 분할)
            if self.person_id == "Person1":
                crop = frame[0:h, 0:w//2]
            else:
                crop = frame[0:h, w//2:w]
                
            # 960x1080으로 리사이즈
            return cv2.resize(crop, (960, 1080))
        
        # 얼굴 크기 계산 (고정 크기 계산용)
        x1, y1, x2, y2 = self.last_good_bbox
        face_width = x2 - x1
        face_height = y2 - y1
        face_size = max(face_width, face_height)
        
        # 고정 크기 크롭 크기 계산 (들쭉날쭉 방지)
        crop_size = self._get_fixed_crop_size(frame, face_size)
        
        # 크롭 중심점 (스무딩된 얼굴 중심)
        center_x, center_y = self.smooth_center
        
        # 크롭 영역 계산
        crop_x1 = max(0, int(center_x - crop_size // 2))
        crop_y1 = max(0, int(center_y - crop_size // 2))
        crop_x2 = min(w, crop_x1 + crop_size)
        crop_y2 = min(h, crop_y1 + crop_size)
        
        # 경계 조정 (정사각형 유지)
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = max(0, crop_x2 - crop_size)
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)
        
        # 크롭 추출
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 정사각형 패딩 (필요시) - 고정 크기 보장
        ch, cw = cropped.shape[:2]
        if ch != cw:
            max_size = max(ch, cw)
            pad_h = (max_size - ch) // 2
            pad_w = (max_size - cw) // 2
            cropped = cv2.copyMakeBorder(
                cropped, pad_h, max_size - ch - pad_h, 
                pad_w, max_size - cw - pad_w, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        # 960x1080으로 리사이즈 (최종 고정 크기)
        return cv2.resize(cropped, (960, 1080))
    
    def _apply_smoothing(self, center: Tuple[float, float]) -> Tuple[float, float]:
        """EMA 스무딩 적용"""
        if self.smooth_center is None:
            return center
            
        alpha = self.smoothing_alpha
        smooth_x = alpha * center[0] + (1 - alpha) * self.smooth_center[0]
        smooth_y = alpha * center[1] + (1 - alpha) * self.smooth_center[1]
        
        return (smooth_x, smooth_y)
    
    def should_consider_reassignment(self) -> bool:
        """재할당을 검토해야 하는지 판단 (신뢰도 기반)"""
        if len(self.confidence_history) < self.reassignment_trigger_count:
            return False
        
        # 최근 N개 프레임의 신뢰도가 임계값보다 낮은지 확인
        recent_confidences = self.confidence_history[-self.reassignment_trigger_count:]
        low_confidence_count = sum(1 for conf in recent_confidences 
                                 if conf < self.low_confidence_threshold)
        
        # 80% 이상이 낮은 신뢰도면 재할당 검토
        return low_confidence_count >= (self.reassignment_trigger_count * 0.8)
    
    def get_average_confidence(self) -> float:
        """평균 신뢰도 반환"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """검출 통계 반환"""
        total_detections = self.detection_success_count + self.detection_fail_count
        success_rate = (self.detection_success_count / max(1, total_detections)) * 100
        
        return {
            'person_id': self.person_id,
            'total_frames': self.total_frames,
            'detection_success': self.detection_success_count,
            'detection_fail': self.detection_fail_count,
            'success_rate': success_rate,
            'has_detection': self.face_detection is not None,
            'detection_active': len(self.detection_history) > 0,
            'smooth_center': self.smooth_center,
            'history_length': len(self.detection_history),
            'average_confidence': self.get_average_confidence(),
            'should_reassign': self.should_consider_reassignment(),
            'fixed_crop_size': self.fixed_crop_size
        }


class FaceEmbeddingTracker(FaceTracker):
    """고급 얼굴 임베딩 기반 트래커 (SmartEmbeddingManager + 고급 유사도 함수 활용)"""
    
    def __init__(self, person_id: str, smoothing_alpha: float = 0.3):
        super().__init__(person_id, smoothing_alpha)
        
        # 고급 임베딩 관리자 (SmartEmbeddingManager 활용)
        if MODEL_MANAGER_AVAILABLE:
            self.smart_embedding_manager = SmartEmbeddingManager(max_size=10, ttl_seconds=300)
            print(f"✅ {person_id}: SmartEmbeddingManager 초기화 (max_size=10, ttl=300s)")
        else:
            self.smart_embedding_manager = None
            print(f"⚠️ {person_id}: SmartEmbeddingManager 비활성화")
        
        # 개별 임베딩 추적 (디버깅용)
        self.face_embeddings = []  # 백업 히스토리
        self.reference_embedding = None  # 대표 임베딩 (평균)
        self.max_embeddings = 10  # 최대 10개 임베딩 유지
        self.embedding_threshold = 0.75  # 0.6 → 0.75 (더 엄격한 임계값)
        
        # 고급 통계
        self.embedding_updates = 0
        self.similarity_scores = []  # 최근 유사도 점수들
        self.l2_normalization_enabled = True  # L2 정규화 사용
        
    def add_face_embedding(self, embedding: torch.Tensor) -> None:
        """새 임베딩 추가 및 고급 임베딩 관리자 업데이트"""
        if embedding is None:
            return
            
        # L2 정규화 (고급 유사도 함수 사용)
        normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        # 고급 임베딩 관리자에 추가 (사용 가능한 경우)
        if self.smart_embedding_manager is not None:
            track_id = f"{self.person_id}_{self.embedding_updates}"
            self.smart_embedding_manager.add_embedding(track_id, normalized_embedding)
        
        # 백업 히스토리에도 추가 (호환성)
        self.face_embeddings.append(normalized_embedding)
        if len(self.face_embeddings) > self.max_embeddings:
            self.face_embeddings.pop(0)  # 오래된 것 제거
        
        # 대표 임베딩 업데이트 (평균)
        if len(self.face_embeddings) > 0:
            stacked_embeddings = torch.stack(self.face_embeddings)
            self.reference_embedding = torch.mean(stacked_embeddings, dim=0)
            self.reference_embedding = torch.nn.functional.normalize(self.reference_embedding, p=2, dim=-1)
            
        self.embedding_updates += 1
        
    def compute_face_similarity(self, new_embedding: torch.Tensor) -> float:
        """새 얼굴과의 유사도 계산 (고급 유사도 함수 사용)"""
        if self.reference_embedding is None or new_embedding is None:
            return 0.0
        
        # 고급 유사도 함수 사용 (MODEL_MANAGER_AVAILABLE 확인)
        if MODEL_MANAGER_AVAILABLE:
            try:
                # calculate_face_similarity 함수 사용 (L2 정규화 자동 적용)
                score = calculate_face_similarity(
                    self.reference_embedding.unsqueeze(0), 
                    new_embedding.unsqueeze(0), 
                    use_l2_norm=self.l2_normalization_enabled
                )
            except Exception as e:
                print(f"⚠️ 고급 유사도 계산 실패: {e}, 기본 방법 사용")
                # 백업 방법: 기본 코사인 유사도
                score = torch.cosine_similarity(
                    torch.nn.functional.normalize(self.reference_embedding, p=2, dim=-1).unsqueeze(0),
                    torch.nn.functional.normalize(new_embedding, p=2, dim=-1).unsqueeze(0),
                    dim=-1
                ).item()
        else:
            # 기본 방법: L2 정규화 + 코사인 유사도
            score = torch.cosine_similarity(
                torch.nn.functional.normalize(self.reference_embedding, p=2, dim=-1).unsqueeze(0),
                torch.nn.functional.normalize(new_embedding, p=2, dim=-1).unsqueeze(0),
                dim=-1
            ).item()
        
        # 통계 업데이트
        self.similarity_scores.append(score)
        if len(self.similarity_scores) > 50:  # 최근 50개만 유지
            self.similarity_scores.pop(0)
            
        return score
    
    def compute_hybrid_score(self, face_detection: 'FaceDetection') -> float:
        """하이브리드 점수 계산 (임베딩 유사도 70% + 위치 거리 30%)"""
        if face_detection is None:
            return 0.0
            
        # 1. 얼굴 유사도 (0.0 ~ 1.0)
        face_similarity = 0.0
        if face_detection.embedding is not None and self.reference_embedding is not None:
            face_similarity = self.compute_face_similarity(face_detection.embedding)
        
        # 2. 위치 거리 (0.0 ~ 1.0, 가까울수록 높은 점수)
        position_distance = self.get_distance_to_detection(face_detection)
        position_score = 1.0 / (1.0 + position_distance / 100.0)  # 100px 기준 정규화
        
        # 3. 하이브리드 점수 (가중 평균)
        # 임베딩이 없거나 reference가 없으면 위치 기반 점수만 사용
        if face_detection.embedding is not None and self.reference_embedding is not None:
            hybrid_score = face_similarity * 0.7 + position_score * 0.3
        else:
            hybrid_score = position_score  # 위치 기반만 사용
        
        return hybrid_score
    
    def is_same_person(self, face_detection: 'FaceDetection') -> bool:
        """같은 사람인지 판단 (임베딩 기반)"""
        if face_detection.embedding is None:
            # 임베딩이 없으면 위치 기반으로 판단
            distance = self.get_distance_to_detection(face_detection)
            return distance <= self.position_threshold
            
        # 임베딩 유사도 기반 판단
        similarity = self.compute_face_similarity(face_detection.embedding)
        return similarity >= self.embedding_threshold
    
    def update_with_embedding(self, detection: Optional['FaceDetection']) -> bool:
        """임베딩을 포함한 검출 기반 업데이트"""
        success = self.update_detection(detection)
        
        # 임베딩 추가 (검출 성공 시에만)
        if success and detection is not None and detection.embedding is not None:
            self.add_face_embedding(detection.embedding)
            
        return success
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """고급 임베딩 관련 통계 반환 (SmartEmbeddingManager 포함)"""
        avg_similarity = 0.0
        if len(self.similarity_scores) > 0:
            avg_similarity = sum(self.similarity_scores) / len(self.similarity_scores)
        
        # SmartEmbeddingManager 통계 추가
        smart_stats = {}
        if self.smart_embedding_manager is not None:
            smart_stats = self.smart_embedding_manager.get_stats()
            
        return {
            'embeddings_count': len(self.face_embeddings),
            'embedding_updates': self.embedding_updates,
            'has_reference_embedding': self.reference_embedding is not None,
            'avg_similarity': avg_similarity,
            'recent_similarities': self.similarity_scores[-5:],  # 최근 5개
            'embedding_threshold': self.embedding_threshold,
            'l2_normalization_enabled': self.l2_normalization_enabled,
            'smart_embedding_manager_enabled': self.smart_embedding_manager is not None,
            'smart_embedding_stats': smart_stats  # SmartEmbeddingManager 통계
        }


class DualFaceTrackingSystem:
    """통합 얼굴 트래킹 시스템"""
    
    def __init__(self, args):
        self.input_path = args.input
        self.output_path = args.output
        self.mode = args.mode
        self.gpu_id = args.gpu
        
        # 출력 디렉토리 생성
        Path(self.output_path).parent.mkdir(exist_ok=True)
        
        # 처리 설정
        self.detection_interval = 1   # 매 프레임마다 얼굴 검출 (트래커 없으므로)
        self.margin_factor = 5.0      # 얼굴 크기 대비 크롭 배율 (상반신 포함)
        self.confidence_threshold = 0.1  # 0.3→0.1 (매우 관대한 임계값)
        
        # 모델 초기화
        self.detection_method = None
        self._debug_detection_logged = False
        self._initialize_models()
        
        # 트래커 초기화 (임베딩 기반 트래커 사용)
        self.person1_tracker = FaceEmbeddingTracker("Person1", smoothing_alpha=0.1)
        self.person2_tracker = FaceEmbeddingTracker("Person2", smoothing_alpha=0.1)
        
        # 크기 기반 간단한 할당 시스템
        self.debug_mode = False  # 디버그 모드 (크기 정보 출력)
        self.size_stabilize = False  # 크기 기반 안정화 사용 여부
        
        # FaceNet 모델 초기화 (얼굴 임베딩용)
        self.model_manager = None
        self.resnet = None
        self.face_transform = None
        self._initialize_facenet()
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detection_calls': 0,
            'start_time': time.time(),
            'processing_time': 0.0
        }
        
        print(f"🏗️ DualFaceTrackingSystem 초기화 완료")
        print(f"   📥 입력: {self.input_path}")
        print(f"   📤 출력: {self.output_path}")
        print(f"   🔧 검출 간격: {self.detection_interval}프레임")
        print(f"   📏 크롭 배율: {self.margin_factor}x")
        
    def _initialize_models(self):
        """얼굴 검출 모델 초기화"""
        print("🏗️ 얼굴 검출 모델 초기화 중...")
        
        # 방법 1: OpenCV Haar Cascade (가장 안정적)
        try:
            # 확인된 경로를 직접 사용
            cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            
            if Path(cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.face_cascade.empty():
                    self.detection_method = "haar"
                    print(f"✅ OpenCV Haar Cascade 모델 로드 완료")
                    print(f"   📍 경로: {cascade_path}")
                    return
                else:
                    print(f"⚠️ Haar Cascade 생성 실패: {cascade_path}")
            else:
                print(f"⚠️ Haar Cascade 파일 없음: {cascade_path}")
                
        except Exception as e:
            print(f"⚠️ Haar Cascade 로드 실패: {e}")
        
        # 방법 2: 기존 프로젝트의 MTCNN 시도 (상위 디렉토리)
        try:
            # 상위 프로젝트 경로 추가
            parent_project = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_project))
            sys.path.insert(0, str(parent_project / "src"))
            
            from face_tracker.core.models import ModelManager
            self.mtcnn, self.resnet = ModelManager.get_models()
            self.detection_method = "mtcnn"
            print("✅ 상위 프로젝트 MTCNN + FaceNet 모델 로드 완료")
            return
        except Exception as e:
            print(f"⚠️ 상위 프로젝트 MTCNN 로드 실패: {e}")
        
        # 방법 3: 간단한 MTCNN 직접 로드
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')
            self.detection_method = "mtcnn_direct"
            print("✅ facenet-pytorch MTCNN 직접 로드 완료")
            return
        except Exception as e:
            print(f"⚠️ 직접 MTCNN 로드 실패: {e}")
        
        # 방법 4: MediaPipe 얼굴 검출
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.3)
            self.detection_method = "mediapipe"
            print("✅ MediaPipe 얼굴 검출 모델 로드 완료")
            return
        except Exception as e:
            print(f"⚠️ MediaPipe 로드 실패: {e}")
        
        raise RuntimeError("❌ 모든 얼굴 검출 모델 로드 실패")
    
    def _initialize_facenet(self):
        """FaceNet 모델 초기화 (얼굴 임베딩용)"""
        print("🧠 FaceNet 모델 초기화 중...")
        
        if not MODEL_MANAGER_AVAILABLE:
            print("⚠️ ModelManager를 사용할 수 없음. 임베딩 기능 비활성화")
            return
            
        try:
            # ModelManager로 FaceNet 모델 로드
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_manager = ModelManager(device)
            self.resnet = self.model_manager.get_resnet()
            
            # 얼굴 전처리 변환 (FaceNet용)
            self.face_transform = transforms.Compose([
                transforms.Resize((160, 160)),  # FaceNet 입력 크기
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            print("✅ FaceNet 모델 로드 완료")
            print(f"   📍 디바이스: {device}")
            print("   🧠 얼굴 임베딩 기능 활성화")
            
        except Exception as e:
            print(f"⚠️ FaceNet 모델 초기화 실패: {e}")
            print("   🔄 임베딩 없이 위치 기반 추적만 사용")
            self.model_manager = None
            self.resnet = None
            self.face_transform = None
    
    def generate_face_embedding(self, face_crop: np.ndarray) -> Optional[torch.Tensor]:
        """얼굴 크롭에서 임베딩 생성"""
        if self.resnet is None or self.face_transform is None:
            return None
            
        try:
            # OpenCV BGR -> PIL RGB 변환
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)
            
            # 전처리 및 배치 차원 추가
            face_tensor = self.face_transform(pil_image).unsqueeze(0)
            
            # GPU로 이동 (사용 가능한 경우)
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            
            # 임베딩 생성 (그래디언트 비활성화)
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
                # L2 정규화
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.squeeze(0)  # 배치 차원 제거
            
        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패: {e}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """프레임에서 얼굴 검출"""
        faces = []
        
        try:
            # 디버깅: 검출 방법 확인
            if not self._debug_detection_logged:
                print(f"🔧 검출 방법: {self.detection_method}")
                self._debug_detection_logged = True
            
            if self.detection_method == "mtcnn":
                faces = self._detect_faces_mtcnn(frame)
            elif self.detection_method == "mtcnn_direct":
                faces = self._detect_faces_mtcnn_direct(frame)
            elif self.detection_method == "haar":
                faces = self._detect_faces_haar(frame)
            elif self.detection_method == "mediapipe":
                faces = self._detect_faces_mediapipe(frame)
            elif self.detection_method == "dnn":
                faces = self._detect_faces_dnn(frame)
            else:
                print(f"❌ 알 수 없는 검출 방법: {self.detection_method}")
                return []
            
            # 신뢰도 순으로 정렬 (높은 것부터)
            faces.sort(key=lambda x: x.confidence, reverse=True)
            
            # 최대 2개까지만 선택
            faces = faces[:2]
            
            # 각 얼굴에 대해 임베딩 생성
            for face in faces:
                try:
                    # 얼굴 영역 크롭
                    x1, y1, x2, y2 = face.bbox
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # 유효한 크롭인지 확인
                    if face_crop.size > 0:
                        # 임베딩 생성
                        embedding = self.generate_face_embedding(face_crop)
                        face.embedding = embedding
                        
                except Exception as e:
                    print(f"⚠️ 얼굴 임베딩 생성 실패: {e}")
                    face.embedding = None
            
            return faces
            
        except Exception as e:
            print(f"⚠️ 얼굴 검출 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_faces_mtcnn(self, frame: np.ndarray) -> List[FaceDetection]:
        """MTCNN으로 얼굴 검출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        boxes_list, probs_list = self.mtcnn.detect([pil_image])
        faces = []
        
        if boxes_list[0] is not None and probs_list[0] is not None:
            boxes = boxes_list[0]
            probs = probs_list[0]
            
            for box, prob in zip(boxes, probs):
                if prob > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    faces.append(FaceDetection((x1, y1, x2, y2), prob))
        
        return faces
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        """Haar Cascade로 얼굴 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 균형잡힌 파라미터로 안정적인 얼굴 검출
        detected_faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.08,    # 1.1→1.08 (조금 더 세밀한 검출)
            minNeighbors=4,      # 5→4 (약간 더 관대하게)
            minSize=(25, 25),    # 30→25 (조금 더 작은 얼굴도 포함)
            maxSize=(180, 180),  # 200→180 (더 적절한 최대 크기)
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in detected_faces:
            # 얼굴 크기 및 종횡비 검증
            if w > 20 and h > 20 and w < 150 and h < 150:
                # 종횡비 체크 (얼굴은 대략 정사각형)
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.3:
                    confidence = 0.9  # Haar는 신뢰도가 없으므로 고정값
                    faces.append(FaceDetection((x, y, x + w, y + h), confidence))
        
        return faces
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> List[FaceDetection]:
        """DNN으로 얼굴 검출"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append(FaceDetection((x1, y1, x2, y2), confidence))
        
        return faces
    
    def _detect_faces_mtcnn_direct(self, frame: np.ndarray) -> List[FaceDetection]:
        """facenet-pytorch MTCNN으로 직접 얼굴 검출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # MTCNN 검출
        boxes, probs = self.mtcnn.detect(pil_image)
        faces = []
        
        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                if prob > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    faces.append(FaceDetection((x1, y1, x2, y2), prob))
        
        return faces
    
    def _detect_faces_mediapipe(self, frame: np.ndarray) -> List[FaceDetection]:
        """MediaPipe로 얼굴 검출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            
            for detection in results.detections:
                # MediaPipe bounding box (normalized coordinates)
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # 신뢰도 점수
                confidence = detection.score[0]
                
                if confidence > self.confidence_threshold:
                    faces.append(FaceDetection((x1, y1, x2, y2), confidence))
        
        return faces
    
    def _assign_by_size(self, faces: List[FaceDetection]) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """크기 기반으로 Person1(가장 큰 얼굴), Person2(두 번째 큰 얼굴) 할당"""
        if not faces:
            return None, None
        
        # 얼굴 크기(area)로 정렬 - 큰 순서대로
        sorted_faces = sorted(faces, key=lambda f: f.area, reverse=True)
        
        # 가장 큰 얼굴 → Person1 (왼쪽)
        person1_face = sorted_faces[0] if len(sorted_faces) >= 1 else None
        
        # 두 번째로 큰 얼굴 → Person2 (오른쪽)
        person2_face = sorted_faces[1] if len(sorted_faces) >= 2 else None
        
        # 디버그 정보 출력
        if self.debug_mode and person1_face and person2_face:
            size_ratio = person1_face.area / person2_face.area
            print(f"📊 크기 비교: P1={person1_face.area:.0f}, P2={person2_face.area:.0f}, 비율={size_ratio:.2f}")
        
        return person1_face, person2_face
    
    def assign_face_ids(self, faces: List[FaceDetection], frame_idx: int) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """크기 기반 단순 할당: 가장 큰 얼굴=Person1, 두 번째=Person2"""
        if len(faces) == 0:
            return None, None
        
        # 단순하게 크기만으로 할당
        person1_face, person2_face = self._assign_by_size(faces)
        
        # 각 트래커에 업데이트
        if person1_face:
            self.person1_tracker.update_detection(person1_face)
        if person2_face:
            self.person2_tracker.update_detection(person2_face)
        
        # 디버그 정보 (30프레임마다)
        if self.debug_mode and frame_idx % 30 == 0:
            if person1_face and person2_face:
                size_ratio = person1_face.area / person2_face.area
                print(f"📊 프레임 {frame_idx}: P1={person1_face.area:.0f}, P2={person2_face.area:.0f}, 비율={size_ratio:.2f}")
            elif person1_face:
                print(f"📊 프레임 {frame_idx}: P1={person1_face.area:.0f}, P2=없음")
        
        return person1_face, person2_face
    
    # 컴팩트한 크기 기반 시스템으로 교체된 복잡한 로직들은 제거됨
    
    def create_split_screen(self, crop1: np.ndarray, crop2: np.ndarray) -> np.ndarray:
        """스플릿 스크린 생성 (1920x1080)"""
        # 1920x1080 캔버스 생성
        split_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # 왼쪽: Person1 (0:960)
        split_screen[0:1080, 0:960] = crop1
        
        # 오른쪽: Person2 (960:1920)
        split_screen[0:1080, 960:1920] = crop2
        
        return split_screen
    
    def add_overlay_info(self, frame: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
        """오버레이 정보 추가"""
        # 진행률 계산
        progress = (frame_idx / max(1, self.stats['total_frames'])) * 100
        
        # 트래커 통계
        p1_stats = self.person1_tracker.get_stats()
        p2_stats = self.person2_tracker.get_stats()
        
        # 텍스트 정보
        texts = [
            f"Frame: {frame_idx}/{self.stats['total_frames']}",
            f"Progress: {progress:.1f}%",
            f"FPS: {fps:.1f}",
            f"P1 Track: {p1_stats['success_rate']:.1f}%",
            f"P2 Track: {p2_stats['success_rate']:.1f}%",
            "Dual-Face Tracking v5.0"
        ]
        
        # 텍스트 그리기
        for i, text in enumerate(texts):
            y_pos = 30 + i * 35
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def process(self):
        """메인 처리 함수"""
        print(f"🎬 비디오 처리 시작")
        
        # 비디오 열기
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오 열기 실패: {self.input_path}")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        self.stats['total_frames'] = total_frames
        
        print(f"   📹 비디오 정보: {width}x{height}, {fps:.1f}fps")
        print(f"   ⏱️ 지속시간: {duration:.1f}초, {total_frames}프레임")
        
        # 출력 비디오 설정 (1920x1080 스플릿 스크린)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (1920, 1080))
        
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"❌ 비디오 라이터 생성 실패: {self.output_path}")
        
        # 진행률 표시
        pbar = tqdm(total=total_frames, desc="🎯 얼굴 트래킹", ncols=80, leave=True)
        
        try:
            frame_idx = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # 1. 얼굴 검출 (매 프레임 또는 N프레임마다)
                if self.detection_interval == 1 or frame_idx % self.detection_interval == 0:
                    # 디버깅: 검출 전 프레임 정보
                    if frame_idx <= 10:
                        print(f"🔍 프레임 {frame_idx} 검출 시도 - 프레임 크기: {frame.shape}")
                    
                    faces = self.detect_faces(frame)
                    self.stats['detection_calls'] += 1
                    
                    # 디버깅: 얼굴 검출 결과 로그 (매 프레임 출력)
                    if frame_idx <= 50 or frame_idx % 200 == 1:  # 처음 50프레임과 200프레임마다
                        print(f"🔍 프레임 {frame_idx}: {len(faces)}개 얼굴 검출 (방법: {self.detection_method})")
                        for i, face in enumerate(faces):
                            print(f"   얼굴 {i+1}: ({face.x1},{face.y1})-({face.x2},{face.y2}) conf={face.confidence:.2f}")
                    
                    # 얼굴 ID 할당 (일관성 강화)
                    person1_face, person2_face = self.assign_face_ids(faces, frame_idx)
                    
                    # 디버깅: ID 할당 결과 (일관성 추적)
                    if frame_idx <= 60:  # 처음 60프레임 동안 상세 로그
                        p1_assigned = person1_face is not None
                        p2_assigned = person2_face is not None
                        print(f"   🎯 ID 할당 (프레임 {frame_idx}): P1={p1_assigned}, P2={p2_assigned}")
                        
                        if person1_face:
                            print(f"     P1 얼굴: center={person1_face.center} conf={person1_face.confidence:.2f}")
                        if person2_face:
                            print(f"     P2 얼굴: center={person2_face.center} conf={person2_face.confidence:.2f}")
                            
                        # 기존 초기 위치 기반 코드 제거됨 (크기 기반 시스템에서 불필요)
                    
                    # 검출 기반 업데이트 (트래커 없음)
                    if person1_face:
                        result1 = self.person1_tracker.update_detection(person1_face)
                        if frame_idx <= 50:
                            print(f"   ✅ P1 검출 업데이트: {result1}")
                    else:
                        self.person1_tracker.update_detection(None)
                        
                    if person2_face:
                        result2 = self.person2_tracker.update_detection(person2_face)
                        if frame_idx <= 50:
                            print(f"   ✅ P2 검출 업데이트: {result2}")
                    else:
                        self.person2_tracker.update_detection(None)
                
                # 3. 크롭 영역 생성
                crop1 = self.person1_tracker.get_crop_region(frame, self.margin_factor)
                crop2 = self.person2_tracker.get_crop_region(frame, self.margin_factor)
                
                # 4. 스플릿 스크린 생성
                split_screen = self.create_split_screen(crop1, crop2)
                
                # 5. 오버레이 정보 추가
                current_fps = frame_idx / max(0.01, time.time() - start_time)
                split_screen = self.add_overlay_info(split_screen, frame_idx, current_fps)
                
                # 6. 출력
                writer.write(split_screen)
                
                # 진행률 업데이트
                self.stats['processed_frames'] = frame_idx
                pbar.update(1)
                
                # 중간 진행률 출력 (100프레임마다)
                if frame_idx % 100 == 0:
                    p1_rate = self.person1_tracker.get_stats()['success_rate']
                    p2_rate = self.person2_tracker.get_stats()['success_rate']
                    pbar.set_postfix({
                        'P1': f'{p1_rate:.1f}%',
                        'P2': f'{p2_rate:.1f}%',
                        'FPS': f'{current_fps:.1f}'
                    })
        
        finally:
            cap.release()
            writer.release()
            pbar.close()
        
        # 최종 통계
        self.stats['processing_time'] = time.time() - start_time
        self._print_final_stats()
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        print("\n" + "=" * 60)
        print("📊 Dual-Face Tracking 최종 결과")
        print("=" * 60)
        
        # 전체 통계
        total_time = self.stats['processing_time']
        total_frames = self.stats['processed_frames']
        detection_calls = self.stats['detection_calls']
        avg_fps = total_frames / max(0.01, total_time)
        
        print(f"🎯 처리 완료: {self.output_path}")
        print(f"⏱️ 총 처리 시간: {total_time:.1f}초")
        print(f"📊 평균 FPS: {avg_fps:.1f}")
        print(f"🔍 얼굴 검출 호출: {detection_calls}회")
        
        # 개별 검출 통계
        p1_stats = self.person1_tracker.get_stats()
        p2_stats = self.person2_tracker.get_stats()
        
        print(f"\n👤 Person1 검출:")
        print(f"   성공률: {p1_stats['success_rate']:.1f}%")
        print(f"   성공/실패: {p1_stats['detection_success']}/{p1_stats['detection_fail']}")
        print(f"   검출 상태: {'✅' if p1_stats['has_detection'] else '❌'}")
        print(f"   히스토리 길이: {p1_stats['history_length']}")
        
        print(f"\n👤 Person2 검출:")
        print(f"   성공률: {p2_stats['success_rate']:.1f}%")
        print(f"   성공/실패: {p2_stats['detection_success']}/{p2_stats['detection_fail']}")
        print(f"   검출 상태: {'✅' if p2_stats['has_detection'] else '❌'}")
        print(f"   히스토리 길이: {p2_stats['history_length']}")
        
        print(f"\n🎬 출력 정보:")
        if Path(self.output_path).exists():
            file_size = Path(self.output_path).stat().st_size / 1024**2
            print(f"   📄 파일: {self.output_path}")
            print(f"   💾 크기: {file_size:.1f}MB")
            print(f"   📺 해상도: 1920x1080 (스플릿 스크린)")
        else:
            print(f"   ❌ 출력 파일이 생성되지 않았습니다")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Dual-Face Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (크기 기반 자동 할당)
  python3 face_tracking_system.py
  
  # 디버그 모드로 크기 정보 확인
  python3 face_tracking_system.py --debug
  
  # 사용자 지정 비디오
  python3 face_tracking_system.py --input tests/videos/sample.mp4 --output output/tracked.mp4 --debug
  
  # 크기 기반 안정화 사용
  python3 face_tracking_system.py --size-stabilize --debug
        """
    )
    
    parser.add_argument("--input", 
                       default="tests/videos/2people_sample1.mp4",
                       help="입력 비디오 경로 (기본값: tests/videos/2people_sample1.mp4)")
    parser.add_argument("--output", 
                       default="output/2people_sample1_tracked.mp4",
                       help="출력 비디오 경로 (기본값: output/2people_sample1_tracked.mp4)")
    parser.add_argument("--mode", 
                       default="dual_face", 
                       choices=["dual_face", "single"],
                       help="처리 모드 (기본값: dual_face)")
    parser.add_argument("--gpu", 
                       type=int, 
                       default=0,
                       help="GPU ID (기본값: 0)")
    parser.add_argument("--debug", 
                       action="store_true",
                       help="디버그 모드 (크기 비교 정보 출력)")
    parser.add_argument("--size-stabilize", 
                       action="store_true",
                       help="크기 기반 할당 안정화 사용")
    
    args = parser.parse_args()
    
    print("🚀 Dual-Face Tracking System v6.0 (크기 기반)")
    print("=" * 50)
    print(f"   📥 입력: {args.input}")
    print(f"   📤 출력: {args.output}")
    print(f"   🔧 모드: {args.mode}")
    print(f"   🖥️ GPU: {args.gpu}")
    print(f"   🔍 디버그: {args.debug}")
    print(f"   ⚙️ 안정화: {args.size_stabilize}")
    print("=" * 50)
    
    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"❌ 입력 비디오 파일 없음: {args.input}")
        sys.exit(1)
    
    # 시스템 실행
    try:
        system = DualFaceTrackingSystem(args)
        system.debug_mode = args.debug
        system.size_stabilize = args.size_stabilize
        system.process()
        
        print("\n🎉 Phase 5 완료: 얼굴 트래킹 시스템 성공!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()