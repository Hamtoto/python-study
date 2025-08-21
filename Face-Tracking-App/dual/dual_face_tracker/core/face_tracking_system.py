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
import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "src"))

# 로깅 시스템 설정
from ..utils.logger import get_logger

# TqdmLoggingHandler for progress bar compatibility
class TqdmLoggingHandler(logging.Handler):
    """tqdm 프로그레스바와 호환되는 로깅 핸들러"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except:
            self.handleError(record)

# 로거 초기화
logger = get_logger(__name__, level=logging.INFO)

# AutoSpeakerDetector import
from .auto_speaker_detector import AutoSpeakerDetector

# GPU 설정 및 딥러닝 모델
import torch
import torch.nn.functional as F
from torchvision import transforms
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    logger.info(f"GPU 설정: {torch.cuda.get_device_name(0)}")

# 설정 import
from ..config.dual_config import (
    PRESERVE_AUDIO, ENABLE_FFMPEG_POST_PROCESSING,
    TRIM_UNDETECTED_SEGMENTS, UNDETECTED_THRESHOLD_SECONDS, 
    TRIM_BUFFER_SECONDS, REQUIRE_BOTH_PERSONS,
    FFMPEG_PRESET, FFMPEG_CRF, VIDEO_CODEC, AUDIO_CODEC
)

# 프로젝트 모델 import (conditional) 
MODEL_MANAGER_AVAILABLE = True
logger.info("DUAL 독립 버전 모드 (ModelManager 활성화)")


class FaceDetection:
    """
    얼굴 검출 결과 클래스
    
    MTCNN, Haar Cascade, DNN 등 다양한 검출 방법으로부터 얻은 얼굴 정보를 저장합니다.
    FaceNet 임베딩과 Person1/Person2 매칭 점수를 포함합니다.
    
    Attributes:
        x1, y1, x2, y2: 얼굴 바운딩 박스 좌표
        confidence: 검출 신뢰도 (0.0 ~ 1.0)
        width, height: 얼굴 크기
        center_x, center_y: 얼굴 중심 좌표
        area: 얼굴 영역 넓이
        embedding: FaceNet 임베딩 벡터 (torch.Tensor)
        p1_score, p2_score: Person1/Person2와의 하이브리드 매칭 점수
    """
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
    """
    검출 기반 얼굴 추적 클래스
    
    프레임별 얼굴 검출 결과를 바탕으로 부드러운 추적을 수행합니다.
    위치 기반 스무딩과 검출 히스토리를 통해 안정적인 추적을 제공합니다.
    
    Attributes:
        person_id: 추적 대상 ID ('person1' 또는 'person2')
        smoothing_alpha: 위치 스무딩 계수 (0.0 ~ 1.0)
        face_detection: 현재 프레임의 얼굴 검출 결과
        smooth_center: 스무딩된 얼굴 중심 좌표
        detection_history: 최근 검출 위치 히스토리
        tracking_confidence: 추적 신뢰도
    """
    
    def __init__(self, person_id: str, smoothing_alpha: float = 0.15):
        self.person_id = person_id
        self.smoothing_alpha = smoothing_alpha  # 0.3 → 0.15로 더 부드럽게
        
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
        
        # 크롭 크기 관련 - 완전 고정
        self.fixed_crop_size = None  # 첫 프레임에서 설정 후 불변
        
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
        """
        얼굴 검출 결과로 추적 상태 업데이트
        
        새로운 얼굴 검출 결과를 받아 추적 상태를 업데이트합니다.
        위치 스무딩, 히스토리 관리, 통계 업데이트를 수행합니다.
        
        Args:
            detection: 새로운 얼굴 검출 결과, None이면 검출 실패
            
        Returns:
            업데이트 성공 여부 (True: 검출 성공, False: 검출 실패)
        """
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
        """프레임 크기 기반 완전 고정 크롭 크기"""
        h, w = frame.shape[:2]
        
        # 완전 고정 크기: 프레임 높이의 45%로 고정
        if self.fixed_crop_size is None:
            self.fixed_crop_size = int(h * 0.45)  # 초기화시 한번만 설정
        
        return self.fixed_crop_size  # 항상 동일한 값 반환
    
    def get_crop_region(self, frame: np.ndarray, margin_factor: float = 2.5) -> np.ndarray:
        """
        고정 크기 얼굴 크롭 영역 추출
        
        현재 추적 중인 얼굴을 중심으로 고정된 크기의 크롭 영역을 반환합니다.
        얼굴이 검출되지 않은 경우 Person ID에 따라 화면의 좌반부 또는 우반부를 반환합니다.
        
        Args:
            frame: 입력 프레임 이미지
            margin_factor: 얼굴 크기 대비 크롭 마진 배율 (기본값: 2.5)
            
        Returns:
            크롭된 이미지 영역 (numpy 배열)
        """
        h, w = frame.shape[:2]
        
        # 얼굴이 없으면 기본 영역
        if self.smooth_center is None or self.last_good_bbox is None:
            if self.person_id == "Person1":
                crop = frame[0:h, 0:w//2]
            else:
                crop = frame[0:h, w//2:w]
            return cv2.resize(crop, (960, 1080))
        
        # 완전 고정 크롭 크기 (첫 프레임에서 한번만 설정)
        if self.fixed_crop_size is None:
            self.fixed_crop_size = int(h * 0.45)
        
        crop_size = self.fixed_crop_size  # 항상 동일
        
        # 크롭 중심점 (부드럽게 스무딩된 얼굴 중심)
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
    
    def get_adaptive_crop_region(self, frame: np.ndarray, face_size: float) -> np.ndarray:
        """얼굴 크기에 따라 적응적 크롭 영역 생성"""
        # 얼굴 크기에 따라 마진 동적 조정
        if face_size < 5000:  # 작은 얼굴 (멀리 있음)
            margin_factor = 3.5  # 더 큰 마진으로 주변 정보 포함
        elif face_size < 15000:  # 중간 얼굴
            margin_factor = 2.5  # 기본 마진
        else:  # 큰 얼굴 (가까이 있음)
            margin_factor = 2.0  # 더 작은 마진으로 얼굴에 집중
        
        # 기존 get_crop_region을 동적 마진으로 호출
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
        calculated_face_size = max(face_width, face_height)
        
        # 적응적 크롭 크기 계산 (얼굴 크기에 따라 마진 조정)
        adaptive_crop_size = int(calculated_face_size * margin_factor)
        adaptive_crop_size = max(100, min(min(w, h), adaptive_crop_size))  # 최소/최대 크기 제한
        
        # 크롭 중심점 (스무딩된 얼굴 중심)
        center_x, center_y = self.smooth_center
        
        # 크롭 영역 계산
        crop_x1 = max(0, int(center_x - adaptive_crop_size // 2))
        crop_y1 = max(0, int(center_y - adaptive_crop_size // 2))
        crop_x2 = min(w, crop_x1 + adaptive_crop_size)
        crop_y2 = min(h, crop_y1 + adaptive_crop_size)
        
        # 경계 조정 (정사각형 유지)
        if crop_x2 - crop_x1 < adaptive_crop_size:
            crop_x1 = max(0, crop_x2 - adaptive_crop_size)
        if crop_y2 - crop_y1 < adaptive_crop_size:
            crop_y1 = max(0, crop_y2 - adaptive_crop_size)
        
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
        """EMA 스무딩 적용 (더 부드러운 움직임)"""
        if self.smooth_center is None:
            return center
            
        # alpha를 0.15로 낮춰서 더 부드럽게 (기존 0.2)
        alpha = 0.15  # self.smoothing_alpha → 0.15 고정
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
    """
    임베딩 기반 고급 얼굴 추적 클래스
    
    FaceNet 임베딩 벡터를 사용하여 얼굴 ID를 지속적으로 추적합니다.
    임베딩 히스토리 관리와 유사도 계산을 통해 동일 인물 여부를 판단합니다.
    
    Attributes:
        face_embeddings: 임베딩 벡터 히스토리 리스트
        reference_embedding: 대표 임베딩 (평균)
        embedding_threshold: 동일 인물 판단 임계값
        max_embeddings: 유지할 최대 임베딩 개수
        similarity_scores: 최근 유사도 점수 히스토리
    """
    
    def __init__(self, person_id: str, smoothing_alpha: float = 0.3):
        super().__init__(person_id, smoothing_alpha)
        
        # 고급 임베딩 관리자 (SmartEmbeddingManager 비활성화)
        self.smart_embedding_manager = None
        logger.warning(f"{person_id}: SmartEmbeddingManager 비활성화")
        
        # 개별 임베딩 추적 (디버깅용)
        self.face_embeddings = []  # 백업 히스토리
        self.reference_embedding = None  # 대표 임베딩 (평균)
        self.max_embeddings = 10  # 최대 10개 임베딩 유지
        self.embedding_threshold = 0.35  # 임베딩 매칭 임계값
        
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
            except (RuntimeError, AttributeError, ValueError, TypeError) as e:
                logger.warning(f"고급 유사도 계산 실패: {e}, 기본 방법 사용")
            except Exception as e:
                logger.error(f"예상치 못한 유사도 계산 오류: {e}, 기본 방법 사용")
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


class SimplePreScanner:
    """
    비디오 사전 스캔 클래스
    
    비디오 전체를 빠르게 스캔하여 주요 2명의 평균 위치를 찾습니다.
    일정 간격으로 프레임을 샘플링하여 계산 비용을 최소화합니다.
    
    Attributes:
        debug_mode: 디버그 출력 활성화 여부
        face_positions: 수집된 얼굴 위치 데이터 리스트
    
    Methods:
        quick_scan: 비디오를 빠르게 스캔하여 2명의 평균 위치 반환
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.face_positions = []  # [(x, y, area, frame_idx), ...]
        
    def quick_scan(self, video_path: str, sample_rate: int = 30) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        빠른 사전 스캔으로 주요 2명의 평균 위치 찾기
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 샘플링 레이트 (기본 30프레임마다)
            
        Returns:
            (person1_avg_pos, person2_avg_pos) 또는 None
        """
        import cv2
        
        logger.debug(f"사전 분석 시작: {sample_rate}프레임마다 샘플링")
        
        # 얼굴 검출기 로드 (기존 시스템과 동일한 경로 사용)
        face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"비디오 열기 실패: {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"총 {total_frames}프레임, {total_frames/fps:.1f}초")
        logger.debug(f"분석할 프레임: {total_frames//sample_rate}개")
        
        self.face_positions = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 샘플링: sample_rate마다만 처리
                if frame_idx % sample_rate == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in faces:
                        center_x = x + w / 2
                        center_y = y + h / 2
                        area = w * h
                        self.face_positions.append((center_x, center_y, area, frame_idx))
                
                frame_idx += 1
                
                # 진행률 표시 (매 300프레임마다)
                if frame_idx % 300 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.debug(f"진행률: {progress:.1f}% ({len(self.face_positions)}개 얼굴 발견)")
        
        finally:
            cap.release()
        
        logger.info(f"스캔 완료: {len(self.face_positions)}개 얼굴 발견")
        
        if len(self.face_positions) < 10:
            logger.warning("충분한 얼굴 데이터 없음")
            return None
        
        # 좌우 기반으로 클러스터링
        return self._cluster_by_position()
    
    def _cluster_by_position(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """위치 기반 단순 클러스터링"""
        if not self.face_positions:
            return None
        
        # X 좌표로 정렬
        sorted_faces = sorted(self.face_positions, key=lambda x: x[0])
        
        # 중간값으로 좌우 구분
        mid_x = sorted_faces[len(sorted_faces)//2][0]
        
        left_faces = [f for f in self.face_positions if f[0] < mid_x]
        right_faces = [f for f in self.face_positions if f[0] >= mid_x]
        
        if len(left_faces) < 3 or len(right_faces) < 3:
            logger.warning("좌우 얼굴 데이터 부족")
            return None
        
        # 각 그룹의 평균 위치 계산
        left_avg_x = sum(f[0] for f in left_faces) / len(left_faces)
        left_avg_y = sum(f[1] for f in left_faces) / len(left_faces)
        
        right_avg_x = sum(f[0] for f in right_faces) / len(right_faces)
        right_avg_y = sum(f[1] for f in right_faces) / len(right_faces)
        
        person1_pos = (left_avg_x, left_avg_y)
        person2_pos = (right_avg_x, right_avg_y)
        
        logger.debug(f"사전 분석 결과:")
        logger.debug(f"왼쪽 그룹: {len(left_faces)}개, 평균 위치={person1_pos}")
        logger.debug(f"오른쪽 그룹: {len(right_faces)}개, 평균 위치={person2_pos}")
        
        return person1_pos, person2_pos


class StablePositionTracker:
    """
    위치 연속성 기반 안정적 얼굴 추적 클래스
    
    얼굴의 위치 히스토리를 분석하여 Person1/Person2를 일관되게 할당합니다.
    초기화 단계에서 크기 기반 할당을 수행하고, 이후 위치 연속성을 유지합니다.
    
    Attributes:
        person1_history, person2_history: 각 인물의 위치 히스토리
        history_size: 저장할 최대 위치 개수
        init_threshold: 초기화에 필요한 프레임 수
        max_distance_threshold: 동일 인물로 인식할 최대 거리
        debug_mode: 디버그 출력 활성화 여부
        prescan_profiles: 사전 스캔된 평균 위치 프로파일
    """
    
    def __init__(self, debug_mode: bool = False, prescan_profiles: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None):
        # 위치 히스토리
        self.person1_history = []  # [(x, y), ...] 최근 위치들
        self.person2_history = []
        self.history_size = 10  # 최대 10개 위치 저장
        
        # 신뢰도 히스토리 (스무딩용)
        self.person1_confidence_history = []  # List[float]
        self.person2_confidence_history = []  # List[float]
        
        # 초기화 관리
        self.init_frames = 0
        self.init_threshold = 30  # 30프레임 동안 크기 기반 초기화
        self.is_initialized = False
        
        # 디버그
        self.debug_mode = debug_mode
        
        # 토레루스 설정
        self.min_confidence = 0.3  # 최소 신뢰도
        self.smoothing_factor = 0.7  # 위치 스무딩 (0.7 = 70% 이전, 30% 새로운)
        self.max_distance_threshold = 200  # 최대 허용 거리 (픽셀) - 기존 100에서 200으로 완화
        
        # Speaker reference 임베딩 (외부에서 설정)
        self.speaker1_reference = None
        self.speaker2_reference = None
        self.model_manager = None  # ModelManager 접근용
        
        # 사전 스캔 결과로 초기화
        if prescan_profiles:
            person1_pos, person2_pos = prescan_profiles
            # 히스토리를 사전 스캔 결과로 미리 채우기
            self.person1_history = [person1_pos] * 5
            self.person2_history = [person2_pos] * 5
            self.is_initialized = True
            self.init_frames = self.init_threshold  # 초기화 스킵
            logger.info(f"사전 스캔 프로파일로 초기화: P1={person1_pos}, P2={person2_pos}")
        else:
            logger.debug(f"StablePositionTracker 초기화 (초기화 {self.init_threshold}프레임, 히스토리 {self.history_size}개)")
    
    def track_faces(self, faces: List[FaceDetection], frame_idx: int, frame: np.ndarray = None) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """
        위치 기반 안정적 얼굴 추적 메인 메서드
        
        초기화 단계에서는 크기 기반 할당을 수행하고,
        이후에는 위치 연속성을 기반으로 안정적인 추적을 수행합니다.
        
        Args:
            faces: 검출된 얼굴 리스트
            frame_idx: 현재 프레임 번호
            frame: 현재 프레임 이미지 (옵션)
            
        Returns:
            (Person1 얼굴, Person2 얼굴) 튜플
        """
        
        if not faces:
            return None, None
        
        # 초기화 단계: 크기 기반 안정화
        if self.init_frames < self.init_threshold:
            return self._initialize_tracking(faces, frame_idx, frame)
        
        # 추적 단계: 위치 연속성 기반
        return self._track_by_position(faces, frame_idx)
    
    def _initialize_tracking(self, faces: List[FaceDetection], frame_idx: int, frame: np.ndarray = None) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """초기 30프레임: 크기 기반으로 안정적인 초기화"""
        self.init_frames += 1
        
        # 사전 스캔 프로파일이 있으면 위치 기반 할당 우선
        if self.person1_history and self.person2_history and len(self.person1_history) >= 5:
            person1_face, person2_face = self._assign_by_expected_position(faces)
        elif self.speaker1_reference is not None and self.speaker2_reference is not None:
            # 임베딩 기반 순수 할당 (위치 무관)
            person1_face, person2_face = self._assign_by_embedding_only(faces, frame)
        else:
            # 폴백: 중요도 기반 할당 (중앙 + 크기 + 신뢰도)
            person1_face, person2_face = self._assign_by_importance(faces)
        
        # 위치 히스토리 구축
        if person1_face:
            self._update_person_history(1, person1_face)
        
        if person2_face:
            self._update_person_history(2, person2_face)
        
        # 초기화 완료 체크
        if self.init_frames >= self.init_threshold:
            self.is_initialized = True
            logger.debug(f"초기화 완료 (프레임 {frame_idx}): P1 히스토리={len(self.person1_history)}, P2 히스토리={len(self.person2_history)}")
        
        if self.debug_mode and frame_idx % 10 == 0:
            size1 = person1_face.area if person1_face else 0
            size2 = person2_face.area if person2_face else 0
            logger.debug(f"초기화 {self.init_frames}/{self.init_threshold}: P1=위치{person1_face.center if person1_face else None}, P2=위치{person2_face.center if person2_face else None}")
        
        return person1_face, person2_face
    
    def _track_by_position(self, faces: List[FaceDetection], frame_idx: int) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """위치 연속성 기반 추적 (나중에 임베딩 하이브리드 가능)"""
        
        # TODO: 임베딩 사용 가능하면 하이브리드
        # if self.embedding_enabled:
        #     embedding_scores = self._calculate_embedding_scores(faces)
        #     position_scores = self._calculate_position_scores(faces)
        #     final_scores = 0.7 * position_scores + 0.3 * embedding_scores
        # else:
        #     final_scores = self._calculate_position_scores(faces)
        
        # 현재는 위치만 사용
        return self._assign_by_position(faces, frame_idx)
    
    def _assign_by_position(self, faces: List[FaceDetection], frame_idx: int) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """위치 기반 얼굴 매칭"""
        
        # 사용된 얼굴 추적
        used_faces = set()
        person1_face = None
        person2_face = None
        
        # Person1 찾기: 가장 가까운 얼굴
        if self.person1_history:
            predicted_p1_pos = self._get_predicted_position(self.person1_history)
            min_dist = float('inf')
            best_face_idx = -1
            
            for i, face in enumerate(faces):
                if face.confidence < self.min_confidence:
                    continue
                
                dist = self._calculate_distance(face.center, predicted_p1_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_face_idx = i
            
            if best_face_idx >= 0 and min_dist <= self.max_distance_threshold:
                person1_face = faces[best_face_idx]
                used_faces.add(best_face_idx)
                
                # 위치 히스토리 업데이트
                self._update_person_history(1, person1_face)
                    
                if self.debug_mode:
                    logger.debug(f"P1 거리 매칭: {min_dist:.1f}px <= {self.max_distance_threshold}px")
            elif self.debug_mode:
                logger.error(f"P1 거리 초과: {min_dist:.1f}px > {self.max_distance_threshold}px")
        
        # Person2 찾기: 남은 얼굴 중 가장 가까운
        if self.person2_history:
            predicted_p2_pos = self._get_predicted_position(self.person2_history)
            min_dist = float('inf')
            best_face_idx = -1
            
            for i, face in enumerate(faces):
                if i in used_faces or face.confidence < self.min_confidence:
                    continue
                
                dist = self._calculate_distance(face.center, predicted_p2_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_face_idx = i
            
            if best_face_idx >= 0 and min_dist <= self.max_distance_threshold:
                person2_face = faces[best_face_idx]
                
                # 위치 히스토리 업데이트
                self._update_person_history(2, person2_face)
                    
                if self.debug_mode:
                    logger.debug(f"P2 거리 매칭: {min_dist:.1f}px <= {self.max_distance_threshold}px")
            elif self.debug_mode:
                logger.error(f"P2 거리 초과: {min_dist:.1f}px > {self.max_distance_threshold}px")
        
        # 단일 얼굴 처리: Person2 가능성 체크
        if len(faces) == 1 and person1_face and not person2_face:
            face = faces[0]
            if self._is_closer_to_person2(face):
                if self.debug_mode:
                    logger.debug(f"단일얼굴을 P2로 재할당: {face.center}")
                person1_face = None
                person2_face = face
                # Person2 히스토리 업데이트
                self._update_person_history(2, person2_face)
        
        # Person2 폴백 메커니즘: 두 번째로 큰 얼굴 자동 할당
        if not person2_face and len(faces) >= 2 and person1_face:
            # Person1으로 사용되지 않은 얼굴들 중 가장 큰 것을 Person2로
            unused_faces = [(i, face) for i, face in enumerate(faces) if i not in used_faces and face.confidence >= self.min_confidence]
            if unused_faces:
                # 크기순으로 정렬하여 가장 큰 얼굴 선택
                unused_faces.sort(key=lambda x: x[1].area, reverse=True)
                person2_face = unused_faces[0][1]
                
                # Person2 히스토리 업데이트
                self._update_person_history(2, person2_face)
                
                if self.debug_mode:
                    logger.debug(f"폴백 메커니즘: P2에 두 번째 큰 얼굴 할당 {person2_face.center}")
        
        # 디버그 정보 출력
        if self.debug_mode and frame_idx % 30 == 0:
            p1_info = f"pos={person1_face.center}, size={person1_face.area:.0f}" if person1_face else "없음"
            p2_info = f"pos={person2_face.center}, size={person2_face.area:.0f}" if person2_face else "없음"
            logger.debug(f"프레임 {frame_idx} 추적: P1={p1_info}, P2={p2_info}")
            
            if person1_face and person2_face:
                dist_p1 = self._calculate_distance(person1_face.center, self._get_predicted_position(self.person1_history[:-1]))
                dist_p2 = self._calculate_distance(person2_face.center, self._get_predicted_position(self.person2_history[:-1]))
                logger.debug(f"이동 거리: P1={dist_p1:.1f}px, P2={dist_p2:.1f}px")
        
        return person1_face, person2_face
    
    def _assign_by_expected_position(self, faces: List[FaceDetection]) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """사전 스캔 결과를 바탕으로 얼굴을 예상 위치에 할당"""
        if not faces:
            if self.debug_mode:
                logger.debug("_assign_by_expected_position: 얼굴 없음")
            return None, None
        
        person1_face = None
        person2_face = None
        used_faces = set()
        
        if self.debug_mode:
            logger.debug(f"_assign_by_expected_position: {len(faces)}개 얼굴, P1히스토리={len(self.person1_history)}, P2히스토리={len(self.person2_history)}")
        
        # Person1 할당 (왼쪽, 첫 번째 사람)
        if self.person1_history:
            predicted_p1_pos = self._get_predicted_position(self.person1_history)
            min_dist = float('inf')
            best_face_idx = -1
            
            if self.debug_mode:
                logger.debug(f"P1 예상위치: {predicted_p1_pos}")
            
            for i, face in enumerate(faces):
                if face.confidence < self.min_confidence:
                    if self.debug_mode:
                        print(f"   P1 얼굴{i}: 신뢰도 부족 {face.confidence} < {self.min_confidence}")
                    continue
                
                dist = self._calculate_distance(face.center, predicted_p1_pos)
                if self.debug_mode:
                    print(f"   P1 얼굴{i}: pos={face.center}, 거리={dist:.1f}px")
                if dist < min_dist:
                    min_dist = dist
                    best_face_idx = i
            
            if best_face_idx >= 0:
                person1_face = faces[best_face_idx]
                used_faces.add(best_face_idx)
                if self.debug_mode:
                    logger.debug(f"P1 할당: 얼굴{best_face_idx}, 거리={min_dist:.1f}px")
            elif self.debug_mode:
                logger.error("P1 할당 실패: 모든 얼굴이 조건 불만족")
        
        # Person2 할당 (오른쪽, 두 번째 사람)
        if self.person2_history:
            predicted_p2_pos = self._get_predicted_position(self.person2_history)
            min_dist = float('inf')
            best_face_idx = -1
            
            if self.debug_mode:
                logger.debug(f"P2 예상위치: {predicted_p2_pos}")
            
            for i, face in enumerate(faces):
                if i in used_faces:
                    if self.debug_mode:
                        print(f"   P2 얼굴{i}: 이미 P1에서 사용됨")
                    continue
                if face.confidence < self.min_confidence:
                    if self.debug_mode:
                        print(f"   P2 얼굴{i}: 신뢰도 부족 {face.confidence} < {self.min_confidence}")
                    continue
                
                dist = self._calculate_distance(face.center, predicted_p2_pos)
                if self.debug_mode:
                    print(f"   P2 얼굴{i}: pos={face.center}, 거리={dist:.1f}px")
                if dist < min_dist:
                    min_dist = dist
                    best_face_idx = i
            
            if best_face_idx >= 0:
                person2_face = faces[best_face_idx]
                if self.debug_mode:
                    logger.debug(f"P2 할당: 얼굴{best_face_idx}, 거리={min_dist:.1f}px")
            else:
                if self.debug_mode:
                    logger.error(f"P2 할당 실패: 사용가능한 얼굴 없음 (총 {len(faces)}개, 사용됨 {used_faces})")
        else:
            if self.debug_mode:
                logger.error("P2 히스토리 없음: 초기화되지 않음")
        
        return person1_face, person2_face
    
    def _assign_by_left_right_position(self, faces: List[FaceDetection]) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """좌우 위치 기반 할당 (좌우 섞임 방지)"""
        if not faces:
            return None, None
        
        # 1920x1080 기준 중앙선(x=960)으로 좌우 구분
        left_faces = []  # 왼쪽 (Person1 후보)
        right_faces = []  # 오른쪽 (Person2 후보)
        center_line = 960  # 화면 중앙
        
        for face in faces:
            if face.confidence < self.min_confidence:
                continue
                
            if face.center_x < center_line:
                left_faces.append(face)
            else:
                right_faces.append(face)
        
        # 각 영역에서 가장 큰 얼굴 선택 (앞사람 우선)
        person1_face = None
        person2_face = None
        
        if left_faces:
            # 왼쪽에서 가장 큰 얼굴 = Person1
            left_faces.sort(key=lambda f: f.area, reverse=True)
            person1_face = left_faces[0]
            
        if right_faces:
            # 오른쪽에서 가장 큰 얼굴 = Person2
            right_faces.sort(key=lambda f: f.area, reverse=True)
            person2_face = right_faces[0]
        
        # 한쪽에만 얼굴이 있는 경우 크기순으로 할당
        if person1_face is None and person2_face is None:
            # 모든 얼굴이 중앙선 근처 -> 크기순 할당
            all_valid_faces = [f for f in faces if f.confidence >= self.min_confidence]
            if len(all_valid_faces) >= 2:
                all_valid_faces.sort(key=lambda f: f.area, reverse=True)
                person1_face = all_valid_faces[0]
                person2_face = all_valid_faces[1]
            elif len(all_valid_faces) == 1:
                person1_face = all_valid_faces[0]
        elif person1_face is None and person2_face is not None:
            # 오른쪽에만 있음 -> 가장 큰것을 Person1으로, 두번째를 Person2로
            if len(right_faces) >= 2:
                person1_face = right_faces[0]  # 가장 큰것
                person2_face = right_faces[1]  # 두번째
        elif person1_face is not None and person2_face is None:
            # 왼쪽에만 있음 -> 가장 큰것을 Person1으로, 두번째를 Person2로
            if len(left_faces) >= 2:
                person1_face = left_faces[0]  # 가장 큰것
                person2_face = left_faces[1]  # 두번째
        
        if self.debug_mode:
            logger.debug(f"좌우 기반 할당: 왼쪽={len(left_faces)}개, 오른쪽={len(right_faces)}개")
            if person1_face:
                print(f"   P1: 위치{person1_face.center}, 크기{person1_face.area:.0f}")
            if person2_face:
                print(f"   P2: 위치{person2_face.center}, 크기{person2_face.area:.0f}")
        
        return person1_face, person2_face
    
    def _assign_by_embedding_only(self, faces: List[FaceDetection], frame: np.ndarray = None) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """순수 임베딩 기반 할당 (위치 무관)"""
        best_p1_face = None
        best_p2_face = None
        best_p1_score = -1
        best_p2_score = -1
        
        # Speaker reference 임베딩이 필요
        if not hasattr(self, 'speaker1_reference') or self.speaker1_reference is None:
            return None, None
        if not hasattr(self, 'speaker2_reference') or self.speaker2_reference is None:
            return None, None
        
        import torch.nn.functional as F
        
        for face in faces:
            if face.confidence < self.min_confidence:
                continue
            
            # 얼굴에서 임베딩 추출 (model_manager를 통해 실시간 추출)
            face_embedding = None
            if frame is not None and hasattr(self, 'model_manager') and self.model_manager is not None:
                try:
                    face_crop = self.model_manager.extract_face_crop(face, frame)
                    if face_crop is not None:
                        face_embedding = self.model_manager.get_embedding(face_crop)
                except (AttributeError, RuntimeError, ValueError) as e:
                    logger.debug(f"임베딩 추출 실패 (계속 진행): {e}")
                except Exception as e:
                    logger.warning(f"예상치 못한 임베딩 추출 오류: {e}")
            
            if face_embedding is not None:
                # 임베딩 유사도 계산 (코사인 유사도)
                p1_sim = F.cosine_similarity(
                    face_embedding.unsqueeze(0), 
                    self.speaker1_reference.unsqueeze(0)
                ).item()
                p2_sim = F.cosine_similarity(
                    face_embedding.unsqueeze(0), 
                    self.speaker2_reference.unsqueeze(0)
                ).item()
                
                # 정규화 (-1~1 → 0~1)
                p1_sim = (p1_sim + 1) / 2
                p2_sim = (p2_sim + 1) / 2
                
                # 임계값 0.4 이상인 경우만 고려
                if p1_sim > best_p1_score and p1_sim > 0.4:
                    best_p1_face = face
                    best_p1_score = p1_sim
                    
                if p2_sim > best_p2_score and p2_sim > 0.4:
                    best_p2_face = face
                    best_p2_score = p2_sim
        
        # 동일 얼굴 중복 방지
        if best_p1_face == best_p2_face and best_p1_face is not None:
            if best_p1_score > best_p2_score:
                best_p2_face = None
            else:
                best_p1_face = None
        
        if self.debug_mode:
            logger.debug(f"임베딩 기반 할당: P1점수={best_p1_score:.3f}, P2점수={best_p2_score:.3f}")
        
        return best_p1_face, best_p2_face
    
    def _assign_by_importance(self, faces: List[FaceDetection]) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """중요도 기반 할당 (크기 + 중앙 + 신뢰도)"""
        if not faces:
            return None, None
        
        scores = []
        for face in faces:
            if face.confidence < self.min_confidence:
                continue
            
            # 중앙 거리 점수 (1920x1080 기준)
            center_dist = ((face.center_x - 960)**2 + (face.center_y - 540)**2)**0.5
            center_score = max(0, 1 - center_dist / 800)
            
            # 크기 점수 (정규화, 50000px² 기준)
            size_score = min(face.area / 50000, 1.0)
            
            # 최종 점수 (중앙 50% + 크기 30% + 신뢰도 20%)
            importance = (0.5 * center_score + 
                         0.3 * size_score + 
                         0.2 * face.confidence)
            scores.append((face, importance))
        
        if not scores:
            return None, None
        
        # 점수 순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        
        person1_face = scores[0][0] if len(scores) >= 1 else None
        person2_face = scores[1][0] if len(scores) >= 2 else None
        
        if self.debug_mode:
            if len(scores) >= 2:
                logger.debug(f"중요도 기반 할당: P1점수={scores[0][1]:.3f}, P2점수={scores[1][1]:.3f}")
            elif len(scores) == 1:
                logger.debug(f"중요도 기반 할당: P1점수={scores[0][1]:.3f}, P2없음")
            else:
                print(f"🔍 중요도 기반 할당: 점수 계산 실패")
        
        return person1_face, person2_face
    
    def _is_closer_to_person2(self, face: FaceDetection) -> bool:
        """단일 얼굴이 Person2에 더 가까운지 판단"""
        if not self.person1_history or not self.person2_history:
            return False
        
        p1_pos = self._get_predicted_position(self.person1_history)
        p2_pos = self._get_predicted_position(self.person2_history)
        
        dist_to_p1 = self._calculate_distance(face.center, p1_pos)
        dist_to_p2 = self._calculate_distance(face.center, p2_pos)
        
        if self.debug_mode:
            logger.debug(f"단일얼굴 거리비교: P1={dist_to_p1:.1f}px, P2={dist_to_p2:.1f}px")
        
        # Person2가 더 가까우면 True
        return dist_to_p2 < dist_to_p1
    
    def _get_predicted_position(self, history: List[Tuple[float, float]]) -> Tuple[float, float]:
        """위치 히스토리로 다음 위치 예측 (속도 벡터 + 스무딩)"""
        if not history:
            return (0, 0)
        
        if len(history) == 1:
            return history[0]
        
        # 1. 기본 가중평균 예측
        recent = history[-3:] if len(history) >= 3 else history
        weights = [0.2, 0.3, 0.5] if len(recent) == 3 else [0.4, 0.6] if len(recent) == 2 else [1.0]
        
        weighted_x = sum(pos[0] * weight for pos, weight in zip(recent, weights[-len(recent):]))
        weighted_y = sum(pos[1] * weight for pos, weight in zip(recent, weights[-len(recent):]))
        
        # 2. 속도 벡터 기반 예측 (5개 이상 히스토리가 있을 때)
        if len(history) >= 5:
            # 최근 4개 프레임의 속도 벡터 계산
            recent_positions = history[-4:]
            velocities = []
            
            for i in range(1, len(recent_positions)):
                vx = recent_positions[i][0] - recent_positions[i-1][0]
                vy = recent_positions[i][1] - recent_positions[i-1][1]
                velocities.append((vx, vy))
            
            # 평균 속도 벡터
            if velocities:
                avg_vx = sum(v[0] for v in velocities) / len(velocities)
                avg_vy = sum(v[1] for v in velocities) / len(velocities)
                
                # 속도 기반 예측 위치
                last_pos = history[-1]
                velocity_predicted_x = last_pos[0] + avg_vx * 1.5  # 1.5프레임 앞 예측
                velocity_predicted_y = last_pos[1] + avg_vy * 1.5
                
                # 가중평균 50% + 속도예측 50% 결합
                final_x = 0.5 * weighted_x + 0.5 * velocity_predicted_x
                final_y = 0.5 * weighted_y + 0.5 * velocity_predicted_y
                
                return (final_x, final_y)
        
        # 기본 가중평균 반환
        return (weighted_x, weighted_y)
    
    def _update_person_history(self, person_id: int, face: FaceDetection):
        """신뢰도와 함께 위치 히스토리 업데이트"""
        if person_id == 1:
            self.person1_history.append(face.center)
            self.person1_confidence_history.append(face.confidence)
            if len(self.person1_history) > self.history_size:
                self.person1_history.pop(0)
                self.person1_confidence_history.pop(0)
        elif person_id == 2:
            self.person2_history.append(face.center)
            self.person2_confidence_history.append(face.confidence)
            if len(self.person2_history) > self.history_size:
                self.person2_history.pop(0)
                self.person2_confidence_history.pop(0)
    
    def get_average_confidence(self, person_id: int) -> float:
        """최근 신뢰도 평균 반환"""
        if person_id == 1 and self.person1_confidence_history:
            return sum(self.person1_confidence_history) / len(self.person1_confidence_history)
        elif person_id == 2 and self.person2_confidence_history:
            return sum(self.person2_confidence_history) / len(self.person2_confidence_history)
        return 0.5  # 기본값
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """두 위치 간 유클리드 거리 계산"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """추적 통계 정보 반환"""
        return {
            'is_initialized': self.is_initialized,
            'init_frames': self.init_frames,
            'init_threshold': self.init_threshold,
            'person1_history_length': len(self.person1_history),
            'person2_history_length': len(self.person2_history),
            'person1_last_position': self.person1_history[-1] if self.person1_history else None,
            'person2_last_position': self.person2_history[-1] if self.person2_history else None
        }


class DualFaceTrackingSystem:
    """
    듀얼 얼굴 추적 메인 시스템 클래스
    
    2명의 얼굴을 검출, 추적, 분류하여 1920x1080 스플릿 스크린 비디오를 생성합니다.
    MTCNN 얼굴 검출, FaceNet 임베딩, 위치 기반 추적을 통합하여 안정적인 결과를 제공합니다.
    
    주요 기능:
    - 다중 검출 방법 지원 (MTCNN, Haar, DNN, MediaPipe)
    - 하이브리드 얼굴 매칭 (임베딩 + 위치 + 크기)
    - 실시간 얼굴 크롭 및 스플릿 스크린 합성
    - 오디오 보존 및 FFmpeg 후처리
    - 검출되지 않은 구간 자동 트리밍
    
    Attributes:
        input_path: 입력 비디오 경로
        output_path: 출력 비디오 경로  
        mode: 추적 모드 ('auto', 'manual' 등)
        detection_methods: 사용할 검출 방법 리스트
        stable_tracker: 위치 기반 안정적 추적기
        timeline: 검출 타임라인 관리자
    """
    
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
        
        # 위치 기반 안정적 추적 시스템
        self.debug_mode = False  # 디버그 모드 (크기 정보 출력)
        self.size_stabilize = False  # 크기 기반 안정화 사용 여부
        self.prescan_enabled = getattr(args, 'prescan', False)  # 사전 스캔 옵션
        
        # AutoSpeakerDetector 옵션 (새로운 자동 화자 선정 시스템)
        self.auto_speaker_enabled = getattr(args, 'auto_speaker', True)  # 기본적으로 활성화
        
        # 1분 분석 모드 옵션
        self.one_minute_mode = getattr(args, 'one_minute', False)  # 1분 집중 분석 모드
        
        # Phase 3: Hungarian Matching 옵션
        self.hungarian_mode = getattr(args, 'hungarian', False)  # Hungarian Matching 사용
        
        # 위치 기반 추적자는 process()에서 프로파일과 함께 초기화
        self.position_tracker = None
        
        # 자동 선정된 화자 정보 (Reference 임베딩)
        self.speaker1_reference = None  # 화자1 대표 임베딩
        self.speaker2_reference = None  # 화자2 대표 임베딩
        self.speaker_similarity_threshold = 0.35  # Reference와 매칭 임계값 (0.5 → 0.35, 퀄리티 개선)
        
        # Phase 1: Identity-based tracking 강화
        self.MIN_FACE_SIZE = 120  # 픽셀 단위, 작은 얼굴 무시 (배경 인물 필터링)
        
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
        
        # 검출 타임라인 (오디오/트리밍용)
        self.detection_timeline = []  # [(frame_idx, has_person1, has_person2), ...]
        
        print(f"🏗️ DualFaceTrackingSystem 초기화 완료")
        print(f"   📥 입력: {self.input_path}")
        print(f"   📤 출력: {self.output_path}")
        logger.debug(f"검출 간격: {self.detection_interval}프레임")
        print(f"   📏 크롭 배율: {self.margin_factor}x")
        
    def _initialize_models(self):
        """얼굴 검출 모델 초기화"""
        print("🏗️ 얼굴 검출 모델 초기화 중...")
        
        # 방법 1: 우리의 ModelManager 사용 (최우선)
        if MODEL_MANAGER_AVAILABLE:
            try:
                from .model_manager import ModelManager
                model_manager = ModelManager()
                if model_manager.mtcnn is not None:
                    self.mtcnn = model_manager.mtcnn
                    self.detection_method = "mtcnn_manager"
                    logger.info("ModelManager MTCNN 모델 로드 완료")
                    print(f"   📍 디바이스: {model_manager.device}")
                    print("   🧠 고성능 얼굴 검출 활성화 (MTCNN)")
                    return
            except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
                logger.warning(f"ModelManager MTCNN 로드 실패: {e}")
            except Exception as e:
                logger.error(f"예상치 못한 ModelManager 로드 오류: {e}")
        
        # 방법 2: 직접 MTCNN 로드
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                keep_all=True,
                selection_method='largest_over_threshold'
            )
            self.detection_method = "mtcnn_direct"
            logger.info("facenet-pytorch MTCNN 직접 로드 완료")
            return
        except (ImportError, ModuleNotFoundError, RuntimeError, TypeError) as e:
            logger.warning(f"직접 MTCNN 로드 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 MTCNN 로드 오류: {e}")
        
        # 방법 3: 기존 프로젝트의 MTCNN 시도 (상위 디렉토리)
        try:
            # 상위 프로젝트 경로 추가
            parent_project = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_project))
            sys.path.insert(0, str(parent_project / "src"))
            
            from face_tracker.core.models import ModelManager
            self.mtcnn, self.resnet = ModelManager.get_models()
            self.detection_method = "mtcnn"
            logger.info("상위 프로젝트 MTCNN + FaceNet 모델 로드 완료")
            return
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
            logger.warning(f"상위 프로젝트 MTCNN 로드 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 상위 프로젝트 로드 오류: {e}")
        
        # 방법 4: OpenCV Haar Cascade (폴백)
        try:
            # 확인된 경로를 직접 사용
            cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            
            if Path(cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.face_cascade.empty():
                    self.detection_method = "haar"
                    logger.info("OpenCV Haar Cascade 모델 로드 완료 (폴백)")
                    print(f"   📍 경로: {cascade_path}")
                    logger.warning("성능 제한: MTCNN 대신 Haar Cascade 사용")
                    return
                else:
                    logger.warning(f"Haar Cascade 생성 실패: {cascade_path}")
            else:
                logger.warning(f"Haar Cascade 파일 없음: {cascade_path}")
                
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Haar Cascade 로드 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 Haar Cascade 로드 오류: {e}")
        
        # 방법 5: MediaPipe 얼굴 검출
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.3)
            self.detection_method = "mediapipe"
            logger.info("MediaPipe 얼굴 검출 모델 로드 완료")
            return
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as e:
            logger.warning(f"MediaPipe 로드 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 MediaPipe 로드 오류: {e}")
        
        raise RuntimeError("❌ 모든 얼굴 검출 모델 로드 실패")
    
    def _initialize_facenet(self):
        """FaceNet 모델 초기화 (얼굴 임베딩용)"""
        print("🧠 FaceNet 모델 초기화 중...")
        
        if not MODEL_MANAGER_AVAILABLE:
            logger.warning("ModelManager를 사용할 수 없음. 임베딩 기능 비활성화")
            return
            
        try:
            # ModelManager로 FaceNet 모델 로드
            from .model_manager import ModelManager
            self.model_manager = ModelManager()
            self.resnet = self.model_manager.facenet
            
            # 얼굴 전처리 변환 (FaceNet용)
            self.face_transform = transforms.Compose([
                transforms.Resize((160, 160)),  # FaceNet 입력 크기
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            logger.info("FaceNet 모델 로드 완료")
            print(f"   📍 디바이스: {self.model_manager.device}")
            print("   🧠 얼굴 임베딩 기능 활성화")
            
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as e:
            logger.warning(f"FaceNet 모델 초기화 실패: {e}")
            print("   🔄 임베딩 없이 위치 기반 추적만 사용")
            self.model_manager = None
        except Exception as e:
            logger.error(f"예상치 못한 FaceNet 초기화 오류: {e}")
            print("   🔄 임베딩 없이 위치 기반 추적만 사용")
            self.model_manager = None
            self.resnet = None
            self.face_transform = None
    
    def generate_face_embedding(self, face_crop: np.ndarray) -> Optional[torch.Tensor]:
        """
        얼굴 크롭 이미지에서 FaceNet 임베딩 생성
        
        입력된 얼굴 크롭 이미지를 FaceNet 모델에 통과시켜 512차원 임베딩 벡터를 생성합니다.
        BGR → RGB 변환, 전처리, GPU 이동 등의 과정을 거쳐 최종 임베딩을 반환합니다.
        
        Args:
            face_crop: 얼굴 크롭 이미지 (BGR 형식 numpy 배열)
            
        Returns:
            512차원 임베딩 텐서, 실패시 None
            
        Raises:
            Exception: 임베딩 생성 과정에서 오류 발생시
        """
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
            logger.warning(f"임베딩 생성 실패: {e}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        프레임에서 얼굴 검출 수행
        
        설정된 검출 방법(MTCNN, Haar, DNN, MediaPipe)을 사용하여 얼굴을 검출합니다.
        검출된 얼굴들은 신뢰도 순으로 정렬되고 최소 크기 필터링을 거칩니다.
        
        Args:
            frame: 입력 프레임 (numpy 배열)
            
        Returns:
            검출된 얼굴 리스트 (FaceDetection 객체들)
        """
        faces = []
        
        try:
            # 디버깅: 검출 방법 확인
            if not self._debug_detection_logged:
                logger.debug(f"검출 방법: {self.detection_method}")
                self._debug_detection_logged = True
            
            if self.detection_method == "mtcnn":
                faces = self._detect_faces_mtcnn(frame)
            elif self.detection_method == "mtcnn_manager":
                faces = self._detect_faces_mtcnn_direct(frame)
            elif self.detection_method == "mtcnn_direct":
                faces = self._detect_faces_mtcnn_direct(frame)
            elif self.detection_method == "haar":
                faces = self._detect_faces_haar(frame)
            elif self.detection_method == "mediapipe":
                faces = self._detect_faces_mediapipe(frame)
            elif self.detection_method == "dnn":
                faces = self._detect_faces_dnn(frame)
            else:
                logger.error(f"알 수 없는 검출 방법: {self.detection_method}")
                return []
            
            # 신뢰도 순으로 정렬 (높은 것부터)
            faces.sort(key=lambda x: x.confidence, reverse=True)
            
            # 작은 얼굴 필터링 (배경 인물 제거)
            faces = [face for face in faces if face.width >= self.MIN_FACE_SIZE and face.height >= self.MIN_FACE_SIZE]
            
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
                    logger.warning(f"얼굴 임베딩 생성 실패: {e}")
                    face.embedding = None
            
            return faces
            
        except Exception as e:
            logger.warning(f"얼굴 검출 실패: {e}")
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

    def assign_face_ids(self, faces: List[FaceDetection], frame: np.ndarray, frame_idx: int) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """
        검출된 얼굴들을 Person1/Person2에 할당
        
        하이브리드 매칭 알고리즘을 사용하여 얼굴을 Person1과 Person2에 할당합니다.
        Reference 임베딩이 있으면 임베딩 유사도(70%) + 위치 기반(30%) 매칭을 사용하고,
        없으면 위치 기반 추적이나 크기 기반 할당을 사용합니다.
        
        Args:
            faces: 검출된 얼굴 리스트
            frame: 현재 프레임 이미지
            frame_idx: 프레임 번호
            
        Returns:
            (Person1 얼굴, Person2 얼굴) 튜플, 없으면 None
        """
        if len(faces) == 0:
            return None, None
        
        # 1. Reference embedding이 있으면 하이브리드 매칭 사용
        if self.speaker1_reference is not None and self.speaker2_reference is not None:
            person1_face, person2_face = self._hybrid_face_matching(faces, frame, frame_idx)
        else:
            # 2. Reference embedding이 없으면 기존 위치 기반 추적
            person1_face, person2_face = self.position_tracker.track_faces(faces, frame_idx, frame)
        
        # 각 트래커에 업데이트
        if person1_face:
            self.person1_tracker.update_detection(person1_face)
        if person2_face:
            self.person2_tracker.update_detection(person2_face)
        
        # 추가 디버그 정보 (StablePositionTracker 내부 디버그와 별도)
        if self.debug_mode and frame_idx % 30 == 0:
            tracking_stats = self.position_tracker.get_tracking_stats()
            print(f"🔄 추적 상태: 초기화={tracking_stats['is_initialized']}, P1히스토리={tracking_stats['person1_history_length']}, P2히스토리={tracking_stats['person2_history_length']}")
            
            if person1_face and person2_face:
                size_ratio = person1_face.area / person2_face.area
                print(f"📊 프레임 {frame_idx}: P1=위치{person1_face.center} 크기{person1_face.area:.0f}, P2=위치{person2_face.center} 크기{person2_face.area:.0f}, 비율={size_ratio:.2f}")
            elif person1_face:
                print(f"📊 프레임 {frame_idx}: P1=위치{person1_face.center} 크기{person1_face.area:.0f}, P2=없음")
        
        return person1_face, person2_face
    
    def _hybrid_face_matching(self, faces: List[FaceDetection], frame: np.ndarray, frame_idx: int) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """
        하이브리드 얼굴 매칭 알고리즘
        
        임베딩 유사도(90%)와 위치 매칭(10%)을 결합한 고급 매칭 알고리즘입니다.
        각 검출된 얼굴에 대해 reference 임베딩과의 유사도를 계산하고,
        위치 정보를 추가로 고려하여 최적의 Person1/Person2 할당을 수행합니다.
        
        Args:
            faces: 검출된 얼굴 리스트
            frame: 현재 프레임 이미지
            frame_idx: 프레임 번호
            
        Returns:
            (Person1 얼굴, Person2 얼굴) 튜플
        """
        import torch.nn.functional as F
        
        if len(faces) == 0:
            return None, None
        
        # ModelManager가 없으면 위치 기반 폴백
        if not hasattr(self, 'model_manager') or self.model_manager is None:
            return self.position_tracker.track_faces(faces, frame_idx, frame)
        
        # 각 얼굴에 대해 임베딩 추출
        face_embeddings = []
        valid_faces = []
        
        for face in faces:
            try:
                # 얼굴 크롭 추출 (프레임 전달)
                face_crop = self.model_manager.extract_face_crop(face, frame)
                if face_crop is not None:
                    # 임베딩 계산
                    embedding = self.model_manager.get_embedding(face_crop)
                    if embedding is not None:
                        face_embeddings.append(embedding)
                        valid_faces.append(face)
            except Exception as e:
                if self.debug_mode:
                    logger.warning(f"임베딩 추출 실패 (얼굴 {face.center}): {e}")
                continue
        
        # 임베딩이 없으면 위치 기반 폴백
        if len(face_embeddings) == 0:
            return self.position_tracker.track_faces(faces, frame_idx, frame)
        
        # 각 얼굴에 대해 Speaker1, Speaker2와의 하이브리드 점수 계산
        speaker1_scores = []
        speaker2_scores = []
        
        for i, (face, embedding) in enumerate(zip(valid_faces, face_embeddings)):
            # 1. 임베딩 유사도 점수 (코사인 유사도, 0~1)
            embedding_sim1 = F.cosine_similarity(embedding.unsqueeze(0), self.speaker1_reference.unsqueeze(0)).item()
            embedding_sim2 = F.cosine_similarity(embedding.unsqueeze(0), self.speaker2_reference.unsqueeze(0)).item()
            
            # 코사인 유사도를 0~1 범위로 정규화 (-1~1 → 0~1)
            embedding_sim1 = (embedding_sim1 + 1) / 2
            embedding_sim2 = (embedding_sim2 + 1) / 2
            
            # 2. 위치 유사도 점수 (거리 기반, 0~1) - 직접 계산
            # Person1, Person2 예상 위치 가져오기
            if hasattr(self.position_tracker, 'person1_history') and self.position_tracker.person1_history:
                predicted_p1_pos = self.position_tracker._get_predicted_position(self.position_tracker.person1_history)
                dist_to_p1 = self.position_tracker._calculate_distance(face.center, predicted_p1_pos)
                # 거리 기반 점수 (가까울수록 높은 점수, 200px 기준)
                position_score1 = max(0, 1 - dist_to_p1 / 200)
            else:
                position_score1 = 0.5
                
            if hasattr(self.position_tracker, 'person2_history') and self.position_tracker.person2_history:
                predicted_p2_pos = self.position_tracker._get_predicted_position(self.position_tracker.person2_history)
                dist_to_p2 = self.position_tracker._calculate_distance(face.center, predicted_p2_pos)
                # 거리 기반 점수 (가까울수록 높은 점수, 200px 기준)
                position_score2 = max(0, 1 - dist_to_p2 / 200)
            else:
                position_score2 = 0.5
            
            # 3. 하이브리드 점수 계산 (임베딩 90% + 위치 10%) - 퀄리티 개선
            hybrid_score1 = 0.9 * embedding_sim1 + 0.1 * position_score1
            hybrid_score2 = 0.9 * embedding_sim2 + 0.1 * position_score2
            
            speaker1_scores.append(hybrid_score1)
            speaker2_scores.append(hybrid_score2)
            
            if self.debug_mode and frame_idx % 60 == 0:  # 2초마다 출력
                print(f"🔍 하이브리드 매칭 (프레임 {frame_idx}): 얼굴{i} → "
                      f"S1점수={hybrid_score1:.3f}(임베딩={embedding_sim1:.3f}+위치={position_score1:.3f}), "
                      f"S2점수={hybrid_score2:.3f}(임베딩={embedding_sim2:.3f}+위치={position_score2:.3f})")
        
        # 임계값 기반 필터링
        threshold = self.speaker_similarity_threshold  # 0.5
        
        # Speaker1, Speaker2에 가장 적합한 얼굴 선택
        person1_candidates = [(i, score) for i, score in enumerate(speaker1_scores) if score >= threshold]
        person2_candidates = [(i, score) for i, score in enumerate(speaker2_scores) if score >= threshold]
        
        # 최고 점수 얼굴 선택
        person1_face = None
        person2_face = None
        
        if person1_candidates:
            best_idx = max(person1_candidates, key=lambda x: x[1])[0]
            person1_face = valid_faces[best_idx]
        
        if person2_candidates:
            # Person1과 중복되지 않는 얼굴 선택
            available_candidates = [c for c in person2_candidates if valid_faces[c[0]] != person1_face]
            if available_candidates:
                best_idx = max(available_candidates, key=lambda x: x[1])[0]
                person2_face = valid_faces[best_idx]
        
        if self.debug_mode and frame_idx % 60 == 0:
            p1_status = f"P1=임계값통과" if person1_face else "P1=임계값미달"
            p2_status = f"P2=임계값통과" if person2_face else "P2=임계값미달"
            logger.debug(f"하이브리드 결과: {p1_status}, {p2_status} (임계값={threshold})")
        
        return person1_face, person2_face
    
    def create_adaptive_split_screen(self, crop1: np.ndarray, crop2: np.ndarray, 
                                   face1_size: float, face2_size: float) -> Tuple[np.ndarray, float]:
        """
        얼굴 크기 기반 적응형 스플릿 스크린 생성
        
        두 얼굴의 크기 비율에 따라 스플릿 비율을 동적으로 조정합니다.
        큰 얼굴에게 더 많은 화면 공간을 할당하여 균형잡힌 출력을 생성합니다.
        
        Args:
            crop1: Person1 크롭 이미지
            crop2: Person2 크롭 이미지  
            face1_size: Person1 얼굴 크기
            face2_size: Person2 얼굴 크기
            
        Returns:
            (합성된 스플릿 스크린, 사용된 스플릿 비율)
        """
        # 얼굴 크기 비율 계산 (30:70 ~ 70:30 제한)
        total_size = face1_size + face2_size
        if total_size > 0:
            ratio1 = face1_size / total_size
            ratio1 = max(0.3, min(0.7, ratio1))  # 30~70% 범위 제한
        else:
            ratio1 = 0.5  # 기본값
        
        ratio2 = 1 - ratio1
        
        # 화면 너비 분할 계산
        width1 = int(1920 * ratio1)
        width2 = 1920 - width1
        
        # 동적 리사이즈 (얼굴 크기에 따라 화면 영역 할당)
        resized_crop1 = cv2.resize(crop1, (width1, 1080))
        resized_crop2 = cv2.resize(crop2, (width2, 1080))
        
        # 스플릿 스크린 생성
        split_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        split_screen[0:1080, 0:width1] = resized_crop1
        split_screen[0:1080, width1:1920] = resized_crop2
        
        # 구분선 추가 (시각적 구분)
        cv2.line(split_screen, (width1, 0), (width1, 1080), (128, 128, 128), 2)
        
        return split_screen, ratio1
    
    def create_split_screen(self, crop1: np.ndarray, crop2: np.ndarray) -> np.ndarray:
        """스플릿 스크린 생성 (1920x1080) - 50:50 고정 비율"""
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
        logger.info("비디오 처리 시작")
        
        # 🆕 새로운 1단계: 자동 화자 선정 (AutoSpeakerDetector or OneMinuteAnalyzer)
        prescan_profiles = None
        
        if self.one_minute_mode:
            print("\n" + "=" * 50)
            print("🎯 1단계: 1분 집중 분석 (OneMinuteAnalyzer)")
            print("=" * 50)
            
            from .auto_speaker_detector import OneMinuteAnalyzer
            analyzer = OneMinuteAnalyzer(debug_mode=self.debug_mode)
            person1_profile, person2_profile = analyzer.analyze_first_minute(self.input_path)
            
            if person1_profile and person2_profile:
                # 화자 정보 저장
                self.speaker1_reference = person1_profile['reference_embedding']
                self.speaker2_reference = person2_profile['reference_embedding']
                
                # 위치 정보를 prescan_profiles로 변환
                prescan_profiles = (
                    person1_profile['average_position'],
                    person2_profile['average_position']
                )
                
                # Phase 2: IdentityBank 저장 (SimpleConsistentTracker와 공유)
                self.identity_bank = analyzer.identity_bank
                
                # SimpleConsistentTracker용 프로파일 저장
                self.person1_profile = person1_profile
                self.person2_profile = person2_profile
                
                logger.info("1분 집중 분석 완료:")
                print(f"   Person1: {person1_profile['appearance_count']}개 얼굴, IdentityBank: {person1_profile['identity_bank_size']}개")
                print(f"   Person2: {person2_profile['appearance_count']}개 얼굴, IdentityBank: {person2_profile['identity_bank_size']}개") 
                print(f"   위치: P1={prescan_profiles[0]}, P2={prescan_profiles[1]}")
                print(f"   💪 IdentityBank 준비: A슬롯={len(self.identity_bank.bank['A'])}개, B슬롯={len(self.identity_bank.bank['B'])}개")
                
            else:
                print("⚠️ 1분 집중 분석 실패, 폴백 모드로 처리")
                self.one_minute_mode = False  # 폴백으로 기본 모드 사용
                
        elif self.auto_speaker_enabled:
            print("\n" + "=" * 50)
            print("🎯 1단계: 자동 화자 선정 (AutoSpeakerDetector)")
            print("=" * 50)
            
            auto_detector = AutoSpeakerDetector(debug_mode=self.debug_mode)
            speaker1_info, speaker2_info = auto_detector.analyze_video(self.input_path)
            
            if speaker1_info and speaker2_info:
                # 화자 정보 저장
                self.speaker1_reference = speaker1_info['representative_embedding']
                self.speaker2_reference = speaker2_info['representative_embedding']
                
                # 위치 정보를 prescan_profiles로 변환
                prescan_profiles = (
                    speaker1_info['average_position'],
                    speaker2_info['average_position']
                )
                
                logger.info("자동 화자 선정 완료:")
                print(f"   화자1: {speaker1_info['appearance_count']}회 등장, 점수 {speaker1_info['importance_score']:.3f}")
                print(f"   화자2: {speaker2_info['appearance_count']}회 등장, 점수 {speaker2_info['importance_score']:.3f}")
                print(f"   위치: P1={prescan_profiles[0]}, P2={prescan_profiles[1]}")
                
            else:
                print("⚠️ 자동 화자 선정 실패, 폴백 모드로 처리")
                self.speaker1_reference = None
                self.speaker2_reference = None
        
        # 폴백: 기존 사전 분석 (prescan 옵션이 켜져있거나 자동 화자 선정 실패시)
        elif self.prescan_enabled or (self.auto_speaker_enabled and prescan_profiles is None):
            print("\n" + "=" * 50)
            print("🔍 폴백: 기존 사전 분석 (SimplePreScanner)")
            print("=" * 50)
            scanner = SimplePreScanner(debug_mode=self.debug_mode)
            prescan_profiles = scanner.quick_scan(self.input_path, sample_rate=30)
            
            if prescan_profiles:
                p1, p2 = prescan_profiles
                logger.info("타겟 프로파일 확정:")
                print(f"   Person1: 위치 {p1}")
                print(f"   Person2: 위치 {p2}")
            else:
                print("⚠️ 사전 분석 실패, 기본 모드로 처리")
        
        # 메인 처리 단계 표시
        print("\n" + "=" * 50)
        print("🎬 2단계: 메인 처리")
        print("=" * 50)
        
        # 위치 기반 추적자 초기화 (프로파일과 함께)
        self.position_tracker = StablePositionTracker(
            debug_mode=self.debug_mode,
            prescan_profiles=prescan_profiles
        )
        
        # Speaker reference 임베딩 및 ModelManager 전달
        if hasattr(self, 'speaker1_reference') and hasattr(self, 'speaker2_reference'):
            self.position_tracker.speaker1_reference = self.speaker1_reference
            self.position_tracker.speaker2_reference = self.speaker2_reference
        if hasattr(self, 'model_manager'):
            self.position_tracker.model_manager = self.model_manager
        
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
        
        # SimpleConsistentTracker 초기화 (1분 분석 모드)
        simple_tracker = None
        if self.one_minute_mode and hasattr(self, 'person1_profile') and hasattr(self, 'person2_profile'):
            # Phase 2: IdentityBank를 SimpleConsistentTracker에 전달
            identity_bank = getattr(self, 'identity_bank', None)
            simple_tracker = SimpleConsistentTracker(
                self.person1_profile, 
                self.person2_profile, 
                debug_mode=self.debug_mode,
                identity_bank=identity_bank
            )
            print(f"✅ SimpleConsistentTracker 초기화 완료 (IdentityBank: {'연결됨' if identity_bank else '없음'})")
        
        # Phase 3: HungarianFaceAssigner 초기화
        hungarian_assigner = None
        if self.hungarian_mode and hasattr(self, 'identity_bank') and self.identity_bank is not None:
            hungarian_assigner = HungarianFaceAssigner(
                self.identity_bank,
                debug_mode=self.debug_mode
            )
            print(f"✅ HungarianFaceAssigner 초기화 완료 (IdentityBank A:{len(self.identity_bank.bank['A'])}, B:{len(self.identity_bank.bank['B'])})")
        elif self.hungarian_mode:
            print(f"⚠️ Hungarian 모드 요청되었으나 IdentityBank 없음. 기본 모드로 진행")
            self.hungarian_mode = False
        
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
                    
                    # 얼굴 ID 할당 (Hungarian > SimpleTracker > 기존 모드 순서)
                    if hungarian_assigner:
                        # Phase 3: Hungarian Matching 할당
                        person1_face, person2_face = hungarian_assigner.assign_faces(faces, frame)
                        if frame_idx <= 60:
                            print(f"   🧮 Hungarian 할당 (프레임 {frame_idx}): P1={'✅' if person1_face else '❌'}, P2={'✅' if person2_face else '❌'}")
                    elif simple_tracker:
                        # 1분 분석 기반 단순 추적
                        person1_face, person2_face = simple_tracker.track_frame(faces, frame)
                        if frame_idx <= 60:
                            print(f"   🎯 SimpleTracker 할당 (프레임 {frame_idx}): P1={'✅' if person1_face else '❌'}, P2={'✅' if person2_face else '❌'}")
                    else:
                        # 기존 복잡한 할당 시스템
                        person1_face, person2_face = self.assign_face_ids(faces, frame, frame_idx)
                    
                    # 디버깅: ID 할당 결과 (일관성 추적)
                    if frame_idx <= 60:  # 처음 60프레임 동안 상세 로그
                        p1_assigned = person1_face is not None
                        p2_assigned = person2_face is not None
                        print(f"   🎯 ID 할당 (프레임 {frame_idx}): P1={p1_assigned}, P2={p2_assigned}")
                        
                        if person1_face:
                            print(f"     P1 얼굴: center={person1_face.center} conf={person1_face.confidence:.2f}")
                        if person2_face:
                            print(f"     P2 얼굴: center={person2_face.center} conf={person2_face.confidence:.2f}")
                    
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
                
                # 검출 타임라인 기록 (오디오/트리밍용)
                has_person1 = person1_face is not None if 'person1_face' in locals() else False
                has_person2 = person2_face is not None if 'person2_face' in locals() else False
                self.detection_timeline.append((frame_idx, has_person1, has_person2))
                
                # 3. 크롭 영역 생성 (고정 마진)
                crop1 = self.person1_tracker.get_crop_region(frame, self.margin_factor)
                crop2 = self.person2_tracker.get_crop_region(frame, self.margin_factor)
                
                # 4. 스플릿 스크린 생성 (50:50 고정 비율)
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
        
        # FFmpeg 후처리 (오디오 병합 + 트리밍)
        if ENABLE_FFMPEG_POST_PROCESSING:
            print(f"\n🔄 FFmpeg 후처리 시작...")
            success = self._post_process_with_ffmpeg(fps)
            if success:
                logger.info("FFmpeg 후처리 완료!")
            else:
                print(f"⚠️ FFmpeg 후처리 실패, 기본 비디오 유지")
        
        self._print_final_stats()
    
    def _post_process_with_ffmpeg(self, fps: float) -> bool:
        """FFmpeg를 사용한 후처리 (오디오 병합 + 트리밍)
        
        Args:
            fps: 비디오 프레임 레이트
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 0. 오디오 트랙 존재 여부 확인
            has_audio = self._check_audio_track(self.input_path)
            print(f"🔊 오디오 트랙: {'있음' if has_audio else '없음'}")
            
            # 1. 트리밍 구간 계산
            keep_segments = self._calculate_trim_segments(fps)
            
            if not keep_segments:
                logger.warning("유지할 구간이 없음, 후처리 건너뜀")
                return False
            
            # 2. 임시 파일 경로 설정
            temp_video = self.output_path + ".temp_no_audio.mp4"
            final_output = self.output_path
            
            # 3. 기존 출력을 임시 파일로 이동
            if os.path.exists(final_output):
                os.rename(final_output, temp_video)
            else:
                print("⚠️ 처리할 비디오 파일이 없음")
                return False
            
            # 4. FFmpeg 처리
            if not has_audio:
                # 오디오가 없는 경우 - 비디오만 처리
                if TRIM_UNDETECTED_SEGMENTS and len(keep_segments) < len(self.detection_timeline):
                    success = self._ffmpeg_trim_video_only(temp_video, final_output, keep_segments)
                else:
                    # 트리밍 없이 그냥 이동
                    os.rename(temp_video, final_output)
                    success = True
                    print("✅ 오디오 없는 비디오 처리 완료")
            elif TRIM_UNDETECTED_SEGMENTS and len(keep_segments) < len(self.detection_timeline):
                # 오디오 있고 트리밍이 필요한 경우
                success = self._ffmpeg_trim_and_merge_audio(temp_video, final_output, keep_segments)
            else:
                # 오디오 있고 트리밍 없이 오디오만 병합
                success = self._ffmpeg_merge_audio_only(temp_video, final_output)
            
            # 5. 임시 파일 정리
            if success and os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logger.warning(f"임시 파일 삭제 실패 (무시): {temp_video} - {e}")
                except Exception as e:
                    logger.debug(f"예상치 못한 파일 삭제 오류: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"FFmpeg 후처리 오류: {e}")
            return False
    
    def _check_audio_track(self, video_path: str) -> bool:
        """비디오 파일에 오디오 트랙이 있는지 확인
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            bool: 오디오 트랙 존재 여부
        """
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0', 
                '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and result.stdout.strip() == 'audio'
            
        except Exception as e:
            print(f"⚠️ 오디오 체크 실패: {e}")
            return False
    
    def _calculate_trim_segments(self, fps: float) -> List[Tuple[float, float]]:
        """검출 타임라인을 기반으로 유지할 구간 계산
        
        Args:
            fps: 비디오 프레임 레이트
            
        Returns:
            List[Tuple[float, float]]: [(시작_시간, 끝_시간), ...] 유지할 구간들
        """
        if not self.detection_timeline:
            return []
        
        print(f"🔍 트리밍 구간 계산 중... (총 {len(self.detection_timeline)}개 프레임)")
        
        segments_to_keep = []
        total_duration = len(self.detection_timeline) / fps
        
        # 1. 첫 프레임부터 분석하여 초기 구간 결정
        current_segment_start = 0  # 항상 0부터 시작
        current_undetected_start = None
        
        for i, (frame_idx, has_p1, has_p2) in enumerate(self.detection_timeline):
            current_time = frame_idx / fps
            
            # 검출 여부 확인 (REQUIRE_BOTH_PERSONS 설정에 따라)
            if REQUIRE_BOTH_PERSONS:
                detected = has_p1 and has_p2  # 둘 다 검출되어야 함
            else:
                detected = has_p1 or has_p2   # 하나라도 검출되면 됨
            
            if detected:
                # 검출된 상태
                if current_undetected_start is not None:
                    # 미검출 구간 종료 - 2초 이상이었는지 확인
                    undetected_duration = current_time - current_undetected_start
                    if undetected_duration >= UNDETECTED_THRESHOLD_SECONDS:
                        # 2초 이상 미검출이었음 -> 이전 구간 종료하고 새 구간 시작
                        end_time = max(current_segment_start, current_undetected_start - TRIM_BUFFER_SECONDS)
                        if end_time > current_segment_start:
                            segments_to_keep.append((current_segment_start, end_time))
                        
                        # 새로운 구간 시작 (버퍼 고려)
                        current_segment_start = max(0, current_time - TRIM_BUFFER_SECONDS)
                    
                    # 미검출 구간 종료 (2초 미만이면 구간 유지됨)
                    current_undetected_start = None
                
                # 검출 중이므로 현재 구간 계속 유지
            else:
                # 미검출 상태 - 미검출 구간 시작점 기록
                if current_undetected_start is None:
                    current_undetected_start = current_time
        
        # 2. 마지막 구간 처리
        if current_undetected_start is not None:
            # 마지막에 미검출 구간이 있음
            final_undetected_duration = total_duration - current_undetected_start
            if final_undetected_duration >= UNDETECTED_THRESHOLD_SECONDS:
                # 마지막 미검출 구간이 2초 이상 -> 그 전까지만 포함
                end_time = max(current_segment_start, current_undetected_start - TRIM_BUFFER_SECONDS)
                if end_time > current_segment_start:
                    segments_to_keep.append((current_segment_start, end_time))
            else:
                # 마지막 미검출 구간이 2초 미만 -> 끝까지 포함
                segments_to_keep.append((current_segment_start, total_duration))
        else:
            # 끝까지 검출됨 -> 현재 구간을 끝까지 포함
            segments_to_keep.append((current_segment_start, total_duration))
        
        # 특별 처리: 전체가 미검출인 경우 원본 유지
        if not segments_to_keep or (len(segments_to_keep) == 1 and segments_to_keep[0][1] - segments_to_keep[0][0] < 0.5):
            # 검출된 얼굴이 전혀 없거나, 유지할 구간이 0.5초 미만인 경우
            total_detected = sum(1 for _, has_p1, has_p2 in self.detection_timeline 
                               if ((has_p1 or has_p2) if not REQUIRE_BOTH_PERSONS else (has_p1 and has_p2)))
            
            if total_detected == 0:
                print(f"⚠️ 전체 비디오에서 얼굴이 검출되지 않음 -> 원본 유지")
                segments_to_keep = [(0, total_duration)]
            else:
                print(f"⚠️ 유지할 구간이 너무 짧음 -> 원본 유지")
                segments_to_keep = [(0, total_duration)]
        
        # 3. 구간 정리 및 검증
        cleaned_segments = []
        for start, end in segments_to_keep:
            # 최소 길이 체크 (0.1초 이상)
            if end - start >= 0.1:
                # 범위 제한
                start = max(0, start)
                end = min(total_duration, end)
                if end > start:
                    cleaned_segments.append((start, end))
        
        print(f"✅ 유지할 구간: {len(cleaned_segments)}개")
        for i, (start, end) in enumerate(cleaned_segments):
            print(f"   구간 {i+1}: {start:.1f}초 - {end:.1f}초 (길이: {end-start:.1f}초)")
        
        return cleaned_segments
    
    def _ffmpeg_trim_and_merge_audio(self, temp_video: str, final_output: str, 
                                   segments: List[Tuple[float, float]]) -> bool:
        """FFmpeg를 사용해서 트리밍 + 오디오 병합
        
        Args:
            temp_video: 임시 비디오 파일 (오디오 없음)
            final_output: 최종 출력 파일
            segments: 유지할 구간들 [(시작, 끝), ...]
            
        Returns:
            bool: 성공 여부
        """
        try:
            if len(segments) == 1:
                # 단일 구간: 간단한 trim + 오디오 병합
                start, end = segments[0]
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-i', temp_video,
                    '-ss', str(start),
                    '-i', self.input_path,
                    '-t', str(end - start),
                    '-c:v', 'copy',  # 비디오 복사 (빠른 처리)
                    '-c:a', AUDIO_CODEC,
                    '-map', '0:v:0',  # 첫 번째 입력의 비디오
                    '-map', '1:a:0',  # 두 번째 입력의 오디오
                    '-preset', FFMPEG_PRESET,
                    final_output
                ]
            else:
                # 다중 구간: filter_complex 사용
                filter_parts = []
                
                # 비디오 세그먼트
                for i, (start, end) in enumerate(segments):
                    filter_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
                
                # 오디오 세그먼트 (원본에서)
                for i, (start, end) in enumerate(segments):
                    filter_parts.append(f"[1:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
                
                # 비디오 concat
                video_inputs = "".join([f"[v{i}]" for i in range(len(segments))])
                filter_parts.append(f"{video_inputs}concat=n={len(segments)}:v=1:a=0[vout]")
                
                # 오디오 concat
                audio_inputs = "".join([f"[a{i}]" for i in range(len(segments))])
                filter_parts.append(f"{audio_inputs}concat=n={len(segments)}:v=0:a=1[aout]")
                
                filter_complex = ";".join(filter_parts)
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,  # 처리된 비디오
                    '-i', self.input_path,  # 원본 (오디오용)
                    '-filter_complex', filter_complex,
                    '-map', '[vout]',
                    '-map', '[aout]',
                    '-c:v', VIDEO_CODEC,
                    '-c:a', AUDIO_CODEC,
                    '-preset', FFMPEG_PRESET,
                    '-crf', str(FFMPEG_CRF),
                    final_output
                ]
            
            print(f"🔄 FFmpeg 트리밍 + 오디오 병합 실행 중...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(final_output):
                    file_size = os.path.getsize(final_output) / 1024 / 1024
                    print(f"✅ 트리밍 + 오디오 병합 완료 ({file_size:.1f}MB)")
                    return True
                else:
                    print("❌ 출력 파일이 생성되지 않음")
                    return False
            else:
                print(f"❌ FFmpeg 오류: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 트리밍 + 오디오 병합 오류: {e}")
            return False
    
    def _ffmpeg_trim_video_only(self, temp_video: str, final_output: str, 
                                keep_segments: List[Tuple[float, float]]) -> bool:
        """비디오만 트리밍 (오디오 없음)
        
        Args:
            temp_video: 임시 비디오 파일
            final_output: 최종 출력 파일
            keep_segments: 유지할 구간들 [(시작시간, 끝시간), ...]
            
        Returns:
            bool: 성공 여부
        """
        try:
            if len(keep_segments) == 1:
                # 단일 구간 트리밍
                start_time, end_time = keep_segments[0]
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-c', 'copy',  # 빠른 복사
                    final_output
                ]
            else:
                # 다중 구간 병합
                filter_parts = []
                for i, (start_time, end_time) in enumerate(keep_segments):
                    duration = end_time - start_time
                    filter_parts.append(f"[0:v]trim=start={start_time:.3f}:duration={duration:.3f},setpts=PTS-STARTPTS[v{i}]")
                
                concat_inputs = "".join(f"[v{i}]" for i in range(len(keep_segments)))
                filter_parts.append(f"{concat_inputs}concat=n={len(keep_segments)}:v=1:a=0[vout]")
                
                filter_complex = ";".join(filter_parts)
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-filter_complex', filter_complex,
                    '-map', '[vout]',
                    '-c:v', VIDEO_CODEC,
                    '-preset', FFMPEG_PRESET,
                    '-crf', str(FFMPEG_CRF),
                    final_output
                ]
            
            print(f"🔄 FFmpeg 비디오 트리밍 실행 중... ({len(keep_segments)}개 구간)")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(final_output):
                    file_size = os.path.getsize(final_output) / 1024 / 1024
                    print(f"✅ 비디오 트리밍 완료 ({file_size:.1f}MB)")
                    return True
                else:
                    print("❌ 출력 파일이 생성되지 않음")
                    return False
            else:
                print(f"❌ FFmpeg 오류: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 비디오 트리밍 오류: {e}")
            return False
    
    def _ffmpeg_merge_audio_only(self, temp_video: str, final_output: str) -> bool:
        """FFmpeg를 사용해서 오디오만 병합 (트리밍 없음)
        
        Args:
            temp_video: 임시 비디오 파일 (오디오 없음)
            final_output: 최종 출력 파일
            
        Returns:
            bool: 성공 여부
        """
        try:
            if not PRESERVE_AUDIO:
                # 오디오 보존 비활성화시 그냥 이름만 변경
                os.rename(temp_video, final_output)
                return True
            
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,      # 처리된 비디오 (오디오 없음)
                '-i', self.input_path,  # 원본 비디오 (오디오 포함)
                '-c:v', 'copy',        # 비디오 복사 (빠른 처리)
                '-c:a', AUDIO_CODEC,   # 오디오 인코딩
                '-map', '0:v:0',       # 첫 번째 입력의 비디오
                '-map', '1:a:0',       # 두 번째 입력의 오디오
                '-shortest',           # 더 짧은 스트림에 맞춤
                final_output
            ]
            
            print(f"🔄 FFmpeg 오디오 병합 실행 중...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(final_output):
                    file_size = os.path.getsize(final_output) / 1024 / 1024
                    print(f"✅ 오디오 병합 완료 ({file_size:.1f}MB)")
                    return True
                else:
                    print("❌ 출력 파일이 생성되지 않음")
                    return False
            else:
                print(f"❌ FFmpeg 오류: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 오디오 병합 오류: {e}")
            return False
    
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
            print(f"   🎵 오디오: {'✅ 포함' if PRESERVE_AUDIO else '❌ 없음'}")
            
            # 트리밍 정보
            if TRIM_UNDETECTED_SEGMENTS and self.detection_timeline:
                total_frames = len(self.detection_timeline)
                detected_frames = sum(1 for _, has_p1, has_p2 in self.detection_timeline 
                                    if ((has_p1 or has_p2) if not REQUIRE_BOTH_PERSONS else (has_p1 and has_p2)))
                detection_rate = detected_frames / total_frames * 100 if total_frames > 0 else 0
                print(f"   ✂️ 트리밍: 활성화 (임계값: {UNDETECTED_THRESHOLD_SECONDS}초)")
                print(f"   📊 검출률: {detection_rate:.1f}% ({detected_frames}/{total_frames} 프레임)")
            else:
                print(f"   ✂️ 트리밍: 비활성화")
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
  python3 face_tracking_system.py --input input/sample.mp4 --output output/tracked.mp4 --debug
  
  # 크기 기반 안정화 사용
  python3 face_tracking_system.py --size-stabilize --debug
        """
    )
    
    parser.add_argument("--input", 
                       default="input/2people_sample1.mp4",
                       help="입력 비디오 경로 (기본값: input/2people_sample1.mp4)")
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
    parser.add_argument("--prescan", 
                       action="store_true",
                       help="사전 분석 모드 (정확도 향상, 15초 추가)")
    parser.add_argument("--quick", 
                       action="store_true",
                       help="빠른 모드 (사전 분석 스킵)")
    parser.add_argument("--auto-speaker", 
                       action="store_true",
                       default=True,
                       help="자동 화자 선정 (AutoSpeakerDetector, 기본값: True)")
    parser.add_argument("--one-minute", 
                       action="store_true",
                       help="1분 집중 분석 모드 (1분 분석 + 간단 추적)")
    parser.add_argument("--hungarian", 
                       action="store_true",
                       help="Phase 3: Hungarian Matching 사용 (고급 할당 시스템)")
    
    args = parser.parse_args()
    
    # prescan과 quick 옵션 충돌 체크
    if args.prescan and args.quick:
        print("❌ --prescan과 --quick 옵션은 동시에 사용할 수 없습니다")
        sys.exit(1)
    
    mode_str = "정확 모드" if args.prescan else "빠른 모드" if args.quick else "기본 모드"
    print("")
    print(f"🚀 Dual-Face Tracking System v6.1 ({mode_str})")
    print("=" * 50)
    print(f"   📥 입력: {args.input}")
    print(f"   📤 출력: {args.output}")
    logger.debug(f"모드: {args.mode}")
    print(f"   🖥️ GPU: {args.gpu}")
    logger.debug(f"디버그: {args.debug}")
    print(f"   ⚙️ 안정화: {args.size_stabilize}")
    logger.debug(f"사전 분석: {args.prescan}")
    logger.debug(f"빠른 모드: {args.quick}")
    print(f"   🤖 자동 화자: {args.auto_speaker}")
    logger.debug(f"1분 분석: {args.one_minute}")
    logger.debug(f"Hungarian: {args.hungarian}")
    print("=" * 50)
    print("")
    
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
        
        # position_tracker 디버그 모드 업데이트 (process() 후에)
        if system.position_tracker:
            system.position_tracker.debug_mode = args.debug
        
        print("\n🎉 Phase 5 완료: 얼굴 트래킹 시스템 성공!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


class SimpleConsistentTracker:
    """
    일관성 기반 단순 추적 클래스
    
    1분 분석 결과로 생성된 인물 프로파일을 바탕으로 일관된 추적을 수행합니다.
    임베딩 기반 매칭과 Identity Bank를 통해 안정적인 인물 식별을 제공합니다.
    
    Attributes:
        p1_profile, p2_profile: Person1/Person2 인물 프로파일
        identity_bank: 얼굴 임베딩 데이터베이스
        embedding_threshold: 임베딩 매칭 임계값
        identity_threshold: Identity Bank 거리 임계값
    """
    
    def __init__(self, person1_profile: Dict[str, Any], person2_profile: Dict[str, Any], 
                 debug_mode: bool = False, identity_bank=None):
        self.p1_profile = person1_profile
        self.p2_profile = person2_profile
        self.debug_mode = debug_mode
        
        # 매칭 임계값 설정
        self.embedding_threshold = 0.35  # Phase 1: 0.65 → 0.35 (Identity-based 강화)
        self.position_tolerance = 200    # 200px 이내 위치 변화
        self.size_tolerance = 0.5        # 50% 크기 변화 허용
        
        # 폴백 카운터
        self.p1_fallback_count = 0
        self.p2_fallback_count = 0
        
        # Phase 2: 외부 IdentityBank 사용 (OneMinuteAnalyzer에서 전달됨)
        if identity_bank is not None:
            self.identity_bank = identity_bank
            if self.debug_mode:
                print(f"✅ Phase 2: 외부 IdentityBank 연결됨 (A:{len(identity_bank.bank['A'])}, B:{len(identity_bank.bank['B'])})")
        else:
            # Phase 1: 로컬 Identity Bank (임베딩 뱅크) - 폴백 모드
            from collections import deque
            max_embeddings = 64  # 최대 64개 임베딩 저장
            self.p1_embedding_bank = deque(maxlen=max_embeddings)
            self.p2_embedding_bank = deque(maxlen=max_embeddings)
            self.identity_bank = None
            
            # 초기 프로토타입 설정
            if person1_profile.get('reference_embedding') is not None:
                self.p1_embedding_bank.append(self._normalize_embedding(person1_profile['reference_embedding']))
            if person2_profile.get('reference_embedding') is not None:
                self.p2_embedding_bank.append(self._normalize_embedding(person2_profile['reference_embedding']))
            
            if self.debug_mode:
                print(f"⚠️ Phase 2: 로컬 임베딩 뱅크 사용 (폴백 모드)")
        
        # ModelManager 초기화 (임베딩 추출용)
        self.model_manager = None
        try:
            from .model_manager import ModelManager
            self.model_manager = ModelManager()
            if self.debug_mode:
                print("✅ SimpleConsistentTracker: ModelManager 로드 완료")
        except ImportError:
            if self.debug_mode:
                print("⚠️ SimpleConsistentTracker: ModelManager 없음 (폴백 모드)")
    
    def _normalize_embedding(self, embedding):
        """L2 정규화"""
        import torch
        if isinstance(embedding, torch.Tensor):
            norm = torch.norm(embedding) + 1e-8
            return embedding / norm
        else:
            import numpy as np
            norm = np.linalg.norm(embedding) + 1e-8
            return embedding / norm
    
    def _get_prototype_embedding(self, person_num: int):
        """중앙값 기반 프로토타입 임베딩 계산 (노이즈 강건)"""
        # Phase 2: IdentityBank가 있으면 우선 사용
        if self.identity_bank is not None:
            slot = 'A' if person_num == 1 else 'B'
            return self.identity_bank.proto(slot)
        
        # Phase 1: 로컬 뱅크 사용 (폴백)
        bank = self.p1_embedding_bank if person_num == 1 else self.p2_embedding_bank
        
        if len(bank) == 0:
            return None
        elif len(bank) == 1:
            return bank[0]
        else:
            # 중앙값 계산 (노이즈에 강건)
            import torch
            import numpy as np
            
            if isinstance(bank[0], torch.Tensor):
                embeddings = torch.stack(list(bank))
                prototype = torch.median(embeddings, dim=0)[0]  # 중앙값
            else:
                embeddings = np.array(list(bank))
                prototype = np.median(embeddings, axis=0)  # 중앙값
            
            return self._normalize_embedding(prototype)
    
    def _update_embedding_bank(self, person_num: int, embedding):
        """임베딩 뱅크 업데이트 (성공한 매칭만)"""
        # Phase 2: IdentityBank가 있으면 우선 사용
        if self.identity_bank is not None:
            slot = 'A' if person_num == 1 else 'B'
            self.identity_bank.update(slot, embedding)
            
            if self.debug_mode and len(self.identity_bank.bank[slot]) % 10 == 0:
                print(f"🔄 Phase 2: {slot} 슬롯 업데이트: {len(self.identity_bank.bank[slot])}개")
            return
        
        # Phase 1: 로컬 뱅크 업데이트 (폴백)
        normalized_emb = self._normalize_embedding(embedding)
        
        if person_num == 1:
            self.p1_embedding_bank.append(normalized_emb)
            if self.debug_mode and len(self.p1_embedding_bank) % 10 == 0:
                print(f"🔄 P1 로컬 뱅크 업데이트: {len(self.p1_embedding_bank)}개")
        else:
            self.p2_embedding_bank.append(normalized_emb)
            if self.debug_mode and len(self.p2_embedding_bank) % 10 == 0:
                print(f"🔄 P2 로컬 뱅크 업데이트: {len(self.p2_embedding_bank)}개")
    
    def track_frame(self, faces: List[FaceDetection], frame: np.ndarray) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """매 프레임 단순 추적"""
        
        # 1. 좌우 분리
        left_faces = [f for f in faces if f.center_x < 960]
        right_faces = [f for f in faces if f.center_x >= 960]
        
        # 2. Person1 찾기 (왼쪽에서)
        person1, p1_embedding = self._find_best_match(left_faces, self.p1_profile, frame, person_num=1)
        
        # 성공적 매칭시 임베딩 뱅크 업데이트
        if person1 and p1_embedding is not None:
            self._update_embedding_bank(1, p1_embedding)
        
        # 못 찾으면 왼쪽에서 가장 큰 얼굴 (폴백)
        if person1 is None and left_faces:
            person1 = max(left_faces, key=lambda f: f.area)
            self.p1_fallback_count += 1
            if self.debug_mode and self.p1_fallback_count % 30 == 1:  # 1초마다 한번
                print(f"⚠️ Person1 폴백: 가장 큰 얼굴 사용 ({self.p1_fallback_count}회)")
        
        # 3. Person2 찾기 (오른쪽에서)
        person2, p2_embedding = self._find_best_match(right_faces, self.p2_profile, frame, person_num=2)
        
        # 성공적 매칭시 임베딩 뱅크 업데이트
        if person2 and p2_embedding is not None:
            self._update_embedding_bank(2, p2_embedding)
        
        # 못 찾으면 오른쪽에서 가장 큰 얼굴 (폴백)
        if person2 is None and right_faces:
            person2 = max(right_faces, key=lambda f: f.area)
            self.p2_fallback_count += 1
            if self.debug_mode and self.p2_fallback_count % 30 == 1:  # 1초마다 한번
                print(f"⚠️ Person2 폴백: 가장 큰 얼굴 사용 ({self.p2_fallback_count}회)")
        
        return person1, person2
    
    def _find_best_match(self, faces: List[FaceDetection], profile: Dict[str, Any], frame: np.ndarray, person_num: int) -> Tuple[Optional[FaceDetection], Optional[Any]]:
        """프로파일과 가장 일치하는 얼굴 찾기 + 임베딩 반환"""
        if not faces:
            return None, None
        
        best_face = None
        best_score = 0
        best_embedding = None
        
        # Phase 1: 프로토타입 임베딩 사용 (중앙값 기반)
        prototype_embedding = self._get_prototype_embedding(person_num)
        
        for face in faces:
            total_score = 0
            score_count = 0
            face_embedding = None
            
            # 1. 임베딩 유사도 (60% 가중치) - 프로토타입 사용
            if self.model_manager and prototype_embedding is not None:
                try:
                    face_crop = self.model_manager.extract_face_crop(face, frame)
                    if face_crop is not None:
                        face_embedding = self.model_manager.get_embedding(face_crop)
                        if face_embedding is not None:
                            face_embedding = self._normalize_embedding(face_embedding)  # 정규화
                            
                            import torch.nn.functional as F
                            if hasattr(face_embedding, 'unsqueeze') and hasattr(prototype_embedding, 'unsqueeze'):
                                similarity = F.cosine_similarity(
                                    face_embedding.unsqueeze(0), 
                                    prototype_embedding.unsqueeze(0)
                                ).item()
                            else:
                                # numpy 배열의 경우
                                import numpy as np
                                similarity = float(np.dot(face_embedding, prototype_embedding))
                            
                            # -1~1 → 0~1 정규화
                            similarity = (similarity + 1) / 2
                            
                            total_score += similarity * 0.6
                            score_count += 0.6
                except (AttributeError, ValueError, TypeError) as e:
                    logger.debug(f"유사도 계산 실패 (무시): {e}")
                except Exception as e:
                    logger.warning(f"예상치 못한 유사도 계산 오류: {e}")
            
            # 2. 크기 일치도 (20% 가중치)
            if profile.get('average_size'):
                size_diff = abs(face.area - profile['average_size']) / profile['average_size']
                size_score = max(0, 1.0 - size_diff)  # 차이가 클수록 점수 낮음
                
                total_score += size_score * 0.2
                score_count += 0.2
            
            # 3. 위치 일치도 (20% 가중치)
            if profile.get('average_position') is not None:
                pos_distance = np.linalg.norm(
                    np.array(face.center) - np.array(profile['average_position'])
                )
                pos_score = max(0, 1.0 - pos_distance / self.position_tolerance)
                
                total_score += pos_score * 0.2
                score_count += 0.2
            
            # 최소 하나의 점수라도 있어야 함
            if score_count > 0:
                final_score = total_score / score_count  # 정규화
                
                if final_score > best_score and final_score > self.embedding_threshold:  # Phase 1: 더 엄격한 임계값 적용
                    best_score = final_score
                    best_face = face
                    best_embedding = face_embedding  # 임베딩도 함께 저장
        
        if self.debug_mode and best_face and best_score > 0.8:  # 고점수일 때만 출력
            logger.debug(f"좋은 매칭: {profile.get('label', f'P{person_num}')} 점수={best_score:.3f}")
        
        return best_face, best_embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """추적 통계 반환"""
        return {
            'p1_fallback_count': self.p1_fallback_count,
            'p2_fallback_count': self.p2_fallback_count,
            'embedding_threshold': self.embedding_threshold,
            'position_tolerance': self.position_tolerance
        }


class HungarianFaceAssigner:
    """
    헝가리안 매칭 기반 얼굴 할당 클래스
    
    헝가리안 알고리즘을 사용하여 검출된 얼굴들을 Person1/Person2에 최적으로 할당합니다.
    임베딩 거리 매트릭스를 기반으로 전역 최적해를 찾아 일관된 할당을 보장합니다.
    
    Attributes:
        identity_bank: 얼굴 임베딩 데이터베이스
        identity_threshold: Identity Bank 매칭 임계값
        debug_mode: 디버그 출력 활성화 여부
    
    Methods:
        assign_faces: 헝가리안 알고리즘으로 얼굴 할당 수행
    """
    
    def __init__(self, identity_bank, debug_mode: bool = False):
        """
        Args:
            identity_bank: IdentityBank 인스턴스
            debug_mode: 디버그 모드
        """
        self.identity_bank = identity_bank
        self.debug_mode = debug_mode
        
        # 가중치 설정
        self.weights = {
            'iou': 0.45,     # IoU (위치 연속성)
            'emb': 0.45,     # 임베딩 거리 (정체성)
            'motion': 0.10   # 모션 (향후 확장)
        }
        
        # 임계값 설정
        self.identity_threshold = 0.45  # 프로토타입과 거리 임계값
        
        # 이전 박스 저장 (연속성을 위해)
        self.prev_boxes = {'A': None, 'B': None}
    
    def assign_faces(self, faces: List[FaceDetection], frame: np.ndarray, 
                    predicted_boxes: Dict[str, Any] = None) -> Tuple[Optional[FaceDetection], Optional[FaceDetection]]:
        """Hungarian Matching으로 A/B 얼굴 할당
        
        Args:
            faces: 검출된 얼굴 리스트
            frame: 현재 프레임
            predicted_boxes: {'A': bbox, 'B': bbox} 예측 박스 (옵션)
            
        Returns:
            (face_A, face_B) 할당된 얼굴 또는 None
        """
        if not faces:
            return None, None
        
        if len(faces) == 1:
            # 얼굴이 하나만 있으면 더 적합한 슬롯에 할당
            face = faces[0]
            best_slot, similarity, _ = self.identity_bank.get_best_match([face], [face.embedding if hasattr(face, 'embedding') else None])
            
            if best_slot == 'A':
                return face, None
            else:
                return None, face
        
        # 비용 행렬 구성
        cost_matrix = self._build_cost_matrix(faces, predicted_boxes)
        
        # Hungarian 할당
        assignment = self._hungarian_assign(cost_matrix)
        
        # 결과 생성
        face_A = faces[assignment['A']] if assignment['A'] != -1 else None
        face_B = faces[assignment['B']] if assignment['B'] != -1 else None
        
        # 이전 박스 업데이트
        if face_A:
            self.prev_boxes['A'] = face_A.bbox
        if face_B:
            self.prev_boxes['B'] = face_B.bbox
        
        if self.debug_mode:
            print(f"🎯 Hungarian 할당: A={'✅' if face_A else '❌'}, B={'✅' if face_B else '❌'}")
        
        return face_A, face_B
    
    def _build_cost_matrix(self, faces: List[FaceDetection], predicted_boxes: Dict[str, Any] = None) -> np.ndarray:
        """비용 행렬 구성 (2×N)"""
        N = len(faces)
        cost_matrix = np.zeros((2, N), dtype=np.float32)
        
        for j, face in enumerate(faces):
            # A/B 각각에 대해 비용 계산
            for i, slot in enumerate(['A', 'B']):
                total_cost = 0.0
                
                # 1. 임베딩 비용 (정체성)
                if hasattr(face, 'embedding') and face.embedding is not None:
                    emb_distance = self.identity_bank.dist(slot, face.embedding)
                    total_cost += emb_distance * self.weights['emb']
                
                # 2. IoU 비용 (위치 연속성)
                if predicted_boxes and slot in predicted_boxes:
                    iou_cost = 1.0 - self._calculate_iou(face, predicted_boxes[slot])
                    total_cost += iou_cost * self.weights['iou']
                elif self.prev_boxes[slot] is not None:
                    iou_cost = 1.0 - self._calculate_iou(face, self.prev_boxes[slot])
                    total_cost += iou_cost * self.weights['iou']
                
                # 3. 모션 비용 (향후 확장)
                # motion_cost = 0.0
                # total_cost += motion_cost * self.weights['motion']
                
                cost_matrix[i, j] = total_cost
        
        # 임계값 기반 패널티 (프로토타입과 너무 먼 경우)
        for j, face in enumerate(faces):
            if hasattr(face, 'embedding') and face.embedding is not None:
                for i, slot in enumerate(['A', 'B']):
                    if self.identity_bank.dist(slot, face.embedding) > self.identity_threshold:
                        cost_matrix[i, j] += 10.0  # 큰 패널티
        
        return cost_matrix
    
    def _calculate_iou(self, face: FaceDetection, box) -> float:
        """IoU 계산"""
        if not hasattr(face, 'bbox'):
            return 0.0
        
        x1, y1, x2, y2 = face.bbox
        
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
    
    def _hungarian_assign(self, cost_matrix: np.ndarray) -> Dict[str, int]:
        """2×N 헝가리언 할당"""
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


if __name__ == "__main__":
    main()