"""
경량 ReID(Re-Identification) 모델 구현.

이 모듈은 얼굴 재식별을 위한 경량 임베딩 추출 모델을 제공합니다.
ONNX Runtime을 사용하여 고속 추론을 수행하며, ConditionalReID 시스템에서
ID 스왑이 감지될 때만 활성화됩니다.
"""

import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .onnx_engine import ONNXRuntimeEngine
from ..utils.logger import UnifiedLogger
from ..utils.exceptions import InferenceError, ModelLoadError, PreprocessingError


class ReIDModel:
    """
    경량 ReID 모델 클래스.
    
    얼굴 이미지로부터 고유한 임베딩 벡터를 추출하여
    동일 인물 여부를 판단할 수 있는 특징을 제공합니다.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        embedding_dim: int = 256,
        input_size: Tuple[int, int] = (112, 112),
        enable_l2_norm: bool = True,
        enable_warmup: bool = True,
        batch_size: int = 8
    ):
        """
        ReID 모델을 초기화합니다.
        
        Args:
            model_path: ONNX 모델 파일 경로 (None이면 모의 모델 사용)
            embedding_dim: 임베딩 벡터 차원 수
            input_size: 모델 입력 이미지 크기 (width, height)
            enable_l2_norm: L2 정규화 활성화 여부
            enable_warmup: 초기화시 워밍업 수행 여부  
            batch_size: 배치 처리 크기
        """
        self.logger = UnifiedLogger("ReIDModel")
        
        # 설정 저장
        self.model_path = Path(model_path) if model_path else None
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.enable_l2_norm = enable_l2_norm
        self.batch_size = batch_size
        
        # 모델 상태
        self.engine = None
        self.use_mock_model = model_path is None
        
        # 성능 통계
        self.stats = {
            'total_extractions': 0,
            'total_time_ms': 0.0,
            'avg_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 초기화
        self._initialize_model()
        
        if enable_warmup:
            self._warmup_model()
    
    def _initialize_model(self):
        """모델을 초기화합니다."""
        try:
            if self.use_mock_model:
                self.logger.info("🔧 Mock ReID 모델 초기화 (실제 모델 없음)")
                # Mock 모델은 별도 초기화 불필요
                return
            
            # 실제 ONNX 모델 로드
            self.logger.stage(f"ReID 모델 로드 중: {self.model_path.name}")
            
            if not self.model_path.exists():
                raise ModelLoadError(f"ReID 모델 파일을 찾을 수 없습니다: {self.model_path}")
            
            # ONNX Runtime 엔진 초기화
            self.engine = ONNXRuntimeEngine(
                model_path=self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                enable_optimization=True,
                enable_profiling=False
            )
            
            self.logger.success(f"ReID 모델 로드 완료: {self.model_path.name}")
            
        except Exception as e:
            self.logger.error(f"ReID 모델 초기화 실패: {e}")
            raise ModelLoadError(f"Failed to initialize ReID model: {e}")
    
    def _warmup_model(self):
        """모델 워밍업을 수행합니다."""
        try:
            self.logger.stage("ReID 모델 워밍업 시작")
            
            # 더미 이미지로 워밍업
            dummy_image = np.random.randint(0, 255, (*self.input_size[::-1], 3), dtype=np.uint8)
            
            warmup_times = []
            for i in range(5):
                start_time = time.perf_counter()
                _ = self.extract_embedding(dummy_image)
                warmup_times.append((time.perf_counter() - start_time) * 1000)
            
            avg_warmup_time = np.mean(warmup_times)
            self.logger.success(f"ReID 모델 워밍업 완료: {avg_warmup_time:.2f}ms")
            
        except Exception as e:
            self.logger.warning(f"ReID 모델 워밍업 실패: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지를 ReID 모델 입력 형태로 전처리합니다.
        
        Args:
            image: 입력 이미지 (BGR 또는 RGB)
            
        Returns:
            np.ndarray: 전처리된 이미지 배열 (1, 3, H, W)
        """
        try:
            if image is None or image.size == 0:
                raise PreprocessingError("입력 이미지가 비어있습니다")
            
            # RGB 변환 (BGR인 경우)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # OpenCV는 BGR이므로 RGB로 변환 
                preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                preprocessed = image.copy()
            
            # 크기 조정
            if preprocessed.shape[:2] != self.input_size[::-1]:  # (H, W)
                preprocessed = cv2.resize(preprocessed, self.input_size)
            
            # 정규화 (0-1 범위로)
            preprocessed = preprocessed.astype(np.float32) / 255.0
            
            # 표준화 (ImageNet 평균/표준편차)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            preprocessed = (preprocessed - mean) / std
            
            # 차원 변경: (H, W, C) -> (1, C, H, W)
            preprocessed = preprocessed.transpose(2, 0, 1)  # (C, H, W)
            preprocessed = np.expand_dims(preprocessed, axis=0)  # (1, C, H, W)
            
            return preprocessed
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            raise PreprocessingError(f"Image preprocessing failed: {e}")
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        단일 이미지로부터 임베딩을 추출합니다.
        
        Args:
            image: 얼굴 이미지 (H, W, C)
            
        Returns:
            np.ndarray: 임베딩 벡터 (embedding_dim,)
        """
        try:
            start_time = time.perf_counter()
            
            if self.use_mock_model:
                # Mock 모델: 랜덤 임베딩 생성
                embedding = self._generate_mock_embedding(image)
            else:
                # 실제 모델 추론
                preprocessed = self.preprocess_image(image)
                
                # ONNX 추론
                outputs = self.engine.run_inference(preprocessed)
                embedding = outputs[0].flatten()  # 첫 번째 출력을 임베딩으로 사용
            
            # L2 정규화
            if self.enable_l2_norm:
                embedding = self._l2_normalize(embedding)
            
            # 통계 업데이트
            inference_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(inference_time)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"임베딩 추출 실패: {e}")
            raise InferenceError(f"Embedding extraction failed: {e}")
    
    def extract_embeddings_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        여러 이미지로부터 배치로 임베딩을 추출합니다.
        
        Args:
            images: 얼굴 이미지 리스트
            
        Returns:
            List[np.ndarray]: 임베딩 벡터 리스트
        """
        try:
            if not images:
                return []
            
            embeddings = []
            
            # 배치 크기로 나누어 처리
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i + self.batch_size]
                
                if self.use_mock_model:
                    # Mock 모델: 각 이미지별로 임베딩 생성
                    batch_embeddings = [self._generate_mock_embedding(img) for img in batch_images]
                else:
                    # 실제 배치 처리
                    batch_embeddings = self._process_batch(batch_images)
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"배치 임베딩 추출 실패: {e}")
            raise InferenceError(f"Batch embedding extraction failed: {e}")
    
    def _process_batch(self, batch_images: List[np.ndarray]) -> List[np.ndarray]:
        """실제 모델로 배치 처리를 수행합니다."""
        # 배치 전처리
        batch_input = []
        for image in batch_images:
            preprocessed = self.preprocess_image(image)
            batch_input.append(preprocessed[0])  # (C, H, W)
        
        # 배치 텐서 생성: (N, C, H, W)
        batch_tensor = np.stack(batch_input, axis=0)
        
        # 배치 추론
        start_time = time.perf_counter()
        outputs = self.engine.run_inference(batch_tensor)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # 결과 분리
        batch_embeddings = outputs[0]  # (N, embedding_dim)
        
        # 개별 임베딩으로 분리
        embeddings = []
        for embedding in batch_embeddings:
            if self.enable_l2_norm:
                embedding = self._l2_normalize(embedding)
            embeddings.append(embedding)
        
        # 통계 업데이트
        self._update_stats(inference_time, batch_size=len(batch_images))
        
        return embeddings
    
    def _generate_mock_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Mock 모델용 임베딩을 생성합니다.
        
        이미지의 간단한 특징을 기반으로 재현 가능한 임베딩을 생성합니다.
        """
        # 이미지의 기본 특징 추출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (32, 32))
        
        # 기본 통계 특징
        mean_intensity = np.mean(resized)
        std_intensity = np.std(resized)
        
        # 히스토그램 특징
        hist = cv2.calcHist([resized], [0], None, [16], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        # LBP 유사 특징 (간단한 텍스처)
        lbp_features = []
        for i in range(1, 31, 3):
            for j in range(1, 31, 3):
                center = resized[i, j]
                neighbors = [
                    resized[i-1, j-1], resized[i-1, j], resized[i-1, j+1],
                    resized[i, j-1], resized[i, j+1],
                    resized[i+1, j-1], resized[i+1, j], resized[i+1, j+1]
                ]
                lbp_value = sum([(1 if n >= center else 0) * (2**k) for k, n in enumerate(neighbors)])
                lbp_features.append(lbp_value / 255.0)
        
        # 특징 결합
        features = [mean_intensity/255.0, std_intensity/255.0] + hist.tolist() + lbp_features[:self.embedding_dim-18]
        
        # 원하는 차원으로 맞추기
        while len(features) < self.embedding_dim:
            features.append(np.random.normal(0, 0.1))
        
        embedding = np.array(features[:self.embedding_dim], dtype=np.float32)
        
        # L2 정규화
        if self.enable_l2_norm:
            embedding = self._l2_normalize(embedding)
        
        return embedding
    
    def _l2_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """임베딩을 L2 정규화합니다."""
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:  # 0으로 나누기 방지
            return embedding / norm
        else:
            return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 간의 유사도를 계산합니다.
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            
        Returns:
            float: 코사인 유사도 (0.0 ~ 1.0, 높을수록 유사)
        """
        try:
            # L2 정규화가 되어있다면 내적이 코사인 유사도
            similarity = np.dot(embedding1, embedding2)
            
            # 수치적 안정성을 위해 [-1, 1] 범위로 클리핑
            similarity = np.clip(similarity, -1.0, 1.0)
            
            # [0, 1] 범위로 변환
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"유사도 계산 실패: {e}")
            return 0.0
    
    def calculate_similarities_batch(self, 
                                   embeddings1: List[np.ndarray], 
                                   embeddings2: List[np.ndarray]) -> np.ndarray:
        """
        두 임베딩 리스트 간의 유사도 행렬을 계산합니다.
        
        Args:
            embeddings1: 첫 번째 임베딩 리스트 (N개)
            embeddings2: 두 번째 임베딩 리스트 (M개)
            
        Returns:
            np.ndarray: 유사도 행렬 (N, M)
        """
        try:
            if not embeddings1 or not embeddings2:
                return np.empty((len(embeddings1), len(embeddings2)), dtype=np.float32)
            
            # NumPy 배열로 변환
            emb1_matrix = np.stack(embeddings1)  # (N, dim)
            emb2_matrix = np.stack(embeddings2)  # (M, dim)
            
            # 배치 내적 계산: (N, dim) @ (dim, M) = (N, M)
            similarity_matrix = np.dot(emb1_matrix, emb2_matrix.T)
            
            # 수치적 안정성
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
            
            # [0, 1] 범위로 변환
            similarity_matrix = (similarity_matrix + 1.0) / 2.0
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"배치 유사도 계산 실패: {e}")
            return np.zeros((len(embeddings1), len(embeddings2)), dtype=np.float32)
    
    def _update_stats(self, inference_time_ms: float, batch_size: int = 1):
        """통계를 업데이트합니다."""
        self.stats['total_extractions'] += batch_size
        self.stats['total_time_ms'] += inference_time_ms
        
        if self.stats['total_extractions'] > 0:
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['total_extractions']
    
    def get_statistics(self) -> Dict[str, Any]:
        """ReID 모델 통계를 반환합니다."""
        stats = self.stats.copy()
        stats['model_type'] = "Mock" if self.use_mock_model else "ONNX"
        stats['embedding_dim'] = self.embedding_dim
        stats['input_size'] = self.input_size
        stats['l2_norm_enabled'] = self.enable_l2_norm
        return stats
    
    def __repr__(self):
        model_type = "Mock" if self.use_mock_model else "ONNX"
        return (f"ReIDModel(type={model_type}, dim={self.embedding_dim}, "
                f"extractions={self.stats['total_extractions']}, "
                f"avg_time={self.stats['avg_time_ms']:.2f}ms)")


class ReIDModelConfig:
    """ReID 모델 설정 클래스."""
    
    def __init__(self):
        self.model_path = None
        self.embedding_dim = 256
        self.input_size = (112, 112)
        self.enable_l2_norm = True
        self.enable_warmup = True
        self.batch_size = 8
    
    @classmethod
    def for_face_reid(cls):
        """얼굴 재식별에 최적화된 설정."""
        config = cls()
        config.embedding_dim = 128  # 얼굴용 경량 임베딩
        config.input_size = (112, 112)  # 표준 얼굴 크기
        config.batch_size = 4  # 얼굴은 보통 적은 수
        return config
    
    @classmethod
    def for_high_performance(cls):
        """고성능 처리를 위한 설정."""
        config = cls()
        config.embedding_dim = 512  # 고차원 임베딩
        config.batch_size = 16  # 큰 배치 사이즈
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환."""
        return {
            'model_path': self.model_path,
            'embedding_dim': self.embedding_dim,
            'input_size': self.input_size,
            'enable_l2_norm': self.enable_l2_norm,
            'enable_warmup': self.enable_warmup,
            'batch_size': self.batch_size
        }