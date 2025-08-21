#!/usr/bin/env python3
"""
ModelManager - MTCNN과 FaceNet 모델 관리
듀얼 페이스 트래킹 시스템의 핵심 모델 로더
"""

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from typing import List, Optional, Tuple, Union
import logging
from ..utils.logger import get_logger

# Logger 설정
logger = get_logger(__name__, level=logging.INFO)

class ModelManager:
    """
    MTCNN과 FaceNet 모델을 관리하는 싱글톤 클래스
    GPU 메모리 효율적 관리 및 배치 처리 지원
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = None
        self.facenet = None
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self._init_models()
        self._initialized = True
        
    def _init_models(self):
        """MTCNN과 FaceNet 모델 초기화"""
        try:
            # MTCNN 초기화 (얼굴 검출)
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True,
                selection_method='largest_over_threshold'
            )
            
            # FaceNet 초기화 (임베딩 추출)
            self.facenet = InceptionResnetV1(
                pretrained='vggface2',
                device=self.device
            ).eval()
            
            # GPU로 이동
            if self.device.type == 'cuda':
                self.facenet = self.facenet.cuda()
                
            self.logger.info(f"✅ ModelManager 초기화 완료 (device: {self.device})")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 초기화 실패: {e}")
            raise
    
    def detect_faces(self, image: Union[np.ndarray, torch.Tensor], min_confidence: float = 0.9):
        """
        이미지에서 얼굴 검출
        
        Args:
            image: 입력 이미지 (H, W, C) numpy array 또는 tensor
            min_confidence: 최소 신뢰도 임계값
            
        Returns:
            boxes: 바운딩 박스 리스트 [(x1, y1, x2, y2), ...]
            probs: 신뢰도 리스트
            landmarks: 랜드마크 리스트 (선택사항)
        """
        if self.mtcnn is None:
            raise RuntimeError("MTCNN 모델이 초기화되지 않음")
            
        try:
            # numpy array를 PIL Image로 변환
            if isinstance(image, np.ndarray):
                from PIL import Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # MTCNN으로 얼굴 검출
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
            
            if boxes is None:
                return [], [], []
                
            # 신뢰도 필터링
            valid_indices = probs >= min_confidence
            
            valid_boxes = boxes[valid_indices] if boxes is not None else []
            valid_probs = probs[valid_indices] if probs is not None else []
            valid_landmarks = landmarks[valid_indices] if landmarks is not None else []
            
            return valid_boxes.tolist(), valid_probs.tolist(), valid_landmarks.tolist()
            
        except Exception as e:
            self.logger.error(f"얼굴 검출 실패: {e}")
            return [], [], []
    
    def extract_embeddings(self, faces: List[np.ndarray], batch_size: int = 8):
        """
        얼굴 이미지들에서 임베딩 벡터 추출
        
        Args:
            faces: 얼굴 이미지 리스트 (각각 160x160 크기)
            batch_size: 배치 크기
            
        Returns:
            embeddings: 임베딩 벡터 리스트 (512차원)
        """
        if self.facenet is None:
            raise RuntimeError("FaceNet 모델이 초기화되지 않음")
            
        if not faces:
            return []
            
        try:
            embeddings = []
            
            # 배치별 처리
            for i in range(0, len(faces), batch_size):
                batch_faces = faces[i:i+batch_size]
                
                # 전처리: numpy → tensor
                batch_tensors = []
                for face in batch_faces:
                    if isinstance(face, np.ndarray):
                        # 0-255 → -1 to 1 정규화
                        face = face.astype(np.float32) / 127.5 - 1.0
                        # HWC → CHW
                        if len(face.shape) == 3:
                            face = np.transpose(face, (2, 0, 1))
                        # Tensor 변환
                        face_tensor = torch.from_numpy(face).to(self.device)
                    else:
                        face_tensor = face.to(self.device)
                    
                    batch_tensors.append(face_tensor)
                
                # 배치 텐서 생성
                batch_tensor = torch.stack(batch_tensors)
                
                # 임베딩 추출
                with torch.no_grad():
                    batch_embeddings = self.facenet(batch_tensor)
                    
                # CPU로 이동하여 저장
                embeddings.extend(batch_embeddings.cpu().numpy())
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"임베딩 추출 실패: {e}")
            return []
    
    def crop_and_align_faces(self, image: Union[np.ndarray, torch.Tensor], boxes: List[List[float]]):
        """
        검출된 얼굴 박스를 기반으로 얼굴을 자르고 정렬
        
        Args:
            image: 원본 이미지
            boxes: 바운딩 박스 리스트 [(x1, y1, x2, y2), ...]
            
        Returns:
            cropped_faces: 자른 얼굴 이미지 리스트 (160x160)
        """
        if not boxes:
            return []
            
        try:
            cropped_faces = []
            
            # PIL Image로 변환
            if isinstance(image, np.ndarray):
                from PIL import Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # 바운딩 박스 크롭
                cropped = pil_image.crop((x1, y1, x2, y2))
                
                # 160x160으로 리사이즈
                cropped = cropped.resize((160, 160))
                
                # numpy array로 변환
                cropped_np = np.array(cropped)
                cropped_faces.append(cropped_np)
            
            return cropped_faces
            
        except Exception as e:
            self.logger.error(f"얼굴 크롭 실패: {e}")
            return []
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 벡터 간의 코사인 유사도 계산
        
        Args:
            embedding1, embedding2: 512차원 임베딩 벡터
            
        Returns:
            similarity: 코사인 유사도 (0~1)
        """
        try:
            # L2 정규화
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # 코사인 유사도
            similarity = np.dot(embedding1, embedding2)
            
            # 0~1 범위로 변환
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"유사도 계산 실패: {e}")
            return 0.0
    
    def process_frame(self, frame: np.ndarray, min_confidence: float = 0.9):
        """
        프레임에서 얼굴 검출 → 크롭 → 임베딩 추출까지 한번에 처리
        
        Args:
            frame: 입력 프레임 (H, W, C)
            min_confidence: 최소 신뢰도
            
        Returns:
            results: [{'box': [x1,y1,x2,y2], 'confidence': float, 'embedding': np.array}, ...]
        """
        # 얼굴 검출
        boxes, probs, _ = self.detect_faces(frame, min_confidence)
        
        if not boxes:
            return []
        
        # 얼굴 크롭
        cropped_faces = self.crop_and_align_faces(frame, boxes)
        
        # 임베딩 추출
        embeddings = self.extract_embeddings(cropped_faces)
        
        # 결과 조합
        results = []
        for i, (box, prob, embedding) in enumerate(zip(boxes, probs, embeddings)):
            results.append({
                'box': box,
                'confidence': prob,
                'embedding': embedding
            })
        
        return results
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            'device': str(self.device),
            'mtcnn_loaded': self.mtcnn is not None,
            'facenet_loaded': self.facenet is not None,
            'cuda_available': torch.cuda.is_available(),
            'gpu_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
    
    def cleanup(self):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("ModelManager 메모리 정리 완료")


# 싱글톤 인스턴스 생성 함수
def get_model_manager() -> ModelManager:
    """ModelManager 싱글톤 인스턴스 반환"""
    return ModelManager()


if __name__ == "__main__":
    # 테스트 코드
    logger.info("ModelManager 테스트")
    
    manager = ModelManager()
    info = manager.get_model_info()
    
    logger.info(f"Device: {info['device']}")
    logger.info(f"MTCNN: {info['mtcnn_loaded']}")
    logger.info(f"FaceNet: {info['facenet_loaded']}")
    logger.info(f"CUDA: {info['cuda_available']}")
    
    logger.info("ModelManager 테스트 완료")