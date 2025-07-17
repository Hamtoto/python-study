"""
ModelManager 클래스 - 모델 싱글톤 매니저
모델 재생성 방지로 성능 향상
"""
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from config import DEVICE, BATCH_SIZE_ANALYZE


class ModelManager:
    """모델 싱글톤 매니저 - 모델 재생성 방지로 성능 향상"""
    _instance = None
    _initialized = False
    
    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device=None):
        if not self._initialized:
            if device is None:
                device = DEVICE
            self.device = device
            print(f"ModelManager 초기화: {device}")
            self.mtcnn = MTCNN(
                keep_all=True, 
                device=device, 
                post_process=False,
                min_face_size=20,  # 기본값 20 -> 더 작은 얼굴 감지
                thresholds=[0.6, 0.7, 0.7]  # 기본값보다 낮춤 (더 관대한 탐지)
            )
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            
            # GPU 메모리 풀 초기화
            self._init_memory_pool()
            self._initialized = True
    
    def _init_memory_pool(self):
        """GPU 메모리 풀 사전 할당"""
        if self.device.type == 'cuda':
            # 배치 텐서 풀 (config 배치 크기 기준)
            self.tensor_pool = torch.zeros(BATCH_SIZE_ANALYZE, 3, 224, 224, device=self.device)
            # 단일 얼굴 임베딩용 텐서 풀
            self.face_tensor_pool = torch.zeros(1, 3, 160, 160, device=self.device)
            # 변환 객체 사전 생성
            self.to_tensor = transforms.ToTensor()
            print(f"GPU 메모리 풀 초기화 완료: {self.device}, 배치크기: {BATCH_SIZE_ANALYZE}")
    
    def get_mtcnn(self):
        return self.mtcnn
    
    def get_resnet(self):
        return self.resnet
    
    def get_tensor_pool(self, batch_size=BATCH_SIZE_ANALYZE):
        """사전 할당된 텐서 풀 반환"""
        if self.device.type == 'cuda':
            if batch_size <= BATCH_SIZE_ANALYZE:
                return self.tensor_pool[:batch_size]
            else:
                # 더 큰 배치 사이즈면 동적 할당
                return torch.zeros(batch_size, 3, 224, 224, device=self.device)
        return None
    
    def get_face_tensor_pool(self):
        """얼굴 임베딩용 텐서 풀 반환"""
        if self.device.type == 'cuda':
            return self.face_tensor_pool
        return None
    
    def get_transform(self):
        """사전 생성된 변환 객체 반환"""
        return self.to_tensor
    
    def opencv_to_tensor_batch(self, frames_list):
        """OpenCV 프레임들을 배치 텐서로 직접 변환 (PIL 건너뛰기)"""
        if not frames_list:
            return None
            
        batch_size = len(frames_list)
        if batch_size == 0:
            return None
            
        # OpenCV BGR → RGB 변환과 동시에 numpy 배열로 변환
        rgb_frames = []
        for frame in frames_list:
            if isinstance(frame, Image.Image):
                # PIL 이미지면 numpy로 변환
                rgb_frame = np.array(frame)
            else:
                # OpenCV 프레임이면 BGR → RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)
        
        # numpy 배열 스택
        batch_array = np.stack(rgb_frames, axis=0)
        
        # numpy → tensor 직접 변환 (HWC → CHW, 0-255 → 0-1)
        batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2).float() / 255.0
        
        # GPU로 전송
        if self.device.type == 'cuda':
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
            
        return batch_tensor