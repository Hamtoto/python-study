"""
ModelManager 클래스 - 모델 싱글톤 매니저
모델 재생성 방지로 성능 향상
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from src.face_tracker.config import DEVICE, BATCH_SIZE_ANALYZE, FACE_EMBEDDING_DIM, USE_HIGH_DIM_EMBEDDING


class ModelManager:
    """모델 싱글톤 매니저 - 모델 재생성 방지로 성능 향상"""
    _instance = None
    _initialized = False
    
    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device=None, multiprocessing_mode=False):
        if not self._initialized:
            if device is None:
                device = DEVICE
            
            # CUDA 가용성 검사 및 안전한 디바이스 설정
            import torch
            if 'cuda' in str(device):
                if not torch.cuda.is_available():
                    print(f"⚠️ CONSOLE: CUDA가 사용할 수 없습니다. CPU로 대체합니다.")
                    device = 'cpu'
                else:
                    # GPU 메모리 상태 확인
                    try:
                        torch.cuda.empty_cache()
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        gpu_allocated = torch.cuda.memory_allocated(0)
                        gpu_available = gpu_memory - gpu_allocated
                        print(f"🔍 CONSOLE: GPU 메모리 - 총 용량: {gpu_memory/1024**3:.1f}GB, 사용 가능: {gpu_available/1024**3:.1f}GB")
                    except Exception as e:
                        print(f"⚠️ CONSOLE: GPU 메모리 확인 실패 - {e}")
            
            self.device = torch.device(device)
            self.multiprocessing_mode = multiprocessing_mode
            
            # 멀티프로세싱 환경에서 메모리 절약 설정
            if multiprocessing_mode:
                batch_size = 32  # 멀티프로세싱시 더 작은 배치
            else:
                batch_size = BATCH_SIZE_ANALYZE
                
            try:
                self.mtcnn = MTCNN(
                    keep_all=True, 
                    device=self.device, 
                    post_process=False,
                    min_face_size=10,  # 15 -> 10으로 훨씬 작은 얼굴도 감지
                    thresholds=[0.4, 0.5, 0.5],  # [0.5,0.6,0.6] -> [0.4,0.5,0.5] 매우 관대한 탐지
                    factor=0.509,  # 0.609 -> 0.509로 더 세밀한 스케일 피라미드 
                    selection_method='probability'  # 확률 기반 얼굴 선택
                )
                print(f"✅ CONSOLE: MTCNN 모델 로드 완료 - device: {self.device}")
            except Exception as e:
                print(f"❌ CONSOLE: MTCNN 로드 실패 - {e}")
                raise
            
            # FaceNet 모델 초기화 (고차원 임베딩 지원)
            try:
                if USE_HIGH_DIM_EMBEDDING:
                    self.resnet = self._create_high_dim_resnet(FACE_EMBEDDING_DIM).to(self.device)
                else:
                    self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                print(f"✅ CONSOLE: FaceNet 모델 로드 완료 - device: {self.device}")
            except Exception as e:
                print(f"❌ CONSOLE: FaceNet 로드 실패 - {e}")
                raise
            
            # GPU 메모리 풀 초기화
            try:
                self._init_memory_pool()
            except Exception as e:
                print(f"❌ CONSOLE: GPU 메모리 풀 초기화 실패 - {e}")
                # 메모리 풀 실패해도 계속 진행
            
            self._initialized = True
            self.batch_size = batch_size  # 배치 크기 저장  # 배치 크기 저장
    
    def _create_high_dim_resnet(self, embedding_dim):
        """고차원 임베딩용 FaceNet 모델 생성"""
        # 기본 모델 로드
        base_resnet = InceptionResnetV1(pretrained='vggface2')
        
        # 기존 512차원 출력을 사용하여 고차원으로 확장하는 추가 레이어 생성
        class HighDimWrapper(nn.Module):
            def __init__(self, base_model, target_dim, device):
                super().__init__()
                self.base_model = base_model
                self._device = device  # device 속성 저장
                # 512차원 → target_dim으로 확장하는 레이어 추가
                self.expansion_layer = nn.Linear(512, target_dim)
                
                # Xavier 초기화
                nn.init.xavier_uniform_(self.expansion_layer.weight)
                nn.init.zeros_(self.expansion_layer.bias)
            
            @property 
            def device(self):
                """device 속성 접근자"""
                return self._device
            
            def forward(self, x):
                # 기존 512차원 임베딩 추출
                base_embedding = self.base_model(x)
                # 고차원으로 확장
                expanded_embedding = self.expansion_layer(base_embedding)
                return expanded_embedding
        
        return HighDimWrapper(base_resnet, embedding_dim, self.device).eval()
    
    def _init_memory_pool(self):
        """GPU 메모리 풀 사전 할당"""
        import torch
        from torchvision import transforms
        
        if self.device.type == 'cuda':
            try:
                # 배치 텐서 풀 (config 배치 크기 기준)
                self.tensor_pool = torch.zeros(BATCH_SIZE_ANALYZE, 3, 224, 224, device=self.device)
                # 단일 얼굴 임베딩용 텐서 풀
                self.face_tensor_pool = torch.zeros(1, 3, 160, 160, device=self.device)
                # 변환 객체 사전 생성
                self.to_tensor = transforms.ToTensor()
                
                # 메모리 풀 성공적으로 할당
                allocated_memory = torch.cuda.memory_allocated(self.device)
                print(f"✅ CONSOLE: GPU 메모리 풀 초기화 완료 - device: {self.device}, 배치크기: {BATCH_SIZE_ANALYZE}")
                print(f"🔍 CONSOLE: 할당된 GPU 메모리: {allocated_memory/1024**2:.1f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ CONSOLE: GPU 메모리 부족으로 메모리 풀 비활성화 - {e}")
                    # 메모리 풀 없이 동작하도록 None 설정
                    self.tensor_pool = None
                    self.face_tensor_pool = None
                    self.to_tensor = transforms.ToTensor()
                    # GPU 캐시 정리
                    torch.cuda.empty_cache()
                else:
                    print(f"❌ CONSOLE: GPU 메모리 풀 초기화 에러 - {e}")
                    raise
            except Exception as e:
                print(f"❌ CONSOLE: 메모리 풀 초기화 예상치 못한 에러 - {e}")
                # 안전을 위해 None으로 설정
                self.tensor_pool = None  
                self.face_tensor_pool = None
                self.to_tensor = transforms.ToTensor()
        else:
            # CPU 모드에서는 메모리 풀 비활성화
            self.tensor_pool = None
            self.face_tensor_pool = None 
            self.to_tensor = transforms.ToTensor()
            print(f"🔍 CONSOLE: CPU 모드 - 메모리 풀 비활성화")
    
    def get_mtcnn(self):
        return self.mtcnn
    
    def detect_faces_multi_scale(self, frame):
        """다중 스케일로 얼굴 감지 (더 강력한 감지)"""
        import cv2
        original_size = frame.shape[:2]  # (height, width)
        
        # 여러 스케일에서 얼굴 감지 시도 (더 많은 스케일)
        scales = [1.0, 0.8, 1.2, 0.6, 1.5, 0.5, 1.8, 0.4]  # 더 다양한 스케일
        all_boxes = []
        all_probs = []
        
        for scale in scales:
            if scale != 1.0:
                # 이미지 크기 조정
                new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
                if new_h < 50 or new_w < 50:  # 너무 작으면 건너뛰기
                    continue
                scaled_frame = cv2.resize(frame, (new_w, new_h))
            else:
                scaled_frame = frame
            
            try:
                # MTCNN으로 얼굴 감지
                boxes, probs = self.mtcnn.detect(scaled_frame)
                
                if boxes is not None and len(boxes) > 0:
                    # 스케일에 따른 좌표 보정
                    if scale != 1.0:
                        boxes = boxes / scale
                    
                    all_boxes.extend(boxes)
                    all_probs.extend(probs)
                    
            except Exception as e:
                # 특정 스케일에서 오류 발생시 건너뛰기
                continue
        
        if len(all_boxes) == 0:
            return None, None
        
        # 중복 제거 및 최고 확률 선택
        import numpy as np
        all_boxes = np.array(all_boxes)
        all_probs = np.array(all_probs)
        
        # 확률 기준으로 정렬
        sorted_indices = np.argsort(all_probs)[::-1]
        
        return all_boxes[sorted_indices], all_probs[sorted_indices]
    
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