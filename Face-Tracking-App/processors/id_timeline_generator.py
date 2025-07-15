"""
ID 타임라인 생성 모듈
"""
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from core.model_manager import ModelManager
from core.embedding_manager import SmartEmbeddingManager
from config import DEVICE, BATCH_SIZE_ID_TIMELINE, SIMILARITY_THRESHOLD
from utils.similarity_utils import find_matching_id_with_best_fallback


def generate_id_timeline(video_path: str, device=DEVICE, batch_size: int = BATCH_SIZE_ID_TIMELINE):
    """
    각 프레임에 대한 track_id (또는 None) 리스트 생성
    배치 처리로 최적화된 버전
    
    Args:
        video_path: 비디오 파일 경로
        device: 사용할 디바이스
        batch_size: 배치 처리 크기
    
    Returns:
        tuple: (ID 타임라인, fps)
    """
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    resnet = model_manager.get_resnet()
    emb_manager = SmartEmbeddingManager()
    next_id = 1

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="[ID 타임라인 생성 - 배치 처리]")
    fps = cap.get(cv2.CAP_PROP_FPS)
    id_timeline = []

    frames_buffer = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_buffer.append(frame)
        frame_count += 1
        
        # 배치가 찼거나 마지막 프레임이면 처리
        if len(frames_buffer) == batch_size or frame_count == total_frames:
            # 배치 BGR→RGB 변환 (중복 제거)
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer]
            pil_images = [Image.fromarray(rgb_frame) for rgb_frame in rgb_frames]
            
            # 배치 얼굴 탐지
            boxes_list, _ = mtcnn.detect(pil_images)
            
            # 각 프레임별 ID 할당
            for i, (frame, boxes) in enumerate(zip(frames_buffer, boxes_list)):
                track_id = None
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    # 이미 변환된 PIL 객체 재사용
                    face_img = pil_images[i]
                    face_crop_pil = face_img.crop((x1, y1, x2, y2)).resize((160, 160))
                    
                    # 최적화된 텐서 변환
                    face_tensor = model_manager.get_face_tensor_pool()
                    if face_tensor is not None:
                        face_array = np.array(face_crop_pil)
                        face_tensor[0] = torch.from_numpy(face_array).permute(2, 0, 1).float() / 255.0
                        emb = resnet(face_tensor)
                    else:
                        emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
                    
                    # ID 할당 (조기 종료 최적화된 유사도 계산)
                    all_embs = emb_manager.get_all_embeddings()
                    track_id = find_matching_id_with_best_fallback(emb, all_embs, SIMILARITY_THRESHOLD)
                    if track_id is None:
                        track_id = next_id
                        next_id += 1
                    emb_manager.add_embedding(track_id, emb)
                
                id_timeline.append(track_id)
                pbar.update(1)
            
            # 버퍼 초기화
            frames_buffer = []
    
    pbar.close()
    cap.release()
    return id_timeline, fps