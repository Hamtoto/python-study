"""
비디오 트래킹 및 크롭 모듈
"""
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from core.model_manager import ModelManager
from core.embedding_manager import SmartEmbeddingManager
from core.tracker_monitor import TrackerMonitor
from utils.bbox_utils import evaluate_bbox_quality
from utils.tracker_utils import get_adaptive_reinit_interval
from config import (
    DEVICE, CROP_SIZE, JUMP_THRESHOLD, EMA_ALPHA, 
    BASE_REINIT_INTERVAL, SIMILARITY_THRESHOLD
)
from utils.similarity_utils import find_matching_id_with_best_fallback


def track_and_crop_video(
    video_path: str,
    output_path: str,
    mtcnn,
    resnet,
    device,
    crop_size: int = CROP_SIZE,
    jump_thresh: float = JUMP_THRESHOLD,
    ema_alpha: float = EMA_ALPHA,
    reinit_interval: int = BASE_REINIT_INTERVAL
):
    """
    얼굴 중심 기준 크롭 (스무딩, 이상치, 하이브리드 트래킹, 클램핑)
    
    Args:
        video_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        mtcnn: 사전 로드된 MTCNN 모델
        resnet: 사전 로드된 ResNet 모델
        device: Torch 디바이스
        crop_size: 크롭 사이즈
        jump_thresh: 점프 임계값
        ema_alpha: EMA 스무딩 계수
        reinit_interval: 재초기화 간격
    """
    # 모델을 인자로 받으므로, 더 이상 ModelManager를 직접 초기화하지 않음.
    emb_manager = SmartEmbeddingManager()
    next_id = 1

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))

    tracker = None
    prev_center = None
    smooth_center = None
    track_id = None

    # 동적 재초기화를 위한 변수들
    tracker_monitor = TrackerMonitor()
    frames_since_reinit = 0
    current_reinit_interval = reinit_interval

    pbar = tqdm(total=total_frames, desc="[크롭 처리 - 동적 최적화]")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        frames_since_reinit += 1
        
        # 기본 중심점 설정 (안전성)
        raw_cx, raw_cy = w // 2, h // 2
        
        # 트래커 업데이트 시도
        tracker_success = False
        if tracker is not None:
            ok, bb = tracker.update(frame)
            if ok:
                bbox_quality = evaluate_bbox_quality(bb, frame.shape)
                tracker_success = bbox_quality
                if tracker_success:
                    x, y, tw, th = bb
                    raw_cx = int(x + tw / 2)
                    raw_cy = int(y + th / 2)
        
        # 성능 모니터링 업데이트
        if tracker is not None:
            tracker_monitor.update(tracker_success)
        
        # 적응적 재초기화 간격 계산
        success_rate = tracker_monitor.get_success_rate()
        current_reinit_interval = get_adaptive_reinit_interval(success_rate, reinit_interval)
        
        # 재초기화 결정
        need_reinit = (
            frames_since_reinit >= current_reinit_interval or
            not tracker_success or
            tracker is None
        )
        
        if need_reinit:
            # BGR→RGB 변환 1번만 실행 (중복 제거)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # MTCNN 얼굴 탐지 실행 (변환된 PIL 객체 재사용)
            boxes_list, _ = mtcnn.detect([pil_image])
            faces = boxes_list[0]
            if faces is not None and len(faces) > 0:
                x1, y1, x2, y2 = faces[0]
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                raw_cx = int((x1 + x2) / 2)
                raw_cy = int((y1 + y2) / 2)
                
                # ResNet 임베딩 계산 (같은 PIL 객체 재사용)
                face_crop_pil = pil_image.crop((x1, y1, x2, y2)).resize((160, 160))
                emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
                
                # ID 할당 (조기 종료 최적화된 유사도 계산)
                all_embs = emb_manager.get_all_embeddings()
                track_id = find_matching_id_with_best_fallback(emb, all_embs, SIMILARITY_THRESHOLD)
                if track_id is None:
                    track_id = next_id
                    next_id += 1
                emb_manager.add_embedding(track_id, emb)
                
                # 재초기화 완료
                frames_since_reinit = 0
                stats = emb_manager.get_stats()
                print(f"  재초기화 완료 (성공률: {success_rate:.2f}, 간격: {current_reinit_interval}, ID수: {stats['count']})")
            else:
                raw_cx, raw_cy = w // 2, h // 2
                track_id = None
                frames_since_reinit = 0
        
        # 스무딩
        if smooth_center is None:
            smooth_center = (raw_cx, raw_cy)
        else:
            scx, scy = smooth_center
            smooth_center = (int(ema_alpha * raw_cx + (1 - ema_alpha) * scx),
                             int(ema_alpha * raw_cy + (1 - ema_alpha) * scy))
        
        # 이상치 제거
        if prev_center is not None:
            dx = smooth_center[0] - prev_center[0]
            dy = smooth_center[1] - prev_center[1]
            if np.hypot(dx, dy) > jump_thresh:
                smooth_center = prev_center
        prev_center = smooth_center
        cx, cy = smooth_center
        
        # 클램핑
        half = crop_size // 2
        x0 = max(min(cx - half, w - crop_size), 0)
        y0 = max(min(cy - half, h - crop_size), 0)
        crop = frame[y0:y0+crop_size, x0:x0+crop_size]
        
        # 패딩
        ch, cw = crop.shape[:2]
        if ch != crop_size or cw != crop_size:
            crop = cv2.copyMakeBorder(crop, 0, crop_size-ch, 0, crop_size-cw, 
                                    cv2.BORDER_CONSTANT, value=[0,0,0])

        # write every frame
        writer.write(crop)

    pbar.close()
    cap.release()
    writer.release()
    print(f"크롭된 영상 저장 완료: {output_path}")