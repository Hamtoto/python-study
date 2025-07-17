"""
얼굴 탐지 및 분석 모듈
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from core.model_manager import ModelManager
from config import DEVICE, BATCH_SIZE_ANALYZE

# OpenCV 최적화 플래그 설정
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;auto'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


def analyze_video_faces(video_path: str, batch_size: int = BATCH_SIZE_ANALYZE, device=DEVICE) -> tuple[list[bool], float]:
    """
    비디오의 각 프레임에서 얼굴 탐지 여부를 분석
    
    Args:
        video_path: 비디오 파일 경로
        batch_size: 배치 처리 크기
        device: 사용할 디바이스 (CPU/GPU)
    
    Returns:
        tuple: (얼굴 탐지 타임라인, fps)
    """
    print("## 1단계: 얼굴 탐지 분석 시작")
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    cap = cv2.VideoCapture(video_path)
    
    # OpenCV 최적화 설정
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_detected_timeline = []
    buffer = []
    
    with tqdm(total=frame_count, desc="[GPU 분석]") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV 프레임을 직접 버퍼에 저장 (PIL 변환 건너뛰기)
            buffer.append(frame)
            pbar.update(1)
            if len(buffer) == batch_size or pbar.n == frame_count:
                # PIL 변환 제거 - numpy 배열 직접 사용
                rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
                
                # MTCNN에 numpy 배열 리스트 전달 (PIL 건너뛰기)
                boxes_list, _ = mtcnn.detect(rgb_frames)
                face_detected_timeline.extend(b is not None for b in boxes_list)
                buffer.clear()
    
    if buffer:
        # 남은 버퍼도 numpy 배열 직접 사용
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
        boxes_list, _ = mtcnn.detect(rgb_frames)
        face_detected_timeline.extend(b is not None for b in boxes_list)
    
    cap.release()
    print("분석 완료")
    return face_detected_timeline, fps