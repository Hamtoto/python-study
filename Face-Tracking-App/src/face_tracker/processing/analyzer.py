"""
얼굴 탐지 및 분석 모듈 - Producer-Consumer 최적화
"""
import os
import cv2
import torch
import numpy as np
import threading
import time
from PIL import Image
from tqdm import tqdm
from src.face_tracker.utils.logging import logger
from queue import Queue, Empty
from src.face_tracker.core.models import ModelManager
from src.face_tracker.config import DEVICE, BATCH_SIZE_ANALYZE

# OpenCV 최적화 플래그 설정
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;auto'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


def calculate_optimal_batch_size(frame_queue, gpu_memory_usage=None):
    """Queue 깊이 기반 동적 배치 크기 계산 - 6시간 영상 최적화"""
    try:
        queue_depth = frame_queue.qsize()
        if queue_depth > 512:  # 큐가 절반 이상 찼을 때 (1024의 50%)
            return 256  # 최대 배치 (GPU 메모리 안전)
        elif queue_depth > 256:  # 큐가 1/4 이상 찼을 때
            return 128  # 중간 배치
        elif queue_depth > 128:  # 큐에 일정량 쌓일 때
            return 64   # 작은 배치
        else:
            return 32   # 최소 배치 (빠른 시작, GPU 유휴 방지)
    except:
        return 32  # 기본값 (안전)


def analyze_video_faces(video_path: str, batch_size: int = BATCH_SIZE_ANALYZE, device=DEVICE) -> tuple[list[bool], float]:
    """
    Producer-Consumer 패턴을 사용한 최적화된 얼굴 탐지 분석
    
    Args:
        video_path: 비디오 파일 경로
        batch_size: 기본 배치 처리 크기 (동적 조정됨)
        device: 사용할 디바이스 (CPU/GPU)
    
    Returns:
        tuple: (얼굴 탐지 타임라인, fps)
    """
    logger.stage("얼굴 분석 시작")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Producer-Consumer 설정 - 6시간 영상 처리를 위해 큐 크기 확대
    frame_queue = Queue(maxsize=1024)  # 512→1024로 확대 (긴 영상 처리)
    face_detected_timeline = []
    timeline_lock = threading.Lock()
    producer_finished = threading.Event()
    
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    pbar = tqdm(total=frame_count, desc="얼굴분석", ncols=60, leave=False)
    
    def producer():
        """Producer Thread: 비디오 프레임 I/O 전담"""
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # OpenCV BGR → RGB 변환 (I/O 스레드에서 처리)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put(rgb_frame, timeout=30)  # 5→30초로 확대 (긴 영상 처리)
                
        except Exception as e:
            logger.error(f"프레임 읽기 오류: {e}")
        finally:
            cap.release()
            producer_finished.set()
    
    def consumer():
        """Consumer Thread: MTCNN GPU 처리 전담"""
        buffer = []
        processed_frames = 0
        
        try:
            while not producer_finished.is_set() or not frame_queue.empty():
                try:
                    # 동적 배치 크기 계산
                    optimal_batch = calculate_optimal_batch_size(frame_queue)
                    
                    # 프레임 수집
                    while len(buffer) < optimal_batch:
                        try:
                            frame = frame_queue.get(timeout=1.0)  # 0.1→1.0초로 증가
                            buffer.append(frame)
                        except Empty:
                            if producer_finished.is_set():
                                break
                            continue
                    
                    if buffer:
                        # MTCNN GPU 배치 처리 + 다중 스케일 폴백
                        start_gpu = time.time()
                        boxes_list, _ = mtcnn.detect(buffer)  # 기본 GPU 처리
                        
                        # 감지 실패한 프레임에 대해 다중 스케일 시도
                        face_results = []
                        for i, (frame, boxes) in enumerate(zip(buffer, boxes_list)):
                            if boxes is not None:
                                face_results.append(True)
                            else:
                                # 다중 스케일로 재시도
                                multi_boxes, _ = model_manager.detect_faces_multi_scale(frame)
                                face_results.append(multi_boxes is not None and len(multi_boxes) > 0)
                        
                        gpu_time = time.time() - start_gpu
                        
                        # 결과 저장 (Thread-safe)
                        with timeline_lock:
                            face_detected_timeline.extend(face_results)
                        
                        processed_frames += len(buffer)
                        pbar.update(len(buffer))
                        
                        
                        buffer.clear()
                        
                except Exception as e:
                    logger.error(f"배치 처리 오류: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"GPU 처리 오류: {e}")
        finally:
            pass
    
    # 스레드 시작
    producer_thread = threading.Thread(target=producer, daemon=True)
    consumer_thread = threading.Thread(target=consumer, daemon=True)
    
    start_time = time.time()
    producer_thread.start()
    consumer_thread.start()
    
    # 완료 대기
    producer_thread.join()
    consumer_thread.join()
    pbar.close()
    
    total_time = time.time() - start_time
    logger.success(f"얼굴 분석 완료 ({len(face_detected_timeline)}프레임, {total_time:.1f}초)")
    
    return face_detected_timeline, fps