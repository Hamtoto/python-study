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
from queue import Queue, Empty
from core.model_manager import ModelManager
from config import DEVICE, BATCH_SIZE_ANALYZE

# OpenCV 최적화 플래그 설정
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;auto'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


def calculate_optimal_batch_size(frame_queue, gpu_memory_usage=None):
    """Queue 깊이 기반 동적 배치 크기 계산 - GPU 메모리 안전"""
    try:
        queue_depth = frame_queue.qsize()
        if queue_depth > 256:  # 충분한 프레임 대기 중
            return 256  # 중간 배치로 제한 (메모리 안전)
        elif queue_depth > 128:
            return 128  # 작은 배치
        else:
            return 64  # 최소 배치 (빠른 시작)
    except:
        return 64  # 기본값 (안전)


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
    print("🎯 refactor.md 얼굴 분석 Producer-Consumer 시작")
    print("🔄 Producer Thread: 비디오 I/O")
    print("⚡ Consumer Thread: MTCNN GPU 처리")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Producer-Consumer 설정
    frame_queue = Queue(maxsize=512)  # refactor.md 기준 큐 크기
    face_detected_timeline = []
    timeline_lock = threading.Lock()
    producer_finished = threading.Event()
    
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    pbar = tqdm(total=frame_count, desc="[GPU 97%+ 분석]")
    
    def producer():
        """Producer Thread: 비디오 프레임 I/O 전담"""
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # OpenCV BGR → RGB 변환 (I/O 스레드에서 처리)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put(rgb_frame, timeout=5)
                
        except Exception as e:
            print(f"Producer 오류: {e}")
        finally:
            cap.release()
            producer_finished.set()
            print("🔄 Producer 완료")
    
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
                            frame = frame_queue.get(timeout=0.1)
                            buffer.append(frame)
                        except Empty:
                            if producer_finished.is_set():
                                break
                            continue
                    
                    if buffer:
                        # MTCNN GPU 배치 처리
                        start_gpu = time.time()
                        boxes_list, _ = mtcnn.detect(buffer)  # GPU 처리
                        gpu_time = time.time() - start_gpu
                        
                        # 결과 저장 (Thread-safe)
                        face_results = [b is not None for b in boxes_list]
                        with timeline_lock:
                            face_detected_timeline.extend(face_results)
                        
                        processed_frames += len(buffer)
                        pbar.update(len(buffer))
                        
                        # 성능 로그 (큰 배치만)
                        if len(buffer) >= 256:
                            print(f"🚀 동적 배치: {len(buffer)}프레임, GPU: {gpu_time:.2f}초")
                        
                        buffer.clear()
                        
                except Exception as e:
                    print(f"Consumer 배치 오류: {e}")
                    continue
                    
        except Exception as e:
            print(f"Consumer 오류: {e}")
        finally:
            print(f"⚡ Consumer 완료: {processed_frames}프레임 처리")
    
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
    print(f"🎯 GPU 최적화 완료: {total_time:.2f}초")
    
    return face_detected_timeline, fps