"""
ID 타임라인 생성 모듈 - Producer-Consumer 최적화
"""
import cv2
import torch
import numpy as np
import threading
import time
from PIL import Image
from tqdm import tqdm
from queue import Queue, Empty
import torch.nn.functional as F
from torchvision import transforms
from core.model_manager import ModelManager
from core.embedding_manager import SmartEmbeddingManager
from config import DEVICE, BATCH_SIZE_ID_TIMELINE, SIMILARITY_THRESHOLD
from utils.similarity_utils import find_matching_id_with_best_fallback


def generate_id_timeline(video_path: str, device=DEVICE, batch_size: int = BATCH_SIZE_ID_TIMELINE):
    """
    Producer-Consumer 패턴을 사용한 최적화된 ID 타임라인 생성
    
    Args:
        video_path: 비디오 파일 경로
        device: 사용할 디바이스
        batch_size: 배치 처리 크기
    
    Returns:
        tuple: (ID 타임라인, fps)
    """
    print("🎯 refactor.md ID 타임라인 Producer-Consumer 시작")
    print("🔄 Producer Thread: 비디오 프레임 I/O") 
    print("⚡ Consumer Thread: MTCNN + ResNet GPU 처리")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Producer-Consumer 설정
    frame_queue = Queue(maxsize=512)  # refactor.md 기준 큐 크기
    id_timeline = []
    timeline_lock = threading.Lock()
    producer_finished = threading.Event()
    
    # 공유 모델 및 데이터
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    resnet = model_manager.get_resnet()
    emb_manager = SmartEmbeddingManager()
    next_id = [1]  # 리스트로 감싸서 스레드 간 공유
    
    pbar = tqdm(total=total_frames, desc="[ID 타임라인 Producer-Consumer]")
    
    def producer():
        """Producer Thread: 비디오 프레임 I/O 전담"""
        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임과 인덱스를 함께 큐에 전달 (순서 보존)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_index, rgb_frame), timeout=30)  # 5→30초로 증가
                frame_index += 1
                
        except Exception as e:
            print(f"Producer 오류: {e}")
        finally:
            cap.release()
            producer_finished.set()
            print("🔄 Producer 완료")
    
    def consumer():
        """Consumer Thread: MTCNN + ResNet GPU 처리 전담"""
        batch_buffer = []
        processed_frames = 0
        
        # 결과를 순서대로 저장하기 위한 딕셔너리
        results_dict = {}
        
        try:
            while not producer_finished.is_set() or not frame_queue.empty():
                try:
                    # 배치 수집
                    while len(batch_buffer) < batch_size:
                        try:
                            frame_data = frame_queue.get(timeout=1.0)  # 0.1→1.0초로 증가
                            batch_buffer.append(frame_data)
                        except Empty:
                            if producer_finished.is_set():
                                break
                            continue
                    
                    if batch_buffer:
                        # 배치 데이터 분리
                        frame_indices, rgb_frames = zip(*batch_buffer)
                        pil_images = [Image.fromarray(rgb_frame) for rgb_frame in rgb_frames]
                        
                        # GPU 배치 처리
                        start_gpu = time.time()
                        
                        # 1. MTCNN 얼굴 감지 (배치)
                        boxes_list, _ = mtcnn.detect(pil_images)
                        
                        # 2. ResNet 얼굴 인식 (배치)
                        valid_faces = []
                        valid_indices = []
                        
                        for i, boxes in enumerate(boxes_list):
                            if boxes is not None and len(boxes) > 0:
                                x1, y1, x2, y2 = boxes[0]
                                face_crop = pil_images[i].crop((x1, y1, x2, y2)).resize((160, 160))
                                valid_faces.append(face_crop)
                                valid_indices.append(i)
                        
                        # ResNet 배치 처리
                        track_ids = []
                        if valid_faces:
                            face_tensors = torch.stack([
                                transforms.ToTensor()(face) for face in valid_faces
                            ]).to(device)
                            
                            embeddings = resnet(face_tensors)  # 배치 GPU 처리
                            
                            # ID 할당
                            for emb in embeddings:
                                all_embs = emb_manager.get_all_embeddings()
                                track_id = find_matching_id_with_best_fallback(
                                    emb.unsqueeze(0), all_embs, SIMILARITY_THRESHOLD
                                )
                                if track_id is None:
                                    track_id = next_id[0]
                                    next_id[0] += 1
                                emb_manager.add_embedding(track_id, emb.unsqueeze(0))
                                track_ids.append(track_id)
                        
                        gpu_time = time.time() - start_gpu
                        
                        # 결과 저장 (Thread-safe)
                        valid_idx = 0
                        for i, frame_idx in enumerate(frame_indices):
                            if i in valid_indices:
                                results_dict[frame_idx] = track_ids[valid_idx]
                                valid_idx += 1
                            else:
                                results_dict[frame_idx] = None
                        
                        processed_frames += len(batch_buffer)
                        pbar.update(len(batch_buffer))
                        
                        # 성능 로그
                        if len(batch_buffer) >= 256:
                            print(f"🚀 ID 배치: {len(batch_buffer)}프레임, GPU: {gpu_time:.2f}초")
                        
                        batch_buffer.clear()
                        
                except Exception as e:
                    print(f"Consumer 배치 오류: {e}")
                    continue
                    
        except Exception as e:
            print(f"Consumer 오류: {e}")
        finally:
            # 결과를 순서대로 재조립
            with timeline_lock:
                if results_dict:  # 결과가 있는 경우에만
                    for i in range(total_frames):
                        id_timeline.append(results_dict.get(i, None))
                else:
                    # 결과가 없으면 모든 프레임을 None으로 설정
                    print("⚠️ 경고: 처리 결과가 없어 모든 프레임을 None으로 설정")
                    for i in range(total_frames):
                        id_timeline.append(None)
            
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
    print(f"🎯 ID 타임라인 최적화 완료: {total_time:.2f}초")
    
    return id_timeline, fps