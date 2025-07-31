"""
GPU 배치 처리 통합 모듈
- GPU 전용 프로세스로 모든 세그먼트를 배치 처리
- CPU 멀티프로세싱과 GPU 단일프로세스 분리
"""

import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import List, Dict, Any
import cv2
import numpy as np
from moviepy import VideoFileClip
import warnings

# MoviePy 경고 억제
warnings.filterwarnings("ignore", message=".*bytes wanted but.*bytes read.*")

from ..core.models import ModelManager
from ..utils.logging import logger
from ..config import DEVICE, TEMP_ROOT


class GPUBatchProcessor:
    """GPU 전용 배치 처리 클래스"""
    
    def __init__(self, device = DEVICE):
        # torch.device 객체든 문자열이든 처리
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        self.model_manager = None  # 프로세스 내에서 초기화
        self.task_queue = None
        self.result_queue = None
        self.is_running = False
        
    def initialize_models(self):
        """GPU 모델 초기화 (프로세스 내에서 실행)"""
        self.model_manager = ModelManager(self.device)
        logger.info(f"GPU 배치 프로세서 초기화 완료 - {self.device}")
        
    def process_batch_segments(self, task_queue: Queue, result_queue: Queue):
        """GPU 배치 처리 워커 프로세스"""
        try:
            # 프로세스 내에서 모델 초기화
            self.initialize_models()
            self.task_queue = task_queue
            self.result_queue = result_queue
            self.is_running = True
            
            logger.info("GPU 배치 처리 워커 시작")
            
            while self.is_running:
                try:
                    # 태스크 배치 수신 (타임아웃 10초)
                    batch_tasks = task_queue.get(timeout=10.0)
                    
                    if batch_tasks == "STOP":
                        break
                        
                    # 배치 처리 실행
                    batch_results = self._process_segment_batch(batch_tasks)
                    
                    # 결과 전송
                    result_queue.put(batch_results)
                    
                except mp.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"GPU 배치 처리 오류: {str(e)}")
                    result_queue.put({"error": str(e)})
                    
        except Exception as e:
            logger.error(f"GPU 워커 프로세스 초기화 오류: {str(e)}")
        finally:
            if self.model_manager:
                # GPU 메모리 정리
                torch.cuda.empty_cache()
            logger.info("GPU 배치 처리 워커 종료")
            
    def _process_segment_batch(self, batch_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """세그먼트 배치 GPU 처리"""
        results = []
        
        try:
            logger.info(f"GPU 배치 처리 시작 - {len(batch_tasks)}개 세그먼트")
            start_time = time.time()
            
            # 배치별로 프레임 로드 및 GPU 처리
            for i, task in enumerate(batch_tasks):
                try:
                    seg_input = task['seg_input']
                    seg_cropped = task['seg_cropped']
                    
                    # 세그먼트 처리
                    success = self._process_single_segment_gpu(seg_input, seg_cropped)
                    
                    results.append({
                        'task_id': i,
                        'seg_fname': task['seg_fname'],
                        'success': success,
                        'error': None if success else "GPU 처리 실패"
                    })
                    
                    # 진행상황 로깅
                    if (i + 1) % 4 == 0 or (i + 1) == len(batch_tasks):
                        logger.info(f"GPU 처리 진행: {i + 1}/{len(batch_tasks)}")
                        
                except Exception as e:
                    logger.error(f"세그먼트 {i} GPU 처리 오류: {str(e)}")
                    results.append({
                        'task_id': i,
                        'seg_fname': task.get('seg_fname', f'segment_{i}'),
                        'success': False,
                        'error': str(e)
                    })
            
            elapsed = time.time() - start_time
            logger.success(f"GPU 배치 처리 완료 - {len(batch_tasks)}개 세그먼트, {elapsed:.1f}초")
            
        except Exception as e:
            logger.error(f"GPU 배치 처리 전체 오류: {str(e)}")
            
        return results
        
    def _process_single_segment_gpu(self, seg_input: str, seg_cropped: str) -> bool:
        """단일 세그먼트 GPU 처리"""
        try:
            # MoviePy로 비디오 클립 로드 (경고 억제됨)
            with VideoFileClip(seg_input) as clip:
                fps = clip.fps
                duration = clip.duration
                
                if duration < 1.0:  # 1초 미만 스킵
                    logger.warning(f"세그먼트 너무 짧음: {duration:.1f}초")
                    return False
                
                # 프레임 추출 및 배치 처리
                frames = []
                for t in np.arange(0, duration, 1.0/fps):  # 1/fps 간격으로 프레임 추출
                    if t >= duration:
                        break
                    try:
                        frame = clip.get_frame(t)
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    except:
                        continue
                
                if not frames:
                    logger.warning("추출된 프레임 없음")
                    return False
                
                # GPU 배치 얼굴 검출 및 크롭
                cropped_frames = self._detect_and_crop_batch(frames)
                
                if not cropped_frames:
                    logger.warning("검출된 얼굴 없음")
                    return False
                
                # 크롭된 프레임으로 비디오 생성
                self._save_cropped_video(cropped_frames, seg_cropped, fps)
                
            return True
            
        except Exception as e:
            logger.error(f"세그먼트 GPU 처리 오류: {str(e)}")
            return False
    
    def _detect_and_crop_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """배치 얼굴 검출 및 크롭"""
        cropped_frames = []
        
        try:
            # 프레임을 배치로 처리
            batch_size = min(32, len(frames))  # 배치 크기 제한
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]
                
                # MTCNN 배치 검출
                batch_boxes, batch_probs = self.model_manager.mtcnn.detect(batch_frames)
                
                # 각 프레임별 크롭
                for j, (frame, boxes, probs) in enumerate(zip(batch_frames, batch_boxes, batch_probs)):
                    if boxes is not None and len(boxes) > 0:
                        # 첫 번째 검출된 얼굴 사용
                        box = boxes[0].astype(int)
                        prob = probs[0]
                        
                        if prob > 0.7:  # 신뢰도 임계값
                            x1, y1, x2, y2 = box
                            
                            # 바운딩 박스 검증 및 조정
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                cropped = frame[y1:y2, x1:x2]
                                if cropped.shape[0] > 50 and cropped.shape[1] > 50:  # 최소 크기 검증
                                    cropped_frames.append(cropped)
                    
        except Exception as e:
            logger.error(f"배치 얼굴 검출 오류: {str(e)}")
            
        return cropped_frames
    
    def _save_cropped_video(self, cropped_frames: List[np.ndarray], output_path: str, fps: float):
        """크롭된 프레임으로 비디오 저장"""
        try:
            if not cropped_frames:
                return
                
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 첫 번째 프레임으로 크기 결정
            h, w = cropped_frames[0].shape[:2]
            
            # VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # 프레임 쓰기
            for frame in cropped_frames:
                # 크기 조정 (필요한 경우)
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
            
            out.release()
            
        except Exception as e:
            logger.error(f"크롭 비디오 저장 오류: {str(e)}")
    


def create_gpu_batch_processor(device: str = DEVICE) -> tuple:
    """GPU 배치 프로세서 생성 및 큐 반환"""
    task_queue = Queue(maxsize=16)  # 최대 16개 배치 대기
    result_queue = Queue(maxsize=16)
    
    processor = GPUBatchProcessor(device)
    gpu_process = Process(
        target=processor.process_batch_segments,
        args=(task_queue, result_queue),
        name="GPUBatchProcessor"
    )
    
    return gpu_process, task_queue, result_queue


def split_tasks_into_batches(tasks: List[Dict[str, Any]], batch_size: int = 4) -> List[List[Dict[str, Any]]]:
    """태스크를 배치로 분할"""
    batches = []
    for i in range(0, len(tasks), batch_size):
        batches.append(tasks[i:i+batch_size])
    return batches