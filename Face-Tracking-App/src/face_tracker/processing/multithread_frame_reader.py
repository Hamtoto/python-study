"""
멀티프로세싱 프레임 읽기 시스템 - Phase 1B CPU I/O 최적화  
7950X3D 16코어 최대 활용을 위한 병렬 프레임 로딩
"""
import os
import time
import threading
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from queue import Queue, Empty
from typing import List, Tuple, Optional
import cv2
from moviepy import VideoFileClip
import warnings

# MoviePy 경고 억제
warnings.filterwarnings("ignore", message=".*bytes wanted but.*bytes read.*")

from ..utils.logging import logger


def _read_single_frame_worker(args):
    """프로세스 워커용 단일 프레임 읽기 함수 (pickle 가능)"""
    video_path, index, timestamp = args
    try:
        # 각 프로세스에서 독립적으로 VideoFileClip 생성
        with VideoFileClip(video_path) as clip:
            frame = clip.get_frame(timestamp)
            return index, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 로깅 생략 (프로세스 간 공유 불가)
        return index, None


class MultiProcessFrameReader:
    """멀티프로세싱 프레임 읽기 시스템"""
    
    def __init__(self, max_workers: int = 4, buffer_size: int = 32):
        """
        Args:
            max_workers: 최대 워커 프로세스 수 (7950X3D 기준 4프로세스 추천)
            buffer_size: 프레임 버퍼 크기 (메모리 사용량과 성능 균형)
        """
        self.max_workers = max_workers
        self.buffer_size = buffer_size
        
    def read_frames_batch(self, video_path: str, time_stamps: List[float]) -> List[np.ndarray]:
        """
        배치 프레임 읽기 - 멀티스레드로 여러 프레임을 동시에 로드
        
        Args:
            video_path: 비디오 파일 경로
            time_stamps: 읽을 프레임의 시간 스탬프 리스트
            
        Returns:
            프레임 배열 리스트
        """
        if not time_stamps:
            return []
            
        logger.info(f"멀티프로세싱 프레임 읽기 시작 - {len(time_stamps)}개 프레임, {self.max_workers}개 프로세스")
        start_time = time.time()
        
        frames = [None] * len(time_stamps)
        
        # ProcessPoolExecutor로 병렬 처리 (독립 프로세스)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_index = {
                executor.submit(_read_single_frame_worker, (video_path, i, ts)): i 
                for i, ts in enumerate(time_stamps)
            }
            
            # 결과 수집
            for future in as_completed(future_to_index):
                try:
                    index, frame = future.result()
                    if frame is not None:
                        frames[index] = frame
                except Exception as e:
                    logger.error(f"스레드 실행 오류: {str(e)}")
        
        # None 제거 (실패한 프레임들)
        valid_frames = [f for f in frames if f is not None]
        
        elapsed = time.time() - start_time
        fps = len(valid_frames) / elapsed if elapsed > 0 else 0
        logger.success(f"멀티프로세싱 프레임 읽기 완료 - {len(valid_frames)}/{len(time_stamps)}개 성공, {elapsed:.1f}초, {fps:.1f} FPS")
        
        return valid_frames
    
    def read_frames_sequential_optimized(self, video_path: str, fps: float, duration: float) -> List[np.ndarray]:
        """
        최적화된 순차 프레임 읽기 - 기존 방식 대비 개선
        
        Args:
            video_path: 비디오 파일 경로
            fps: 비디오 FPS
            duration: 비디오 길이 (초)
            
        Returns:
            프레임 배열 리스트
        """
        # 시간 스탬프 생성 (1/fps 간격)
        time_stamps = []
        t = 0.0
        while t < duration:
            time_stamps.append(t)
            t += 1.0 / fps
            
        return self.read_frames_batch(video_path, time_stamps)
    
    def get_video_info(self, video_path: str) -> Tuple[float, float]:
        """
        비디오 정보 빠른 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            (fps, duration) 튜플
        """
        try:
            with VideoFileClip(video_path) as clip:
                return clip.fps, clip.duration
        except Exception as e:
            logger.error(f"비디오 정보 읽기 실패: {str(e)}")
            return 30.0, 0.0  # 기본값


class FrameBuffer:
    """프레임 버퍼 - 메모리 효율적인 프레임 관리"""
    
    def __init__(self, max_size: int = 64):
        """
        Args:
            max_size: 최대 버퍼 크기 (7950X3D 128MB L3 캐시 고려)
        """
        self.max_size = max_size
        self.buffer = Queue(maxsize=max_size)
        self.lock = threading.Lock()
        
    def put_frame(self, frame: np.ndarray) -> bool:
        """프레임을 버퍼에 추가"""
        try:
            self.buffer.put_nowait(frame)
            return True
        except:
            return False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """버퍼에서 프레임 가져오기"""
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def clear(self):
        """버퍼 비우기"""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except Empty:
                    break


def create_multithread_frame_reader(max_workers: int = None) -> MultiProcessFrameReader:
    """
    멀티프로세싱 프레임 리더 생성
    
    Args:
        max_workers: 최대 워커 수 (None이면 CPU 코어 기준 자동 설정)
        
    Returns:
        MultiProcessFrameReader 인스턴스
    """
    if max_workers is None:
        # 7950X3D 16코어 기준 4프로세스 사용 (디코딩 바운드이므로)
        max_workers = min(4, cpu_count() // 4)
        
    logger.info(f"멀티프로세싱 프레임 리더 생성 - {max_workers}개 워커")
    return MultiProcessFrameReader(max_workers=max_workers)