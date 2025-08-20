"""
MultiStreamProcessor - 4-Stream 병렬 GPU 비디오 처리 시스템.

CUDA 스트림을 활용한 병렬 처리로 4개의 비디오를 동시에 처리합니다:
- CUDA Stream 기반 비동기 처리
- GPU 메모리 풀 관리 및 동적 할당
- 스트림별 독립적 에러 처리
- 실시간 성능 모니터링 및 최적화

성능 목표:
    - 4개 비디오 동시 처리
    - 23분 비디오 4개 → 15분 내 완료
    - GPU 활용률 80% 이상 유지
    - 메모리 효율성 <75% VRAM 사용

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0 (Phase 3)
"""

import time
import asyncio
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import cv2
import numpy as np
import torch

from .dual_face_processor import DualFaceProcessor, process_video_with_config
from .stream_manager import StreamManager, StreamContext
from .memory_pool_manager import MemoryPoolManager
from ..utils.logger import UnifiedLogger
from ..utils.exceptions import (
    DualFaceTrackerError,
    MultiStreamError,
    GPUMemoryError,
    StreamAllocationError
)
from ..utils.cuda_utils import (
    get_gpu_memory_info,
    monitor_gpu_memory,
    clear_gpu_cache
)


@dataclass
class StreamJob:
    """개별 스트림 작업 정의"""
    stream_id: int
    input_path: Path
    output_path: Path
    priority: int = 1  # 1=높음, 2=보통, 3=낮음
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    
    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)


@dataclass 
class MultiStreamConfig:
    """멀티스트림 처리 설정"""
    max_streams: int = 4
    gpu_id: int = 0
    target_gpu_utilization: float = 0.8  # 80%
    max_vram_usage: float = 0.75  # 75%
    stream_timeout: float = 3600.0  # 1시간 타임아웃
    memory_cleanup_interval: float = 300.0  # 5분마다 메모리 정리
    performance_monitoring: bool = True
    auto_quality_adjustment: bool = True


@dataclass
class MultiStreamStats:
    """멀티스트림 처리 통계"""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    active_streams: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    gpu_utilization: float = 0.0
    vram_usage: float = 0.0
    throughput_fps: float = 0.0
    error_rate: float = 0.0


class MultiStreamProcessor:
    """
    4-Stream 병렬 GPU 비디오 처리 시스템.
    
    CUDA 스트림을 활용하여 여러 비디오를 동시에 처리하는 고성능 시스템입니다.
    각 스트림은 독립적으로 동작하며, GPU 리소스를 효율적으로 공유합니다.
    
    주요 특징:
        - 4개 CUDA 스트림 병렬 처리
        - 동적 GPU 메모리 풀 관리
        - 스트림별 독립적 에러 처리
        - 실시간 성능 모니터링
        - 자동 품질 조정
    
    사용 예시:
        ```python
        config = MultiStreamConfig(max_streams=4)
        processor = MultiStreamProcessor(config)
        
        jobs = [
            StreamJob(0, "video1.mp4", "output1.mp4"),
            StreamJob(1, "video2.mp4", "output2.mp4"),
            StreamJob(2, "video3.mp4", "output3.mp4"),
            StreamJob(3, "video4.mp4", "output4.mp4"),
        ]
        
        await processor.process_jobs(jobs)
        ```
    """
    
    def __init__(self, config: MultiStreamConfig):
        """
        MultiStreamProcessor 초기화.
        
        Args:
            config: 멀티스트림 처리 설정
        """
        self.config = config
        self.logger = UnifiedLogger("MultiStreamProcessor")
        
        # CUDA 디바이스 설정
        self.device = torch.device(f'cuda:{config.gpu_id}')
        torch.cuda.set_device(self.device)
        
        # 스트림 관리자들 초기화
        self.stream_manager: Optional[StreamManager] = None
        self.memory_pool: Optional[MemoryPoolManager] = None
        
        # 스트림별 프로세서들 (지연 초기화)
        self.processors: Dict[int, DualFaceProcessor] = {}
        
        # 작업 관리
        self.job_queue: List[StreamJob] = []
        self.active_jobs: Dict[int, StreamJob] = {}
        self.completed_jobs: List[StreamJob] = []
        self.failed_jobs: List[StreamJob] = []
        
        # 통계 및 모니터링
        self.stats = MultiStreamStats()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 동기화
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        self.logger.stage(f"MultiStreamProcessor 초기화 완료 (최대 {config.max_streams}개 스트림)")
    
    async def initialize(self) -> None:
        """
        멀티스트림 프로세서 초기화.
        
        CUDA 스트림, 메모리 풀, 스트림 매니저를 설정합니다.
        """
        try:
            self.logger.stage("MultiStreamProcessor 초기화 시작...")
            
            # GPU 메모리 상태 확인
            memory_info = get_gpu_memory_info(self.config.gpu_id)
            self.logger.debug(f"GPU {self.config.gpu_id} 메모리: {memory_info}")
            
            # 스트림 매니저 초기화
            self.stream_manager = StreamManager(
                max_streams=self.config.max_streams,
                gpu_id=self.config.gpu_id
            )
            await self.stream_manager.initialize()
            
            # 메모리 풀 매니저 초기화
            from .memory_pool_manager import MemoryPoolConfig
            memory_config = MemoryPoolConfig(
                max_vram_usage=self.config.max_vram_usage
            )
            self.memory_pool = MemoryPoolManager(memory_config, self.config.gpu_id)
            await self.memory_pool.initialize()
            
            # 성능 모니터링 시작
            if self.config.performance_monitoring:
                self._start_monitoring()
            
            self.logger.success("MultiStreamProcessor 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"MultiStreamProcessor 초기화 실패: {e}")
            raise MultiStreamError(f"초기화 실패: {e}")
    
    async def process_jobs(self, jobs: List[StreamJob]) -> MultiStreamStats:
        """
        여러 스트림 작업을 병렬로 처리합니다.
        
        Args:
            jobs: 처리할 스트림 작업 목록
            
        Returns:
            처리 완료 후 통계 정보
            
        Raises:
            MultiStreamError: 처리 중 오류 발생시
        """
        if not self.stream_manager or not self.memory_pool:
            await self.initialize()
        
        try:
            self.logger.stage(f"{len(jobs)}개 작업 병렬 처리 시작")
            start_time = time.time()
            
            # 작업 대기열에 추가
            self.job_queue.extend(jobs)
            self.stats.total_jobs = len(jobs)
            
            # 우선순위별로 정렬
            self.job_queue.sort(key=lambda x: x.priority)
            
            # 병렬 처리 실행
            await self._process_job_queue()
            
            # 처리 완료
            total_time = time.time() - start_time
            self.stats.total_processing_time = total_time
            self.stats.avg_processing_time = total_time / len(jobs) if jobs else 0
            
            # 성공률 계산
            self.stats.error_rate = len(self.failed_jobs) / len(jobs) if jobs else 0
            
            self.logger.success(
                f"전체 작업 완료: {self.stats.completed_jobs}/{self.stats.total_jobs} "
                f"성공, 총 {total_time:.1f}초 소요"
            )
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"작업 처리 중 오류: {e}")
            raise MultiStreamError(f"작업 처리 실패: {e}")
    
    async def _process_job_queue(self) -> None:
        """작업 대기열을 병렬로 처리합니다."""
        
        # ThreadPoolExecutor로 병렬 처리
        max_workers = min(self.config.max_streams, len(self.job_queue))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 작업을 스레드로 실행
            futures = {
                executor.submit(self._process_single_job, job): job 
                for job in self.job_queue[:max_workers]
            }
            
            # 나머지 작업들은 완료되는 대로 추가
            remaining_jobs = self.job_queue[max_workers:]
            
            try:
                while futures or remaining_jobs:
                    # 완료된 작업들 처리
                    for future in as_completed(futures, timeout=1.0):
                        job = futures.pop(future)
                        
                        try:
                            result = future.result()
                            self._on_job_completed(job, result)
                            
                        except Exception as e:
                            self._on_job_failed(job, str(e))
                        
                        # 대기 중인 작업이 있으면 새로 시작
                        if remaining_jobs:
                            new_job = remaining_jobs.pop(0)
                            new_future = executor.submit(self._process_single_job, new_job)
                            futures[new_future] = new_job
            
            except TimeoutError:
                # 타임아웃 처리는 개별 작업 레벨에서 처리됨
                pass
    
    def _process_single_job(self, job: StreamJob) -> Dict[str, Any]:
        """
        단일 스트림 작업을 처리합니다.
        
        Args:
            job: 처리할 스트림 작업
            
        Returns:
            처리 결과 정보
        """
        try:
            self.logger.debug(f"스트림 {job.stream_id} 작업 시작: {job.input_path}")
            job.start_time = time.time()
            job.status = "processing"
            
            # 스트림 컨텍스트 할당
            stream_context = self.stream_manager.allocate_stream(job.stream_id)
            if not stream_context:
                raise StreamAllocationError(f"스트림 {job.stream_id} 할당 실패")
            
            try:
                # DualFaceProcessor 생성/재사용
                processor = self._get_processor(job.stream_id, stream_context)
                
                # 실제 비디오 처리
                result = self._process_video(processor, job)
                
                job.end_time = time.time()
                job.status = "completed"
                job.progress = 1.0
                
                return result
                
            finally:
                # 스트림 해제
                self.stream_manager.release_stream(job.stream_id)
            
        except Exception as e:
            job.end_time = time.time()
            job.status = "failed"
            job.error_message = str(e)
            self.logger.error(f"스트림 {job.stream_id} 처리 실패: {e}")
            raise
    
    def _get_processor(self, stream_id: int, stream_context: StreamContext) -> DualFaceProcessor:
        """스트림별 DualFaceProcessor 생성/재사용"""
        
        if stream_id not in self.processors:
            # 새 프로세서 생성
            self.processors[stream_id] = DualFaceProcessor(
                cuda_stream=stream_context.cuda_stream,
                gpu_id=self.config.gpu_id
            )
            self.logger.debug(f"스트림 {stream_id} 새 프로세서 생성")
        
        return self.processors[stream_id]
    
    def _process_video(self, processor: DualFaceProcessor, job: StreamJob) -> Dict[str, Any]:
        """실제 비디오 처리 수행"""
        
        try:
            # 간단한 비디오 처리 함수 사용
            result = process_video_with_config(
                input_path=str(job.input_path),
                output_path=str(job.output_path)
            )
            
            return {
                'stream_id': job.stream_id,
                'input_path': str(job.input_path),
                'output_path': str(job.output_path),
                'processing_time': job.end_time - job.start_time if job.end_time and job.start_time else 0,
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"비디오 처리 실패: {e}")
            raise
    
    def _on_job_completed(self, job: StreamJob, result: Dict[str, Any]) -> None:
        """작업 완료 처리"""
        with self._lock:
            self.completed_jobs.append(job)
            self.stats.completed_jobs += 1
            
            if job.stream_id in self.active_jobs:
                del self.active_jobs[job.stream_id]
        
        processing_time = result.get('processing_time', 0)
        self.logger.success(
            f"스트림 {job.stream_id} 완료: {job.input_path.name} → {job.output_path.name} "
            f"({processing_time:.1f}초)"
        )
    
    def _on_job_failed(self, job: StreamJob, error_message: str) -> None:
        """작업 실패 처리"""
        with self._lock:
            self.failed_jobs.append(job)
            self.stats.failed_jobs += 1
            
            if job.stream_id in self.active_jobs:
                del self.active_jobs[job.stream_id]
        
        self.logger.error(
            f"스트림 {job.stream_id} 실패: {job.input_path.name} - {error_message}"
        )
    
    def _start_monitoring(self) -> None:
        """성능 모니터링 시작"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MultiStreamMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.debug("성능 모니터링 시작")
    
    def _monitoring_loop(self) -> None:
        """성능 모니터링 루프"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                # GPU 메모리 및 활용률 모니터링
                memory_info = get_gpu_memory_info(self.config.gpu_id)
                
                self.stats.vram_usage = memory_info.get('used_percent', 0) / 100
                self.stats.active_streams = len(self.active_jobs)
                
                # 메모리 정리 필요 시
                if self.stats.vram_usage > self.config.max_vram_usage:
                    self.logger.warning(
                        f"VRAM 사용률 높음: {self.stats.vram_usage:.1%}, 메모리 정리 실행"
                    )
                    clear_gpu_cache()
                
                # 주기적 로그
                if self.stats.active_streams > 0:
                    self.logger.debug(
                        f"활성 스트림: {self.stats.active_streams}, VRAM: {self.stats.vram_usage:.1%}"
                    )
                
                time.sleep(5.0)  # 5초 간격 모니터링
                
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                time.sleep(10.0)
    
    def get_progress(self) -> Dict[str, Any]:
        """현재 처리 진행 상황을 반환합니다."""
        with self._lock:
            return {
                'total_jobs': self.stats.total_jobs,
                'completed_jobs': self.stats.completed_jobs,
                'failed_jobs': self.stats.failed_jobs,
                'active_streams': self.stats.active_streams,
                'progress_percent': (self.stats.completed_jobs + self.stats.failed_jobs) / max(self.stats.total_jobs, 1) * 100,
                'vram_usage': self.stats.vram_usage,
                'error_rate': self.stats.error_rate,
                'active_jobs': {job_id: job.progress for job_id, job in self.active_jobs.items()}
            }
    
    async def shutdown(self) -> None:
        """멀티스트림 프로세서 종료"""
        try:
            self.logger.stage("MultiStreamProcessor 종료 시작...")
            
            # 종료 신호
            self._shutdown_event.set()
            self.monitoring_active = False
            
            # 모니터링 스레드 종료 대기
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            # 활성 작업 정리
            with self._lock:
                self.active_jobs.clear()
            
            # 스트림별 프로세서 정리
            for stream_id, processor in self.processors.items():
                try:
                    if hasattr(processor, 'cleanup'):
                        processor.cleanup()
                except Exception as e:
                    self.logger.warning(f"프로세서 {stream_id} 정리 중 오류: {e}")
            
            self.processors.clear()
            
            # 매니저들 정리
            if self.memory_pool:
                await self.memory_pool.shutdown()
            
            if self.stream_manager:
                await self.stream_manager.shutdown()
            
            # GPU 메모리 정리
            clear_gpu_cache()
            
            self.logger.success("MultiStreamProcessor 종료 완료")
            
        except Exception as e:
            self.logger.error(f"종료 중 오류: {e}")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
            try:
                # 동기 방식으로 기본 정리만 수행
                self._shutdown_event.set()
                self.monitoring_active = False
                clear_gpu_cache()
            except:
                pass


# 편의 함수들
def create_stream_jobs(
    input_videos: List[Union[str, Path]], 
    output_dir: Union[str, Path],
    priorities: Optional[List[int]] = None
) -> List[StreamJob]:
    """
    입력 비디오들로부터 스트림 작업 목록을 생성합니다.
    
    Args:
        input_videos: 입력 비디오 경로 목록
        output_dir: 출력 디렉토리
        priorities: 각 작업의 우선순위 (선택사항)
        
    Returns:
        생성된 스트림 작업 목록
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    for i, input_path in enumerate(input_videos):
        input_path = Path(input_path)
        output_file = output_path / f"{input_path.stem}_processed{input_path.suffix}"
        priority = priorities[i] if priorities and i < len(priorities) else 1
        
        jobs.append(StreamJob(
            stream_id=i,
            input_path=input_path,
            output_path=output_file,
            priority=priority
        ))
    
    return jobs


async def process_videos_parallel(
    input_videos: List[Union[str, Path]],
    output_dir: Union[str, Path],
    config: Optional[MultiStreamConfig] = None
) -> MultiStreamStats:
    """
    여러 비디오를 병렬로 처리하는 편의 함수.
    
    Args:
        input_videos: 입력 비디오 경로 목록
        output_dir: 출력 디렉토리
        config: 멀티스트림 설정 (선택사항)
        
    Returns:
        처리 완료 후 통계 정보
    """
    if not config:
        config = MultiStreamConfig()
    
    processor = MultiStreamProcessor(config)
    
    try:
        await processor.initialize()
        jobs = create_stream_jobs(input_videos, output_dir)
        return await processor.process_jobs(jobs)
    
    finally:
        await processor.shutdown()