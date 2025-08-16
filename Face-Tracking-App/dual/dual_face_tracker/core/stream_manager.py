"""
StreamManager - CUDA 스트림 동적 할당 및 모니터링 시스템.

4개의 CUDA 스트림을 효율적으로 관리하고 할당하는 시스템입니다:
- CUDA Stream 생성, 할당, 해제 관리
- 스트림별 독립적 처리 상태 추적
- 동적 스트림 재할당 및 로드 밸런싱
- 스트림 상태 실시간 모니터링

주요 기능:
    - 스트림 풀 관리 (4개 스트림)
    - 스트림 할당/해제 자동화
    - 스트림별 처리 상태 추적
    - 부하 균형 최적화
    - 스트림 동기화 관리

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0 (Phase 3)
"""

import time
import asyncio
import threading
from typing import Optional, Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import cv2
import torch

from ..utils.logger import UnifiedLogger
from ..utils.exceptions import (
    DualFaceTrackerError,
    StreamAllocationError,
    StreamSynchronizationError
)
from ..utils.cuda_utils import get_gpu_memory_info


class StreamStatus(Enum):
    """CUDA 스트림 상태"""
    IDLE = "idle"           # 유휴
    ALLOCATED = "allocated" # 할당됨
    BUSY = "busy"          # 처리 중
    SYNCHRONIZING = "sync" # 동기화 중
    ERROR = "error"        # 오류 상태


@dataclass
class StreamContext:
    """CUDA 스트림 컨텍스트"""
    stream_id: int
    cuda_stream: torch.cuda.Stream
    opencv_stream: cv2.cuda.Stream
    status: StreamStatus = StreamStatus.IDLE
    allocated_time: Optional[float] = None
    last_activity: Optional[float] = None
    processing_count: int = 0
    error_count: int = 0
    owner_job_id: Optional[int] = None
    
    def __post_init__(self):
        self.last_activity = time.time()


@dataclass
class StreamPoolStats:
    """스트림 풀 통계"""
    total_streams: int = 0
    idle_streams: int = 0
    allocated_streams: int = 0
    busy_streams: int = 0
    error_streams: int = 0
    avg_allocation_time: float = 0.0
    total_allocations: int = 0
    allocation_failures: int = 0
    synchronization_count: int = 0


class StreamManager:
    """
    CUDA 스트림 관리자.
    
    4개의 CUDA 스트림을 효율적으로 관리하고 할당하는 시스템입니다.
    각 스트림은 독립적으로 동작하며, 동적 할당과 해제를 지원합니다.
    
    주요 특징:
        - CUDA Stream과 OpenCV Stream 연동
        - 스트림별 상태 추적 및 모니터링
        - 자동 로드 밸런싱
        - 에러 복구 및 재할당
        - 비동기 동기화 지원
    
    사용 예시:
        ```python
        manager = StreamManager(max_streams=4)
        await manager.initialize()
        
        # 스트림 할당
        context = manager.allocate_stream(job_id=1)
        if context:
            # 스트림 사용
            with manager.use_stream(context):
                # GPU 작업 수행
                pass
        
        # 스트림 해제
        manager.release_stream(job_id=1)
        ```
    """
    
    def __init__(self, max_streams: int = 4, gpu_id: int = 0):
        """
        StreamManager 초기화.
        
        Args:
            max_streams: 최대 스트림 개수 (기본 4개)
            gpu_id: GPU 디바이스 ID
        """
        self.max_streams = max_streams
        self.gpu_id = gpu_id
        self.logger = UnifiedLogger("StreamManager")
        
        # CUDA 디바이스 설정
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # 스트림 풀
        self.stream_contexts: Dict[int, StreamContext] = {}
        self.allocated_streams: Dict[int, StreamContext] = {}  # job_id -> context
        
        # 통계
        self.stats = StreamPoolStats()
        
        # 동기화
        self._lock = threading.RLock()
        self._allocation_lock = threading.Lock()
        
        # 초기화 상태
        self.initialized = False
        
        self.logger.debug(f"StreamManager 생성 (최대 {max_streams}개 스트림, GPU {gpu_id})")
    
    async def initialize(self) -> None:
        """
        스트림 풀을 초기화합니다.
        
        4개의 CUDA 스트림과 해당하는 OpenCV 스트림을 생성합니다.
        """
        try:
            self.logger.stage("StreamManager 초기화 시작...")
            
            # GPU 디바이스 설정
            torch.cuda.set_device(self.device)
            
            # 스트림 컨텍스트들 생성
            for stream_id in range(self.max_streams):
                try:
                    # CUDA Stream 생성
                    cuda_stream = torch.cuda.Stream(device=self.device)
                    
                    # OpenCV Stream 생성
                    opencv_stream = cv2.cuda.Stream()
                    
                    # 스트림 컨텍스트 생성
                    context = StreamContext(
                        stream_id=stream_id,
                        cuda_stream=cuda_stream,
                        opencv_stream=opencv_stream,
                        status=StreamStatus.IDLE
                    )
                    
                    self.stream_contexts[stream_id] = context
                    self.logger.debug(f"스트림 {stream_id} 생성 완료")
                    
                except Exception as e:
                    self.logger.error(f"스트림 {stream_id} 생성 실패: {e}")
                    raise StreamAllocationError(f"스트림 {stream_id} 생성 실패: {e}")
            
            # 통계 초기화
            self.stats.total_streams = len(self.stream_contexts)
            self.stats.idle_streams = self.stats.total_streams
            
            self.initialized = True
            self.logger.success(f"StreamManager 초기화 완료 ({self.stats.total_streams}개 스트림)")
            
        except Exception as e:
            self.logger.error(f"StreamManager 초기화 실패: {e}")
            raise StreamAllocationError(f"초기화 실패: {e}")
    
    def allocate_stream(self, job_id: int, priority: int = 1) -> Optional[StreamContext]:
        """
        스트림을 할당합니다.
        
        Args:
            job_id: 작업 ID
            priority: 우선순위 (1=높음, 2=보통, 3=낮음)
            
        Returns:
            할당된 스트림 컨텍스트 또는 None (할당 실패시)
        """
        if not self.initialized:
            self.logger.error("StreamManager가 초기화되지 않음")
            return None
        
        with self._allocation_lock:
            try:
                # 이미 할당된 작업 확인
                if job_id in self.allocated_streams:
                    existing_context = self.allocated_streams[job_id]
                    self.logger.warning(f"작업 {job_id}는 이미 스트림 {existing_context.stream_id}에 할당됨")
                    return existing_context
                
                # 사용 가능한 스트림 찾기
                available_context = self._find_available_stream(priority)
                if not available_context:
                    self.stats.allocation_failures += 1
                    self.logger.warning(f"사용 가능한 스트림이 없음 (작업 {job_id})")
                    return None
                
                # 스트림 할당
                available_context.status = StreamStatus.ALLOCATED
                available_context.owner_job_id = job_id
                available_context.allocated_time = time.time()
                available_context.last_activity = time.time()
                
                self.allocated_streams[job_id] = available_context
                
                # 통계 업데이트
                self._update_stats()
                self.stats.total_allocations += 1
                
                self.logger.debug(
                    f"스트림 {available_context.stream_id} 할당됨 → 작업 {job_id} "
                    f"(우선순위 {priority})"
                )
                
                return available_context
                
            except Exception as e:
                self.stats.allocation_failures += 1
                self.logger.error(f"스트림 할당 실패 (작업 {job_id}): {e}")
                return None
    
    def release_stream(self, job_id: int) -> bool:
        """
        스트림을 해제합니다.
        
        Args:
            job_id: 작업 ID
            
        Returns:
            해제 성공 여부
        """
        with self._allocation_lock:
            try:
                if job_id not in self.allocated_streams:
                    self.logger.warning(f"작업 {job_id}에 할당된 스트림이 없음")
                    return False
                
                context = self.allocated_streams[job_id]
                
                # 스트림 동기화 (진행 중인 작업 완료 대기)
                self._synchronize_stream(context)
                
                # 스트림 해제
                context.status = StreamStatus.IDLE
                context.owner_job_id = None
                context.allocated_time = None
                context.last_activity = time.time()
                context.processing_count += 1
                
                del self.allocated_streams[job_id]
                
                # 통계 업데이트
                self._update_stats()
                
                self.logger.debug(f"스트림 {context.stream_id} 해제됨 (작업 {job_id})")
                return True
                
            except Exception as e:
                self.logger.error(f"스트림 해제 실패 (작업 {job_id}): {e}")
                return False
    
    def _find_available_stream(self, priority: int) -> Optional[StreamContext]:
        """
        사용 가능한 스트림을 찾습니다.
        
        우선순위에 따라 최적의 스트림을 선택합니다:
        1. IDLE 상태 스트림 우선
        2. 사용 빈도가 낮은 스트림 우선
        3. 오류 발생이 적은 스트림 우선
        
        Args:
            priority: 요청 우선순위
            
        Returns:
            사용 가능한 스트림 컨텍스트 또는 None
        """
        idle_streams = [
            context for context in self.stream_contexts.values()
            if context.status == StreamStatus.IDLE
        ]
        
        if not idle_streams:
            return None
        
        # 우선순위에 따른 선택 기준
        if priority == 1:  # 높은 우선순위
            # 가장 적게 사용된 스트림
            return min(idle_streams, key=lambda x: (x.processing_count, x.error_count))
        
        elif priority == 2:  # 보통 우선순위
            # 마지막 활동 시간이 오래된 스트림
            return min(idle_streams, key=lambda x: x.last_activity or 0)
        
        else:  # 낮은 우선순위 (3)
            # 가장 먼저 찾은 스트림
            return idle_streams[0]
    
    def _synchronize_stream(self, context: StreamContext) -> None:
        """
        스트림을 동기화합니다.
        
        진행 중인 CUDA 작업이 완료될 때까지 대기합니다.
        
        Args:
            context: 동기화할 스트림 컨텍스트
        """
        try:
            context.status = StreamStatus.SYNCHRONIZING
            
            # CUDA Stream 동기화
            with torch.cuda.device(self.device):
                context.cuda_stream.synchronize()
            
            # OpenCV Stream 동기화 
            context.opencv_stream.waitForCompletion()
            
            self.stats.synchronization_count += 1
            self.logger.debug(f"스트림 {context.stream_id} 동기화 완료")
            
        except Exception as e:
            context.status = StreamStatus.ERROR
            context.error_count += 1
            self.logger.error(f"스트림 {context.stream_id} 동기화 실패: {e}")
            raise StreamSynchronizationError(f"스트림 동기화 실패: {e}")
    
    @contextmanager
    def use_stream(self, context: StreamContext):
        """
        스트림 사용 컨텍스트 매니저.
        
        스트림 사용 중 상태를 BUSY로 설정하고, 완료 후 ALLOCATED로 복원합니다.
        
        Args:
            context: 사용할 스트림 컨텍스트
        """
        old_status = context.status
        
        try:
            context.status = StreamStatus.BUSY
            context.last_activity = time.time()
            
            # CUDA 스트림 컨텍스트 설정
            with torch.cuda.stream(context.cuda_stream):
                yield context
        
        except Exception as e:
            context.status = StreamStatus.ERROR
            context.error_count += 1
            self.logger.error(f"스트림 {context.stream_id} 사용 중 오류: {e}")
            raise
        
        finally:
            # 원래 상태로 복원 (ERROR 상태가 아닌 경우)
            if context.status != StreamStatus.ERROR:
                context.status = old_status
            
            context.last_activity = time.time()
            self._update_stats()
    
    def _update_stats(self) -> None:
        """통계 정보를 업데이트합니다."""
        with self._lock:
            status_counts = {}
            for status in StreamStatus:
                status_counts[status] = 0
            
            for context in self.stream_contexts.values():
                status_counts[context.status] += 1
            
            self.stats.idle_streams = status_counts[StreamStatus.IDLE]
            self.stats.allocated_streams = status_counts[StreamStatus.ALLOCATED] 
            self.stats.busy_streams = status_counts[StreamStatus.BUSY]
            self.stats.error_streams = status_counts[StreamStatus.ERROR]
            
            # 평균 할당 시간 계산
            if self.stats.total_allocations > 0:
                allocation_times = [
                    time.time() - context.allocated_time
                    for context in self.allocated_streams.values()
                    if context.allocated_time
                ]
                if allocation_times:
                    self.stats.avg_allocation_time = sum(allocation_times) / len(allocation_times)
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        현재 스트림 상태를 반환합니다.
        
        Returns:
            스트림 상태 정보
        """
        self._update_stats()
        
        with self._lock:
            stream_details = {}
            for stream_id, context in self.stream_contexts.items():
                stream_details[stream_id] = {
                    'status': context.status.value,
                    'owner_job_id': context.owner_job_id,
                    'processing_count': context.processing_count,
                    'error_count': context.error_count,
                    'allocated_time': context.allocated_time,
                    'last_activity': context.last_activity
                }
            
            return {
                'stats': {
                    'total_streams': self.stats.total_streams,
                    'idle_streams': self.stats.idle_streams,
                    'allocated_streams': self.stats.allocated_streams,
                    'busy_streams': self.stats.busy_streams,
                    'error_streams': self.stats.error_streams,
                    'avg_allocation_time': self.stats.avg_allocation_time,
                    'total_allocations': self.stats.total_allocations,
                    'allocation_failures': self.stats.allocation_failures,
                    'allocation_success_rate': (
                        1 - self.stats.allocation_failures / max(self.stats.total_allocations, 1)
                    ) * 100
                },
                'streams': stream_details
            }
    
    def force_release_all(self) -> int:
        """
        모든 할당된 스트림을 강제로 해제합니다.
        
        비상 상황이나 시스템 종료 시 사용됩니다.
        
        Returns:
            해제된 스트림 개수
        """
        released_count = 0
        
        with self._allocation_lock:
            allocated_job_ids = list(self.allocated_streams.keys())
            
            for job_id in allocated_job_ids:
                try:
                    if self.release_stream(job_id):
                        released_count += 1
                except Exception as e:
                    self.logger.error(f"강제 해제 실패 (작업 {job_id}): {e}")
        
        self.logger.warning(f"강제 해제 완료: {released_count}개 스트림")
        return released_count
    
    def reset_error_streams(self) -> int:
        """
        오류 상태의 스트림들을 재설정합니다.
        
        Returns:
            재설정된 스트림 개수
        """
        reset_count = 0
        
        with self._lock:
            for context in self.stream_contexts.values():
                if context.status == StreamStatus.ERROR:
                    try:
                        # 스트림 동기화 시도
                        self._synchronize_stream(context)
                        context.status = StreamStatus.IDLE
                        context.error_count = 0
                        reset_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"스트림 {context.stream_id} 재설정 실패: {e}")
        
        if reset_count > 0:
            self.logger.success(f"{reset_count}개 오류 스트림 재설정 완료")
            self._update_stats()
        
        return reset_count
    
    async def shutdown(self) -> None:
        """
        StreamManager 종료.
        
        모든 스트림을 해제하고 리소스를 정리합니다.
        """
        try:
            self.logger.stage("StreamManager 종료 시작...")
            
            # 모든 스트림 강제 해제
            released_count = self.force_release_all()
            
            # 모든 스트림 동기화
            for context in self.stream_contexts.values():
                try:
                    if context.status != StreamStatus.ERROR:
                        self._synchronize_stream(context)
                except Exception as e:
                    self.logger.warning(f"스트림 {context.stream_id} 종료 동기화 실패: {e}")
            
            # 스트림 컨텍스트들 정리
            self.stream_contexts.clear()
            self.allocated_streams.clear()
            
            self.initialized = False
            
            self.logger.success(f"StreamManager 종료 완료 ({released_count}개 스트림 해제)")
            
        except Exception as e:
            self.logger.error(f"StreamManager 종료 중 오류: {e}")
    
    def __del__(self):
        """소멸자 - 기본 리소스 정리"""
        if hasattr(self, 'initialized') and self.initialized:
            try:
                self.force_release_all()
            except:
                pass


# 편의 함수들
async def create_stream_manager(max_streams: int = 4, gpu_id: int = 0) -> StreamManager:
    """
    초기화된 StreamManager를 생성합니다.
    
    Args:
        max_streams: 최대 스트림 개수
        gpu_id: GPU 디바이스 ID
        
    Returns:
        초기화된 StreamManager 인스턴스
    """
    manager = StreamManager(max_streams, gpu_id)
    await manager.initialize()
    return manager


def get_stream_allocation_stats() -> Dict[str, Any]:
    """
    전역 스트림 할당 통계를 반환합니다.
    
    Returns:
        스트림 할당 통계 정보
    """
    # GPU 메모리 정보 포함
    memory_info = get_gpu_memory_info(0)  # 기본 GPU
    
    return {
        'gpu_memory': memory_info,
        'timestamp': time.time()
    }