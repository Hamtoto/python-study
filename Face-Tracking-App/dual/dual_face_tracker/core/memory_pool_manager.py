"""
MemoryPoolManager - GPU 메모리 풀 동적 관리 시스템.

GPU 메모리를 효율적으로 할당, 관리, 해제하는 시스템입니다:
- GPU 메모리 풀 사전 할당 및 재사용
- 동적 배치 크기 조정
- 메모리 단편화 방지
- OOM 예방 및 복구 메커니즘

주요 기능:
    - 스마트 메모리 풀 관리
    - 동적 배치 크기 최적화
    - 메모리 사용량 실시간 모니터링
    - 자동 메모리 정리 및 압축
    - OOM 감지 및 예방

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0 (Phase 3)
"""

import time
import threading
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import gc
import cv2
import numpy as np
import torch

from ..utils.logger import UnifiedLogger
from ..utils.exceptions import (
    DualFaceTrackerError,
    GPUMemoryError,
    MemoryPoolError
)
from ..utils.cuda_utils import (
    get_gpu_memory_info,
    monitor_gpu_memory,
    clear_gpu_cache
)


class MemoryPoolType(Enum):
    """메모리 풀 타입"""
    DECODER = "decoder"       # 디코딩용 메모리
    COMPOSER = "composer"     # 합성용 메모리  
    ENCODER = "encoder"       # 인코딩용 메모리
    TEMPORARY = "temporary"   # 임시 메모리
    SHARED = "shared"         # 공유 메모리


class MemoryAllocationStrategy(Enum):
    """메모리 할당 전략"""
    CONSERVATIVE = "conservative"   # 보수적 할당 (50% 사용)
    BALANCED = "balanced"          # 균형 할당 (65% 사용)
    AGGRESSIVE = "aggressive"      # 적극적 할당 (80% 사용)
    ADAPTIVE = "adaptive"          # 적응형 할당 (동적)


@dataclass
class MemoryBlock:
    """메모리 블록 정보"""
    block_id: str
    pool_type: MemoryPoolType
    size_bytes: int
    allocated_time: float
    last_used_time: float
    use_count: int = 0
    is_allocated: bool = False
    tensor_data: Optional[torch.Tensor] = None
    gpu_mat_data: Optional[cv2.cuda.GpuMat] = None
    
    def __post_init__(self):
        if not hasattr(self, 'last_used_time'):
            self.last_used_time = self.allocated_time


@dataclass
class MemoryPoolConfig:
    """메모리 풀 설정"""
    max_vram_usage: float = 0.75  # 최대 VRAM 사용률 (75%)
    allocation_strategy: MemoryAllocationStrategy = MemoryAllocationStrategy.ADAPTIVE
    pool_initial_size_mb: Dict[MemoryPoolType, int] = field(default_factory=lambda: {
        MemoryPoolType.DECODER: 2048,    # 2GB
        MemoryPoolType.COMPOSER: 1024,   # 1GB  
        MemoryPoolType.ENCODER: 1024,    # 1GB
        MemoryPoolType.TEMPORARY: 512,   # 512MB
        MemoryPoolType.SHARED: 1024      # 1GB
    })
    cleanup_threshold: float = 0.9  # 90% 사용 시 정리 시작
    cleanup_interval: float = 60.0  # 1분마다 정리 체크
    fragmentation_threshold: float = 0.3  # 30% 단편화 시 압축
    adaptive_batch_sizes: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        # (min_size, default_size, max_size)
        'decode': (1, 4, 16),
        'compose': (1, 2, 8),
        'encode': (1, 4, 12)
    })


@dataclass
class MemoryPoolStats:
    """메모리 풀 통계"""
    total_allocated_bytes: int = 0
    total_used_bytes: int = 0
    fragmentation_ratio: float = 0.0
    allocation_count: int = 0
    deallocation_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    oom_events: int = 0
    cleanup_events: int = 0
    compaction_events: int = 0
    current_batch_sizes: Dict[str, int] = field(default_factory=dict)


class MemoryPoolManager:
    """
    GPU 메모리 풀 관리자.
    
    여러 타입의 메모리 풀을 관리하고 동적으로 최적화하는 시스템입니다.
    메모리 효율성을 극대화하고 OOM을 예방합니다.
    
    주요 특징:
        - 타입별 메모리 풀 관리
        - 동적 배치 크기 조정
        - 스마트 메모리 재사용
        - 자동 메모리 정리 및 압축
        - OOM 예방 및 복구
    
    사용 예시:
        ```python
        config = MemoryPoolConfig()
        manager = MemoryPoolManager(config)
        await manager.initialize()
        
        # 메모리 할당
        with manager.allocate_memory(MemoryPoolType.DECODER, 1024*1024) as memory:
            # 메모리 사용
            tensor = memory.tensor_data
            
        # 배치 크기 최적화
        optimal_size = manager.get_optimal_batch_size('decode')
        ```
    """
    
    def __init__(self, config: MemoryPoolConfig, gpu_id: int = 0):
        """
        MemoryPoolManager 초기화.
        
        Args:
            config: 메모리 풀 설정
            gpu_id: GPU 디바이스 ID
        """
        self.config = config
        self.gpu_id = gpu_id
        self.logger = UnifiedLogger("MemoryPoolManager")
        
        # CUDA 디바이스 설정
        self.device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(self.device)
        
        # 메모리 풀들
        self.memory_pools: Dict[MemoryPoolType, List[MemoryBlock]] = {
            pool_type: [] for pool_type in MemoryPoolType
        }
        
        # 할당된 메모리 추적
        self.allocated_blocks: Dict[str, MemoryBlock] = {}
        
        # 통계
        self.stats = MemoryPoolStats()
        
        # 동기화
        self._lock = threading.RLock()
        self._allocation_lock = threading.Lock()
        
        # 백그라운드 정리 스레드
        self._cleanup_active = False
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # 초기화 상태
        self.initialized = False
        
        # 현재 배치 크기들 초기화
        for operation, (_, default, _) in config.adaptive_batch_sizes.items():
            self.stats.current_batch_sizes[operation] = default
        
        self.logger.debug(f"MemoryPoolManager 생성 (GPU {gpu_id})")
    
    async def initialize(self) -> None:
        """
        메모리 풀을 초기화합니다.
        
        각 타입별로 초기 메모리 풀을 생성하고 백그라운드 정리를 시작합니다.
        """
        try:
            self.logger.stage("MemoryPoolManager 초기화 시작...")
            
            # GPU 메모리 상태 확인
            memory_info = get_gpu_memory_info(self.gpu_id)
            total_vram = memory_info.get('total', 0)
            available_vram = memory_info.get('free', 0)
            
            self.logger.debug(f"GPU 메모리: {available_vram/1024**3:.1f}GB 사용가능 / {total_vram/1024**3:.1f}GB 전체")
            
            # 각 타입별 메모리 풀 초기화
            for pool_type, initial_size_mb in self.config.pool_initial_size_mb.items():
                try:
                    await self._initialize_pool(pool_type, initial_size_mb)
                    self.logger.debug(f"{pool_type.value} 풀 초기화 완료 ({initial_size_mb}MB)")
                
                except Exception as e:
                    self.logger.error(f"{pool_type.value} 풀 초기화 실패: {e}")
                    # 초기화 실패한 풀은 빈 상태로 유지
            
            # 백그라운드 정리 시작
            self._start_background_cleanup()
            
            self.initialized = True
            self.logger.success("MemoryPoolManager 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"MemoryPoolManager 초기화 실패: {e}")
            raise MemoryPoolError(f"초기화 실패: {e}")
    
    async def _initialize_pool(self, pool_type: MemoryPoolType, size_mb: int) -> None:
        """
        특정 타입의 메모리 풀을 초기화합니다.
        
        Args:
            pool_type: 메모리 풀 타입
            size_mb: 초기 할당 크기 (MB)
        """
        size_bytes = size_mb * 1024 * 1024
        
        try:
            # 메모리 풀용 텐서 생성
            pool_tensor = torch.empty(
                size_bytes // 4,  # float32 기준
                dtype=torch.float32,
                device=self.device
            )
            
            # GPU Mat도 준비 (OpenCV 작업용)
            height, width = 1080, 1920  # 기본 해상도
            gpu_mat = cv2.cuda.GpuMat(height, width, cv2.CV_8UC3)
            
            # 메모리 블록 생성
            block = MemoryBlock(
                block_id=f"{pool_type.value}_initial_{int(time.time())}",
                pool_type=pool_type,
                size_bytes=size_bytes,
                allocated_time=time.time(),
                last_used_time=time.time(),
                tensor_data=pool_tensor,
                gpu_mat_data=gpu_mat
            )
            
            self.memory_pools[pool_type].append(block)
            self.stats.total_allocated_bytes += size_bytes
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.warning(f"{pool_type.value} 풀 초기화 중 OOM, 크기 축소 재시도")
            
            # 크기를 반으로 줄여서 재시도
            reduced_size_mb = max(size_mb // 2, 64)  # 최소 64MB
            if reduced_size_mb < size_mb:
                await self._initialize_pool(pool_type, reduced_size_mb)
            else:
                raise MemoryPoolError(f"{pool_type.value} 풀 초기화 실패: {e}")
    
    @contextmanager
    def allocate_memory(self, pool_type: MemoryPoolType, size_bytes: int):
        """
        메모리를 할당하고 사용 후 자동으로 해제하는 컨텍스트 매니저.
        
        Args:
            pool_type: 메모리 풀 타입
            size_bytes: 필요한 메모리 크기
            
        Yields:
            할당된 메모리 블록
        """
        allocated_block = None
        
        try:
            # 메모리 할당
            allocated_block = self._allocate_block(pool_type, size_bytes)
            if not allocated_block:
                raise MemoryPoolError(f"메모리 할당 실패: {pool_type.value}, {size_bytes} bytes")
            
            yield allocated_block
            
        finally:
            # 메모리 해제
            if allocated_block:
                self._deallocate_block(allocated_block)
    
    def _allocate_block(self, pool_type: MemoryPoolType, size_bytes: int) -> Optional[MemoryBlock]:
        """
        메모리 블록을 할당합니다.
        
        Args:
            pool_type: 메모리 풀 타입
            size_bytes: 필요한 메모리 크기
            
        Returns:
            할당된 메모리 블록 또는 None
        """
        with self._allocation_lock:
            try:
                # 재사용 가능한 블록 찾기
                available_block = self._find_reusable_block(pool_type, size_bytes)
                
                if available_block:
                    # 기존 블록 재사용
                    available_block.is_allocated = True
                    available_block.last_used_time = time.time()
                    available_block.use_count += 1
                    
                    self.allocated_blocks[available_block.block_id] = available_block
                    self.stats.cache_hits += 1
                    self.stats.allocation_count += 1
                    
                    self.logger.debug(f"메모리 블록 재사용: {available_block.block_id}")
                    return available_block
                
                else:
                    # 새 블록 생성
                    new_block = self._create_new_block(pool_type, size_bytes)
                    if new_block:
                        self.stats.cache_misses += 1
                        self.stats.allocation_count += 1
                        return new_block
                
                return None
                
            except torch.cuda.OutOfMemoryError as e:
                self.logger.warning(f"OOM 발생, 메모리 정리 후 재시도: {e}")
                
                # 긴급 메모리 정리
                self._emergency_cleanup()
                
                # 재시도
                try:
                    new_block = self._create_new_block(pool_type, size_bytes)
                    if new_block:
                        self.stats.oom_events += 1
                        return new_block
                
                except Exception as retry_e:
                    self.logger.error(f"OOM 복구 실패: {retry_e}")
                    self.stats.oom_events += 1
                    return None
            
            except Exception as e:
                self.logger.error(f"메모리 할당 실패: {e}")
                return None
    
    def _find_reusable_block(self, pool_type: MemoryPoolType, size_bytes: int) -> Optional[MemoryBlock]:
        """
        재사용 가능한 메모리 블록을 찾습니다.
        
        Args:
            pool_type: 메모리 풀 타입
            size_bytes: 필요한 메모리 크기
            
        Returns:
            재사용 가능한 메모리 블록 또는 None
        """
        pool = self.memory_pools.get(pool_type, [])
        
        # 사용 중이지 않고 크기가 적절한 블록 찾기
        available_blocks = [
            block for block in pool
            if not block.is_allocated and block.size_bytes >= size_bytes
        ]
        
        if not available_blocks:
            return None
        
        # 크기가 가장 적절한 블록 선택 (메모리 효율성)
        best_block = min(available_blocks, key=lambda b: b.size_bytes)
        
        # 크기가 너무 크면 새로 생성하는 것이 나을 수 있음
        size_ratio = best_block.size_bytes / size_bytes
        if size_ratio > 4.0:  # 4배 이상 크면 새로 생성
            return None
        
        return best_block
    
    def _create_new_block(self, pool_type: MemoryPoolType, size_bytes: int) -> Optional[MemoryBlock]:
        """
        새로운 메모리 블록을 생성합니다.
        
        Args:
            pool_type: 메모리 풀 타입
            size_bytes: 메모리 크기
            
        Returns:
            생성된 메모리 블록 또는 None
        """
        try:
            # GPU 메모리 체크
            memory_info = get_gpu_memory_info(self.gpu_id)
            available_bytes = memory_info.get('free', 0)
            
            if available_bytes < size_bytes * 1.2:  # 20% 여유분 필요
                self.logger.warning(f"GPU 메모리 부족: 필요 {size_bytes/1024**2:.1f}MB, 사용가능 {available_bytes/1024**2:.1f}MB")
                return None
            
            # 텐서 생성
            tensor_elements = max(size_bytes // 4, 1)  # float32 기준
            pool_tensor = torch.empty(
                tensor_elements,
                dtype=torch.float32,
                device=self.device
            )
            
            # GPU Mat 생성 (OpenCV용)
            # 적절한 해상도로 계산
            pixels = size_bytes // 3  # RGB 기준
            height = int((pixels / (1920 / 1080)) ** 0.5)
            width = int(height * (1920 / 1080))
            height = max(height, 240)  # 최소 해상도
            width = max(width, 320)
            
            gpu_mat = cv2.cuda.GpuMat(height, width, cv2.CV_8UC3)
            
            # 메모리 블록 생성
            block = MemoryBlock(
                block_id=f"{pool_type.value}_{int(time.time())}_{len(self.memory_pools[pool_type])}",
                pool_type=pool_type,
                size_bytes=size_bytes,
                allocated_time=time.time(),
                last_used_time=time.time(),
                is_allocated=True,
                tensor_data=pool_tensor,
                gpu_mat_data=gpu_mat
            )
            
            # 풀에 추가
            self.memory_pools[pool_type].append(block)
            self.allocated_blocks[block.block_id] = block
            
            self.stats.total_allocated_bytes += size_bytes
            
            self.logger.debug(f"새 메모리 블록 생성: {block.block_id} ({size_bytes/1024**2:.1f}MB)")
            return block
            
        except Exception as e:
            self.logger.error(f"메모리 블록 생성 실패: {e}")
            return None
    
    def _deallocate_block(self, block: MemoryBlock) -> None:
        """
        메모리 블록을 해제합니다.
        
        Args:
            block: 해제할 메모리 블록
        """
        with self._allocation_lock:
            try:
                block.is_allocated = False
                block.last_used_time = time.time()
                
                # 할당 추적에서 제거
                if block.block_id in self.allocated_blocks:
                    del self.allocated_blocks[block.block_id]
                
                self.stats.deallocation_count += 1
                self.stats.total_used_bytes = sum(
                    block.size_bytes for block in self.allocated_blocks.values()
                )
                
                self.logger.debug(f"메모리 블록 해제: {block.block_id}")
                
            except Exception as e:
                self.logger.error(f"메모리 블록 해제 실패: {e}")
    
    def get_optimal_batch_size(self, operation: str) -> int:
        """
        현재 메모리 상태에 기반한 최적 배치 크기를 반환합니다.
        
        Args:
            operation: 작업 타입 ('decode', 'compose', 'encode')
            
        Returns:
            최적 배치 크기
        """
        if operation not in self.config.adaptive_batch_sizes:
            return 4  # 기본값
        
        min_size, default_size, max_size = self.config.adaptive_batch_sizes[operation]
        current_size = self.stats.current_batch_sizes.get(operation, default_size)
        
        try:
            # GPU 메모리 상태 확인
            memory_info = get_gpu_memory_info(self.gpu_id)
            used_ratio = memory_info.get('used_percent', 50) / 100
            
            # 메모리 사용률에 따른 배치 크기 조정
            if used_ratio < 0.4:  # 40% 미만
                # 메모리 여유 있음 - 배치 크기 증가
                new_size = min(current_size + 1, max_size)
            
            elif used_ratio > 0.8:  # 80% 초과
                # 메모리 부족 - 배치 크기 감소
                new_size = max(current_size - 1, min_size)
            
            else:
                # 적절한 수준 - 현재 크기 유지
                new_size = current_size
            
            # OOM 이벤트가 최근에 발생했다면 보수적으로
            if self.stats.oom_events > 0:
                new_size = max(min_size, new_size - 1)
            
            # 업데이트
            self.stats.current_batch_sizes[operation] = new_size
            
            if new_size != current_size:
                self.logger.debug(
                    f"{operation} 배치 크기 조정: {current_size} → {new_size} "
                    f"(메모리 사용률: {used_ratio:.1%})"
                )
            
            return new_size
            
        except Exception as e:
            self.logger.error(f"배치 크기 최적화 실패: {e}")
            return current_size
    
    def _start_background_cleanup(self) -> None:
        """백그라운드 메모리 정리를 시작합니다."""
        self._cleanup_active = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="MemoryPoolCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
        self.logger.debug("백그라운드 메모리 정리 시작")
    
    def _cleanup_loop(self) -> None:
        """백그라운드 메모리 정리 루프"""
        while self._cleanup_active and not self._shutdown_event.is_set():
            try:
                # 메모리 상태 체크
                memory_info = get_gpu_memory_info(self.gpu_id)
                used_ratio = memory_info.get('used_percent', 0) / 100
                
                # 정리 필요 여부 판단
                if used_ratio > self.config.cleanup_threshold:
                    self.logger.debug(f"백그라운드 메모리 정리 시작 (사용률: {used_ratio:.1%})")
                    self._routine_cleanup()
                
                # 단편화 체크
                fragmentation = self._calculate_fragmentation()
                if fragmentation > self.config.fragmentation_threshold:
                    self.logger.debug(f"메모리 압축 시작 (단편화: {fragmentation:.1%})")
                    self._compact_memory()
                
                # 통계 업데이트
                self._update_stats()
                
                # 대기
                self._shutdown_event.wait(self.config.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"백그라운드 정리 중 오류: {e}")
                self._shutdown_event.wait(30.0)  # 오류 시 30초 대기
    
    def _routine_cleanup(self) -> None:
        """일상적인 메모리 정리"""
        with self._lock:
            cleaned_bytes = 0
            cleaned_blocks = 0
            
            current_time = time.time()
            
            for pool_type, pool in self.memory_pools.items():
                # 오래된 미사용 블록들 정리
                blocks_to_remove = []
                
                for block in pool:
                    if (not block.is_allocated and 
                        current_time - block.last_used_time > 300):  # 5분 이상 미사용
                        
                        blocks_to_remove.append(block)
                        cleaned_bytes += block.size_bytes
                        cleaned_blocks += 1
                
                # 블록들 제거
                for block in blocks_to_remove:
                    try:
                        pool.remove(block)
                        # 텐서 정리
                        if block.tensor_data is not None:
                            del block.tensor_data
                        if block.gpu_mat_data is not None:
                            del block.gpu_mat_data
                    except Exception as e:
                        self.logger.warning(f"블록 정리 실패: {e}")
            
            if cleaned_blocks > 0:
                self.stats.cleanup_events += 1
                self.stats.total_allocated_bytes -= cleaned_bytes
                
                self.logger.debug(
                    f"정기 정리 완료: {cleaned_blocks}개 블록, {cleaned_bytes/1024**2:.1f}MB"
                )
            
            # GPU 캐시 정리
            clear_gpu_cache()
    
    def _emergency_cleanup(self) -> None:
        """긴급 메모리 정리 (OOM 시)"""
        with self._lock:
            self.logger.warning("긴급 메모리 정리 시작")
            
            freed_bytes = 0
            freed_blocks = 0
            
            # 모든 풀에서 미사용 블록들 정리
            for pool_type, pool in self.memory_pools.items():
                blocks_to_remove = []
                
                for block in pool:
                    if not block.is_allocated:
                        blocks_to_remove.append(block)
                        freed_bytes += block.size_bytes
                        freed_blocks += 1
                
                # 블록들 즉시 제거
                for block in blocks_to_remove:
                    try:
                        pool.remove(block)
                        if block.tensor_data is not None:
                            del block.tensor_data
                        if block.gpu_mat_data is not None:
                            del block.gpu_mat_data
                    except Exception as e:
                        self.logger.warning(f"긴급 정리 중 블록 제거 실패: {e}")
            
            # GPU 캐시 강제 정리
            clear_gpu_cache()
            
            # Python GC 실행
            gc.collect()
            
            self.stats.total_allocated_bytes -= freed_bytes
            
            self.logger.warning(
                f"긴급 정리 완료: {freed_blocks}개 블록, {freed_bytes/1024**2:.1f}MB 해제"
            )
    
    def _calculate_fragmentation(self) -> float:
        """메모리 단편화 비율을 계산합니다."""
        with self._lock:
            total_allocated = 0
            total_used = 0
            
            for pool in self.memory_pools.values():
                for block in pool:
                    total_allocated += block.size_bytes
                    if block.is_allocated:
                        total_used += block.size_bytes
            
            if total_allocated == 0:
                return 0.0
            
            # 단편화 = (할당됨 - 사용됨) / 할당됨
            fragmentation = (total_allocated - total_used) / total_allocated
            self.stats.fragmentation_ratio = fragmentation
            
            return fragmentation
    
    def _compact_memory(self) -> None:
        """메모리 압축 (단편화 해소)"""
        with self._lock:
            self.logger.debug("메모리 압축 시작")
            
            compacted_pools = 0
            
            for pool_type, pool in self.memory_pools.items():
                # 미사용 블록들을 크기별로 정렬
                unused_blocks = [block for block in pool if not block.is_allocated]
                
                if len(unused_blocks) > 2:  # 압축할 블록이 충분한 경우
                    # 작은 블록들 제거하고 큰 블록으로 통합
                    small_blocks = [b for b in unused_blocks if b.size_bytes < 50*1024*1024]  # 50MB 미만
                    
                    if len(small_blocks) > 1:
                        total_size = sum(b.size_bytes for b in small_blocks)
                        
                        # 작은 블록들 제거
                        for block in small_blocks:
                            try:
                                pool.remove(block)
                                if block.tensor_data is not None:
                                    del block.tensor_data
                                if block.gpu_mat_data is not None:
                                    del block.gpu_mat_data
                            except:
                                pass
                        
                        # 통합된 큰 블록 생성 시도
                        try:
                            new_block = self._create_new_block(pool_type, total_size)
                            if new_block:
                                new_block.is_allocated = False  # 즉시 사용 가능하게
                                compacted_pools += 1
                        except:
                            self.logger.warning(f"{pool_type.value} 풀 압축 실패")
            
            if compacted_pools > 0:
                self.stats.compaction_events += 1
                self.logger.debug(f"메모리 압축 완료: {compacted_pools}개 풀")
    
    def _update_stats(self) -> None:
        """통계 정보 업데이트"""
        with self._lock:
            self.stats.total_used_bytes = sum(
                block.size_bytes for block in self.allocated_blocks.values()
            )
            
            # 단편화 재계산
            self._calculate_fragmentation()
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        현재 메모리 상태를 반환합니다.
        
        Returns:
            메모리 상태 정보
        """
        self._update_stats()
        
        with self._lock:
            # 풀별 상태
            pool_status = {}
            for pool_type, pool in self.memory_pools.items():
                allocated_blocks = [b for b in pool if b.is_allocated]
                free_blocks = [b for b in pool if not b.is_allocated]
                
                pool_status[pool_type.value] = {
                    'total_blocks': len(pool),
                    'allocated_blocks': len(allocated_blocks),
                    'free_blocks': len(free_blocks),
                    'total_size_mb': sum(b.size_bytes for b in pool) / 1024**2,
                    'used_size_mb': sum(b.size_bytes for b in allocated_blocks) / 1024**2,
                    'free_size_mb': sum(b.size_bytes for b in free_blocks) / 1024**2
                }
            
            return {
                'stats': {
                    'total_allocated_mb': self.stats.total_allocated_bytes / 1024**2,
                    'total_used_mb': self.stats.total_used_bytes / 1024**2,
                    'fragmentation_ratio': self.stats.fragmentation_ratio,
                    'allocation_count': self.stats.allocation_count,
                    'deallocation_count': self.stats.deallocation_count,
                    'cache_hit_rate': self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1),
                    'oom_events': self.stats.oom_events,
                    'cleanup_events': self.stats.cleanup_events,
                    'current_batch_sizes': self.stats.current_batch_sizes.copy()
                },
                'pools': pool_status,
                'gpu_memory': get_gpu_memory_info(self.gpu_id)
            }
    
    async def shutdown(self) -> None:
        """
        MemoryPoolManager 종료.
        
        모든 메모리를 정리하고 백그라운드 스레드를 종료합니다.
        """
        try:
            self.logger.stage("MemoryPoolManager 종료 시작...")
            
            # 백그라운드 정리 중지
            self._shutdown_event.set()
            self._cleanup_active = False
            
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5.0)
            
            # 모든 메모리 정리
            with self._lock:
                total_freed_mb = 0
                
                for pool_type, pool in self.memory_pools.items():
                    pool_freed_mb = 0
                    
                    for block in pool[:]:  # 복사본으로 반복
                        try:
                            pool_freed_mb += block.size_bytes / 1024**2
                            
                            if block.tensor_data is not None:
                                del block.tensor_data
                            if block.gpu_mat_data is not None:
                                del block.gpu_mat_data
                        except:
                            pass
                    
                    pool.clear()
                    total_freed_mb += pool_freed_mb
                    
                    self.logger.debug(f"{pool_type.value} 풀 정리: {pool_freed_mb:.1f}MB")
                
                # 할당 추적 정리
                self.allocated_blocks.clear()
                
                # 최종 GPU 캐시 정리
                clear_gpu_cache()
                
                self.initialized = False
                
                self.logger.success(f"MemoryPoolManager 종료 완료 (총 {total_freed_mb:.1f}MB 정리)")
        
        except Exception as e:
            self.logger.error(f"MemoryPoolManager 종료 중 오류: {e}")
    
    def __del__(self):
        """소멸자 - 기본 리소스 정리"""
        if hasattr(self, 'initialized') and self.initialized:
            try:
                self._shutdown_event.set()
                self._cleanup_active = False
                clear_gpu_cache()
            except:
                pass


# 편의 함수들
async def create_memory_pool_manager(
    max_vram_usage: float = 0.75,
    strategy: MemoryAllocationStrategy = MemoryAllocationStrategy.ADAPTIVE,
    gpu_id: int = 0
) -> MemoryPoolManager:
    """
    초기화된 MemoryPoolManager를 생성합니다.
    
    Args:
        max_vram_usage: 최대 VRAM 사용률
        strategy: 메모리 할당 전략
        gpu_id: GPU 디바이스 ID
        
    Returns:
        초기화된 MemoryPoolManager 인스턴스
    """
    config = MemoryPoolConfig(
        max_vram_usage=max_vram_usage,
        allocation_strategy=strategy
    )
    
    manager = MemoryPoolManager(config, gpu_id)
    await manager.initialize()
    return manager