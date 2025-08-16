"""
GPU 메모리 관리 시스템

VRAM 사용량을 모니터링하고 자동으로 메모리 정리 및 최적화를 수행하는 시스템
"""

import gc
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..utils.logger import logger
from ..utils.exceptions import GPUMemoryError, MemoryManagementError


@dataclass
class MemorySnapshot:
    """메모리 스냅샷"""
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_free_mb: float
    system_memory_mb: float
    system_memory_percent: float
    
    @property
    def gpu_utilization_percent(self) -> float:
        total = self.gpu_allocated_mb + self.gpu_free_mb
        if total > 0:
            return (self.gpu_allocated_mb / total) * 100
        return 0


class MemoryPool:
    """GPU 메모리 풀 관리"""
    
    def __init__(self, initial_size_mb: int = 1024):
        self.initial_size_mb = initial_size_mb
        self.allocated_tensors: List[torch.Tensor] = []
        self.free_tensors: List[torch.Tensor] = []
        self._lock = threading.Lock()
        
    def preallocate(self, sizes: List[Tuple[int, ...]], dtype=torch.float32):
        """메모리 사전 할당"""
        if not TORCH_AVAILABLE:
            return
            
        with self._lock:
            for size in sizes:
                try:
                    tensor = torch.empty(size, dtype=dtype, device='cuda')
                    self.free_tensors.append(tensor)
                    logger.debug(f"💾 메모리 풀 할당: {size} ({tensor.numel() * tensor.element_size() / 1024**2:.1f}MB)")
                except Exception as e:
                    logger.warning(f"⚠️ 메모리 풀 할당 실패: {e}")
    
    def get_tensor(self, size: Tuple[int, ...], dtype=torch.float32) -> Optional[torch.Tensor]:
        """풀에서 텐서 가져오기"""
        if not TORCH_AVAILABLE:
            return None
            
        with self._lock:
            # 적합한 크기의 텐서 찾기
            for i, tensor in enumerate(self.free_tensors):
                if tensor.shape == size and tensor.dtype == dtype:
                    self.free_tensors.pop(i)
                    self.allocated_tensors.append(tensor)
                    return tensor
            
            # 없으면 새로 생성
            try:
                tensor = torch.empty(size, dtype=dtype, device='cuda')
                self.allocated_tensors.append(tensor)
                return tensor
            except Exception as e:
                logger.warning(f"⚠️ 텐서 생성 실패: {e}")
                return None
    
    def return_tensor(self, tensor: torch.Tensor):
        """풀에 텐서 반환"""
        if not TORCH_AVAILABLE:
            return
            
        with self._lock:
            if tensor in self.allocated_tensors:
                self.allocated_tensors.remove(tensor)
                self.free_tensors.append(tensor)
    
    def clear_pool(self):
        """풀 완전 정리"""
        with self._lock:
            self.allocated_tensors.clear()
            self.free_tensors.clear()


class MemoryManager:
    """
    GPU 메모리 관리자
    
    기능:
    - VRAM 사용량 실시간 모니터링
    - 임계값 기반 자동 메모리 정리
    - 메모리 풀 관리
    - 배치 크기 동적 조정
    - 메모리 누수 감지
    - 시스템 메모리 관리
    """
    
    def __init__(self, 
                 threshold_percent: float = 80.0,
                 critical_threshold_percent: float = 95.0,
                 monitoring_interval: float = 10.0):
        """
        Args:
            threshold_percent: 정리 시작 임계값 (%)
            critical_threshold_percent: 긴급 정리 임계값 (%)
            monitoring_interval: 모니터링 간격 (초)
        """
        self.threshold_percent = threshold_percent
        self.critical_threshold_percent = critical_threshold_percent
        self.monitoring_interval = monitoring_interval
        
        # GPU 정보
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.total_gpu_memory_mb = 0
        
        if self.gpu_available:
            # GPU 메모리 총량 확인
            self.total_gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            logger.info(f"🖥️ 총 GPU 메모리: {self.total_gpu_memory_mb/1024:.1f}GB")
        
        # 메모리 히스토리 (최근 100개)
        self.memory_history = deque(maxlen=100)
        
        # 메모리 풀
        self.memory_pool = MemoryPool()
        
        # 정리 콜백 함수들
        self.cleanup_callbacks: List[Callable[[], None]] = []
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 통계
        self.stats = {
            'total_cleanups': 0,
            'emergency_cleanups': 0,
            'bytes_freed': 0,
            'peak_memory_mb': 0,
            'avg_memory_mb': 0
        }
        
        logger.info(f"💾 MemoryManager 초기화 완료")
        logger.info(f"📊 임계값: {threshold_percent}% / 긴급: {critical_threshold_percent}%")
    
    def get_memory_info(self) -> MemorySnapshot:
        """현재 메모리 정보 수집"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated_mb=0,
            gpu_reserved_mb=0,
            gpu_free_mb=0,
            system_memory_mb=0,
            system_memory_percent=0
        )
        
        if self.gpu_available:
            try:
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                free = self.total_gpu_memory_mb - reserved
                
                snapshot.gpu_allocated_mb = allocated
                snapshot.gpu_reserved_mb = reserved
                snapshot.gpu_free_mb = free
                
                # 통계 업데이트
                self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], allocated)
                
            except Exception as e:
                logger.warning(f"⚠️ GPU 메모리 정보 수집 실패: {e}")
        
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                snapshot.system_memory_mb = memory.used / 1024**2
                snapshot.system_memory_percent = memory.percent
            except Exception as e:
                logger.warning(f"⚠️ 시스템 메모리 정보 수집 실패: {e}")
        
        return snapshot
    
    def check_and_cleanup(self, force: bool = False) -> bool:
        """메모리 상태 체크 및 필요시 정리"""
        snapshot = self.get_memory_info()
        self.memory_history.append(snapshot)
        
        if not self.gpu_available:
            return True
        
        current_percent = snapshot.gpu_utilization_percent
        
        # 정리 필요성 판단
        needs_cleanup = (
            force or 
            current_percent > self.threshold_percent or
            current_percent > self.critical_threshold_percent
        )
        
        if needs_cleanup:
            is_emergency = current_percent > self.critical_threshold_percent
            return self._perform_cleanup(snapshot, is_emergency)
        
        return True
    
    def _perform_cleanup(self, snapshot: MemorySnapshot, is_emergency: bool = False) -> bool:
        """메모리 정리 수행"""
        cleanup_type = "긴급" if is_emergency else "일반"
        logger.info(f"🧹 {cleanup_type} 메모리 정리 시작 "
                   f"({snapshot.gpu_utilization_percent:.1f}% 사용)")
        
        initial_allocated = snapshot.gpu_allocated_mb
        cleanup_start = time.time()
        
        try:
            # 1. PyTorch CUDA 캐시 정리
            if self.gpu_available:
                torch.cuda.empty_cache()
                logger.debug("🗑️ CUDA 캐시 정리 완료")
            
            # 2. Python 가비지 컬렉션
            gc.collect()
            logger.debug("🗑️ Python GC 완료")
            
            # 3. 메모리 풀 정리 (긴급시에만)
            if is_emergency:
                self.memory_pool.clear_pool()
                logger.debug("🗑️ 메모리 풀 정리 완료")
            
            # 4. 등록된 정리 콜백 실행
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"⚠️ 정리 콜백 실패: {e}")
            
            # 정리 후 상태 확인
            if self.gpu_available:
                final_allocated = torch.cuda.memory_allocated() / 1024**2
                freed_mb = initial_allocated - final_allocated
                
                logger.info(f"✅ 메모리 정리 완료: {freed_mb:.1f}MB 해제 "
                           f"({initial_allocated:.1f}MB → {final_allocated:.1f}MB)")
                
                # 통계 업데이트
                self.stats['total_cleanups'] += 1
                if is_emergency:
                    self.stats['emergency_cleanups'] += 1
                self.stats['bytes_freed'] += freed_mb * 1024**2
                
                cleanup_time = time.time() - cleanup_start
                logger.debug(f"⏱️ 정리 소요 시간: {cleanup_time:.2f}초")
                
                return True
            
        except Exception as e:
            logger.error(f"❌ 메모리 정리 실패: {e}")
            return False
        
        return True
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """메모리 정리시 실행할 콜백 등록"""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"📝 정리 콜백 등록 (총 {len(self.cleanup_callbacks)}개)")
    
    def unregister_cleanup_callback(self, callback: Callable[[], None]):
        """콜백 등록 해제"""
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)
    
    def get_recommended_batch_size(self, 
                                 base_batch_size: int,
                                 memory_per_item_mb: float) -> int:
        """현재 메모리 상태 기반 권장 배치 크기"""
        if not self.gpu_available:
            return base_batch_size
        
        snapshot = self.get_memory_info()
        available_mb = snapshot.gpu_free_mb * 0.8  # 안전 마진
        
        if memory_per_item_mb > 0:
            max_items = int(available_mb / memory_per_item_mb)
            recommended = min(base_batch_size, max_items)
            
            if recommended < base_batch_size:
                logger.info(f"📉 배치 크기 조정 권장: {base_batch_size} → {recommended} "
                           f"(가용 메모리: {available_mb:.1f}MB)")
            
            return max(1, recommended)
        
        return base_batch_size
    
    def start_monitoring(self):
        """백그라운드 메모리 모니터링 시작"""
        if self.is_monitoring:
            logger.warning("⚠️ 이미 모니터링 중입니다")
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            logger.info("🔍 메모리 모니터링 시작")
            
            while self.is_monitoring:
                try:
                    self.check_and_cleanup()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"❌ 모니터링 루프 에러: {e}")
                    time.sleep(self.monitoring_interval)
            
            logger.info("🔚 메모리 모니터링 종료")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.monitoring_interval + 1)
    
    def detect_memory_leak(self, window_minutes: int = 30) -> bool:
        """메모리 누수 감지"""
        if len(self.memory_history) < 10:
            return False
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_snapshots = [s for s in self.memory_history if s.timestamp > cutoff_time]
        
        if len(recent_snapshots) < 5:
            return False
        
        # 메모리 사용량 트렌드 분석
        memory_values = [s.gpu_allocated_mb for s in recent_snapshots]
        
        # 단순 선형 증가 감지
        increase_count = 0
        for i in range(1, len(memory_values)):
            if memory_values[i] > memory_values[i-1]:
                increase_count += 1
        
        # 70% 이상이 증가 트렌드면 누수 가능성
        leak_threshold = 0.7
        is_leak = (increase_count / (len(memory_values) - 1)) > leak_threshold
        
        if is_leak:
            start_memory = memory_values[0]
            end_memory = memory_values[-1]
            logger.warning(f"⚠️ 메모리 누수 가능성 감지: "
                          f"{start_memory:.1f}MB → {end_memory:.1f}MB "
                          f"({window_minutes}분간)")
        
        return is_leak
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 관리 통계 반환"""
        if not self.memory_history:
            current_snapshot = self.get_memory_info()
        else:
            current_snapshot = self.memory_history[-1]
        
        # 평균 메모리 사용량 계산
        if self.memory_history:
            avg_memory = sum(s.gpu_allocated_mb for s in self.memory_history) / len(self.memory_history)
            self.stats['avg_memory_mb'] = avg_memory
        
        return {
            **self.stats,
            'current_allocated_mb': current_snapshot.gpu_allocated_mb,
            'current_utilization_percent': current_snapshot.gpu_utilization_percent,
            'total_gpu_memory_mb': self.total_gpu_memory_mb,
            'memory_pool_size': len(self.memory_pool.allocated_tensors) + len(self.memory_pool.free_tensors),
            'cleanup_callbacks': len(self.cleanup_callbacks),
            'monitoring_active': self.is_monitoring
        }
    
    def print_memory_summary(self):
        """메모리 상태 요약 출력"""
        stats = self.get_memory_stats()
        
        print(f"""
💾 메모리 관리 요약:
   • 현재 사용량: {stats['current_allocated_mb']:.1f}MB ({stats['current_utilization_percent']:.1f}%)
   • 총 GPU 메모리: {stats['total_gpu_memory_mb']/1024:.1f}GB
   • 평균 사용량: {stats['avg_memory_mb']:.1f}MB
   • 최대 사용량: {stats['peak_memory_mb']:.1f}MB
   • 총 정리 횟수: {stats['total_cleanups']}회
   • 긴급 정리: {stats['emergency_cleanups']}회
   • 해제된 메모리: {stats['bytes_freed']/1024**3:.2f}GB
   • 모니터링: {'활성' if stats['monitoring_active'] else '비활성'}
        """)
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_monitoring()
        # 최종 정리
        self.check_and_cleanup(force=True)


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 MemoryManager 테스트 시작...")
    
    with MemoryManager(threshold_percent=50.0, monitoring_interval=2.0) as manager:
        print("💾 메모리 관리자 실행 중...")
        
        # 가짜 메모리 사용 시뮬레이션
        if TORCH_AVAILABLE and torch.cuda.is_available():
            test_tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000, device='cuda')
                test_tensors.append(tensor)
                print(f"🔄 텐서 {i+1} 생성")
                time.sleep(1)
            
            # 정리 테스트
            manager.check_and_cleanup(force=True)
            
            # 텐서 해제
            del test_tensors
        
        time.sleep(5)
        manager.print_memory_summary()
    
    print("✅ 테스트 완료!")