"""
GPU 하드웨어 모니터링 시스템 - CSV 파일 기반 심플 로깅

실시간 GPU 상태를 추적하고 CSV 파일로 저장하는 경량 모니터링 시스템
"""

import csv
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import logger
from ..utils.exceptions import MonitoringError


class GPUMetrics:
    """GPU 메트릭 데이터 클래스"""
    
    def __init__(self):
        self.gpu_utilization = 0
        self.memory_used = 0
        self.memory_total = 0
        self.memory_percent = 0
        self.temperature = 0
        self.power_draw = 0
        self.nvenc_utilization = 0
        self.nvdec_utilization = 0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gpu_util': self.gpu_utilization,
            'memory_used_mb': self.memory_used,
            'memory_total_mb': self.memory_total,
            'memory_percent': self.memory_percent,
            'temperature': self.temperature,
            'power_draw': self.power_draw,
            'nvenc_util': self.nvenc_utilization,
            'nvdec_util': self.nvdec_utilization
        }


class SystemMetrics:
    """시스템 메트릭 데이터 클래스"""
    
    def __init__(self):
        self.cpu_percent = 0
        self.memory_used = 0
        self.memory_total = 0
        self.memory_percent = 0
        self.disk_io_read = 0
        self.disk_io_write = 0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'sys_memory_used_mb': self.memory_used,
            'sys_memory_total_mb': self.memory_total,
            'sys_memory_percent': self.memory_percent,
            'disk_read_mb': self.disk_io_read,
            'disk_write_mb': self.disk_io_write
        }


class HardwareMonitor:
    """
    하드웨어 모니터링 시스템
    
    기능:
    - GPU 사용률, 메모리, 온도 실시간 추적
    - 시스템 CPU, 메모리, 디스크 I/O 모니터링
    - CSV 파일로 히스토리 저장
    - 콘솔 실시간 출력
    - 성능 트렌드 분석
    """
    
    def __init__(self, 
                 log_dir: str = "monitoring_logs",
                 interval: float = 5.0,
                 max_history: int = 100):
        """
        Args:
            log_dir: 로그 디렉토리 경로
            interval: 모니터링 간격 (초)
            max_history: 메모리에 보관할 최대 메트릭 수
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.interval = interval
        self.max_history = max_history
        
        # 로그 파일 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"hardware_metrics_{timestamp}.csv"
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 메트릭 히스토리
        self.gpu_history = deque(maxlen=max_history)
        self.system_history = deque(maxlen=max_history)
        
        # 통계
        self.stats = {
            'total_samples': 0,
            'monitoring_start_time': None,
            'gpu_peak_util': 0,
            'gpu_peak_memory': 0,
            'cpu_peak_util': 0
        }
        
        # GPU 초기화
        self._init_gpu_monitoring()
        self._init_csv_file()
        
        logger.info(f"🔧 HardwareMonitor 초기화 완료")
        logger.info(f"📁 로그 파일: {self.metrics_file}")
        logger.info(f"⏱️ 모니터링 간격: {interval}초")
    
    def _init_gpu_monitoring(self):
        """GPU 모니터링 초기화"""
        self.gpu_available = False
        self.gpu_handle = None
        
        if not PYNVML_AVAILABLE:
            logger.warning("⚠️ pynvml 라이브러리 없음 - GPU 모니터링 비활성화")
            return
            
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            
            # GPU 정보 출력
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            total_memory_gb = memory_info.total / 1024**3
            
            logger.info(f"🖥️ GPU 감지: {gpu_name}")
            logger.info(f"💾 GPU 메모리: {total_memory_gb:.1f}GB")
            
        except Exception as e:
            logger.warning(f"⚠️ GPU 모니터링 초기화 실패: {e}")
    
    def _init_csv_file(self):
        """CSV 파일 헤더 초기화"""
        headers = [
            'timestamp',
            'elapsed_seconds',
            # GPU 메트릭
            'gpu_util', 'memory_used_mb', 'memory_total_mb', 'memory_percent',
            'temperature', 'power_draw', 'nvenc_util', 'nvdec_util',
            # 시스템 메트릭
            'cpu_percent', 'sys_memory_used_mb', 'sys_memory_total_mb', 
            'sys_memory_percent', 'disk_read_mb', 'disk_write_mb'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def get_gpu_metrics(self) -> GPUMetrics:
        """현재 GPU 메트릭 수집"""
        metrics = GPUMetrics()
        
        if not self.gpu_available:
            return metrics
            
        try:
            # GPU 사용률
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            metrics.gpu_utilization = util.gpu
            
            # 메모리 정보
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            metrics.memory_used = memory_info.used // 1024**2  # MB
            metrics.memory_total = memory_info.total // 1024**2  # MB
            metrics.memory_percent = (memory_info.used / memory_info.total) * 100
            
            # 온도
            try:
                metrics.temperature = pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            # 전력
            try:
                metrics.power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000
            except:
                pass
            
            # NVENC/NVDEC 사용률 (RTX 5090 지원시)
            try:
                encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(self.gpu_handle)
                metrics.nvenc_utilization = encoder_util[0]
                
                decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(self.gpu_handle)
                metrics.nvdec_utilization = decoder_util[0]
            except:
                pass
                
        except Exception as e:
            logger.warning(f"⚠️ GPU 메트릭 수집 실패: {e}")
        
        return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """현재 시스템 메트릭 수집"""
        metrics = SystemMetrics()
        
        try:
            # CPU 사용률
            metrics.cpu_percent = psutil.cpu_percent(interval=None)
            
            # 메모리
            memory = psutil.virtual_memory()
            metrics.memory_used = memory.used // 1024**2  # MB
            metrics.memory_total = memory.total // 1024**2  # MB
            metrics.memory_percent = memory.percent
            
            # 디스크 I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.disk_io_read = disk_io.read_bytes // 1024**2  # MB
                metrics.disk_io_write = disk_io.write_bytes // 1024**2  # MB
                
        except Exception as e:
            logger.warning(f"⚠️ 시스템 메트릭 수집 실패: {e}")
        
        return metrics
    
    def collect_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 수집 및 저장"""
        timestamp = datetime.now()
        elapsed = 0
        if self.stats['monitoring_start_time']:
            elapsed = (timestamp - self.stats['monitoring_start_time']).total_seconds()
        
        gpu_metrics = self.get_gpu_metrics()
        system_metrics = self.get_system_metrics()
        
        # 히스토리에 추가
        self.gpu_history.append(gpu_metrics)
        self.system_history.append(system_metrics)
        
        # 통계 업데이트
        self.stats['total_samples'] += 1
        self.stats['gpu_peak_util'] = max(self.stats['gpu_peak_util'], gpu_metrics.gpu_utilization)
        self.stats['gpu_peak_memory'] = max(self.stats['gpu_peak_memory'], gpu_metrics.memory_percent)
        self.stats['cpu_peak_util'] = max(self.stats['cpu_peak_util'], system_metrics.cpu_percent)
        
        # CSV 데이터 준비
        csv_data = {
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': elapsed,
            **gpu_metrics.to_dict(),
            **system_metrics.to_dict()
        }
        
        return csv_data
    
    def write_metrics_to_csv(self, metrics: Dict[str, Any]):
        """메트릭을 CSV 파일에 저장"""
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    metrics['timestamp'], metrics['elapsed_seconds'],
                    metrics['gpu_util'], metrics['memory_used_mb'], 
                    metrics['memory_total_mb'], metrics['memory_percent'],
                    metrics['temperature'], metrics['power_draw'],
                    metrics['nvenc_util'], metrics['nvdec_util'],
                    metrics['cpu_percent'], metrics['sys_memory_used_mb'],
                    metrics['sys_memory_total_mb'], metrics['sys_memory_percent'],
                    metrics['disk_read_mb'], metrics['disk_write_mb']
                ]
                writer.writerow(row)
        except Exception as e:
            logger.error(f"❌ CSV 쓰기 실패: {e}")
    
    def print_console_summary(self, metrics: Dict[str, Any]):
        """콘솔에 요약 정보 출력"""
        if self.stats['total_samples'] % 3 == 0:  # 15초마다 출력
            elapsed_min = metrics['elapsed_seconds'] / 60
            
            print(f"\n📊 [모니터링 {elapsed_min:.1f}분] "
                  f"GPU: {metrics['gpu_util']}% | "
                  f"VRAM: {metrics['memory_percent']:.1f}% "
                  f"({metrics['memory_used_mb']/1024:.1f}GB) | "
                  f"CPU: {metrics['cpu_percent']:.1f}% | "
                  f"온도: {metrics['temperature']}°C")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("⚠️ 이미 모니터링 중입니다")
            return
        
        self.is_monitoring = True
        self.stats['monitoring_start_time'] = datetime.now()
        
        def monitor_loop():
            logger.info("🚀 하드웨어 모니터링 시작")
            
            while self.is_monitoring:
                try:
                    # 메트릭 수집
                    metrics = self.collect_metrics()
                    
                    # CSV 저장
                    self.write_metrics_to_csv(metrics)
                    
                    # 콘솔 출력
                    self.print_console_summary(metrics)
                    
                    # 대기
                    time.sleep(self.interval)
                    
                except Exception as e:
                    logger.error(f"❌ 모니터링 루프 에러: {e}")
                    time.sleep(self.interval)
            
            logger.info("🔚 하드웨어 모니터링 종료")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.interval + 1)
        
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """모니터링 요약 리포트 생성"""
        if not self.gpu_history or not self.system_history:
            return
        
        # 평균값 계산
        avg_gpu_util = sum(m.gpu_utilization for m in self.gpu_history) / len(self.gpu_history)
        avg_memory_percent = sum(m.memory_percent for m in self.gpu_history) / len(self.gpu_history)
        avg_cpu_util = sum(m.cpu_percent for m in self.system_history) / len(self.system_history)
        
        total_time = (datetime.now() - self.stats['monitoring_start_time']).total_seconds()
        
        report = f"""
================================================================================
📊 하드웨어 모니터링 요약 리포트
================================================================================
⏱️ 모니터링 시간: {total_time/60:.1f}분 ({self.stats['total_samples']}개 샘플)
📁 로그 파일: {self.metrics_file}

🖥️ GPU 성능:
   • 평균 GPU 사용률: {avg_gpu_util:.1f}%
   • 최대 GPU 사용률: {self.stats['gpu_peak_util']}%
   • 평균 VRAM 사용률: {avg_memory_percent:.1f}%
   • 최대 VRAM 사용률: {self.stats['gpu_peak_memory']:.1f}%

💻 시스템 성능:
   • 평균 CPU 사용률: {avg_cpu_util:.1f}%
   • 최대 CPU 사용률: {self.stats['cpu_peak_util']:.1f}%

📈 성능 분석:
   • GPU 활용도: {'🟢 양호' if avg_gpu_util > 70 else '🟡 보통' if avg_gpu_util > 30 else '🔴 낮음'}
   • 메모리 효율성: {'🟢 양호' if avg_memory_percent < 80 else '🟡 보통' if avg_memory_percent < 95 else '🔴 높음'}
================================================================================
        """
        
        print(report)
        
        # 리포트 파일로도 저장
        report_file = self.log_dir / f"monitoring_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📋 모니터링 리포트 저장: {report_file}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        if not self.gpu_history or not self.system_history:
            current_metrics = self.collect_metrics()
            return current_metrics
        
        latest_gpu = self.gpu_history[-1]
        latest_system = self.system_history[-1]
        
        return {
            'gpu_util': latest_gpu.gpu_utilization,
            'memory_percent': latest_gpu.memory_percent,
            'memory_used_gb': latest_gpu.memory_used / 1024,
            'temperature': latest_gpu.temperature,
            'cpu_percent': latest_system.cpu_percent,
            'is_monitoring': self.is_monitoring,
            'total_samples': self.stats['total_samples']
        }
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_monitoring()


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 HardwareMonitor 테스트 시작...")
    
    with HardwareMonitor(interval=2.0) as monitor:
        print("⏰ 10초간 모니터링...")
        time.sleep(10)
    
    print("✅ 테스트 완료!")