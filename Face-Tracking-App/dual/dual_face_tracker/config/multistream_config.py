"""
멀티스트림 처리 최적화 설정

NVENC 세션 제한 문제를 해결한 최적화된 멀티스트림 설정입니다:
- RTX 5090 GPU 특성에 맞춘 세션 제한
- 배치 처리 최적화
- 자동 폴백 설정
- 성능 튜닝 파라미터

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0 (Optimized)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import os


class GPUModel(Enum):
    """지원하는 GPU 모델"""
    RTX_5090 = "rtx_5090"
    RTX_4090 = "rtx_4090"
    RTX_3090 = "rtx_3090"
    RTX_3080 = "rtx_3080"
    UNKNOWN = "unknown"


class ProcessingMode(Enum):
    """처리 모드"""
    BATCH_SEQUENTIAL = "batch_sequential"  # 배치 순차 처리 (권장)
    BATCH_PARALLEL = "batch_parallel"      # 배치 병렬 처리
    SINGLE_QUEUE = "single_queue"          # 단일 큐 처리
    PRIORITY_QUEUE = "priority_queue"      # 우선순위 큐 처리


@dataclass
class GPUOptimizationConfig:
    """GPU별 최적화 설정"""
    max_concurrent_nvenc_sessions: int = 2
    batch_size: int = 2
    memory_buffer_size_mb: int = 1024
    session_timeout: float = 30.0
    cleanup_interval: float = 5.0
    
    # 성능 튜닝
    enable_cuda_streams: bool = True
    cuda_stream_count: int = 2
    memory_pool_size_mb: int = 2048
    
    # 품질 설정
    default_encoding_preset: str = "streaming"
    fallback_quality: str = "balanced"
    
    # 에러 처리
    max_retry_count: int = 3
    retry_delay: float = 1.0
    enable_error_recovery: bool = True


# GPU별 최적화 설정
GPU_CONFIGS = {
    GPUModel.RTX_5090: GPUOptimizationConfig(
        max_concurrent_nvenc_sessions=2,  # RTX 5090 NVENC 제한
        batch_size=2,
        memory_buffer_size_mb=2048,       # 32GB VRAM 활용
        session_timeout=30.0,
        cleanup_interval=3.0,             # 빠른 정리
        
        enable_cuda_streams=True,
        cuda_stream_count=4,              # 4개 스트림 활용
        memory_pool_size_mb=4096,         # 큰 메모리 풀
        
        default_encoding_preset="streaming",
        fallback_quality="balanced",
        
        max_retry_count=3,
        retry_delay=0.5,                  # 빠른 재시도
        enable_error_recovery=True
    ),
    
    GPUModel.RTX_4090: GPUOptimizationConfig(
        max_concurrent_nvenc_sessions=2,
        batch_size=2,
        memory_buffer_size_mb=1536,       # 24GB VRAM
        session_timeout=40.0,
        cleanup_interval=5.0,
        
        enable_cuda_streams=True,
        cuda_stream_count=3,
        memory_pool_size_mb=3072,
        
        default_encoding_preset="balanced",
        fallback_quality="balanced",
        
        max_retry_count=2,
        retry_delay=1.0,
        enable_error_recovery=True
    ),
    
    GPUModel.RTX_3090: GPUOptimizationConfig(
        max_concurrent_nvenc_sessions=1,  # 구형 GPU는 더 보수적
        batch_size=1,
        memory_buffer_size_mb=1024,       # 24GB VRAM
        session_timeout=60.0,
        cleanup_interval=10.0,
        
        enable_cuda_streams=True,
        cuda_stream_count=2,
        memory_pool_size_mb=2048,
        
        default_encoding_preset="balanced",
        fallback_quality="fast",
        
        max_retry_count=2,
        retry_delay=2.0,
        enable_error_recovery=True
    ),
    
    GPUModel.UNKNOWN: GPUOptimizationConfig(
        max_concurrent_nvenc_sessions=1,  # 안전한 기본값
        batch_size=1,
        memory_buffer_size_mb=512,
        session_timeout=60.0,
        cleanup_interval=15.0,
        
        enable_cuda_streams=False,        # 보수적 설정
        cuda_stream_count=1,
        memory_pool_size_mb=1024,
        
        default_encoding_preset="fast",
        fallback_quality="fast",
        
        max_retry_count=1,
        retry_delay=3.0,
        enable_error_recovery=False
    )
}


@dataclass
class MultiStreamOptimizedConfig:
    """최적화된 멀티스트림 설정"""
    # 기본 설정
    processing_mode: ProcessingMode = ProcessingMode.BATCH_SEQUENTIAL
    gpu_model: GPUModel = GPUModel.RTX_5090
    
    # GPU 최적화
    gpu_config: GPUOptimizationConfig = field(default_factory=lambda: GPU_CONFIGS[GPUModel.RTX_5090])
    
    # 폴백 설정
    enable_software_fallback: bool = True
    fallback_threshold: float = 0.8  # NVENC 실패율 80% 초과시 폴백
    max_fallback_ratio: float = 0.5  # 전체의 50%까지 폴백 허용
    
    # 모니터링
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 2.0
    enable_gpu_memory_tracking: bool = True
    
    # 최적화 플래그
    enable_dynamic_batch_sizing: bool = True
    enable_load_balancing: bool = True
    enable_priority_scheduling: bool = False
    
    # 디버깅
    enable_detailed_logging: bool = True
    log_session_events: bool = True
    log_performance_metrics: bool = True
    
    def __post_init__(self):
        """설정 후 처리"""
        # GPU 모델에 따른 설정 적용
        if self.gpu_model in GPU_CONFIGS:
            self.gpu_config = GPU_CONFIGS[self.gpu_model]
        else:
            self.gpu_config = GPU_CONFIGS[GPUModel.UNKNOWN]
    
    @classmethod
    def from_gpu_model(cls, gpu_model: GPUModel) -> 'MultiStreamOptimizedConfig':
        """GPU 모델에서 최적화된 설정 생성"""
        return cls(gpu_model=gpu_model)
    
    @classmethod
    def from_environment(cls) -> 'MultiStreamOptimizedConfig':
        """환경에서 자동 감지하여 설정 생성"""
        gpu_model = detect_gpu_model()
        return cls.from_gpu_model(gpu_model)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'processing_mode': self.processing_mode.value,
            'gpu_model': self.gpu_model.value,
            'gpu_config': {
                'max_concurrent_nvenc_sessions': self.gpu_config.max_concurrent_nvenc_sessions,
                'batch_size': self.gpu_config.batch_size,
                'memory_buffer_size_mb': self.gpu_config.memory_buffer_size_mb,
                'session_timeout': self.gpu_config.session_timeout,
                'cleanup_interval': self.gpu_config.cleanup_interval,
                'enable_cuda_streams': self.gpu_config.enable_cuda_streams,
                'cuda_stream_count': self.gpu_config.cuda_stream_count,
                'memory_pool_size_mb': self.gpu_config.memory_pool_size_mb,
                'default_encoding_preset': self.gpu_config.default_encoding_preset,
                'fallback_quality': self.gpu_config.fallback_quality,
                'max_retry_count': self.gpu_config.max_retry_count,
                'retry_delay': self.gpu_config.retry_delay,
                'enable_error_recovery': self.gpu_config.enable_error_recovery
            },
            'enable_software_fallback': self.enable_software_fallback,
            'fallback_threshold': self.fallback_threshold,
            'max_fallback_ratio': self.max_fallback_ratio,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'enable_gpu_memory_tracking': self.enable_gpu_memory_tracking,
            'enable_dynamic_batch_sizing': self.enable_dynamic_batch_sizing,
            'enable_load_balancing': self.enable_load_balancing,
            'enable_priority_scheduling': self.enable_priority_scheduling,
            'enable_detailed_logging': self.enable_detailed_logging,
            'log_session_events': self.log_session_events,
            'log_performance_metrics': self.log_performance_metrics
        }
    
    def get_session_manager_config(self) -> Dict[str, Any]:
        """세션 관리자용 설정 추출"""
        return {
            'max_concurrent_sessions': self.gpu_config.max_concurrent_nvenc_sessions,
            'session_timeout': self.gpu_config.session_timeout,
            'cleanup_interval': self.gpu_config.cleanup_interval,
            'enable_monitoring': self.enable_performance_monitoring
        }
    
    def get_encoder_config(self) -> Dict[str, Any]:
        """인코더용 설정 추출"""
        return {
            'enable_fallback': self.enable_software_fallback,
            'max_nvenc_sessions': self.gpu_config.max_concurrent_nvenc_sessions,
            'session_timeout': self.gpu_config.session_timeout,
            'default_preset': self.gpu_config.default_encoding_preset,
            'fallback_quality': self.gpu_config.fallback_quality,
            'max_retry_count': self.gpu_config.max_retry_count,
            'retry_delay': self.gpu_config.retry_delay
        }
    
    def get_multistream_config(self) -> Dict[str, Any]:
        """멀티스트림 프로세서용 설정 추출"""
        return {
            'processing_mode': self.processing_mode.value,
            'batch_size': self.gpu_config.batch_size,
            'max_streams': self.gpu_config.cuda_stream_count,
            'enable_cuda_streams': self.gpu_config.enable_cuda_streams,
            'memory_pool_size_mb': self.gpu_config.memory_pool_size_mb,
            'enable_dynamic_batch_sizing': self.enable_dynamic_batch_sizing,
            'enable_load_balancing': self.enable_load_balancing,
            'enable_priority_scheduling': self.enable_priority_scheduling
        }


def detect_gpu_model() -> GPUModel:
    """현재 시스템의 GPU 모델 감지"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            
            if 'rtx 5090' in gpu_name:
                return GPUModel.RTX_5090
            elif 'rtx 4090' in gpu_name:
                return GPUModel.RTX_4090
            elif 'rtx 3090' in gpu_name:
                return GPUModel.RTX_3090
            elif 'rtx 3080' in gpu_name:
                return GPUModel.RTX_3080
        
    except Exception:
        pass
    
    return GPUModel.UNKNOWN


def get_optimized_config(gpu_model: Optional[GPUModel] = None) -> MultiStreamOptimizedConfig:
    """최적화된 설정 반환"""
    if gpu_model is None:
        gpu_model = detect_gpu_model()
    
    return MultiStreamOptimizedConfig.from_gpu_model(gpu_model)


def get_rtx_5090_config() -> MultiStreamOptimizedConfig:
    """RTX 5090 최적화 설정 (기본)"""
    return get_optimized_config(GPUModel.RTX_5090)


# 환경 변수에서 설정 오버라이드
def apply_environment_overrides(config: MultiStreamOptimizedConfig) -> MultiStreamOptimizedConfig:
    """환경 변수로 설정 오버라이드"""
    
    # NVENC 세션 수
    if 'NVENC_MAX_SESSIONS' in os.environ:
        try:
            config.gpu_config.max_concurrent_nvenc_sessions = int(os.environ['NVENC_MAX_SESSIONS'])
        except ValueError:
            pass
    
    # 배치 크기
    if 'MULTISTREAM_BATCH_SIZE' in os.environ:
        try:
            config.gpu_config.batch_size = int(os.environ['MULTISTREAM_BATCH_SIZE'])
        except ValueError:
            pass
    
    # 폴백 활성화
    if 'ENABLE_SOFTWARE_FALLBACK' in os.environ:
        config.enable_software_fallback = os.environ['ENABLE_SOFTWARE_FALLBACK'].lower() in ['true', '1', 'yes']
    
    # 상세 로그
    if 'ENABLE_DETAILED_LOGGING' in os.environ:
        config.enable_detailed_logging = os.environ['ENABLE_DETAILED_LOGGING'].lower() in ['true', '1', 'yes']
    
    return config


# 기본 설정 인스턴스 (전역 사용)
DEFAULT_CONFIG = get_rtx_5090_config()
DEFAULT_CONFIG = apply_environment_overrides(DEFAULT_CONFIG)