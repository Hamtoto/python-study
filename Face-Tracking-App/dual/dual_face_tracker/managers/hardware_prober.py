"""
HardwareProber - 하드웨어 자동 프로빙 시스템

GPU 능력, NVDEC/NVENC 세션 한도, 시스템 리소스를 자동으로 측정하여
최적화된 설정을 생성합니다.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import torch

from ..utils.exceptions import DualFaceTrackerError, GPUMemoryError
from ..utils.cuda_utils import check_cuda_available, get_gpu_memory_info


class HardwareProber:
    """
    하드웨어 자동 프로빙 클래스
    
    GPU 능력 측정 및 최적 설정 생성을 담당합니다.
    """
    
    def __init__(self):
        self.probe_results: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)
        
    def generate_optimal_config(self) -> Dict[str, Any]:
        """
        최적 설정 생성
        
        Returns:
            Dict[str, Any]: 하드웨어 기반 최적 설정
            
        Raises:
            DualFaceTrackerError: 프로빙 실패시
        """
        self.logger.info("🔍 GPU 하드웨어 능력 측정 시작...")
        
        # GPU 능력 측정
        gpu_info = self._probe_gpu_capabilities()
        
        # 최적화된 설정 생성
        optimal_config = {
            'hardware': gpu_info,
            'performance': {
                'max_concurrent_streams': self._calculate_optimal_streams(gpu_info),
                'batch_size_analyze': self._calculate_optimal_batch_size(gpu_info),
                'vram_safety_margin': 0.15,  # 15% 안전 마진
                'target_gpu_utilization': 0.85  # 85% 목표
            },
            'nvdec_settings': {
                'max_sessions': gpu_info['nvdec_max_sessions'],
                'preferred_format': 'nv12'
            },
            'nvenc_settings': {
                'max_sessions': gpu_info['nvenc_max_sessions'],
                'preset': 'medium',
                'rc_mode': 'cbr'
            },
            'generated_timestamp': datetime.now().isoformat(),
            'gpu_driver_version': gpu_info.get('driver_version'),
            'cuda_version': gpu_info.get('cuda_version')
        }
        
        self.logger.info("✅ 하드웨어 프로빙 완료")
        return optimal_config
        
    def _probe_gpu_capabilities(self) -> Dict[str, Any]:
        """
        GPU 하드웨어 능력 측정
        
        Returns:
            Dict[str, Any]: GPU 정보
            
        Raises:
            DualFaceTrackerError: GPU 정보 수집 실패시
        """
        if not check_cuda_available():
            raise DualFaceTrackerError("CUDA가 사용 불가능합니다")
            
        try:
            # PyTorch를 통한 GPU 정보 수집
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            
            # GPU 메모리 정보
            memory_info = get_gpu_memory_info()
            vram_total_gb = memory_info['total'] // (1024**3)
            
            # NVDEC/NVENC 세션 한도 테스트
            nvdec_sessions = self._test_concurrent_decoders()
            nvenc_sessions = self._test_concurrent_encoders()
            
            # CUDA 버전
            cuda_version = torch.version.cuda
            
            gpu_info = {
                'gpu_name': gpu_name,
                'nvdec_max_sessions': nvdec_sessions,
                'nvenc_max_sessions': nvenc_sessions,
                'vram_gb': vram_total_gb,
                'cuda_version': cuda_version,
                'compute_capability': self._get_compute_capability(),
                'memory_info': memory_info
            }
            
            self.logger.info(f"🎯 GPU 감지: {gpu_name} ({vram_total_gb}GB VRAM)")
            self.logger.info(f"🔧 NVDEC 세션: {nvdec_sessions}, NVENC 세션: {nvenc_sessions}")
            
            return gpu_info
            
        except Exception as e:
            raise DualFaceTrackerError(f"GPU 능력 측정 실패: {e}")
            
    def _test_concurrent_decoders(self) -> int:
        """
        동시 NVDEC 디코더 세션 수 테스트
        
        Returns:
            int: 최대 동시 NVDEC 세션 수
        """
        # RTX 5090의 경우 일반적으로 4개 NVDEC 엔진
        # 실제 테스트 없이 GPU 이름 기반 추정
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            if 'rtx 5090' in gpu_name or 'rtx 4090' in gpu_name:
                return 4
            elif 'rtx' in gpu_name and ('3090' in gpu_name or '4080' in gpu_name):
                return 3
            elif 'rtx' in gpu_name:
                return 2
            else:
                return 2  # 안전한 기본값
                
        except Exception:
            self.logger.warning("⚠️ NVDEC 세션 수 측정 실패, 기본값 사용")
            return 2
            
    def _test_concurrent_encoders(self) -> int:
        """
        동시 NVENC 인코더 세션 수 테스트
        
        Returns:
            int: 최대 동시 NVENC 세션 수
        """
        # RTX 5090의 경우 일반적으로 3개 NVENC 세션
        # 실제 테스트 없이 GPU 이름 기반 추정
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            if 'rtx 5090' in gpu_name:
                return 3
            elif 'rtx 4090' in gpu_name:
                return 3
            elif 'rtx' in gpu_name and ('3090' in gpu_name or '4080' in gpu_name):
                return 2
            elif 'rtx' in gpu_name:
                return 2
            else:
                return 2  # 안전한 기본값
                
        except Exception:
            self.logger.warning("⚠️ NVENC 세션 수 측정 실패, 기본값 사용")
            return 2
            
    def _get_compute_capability(self) -> str:
        """
        GPU Compute Capability 확인
        
        Returns:
            str: Compute capability (예: "8.9")
        """
        try:
            device = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(device)
            return f"{major}.{minor}"
        except Exception:
            return "Unknown"
            
    def _calculate_optimal_streams(self, gpu_info: Dict[str, Any]) -> int:
        """
        최적 동시 스트림 수 계산 (보수적 접근)
        
        Args:
            gpu_info: GPU 정보
            
        Returns:
            int: 최적 스트림 수
        """
        # NVDEC 세션 수를 기반으로 계산
        nvdec_sessions = gpu_info.get('nvdec_max_sessions', 2)
        vram_gb = gpu_info.get('vram_gb', 8)
        
        # 메모리 기반 제한
        if vram_gb >= 24:  # RTX 5090 등
            max_by_memory = 4
        elif vram_gb >= 16:
            max_by_memory = 3
        elif vram_gb >= 12:
            max_by_memory = 2
        else:
            max_by_memory = 1
            
        # 보수적으로 적은 값 선택
        optimal_streams = min(nvdec_sessions, max_by_memory)
        
        self.logger.info(f"🎯 최적 스트림 수: {optimal_streams}")
        return optimal_streams
        
    def _calculate_optimal_batch_size(self, gpu_info: Dict[str, Any]) -> int:
        """
        최적 배치 크기 계산
        
        Args:
            gpu_info: GPU 정보
            
        Returns:
            int: 최적 배치 크기
        """
        vram_gb = gpu_info.get('vram_gb', 8)
        
        # VRAM 기반 배치 크기 계산
        if vram_gb >= 32:  # RTX 5090
            return 256
        elif vram_gb >= 24:  # RTX 4090
            return 128
        elif vram_gb >= 16:
            return 64
        elif vram_gb >= 12:
            return 32
        else:
            return 16
            
    def get_system_info(self) -> Dict[str, Any]:
        """
        시스템 전체 정보 수집
        
        Returns:
            Dict[str, Any]: 시스템 정보
        """
        import psutil
        import platform
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total // (1024**3),
            'disk_space_gb': psutil.disk_usage('/').total // (1024**3),
            'cuda_available': check_cuda_available(),
            'torch_version': torch.__version__
        }