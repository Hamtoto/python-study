"""
CUDA utility functions for GPU operations and memory management.

Provides helper functions for CUDA availability checking, memory monitoring,
and GPU resource management.
"""

import torch
import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, Any
from .exceptions import GPUMemoryError


def check_cuda_available() -> bool:
    """
    Check if CUDA is available and functional.
    
    Returns:
        bool: True if CUDA is available and working
    """
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, int]:
    """
    Get GPU memory information.
    
    Args:
        device: GPU device ID. If None, uses current device.
        
    Returns:
        Dict[str, int]: Memory info with 'total', 'allocated', 'cached', 'free' in bytes
        
    Raises:
        GPUMemoryError: If memory info cannot be retrieved
    """
    if not check_cuda_available():
        raise GPUMemoryError("CUDA not available")
        
    try:
        if device is not None:
            torch.cuda.set_device(device)
            
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        free = total - allocated
        
        return {
            'total': total,
            'allocated': allocated, 
            'cached': cached,
            'free': free
        }
    except Exception as e:
        raise GPUMemoryError(f"Failed to get GPU memory info: {e}")


def get_gpu_utilization() -> float:
    """
    Get current GPU utilization percentage.
    
    Returns:
        float: GPU utilization percentage (0.0 to 100.0)
        
    Note:
        This is a simplified version. For production use, consider using nvidia-ml-py
    """
    try:
        # Simple estimation based on memory usage
        memory_info = get_gpu_memory_info()
        utilization = (memory_info['allocated'] / memory_info['total']) * 100
        return min(utilization, 100.0)
    except Exception:
        return 0.0


def clear_gpu_cache() -> None:
    """
    Clear GPU memory cache.
    
    Raises:
        GPUMemoryError: If cache clearing fails
    """
    if not check_cuda_available():
        raise GPUMemoryError("CUDA not available")
        
    try:
        torch.cuda.empty_cache()
        logging.getLogger(__name__).debug("🧹 GPU cache cleared")
    except Exception as e:
        raise GPUMemoryError(f"Failed to clear GPU cache: {e}")


def get_device_properties(device: Optional[int] = None) -> Dict[str, any]:
    """
    Get CUDA device properties.
    
    Args:
        device: GPU device ID. If None, uses current device.
        
    Returns:
        Dict[str, any]: Device properties
        
    Raises:
        GPUMemoryError: If device properties cannot be retrieved
    """
    if not check_cuda_available():
        raise GPUMemoryError("CUDA not available")
        
    try:
        if device is None:
            device = torch.cuda.current_device()
            
        props = torch.cuda.get_device_properties(device)
        
        return {
            'name': props.name,
            'total_memory': props.total_memory,
            'multi_processor_count': props.multi_processor_count,
            'major': props.major,
            'minor': props.minor,
            'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
            'max_threads_per_block': props.max_threads_per_block
        }
    except Exception as e:
        raise GPUMemoryError(f"Failed to get device properties: {e}")


def estimate_batch_size_for_memory(model_memory_mb: int, target_memory_usage: float = 0.8) -> int:
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model_memory_mb: Memory required per model inference in MB
        target_memory_usage: Target GPU memory usage ratio (0.0 to 1.0)
        
    Returns:
        int: Estimated optimal batch size
        
    Raises:
        GPUMemoryError: If memory estimation fails
    """
    try:
        memory_info = get_gpu_memory_info()
        available_mb = (memory_info['free'] * target_memory_usage) / (1024 * 1024)
        
        batch_size = max(1, int(available_mb // model_memory_mb))
        
        logging.getLogger(__name__).debug(
            f"🎯 Estimated batch size: {batch_size} "
            f"(Available: {available_mb:.1f}MB, Model: {model_memory_mb}MB)"
        )
        
        return batch_size
    except Exception as e:
        raise GPUMemoryError(f"Failed to estimate batch size: {e}")


def monitor_gpu_memory(operation_name: str) -> None:
    """
    Log current GPU memory usage for monitoring.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    try:
        memory_info = get_gpu_memory_info()
        total_gb = memory_info['total'] / (1024**3)
        allocated_gb = memory_info['allocated'] / (1024**3)
        utilization = (allocated_gb / total_gb) * 100
        
        logging.getLogger(__name__).debug(
            f"🔍 {operation_name} - GPU Memory: {allocated_gb:.1f}/{total_gb:.1f}GB ({utilization:.1f}%)"
        )
    except Exception:
        # Silent failure for monitoring
        pass


def check_cuda_memory(min_free_mb: int = 500) -> None:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        min_free_mb: Minimum required free memory in MB
        
    Raises:
        GPUMemoryError: If insufficient memory is available
    """
    if not check_cuda_available():
        raise GPUMemoryError("CUDA not available")
        
    try:
        memory_info = get_gpu_memory_info()
        free_mb = memory_info['free'] / (1024 * 1024)
        
        if free_mb < min_free_mb:
            raise GPUMemoryError(
                f"Insufficient GPU memory: {free_mb:.1f}MB available, {min_free_mb}MB required"
            )
            
    except GPUMemoryError:
        raise
    except Exception as e:
        raise GPUMemoryError(f"Failed to check GPU memory: {e}")


def safe_upload_to_gpu(gpu_mat: cv2.cuda.GpuMat, data: Union[np.ndarray, Any], cuda_stream: Optional[cv2.cuda.Stream] = None) -> bool:
    """
    OpenCV 4.13 호환 안전한 GPU 업로드 함수.
    
    Context7 검증된 여러 오버로드 지원:
    - numpy array 우선 변환
    - 다중 upload() 방식 시도
    - stream 파라미터 유연 처리
    
    Args:
        gpu_mat: 대상 GPU 매트릭스
        data: 업로드할 데이터 
        cuda_stream: CUDA 스트림 (선택적)
        
    Returns:
        bool: 업로드 성공 여부
        
    Raises:
        GPUMemoryError: 모든 방법 실패 시
    """
    # 1단계: numpy array로 변환 보장
    if not isinstance(data, np.ndarray):
        if hasattr(data, 'to_ndarray'):
            data = data.to_ndarray(format='rgb24')
        elif hasattr(data, 'numpy'):
            data = data.numpy()
        else:
            raise GPUMemoryError(f"지원하지 않는 데이터 타입: {type(data)}")
    
    # 2단계: 연속 메모리 보장 (OpenCV 요구사항)
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    # 3단계: 다중 upload() 방식 시도
    upload_methods = [
        lambda: gpu_mat.upload(data),  # 기본 방식 (가장 일반적)
        lambda: gpu_mat.upload(data, None),  # None 스트림
        lambda: gpu_mat.upload(data, cuda_stream) if cuda_stream else None,  # 사용자 스트림
        lambda: gpu_mat.upload(data, cv2.cuda.Stream())  # 새 스트림
    ]
    
    for i, method in enumerate(upload_methods):
        if method is None:
            continue
        try:
            method()
            if i > 0:  # 기본 방식이 아닐 때 로그
                logging.getLogger(__name__).debug(f"🔧 upload() 방식 {i} 성공")
            return True
        except Exception as e:
            if i == len(upload_methods) - 1:  # 마지막 시도
                raise GPUMemoryError(f"모든 upload() 방식 실패: {e}")
            continue
    
    return False


def ensure_gpu_mat(frame: Union[np.ndarray, cv2.cuda.GpuMat]) -> cv2.cuda.GpuMat:
    """
    Ensure the frame is a GPU Mat (GpuMat).
    
    Args:
        frame: Input frame (CPU Mat or GPU GpuMat)
        
    Returns:
        cv2.cuda.GpuMat: GPU Mat version of the frame
        
    Raises:
        GPUMemoryError: If GPU upload fails
    """
    if isinstance(frame, cv2.cuda.GpuMat):
        return frame
        
    # PyAV VideoFrame을 numpy array로 변환
    if hasattr(frame, 'to_ndarray'):
        frame = frame.to_ndarray(format='rgb24')
    elif not isinstance(frame, np.ndarray):
        raise GPUMemoryError(f"Unsupported frame type: {type(frame)}")
        
    try:
        gpu_mat = cv2.cuda.GpuMat()
        # OpenCV 4.13 호환 안전한 업로드 사용
        if not safe_upload_to_gpu(gpu_mat, frame):
            raise GPUMemoryError("Safe upload failed")
        return gpu_mat
    except Exception as e:
        raise GPUMemoryError(f"Failed to upload frame to GPU: {e}")


def ensure_cpu_mat(frame: Union[np.ndarray, cv2.cuda.GpuMat]) -> np.ndarray:
    """
    Ensure the frame is a CPU Mat (ndarray).
    
    Args:
        frame: Input frame (CPU Mat or GPU GpuMat)
        
    Returns:
        np.ndarray: CPU Mat version of the frame
    """
    if isinstance(frame, np.ndarray):
        return frame
        
    try:
        return frame.download()
    except Exception as e:
        raise GPUMemoryError(f"Failed to download frame from GPU: {e}")


def get_optimal_cuda_streams() -> int:
    """
    Get optimal number of CUDA streams based on GPU capabilities.
    
    Returns:
        int: Optimal number of streams
    """
    try:
        if not check_cuda_available():
            return 1
            
        props = get_device_properties()
        # Estimate based on multiprocessor count
        mp_count = props.get('multi_processor_count', 1)
        
        # Conservative estimate: 2-4 streams per streaming multiprocessor
        optimal_streams = min(max(2, mp_count // 4), 8)
        
        logging.getLogger(__name__).debug(f"🔧 Optimal CUDA streams: {optimal_streams}")
        return optimal_streams
        
    except Exception:
        return 2  # Safe fallback


def ensure_cuda_context(device: Optional[int] = None) -> None:
    """
    Ensure CUDA context is properly initialized.
    
    Args:
        device: GPU device ID. If None, uses current device.
        
    Raises:
        GPUMemoryError: If CUDA context initialization fails
    """
    if not check_cuda_available():
        raise GPUMemoryError("CUDA not available")
    
    try:
        if device is not None:
            torch.cuda.set_device(device)
        
        # Initialize CUDA context by creating a small tensor
        dummy = torch.cuda.FloatTensor(1)
        del dummy
        torch.cuda.synchronize()
        
        logging.getLogger(__name__).debug("🔧 CUDA context initialized")
        
    except Exception as e:
        raise GPUMemoryError(f"Failed to initialize CUDA context: {e}")