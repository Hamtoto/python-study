"""
Utility functions and helper classes.

Common utilities for logging, error handling, and system operations
across the dual-face tracking pipeline.
"""

from .logger import get_logger
from .exceptions import DualFaceTrackerError, GPUMemoryError, DecodingError
from .cuda_utils import check_cuda_available, get_gpu_memory_info

# Create default logger instance
logger = get_logger(__name__)

__all__ = [
    'logger',
    'get_logger',
    'DualFaceTrackerError',
    'GPUMemoryError', 
    'DecodingError',
    'check_cuda_available',
    'get_gpu_memory_info'
]