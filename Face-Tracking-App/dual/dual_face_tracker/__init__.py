"""
Dual-Face High-Speed Video Processing System

A revolutionary GPU-optimized video processing system that processes dual-face videos 
through a complete zero-copy pipeline using CUDA streams for parallel processing.

Architecture:
    PyAV NVDEC → TensorRT → GPU Composition → NVENC
    
Performance Target: 5-8x throughput improvement vs existing CPU pipeline
"""

__version__ = "0.1.0"
__author__ = "Dual-Face Development Team"

# Core modules
from .core import *
from .managers import HybridConfigManager
# from .decoders import NvDecoder  # 임시 주석 처리 - import 문제
from .utils import logger

__all__ = [
    'HybridConfigManager',
    # 'NvDecoder',  # 임시 주석 처리
    'logger'
]