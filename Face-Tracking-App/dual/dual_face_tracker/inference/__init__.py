"""
ONNX Runtime inference engines for face detection and recognition.

Optimized inference using ONNX Runtime GPU acceleration for YOLO face detection
and conditional ReID systems. Supports CUDA Execution Provider with CPU fallback.
"""

from .onnx_engine import ONNXRuntimeEngine
from .face_detector import FaceDetector
from .reid_model import ReIDModel, ReIDModelConfig  # D10에서 구현 완료

__all__ = [
    'ONNXRuntimeEngine',
    'FaceDetector', 
    'ReIDModel',      # D10에서 구현 완료
    'ReIDModelConfig' # D10에서 구현 완료
]