"""
Custom exceptions for dual-face tracker system.

Provides specialized exception classes for different error scenarios
in the GPU processing pipeline.
"""


class DualFaceTrackerError(Exception):
    """Base exception for dual-face tracker system."""
    pass


class GPUMemoryError(DualFaceTrackerError):
    """Raised when GPU memory allocation or management fails."""
    pass


class DecodingError(DualFaceTrackerError):
    """Raised when video decoding fails."""
    pass


class EncodingError(DualFaceTrackerError):
    """Raised when video encoding fails."""
    pass


class ConfigurationError(DualFaceTrackerError):
    """Raised when configuration loading or validation fails."""
    pass


class HardwareProbeError(DualFaceTrackerError):
    """Raised when hardware probing fails."""
    pass


class TensorRTError(DualFaceTrackerError):
    """Raised when TensorRT operations fail."""
    pass


class StreamProcessingError(DualFaceTrackerError):
    """Raised when CUDA stream processing fails."""
    pass


class InferenceError(DualFaceTrackerError):
    """Raised when model inference fails."""
    pass


class ModelLoadError(DualFaceTrackerError):
    """Raised when model loading fails."""
    pass


class PreprocessingError(DualFaceTrackerError):
    """Raised when image preprocessing fails."""
    pass


class PostprocessingError(DualFaceTrackerError):
    """Raised when inference output postprocessing fails."""
    pass


class CompositionError(DualFaceTrackerError):
    """Raised when frame composition or tile processing fails."""
    pass


class ResizeError(DualFaceTrackerError):
    """Raised when image resizing operations fail."""
    pass


class BufferError(DualFaceTrackerError):
    """Raised when buffer management operations fail."""
    pass


class CUDAStreamError(DualFaceTrackerError):
    """Raised when CUDA stream operations fail."""
    pass


class HardwareError(DualFaceTrackerError):
    """Raised when hardware detection or capability checks fail."""
    pass


class ResourceError(DualFaceTrackerError):
    """Raised when system resource allocation or management fails."""
    pass


# Phase 3 MultiStream 전용 예외들
class MultiStreamError(DualFaceTrackerError):
    """Raised when multi-stream processing fails."""
    pass


class StreamAllocationError(DualFaceTrackerError):
    """Raised when CUDA stream allocation fails."""
    pass


class StreamSynchronizationError(DualFaceTrackerError):
    """Raised when CUDA stream synchronization fails."""
    pass


class MemoryPoolError(DualFaceTrackerError):
    """Raised when memory pool operations fail."""
    pass


class FFmpegError(DualFaceTrackerError):
    """Raised when FFmpeg operations fail."""
    pass


class VideoProcessingError(DualFaceTrackerError):
    """Raised when video processing operations fail."""
    pass