"""
GPU-based video composition and tile management.

Handles real-time frame composition using CUDA kernels for
dual-face split-screen video generation.
"""

from .tile_composer import TileComposer
from .gpu_resizer import GpuResizer
from .composition_policy import TileCompositionErrorPolicy

__all__ = [
    'TileComposer',
    'GpuResizer', 
    'TileCompositionErrorPolicy'
]