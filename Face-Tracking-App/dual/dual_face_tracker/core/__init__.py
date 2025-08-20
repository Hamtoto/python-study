"""
Core processing modules for dual-face tracking pipeline.

This module contains the fundamental processing components that coordinate
the entire GPU pipeline workflow.
"""

# ByteTrack 다중 객체 추적 시스템 (D9 완료)
from .bytetrack import ByteTracker, ByteTrackConfig
from .tracking_structures import Detection, Track, TrackState
from .matching import MatchingEngine

# ConditionalReID 시스템 (D10 완료)
from .conditional_reid import ConditionalReID, ConditionalReIDResult, ReIDActivationRecord
from .id_swap_detector import IDSwapDetector, SwapDetectionResult, SwapIndicator
from .embedding_manager import EmbeddingManager, EmbeddingRecord, MatchingResult

# DualFaceProcessor 통합 파이프라인 (D13 완료)
from .dual_face_processor import (
    DualFaceProcessor,
    DualFaceConfig,
    ProcessingMetrics,
    PipelineErrorHandler,
    create_default_config,
    process_video_simple
)

# MultiStreamProcessor - Phase 3 병렬 처리 시스템 (D14+ 완료)
from .multi_stream_processor import (
    MultiStreamProcessor,
    MultiStreamConfig,
    StreamJob,
    MultiStreamStats,
    create_stream_jobs,
    process_videos_parallel
)
from .stream_manager import (
    StreamManager,
    StreamContext,
    StreamStatus,
    StreamPoolStats,
    create_stream_manager
)
from .memory_pool_manager import (
    MemoryPoolManager,
    MemoryPoolConfig,
    MemoryPoolType,
    MemoryAllocationStrategy,
    MemoryBlock,
    MemoryPoolStats,
    create_memory_pool_manager
)

__all__ = [
    # ByteTrack 추적 시스템 (D9)
    'ByteTracker',
    'ByteTrackConfig', 
    'Detection',
    'Track',
    'TrackState',
    'MatchingEngine',
    
    # ConditionalReID 시스템 (D10)
    'ConditionalReID',
    'ConditionalReIDResult', 
    'ReIDActivationRecord',
    'IDSwapDetector',
    'SwapDetectionResult',
    'SwapIndicator',
    'EmbeddingManager',
    'EmbeddingRecord',
    'MatchingResult',
    
    # DualFaceProcessor 통합 파이프라인 (D13)
    'DualFaceProcessor',
    'DualFaceConfig',
    'ProcessingMetrics',
    'PipelineErrorHandler',
    'create_default_config',
    'process_video_simple',
    
    # MultiStreamProcessor - Phase 3 병렬 처리 시스템 (D14+)
    'MultiStreamProcessor',
    'MultiStreamConfig',
    'StreamJob',
    'MultiStreamStats',
    'create_stream_jobs',
    'process_videos_parallel',
    'StreamManager',
    'StreamContext',
    'StreamStatus',
    'StreamPoolStats',
    'create_stream_manager',
    'MemoryPoolManager',
    'MemoryPoolConfig',
    'MemoryPoolType',
    'MemoryAllocationStrategy',
    'MemoryBlock',
    'MemoryPoolStats',
    'create_memory_pool_manager'
]