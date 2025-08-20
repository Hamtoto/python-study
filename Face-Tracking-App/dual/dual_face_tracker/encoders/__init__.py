"""
NVENC Hardware Encoding Package.

This package provides hardware-accelerated video encoding using NVIDIA NVENC.
"""

from .nvencoder import (
    NvEncoder,
    AdaptiveNvEncoder,
    create_nvenc_encoder
)

from .encoding_config import (
    EncodingProfile,
    EncodingProfileManager,
    Codec,
    Preset,
    Profile,
    RateControlMode,
    TuneMode,
    BitrateConfig,
    GOPConfig,
    QualityConfig,
    HardwareConfig,
    get_default_profile,
    get_streaming_profile,
    get_quality_profile,
    create_custom_profile
)

__all__ = [
    # Encoders
    'NvEncoder',
    'AdaptiveNvEncoder',
    'create_nvenc_encoder',
    
    # Configuration
    'EncodingProfile',
    'EncodingProfileManager',
    'Codec',
    'Preset',
    'Profile',
    'RateControlMode',
    'TuneMode',
    'BitrateConfig',
    'GOPConfig',
    'QualityConfig',
    'HardwareConfig',
    
    # Convenience functions
    'get_default_profile',
    'get_streaming_profile',
    'get_quality_profile',
    'create_custom_profile'
]