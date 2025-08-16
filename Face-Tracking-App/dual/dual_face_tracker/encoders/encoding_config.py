"""
Encoding configuration management for NVENC hardware encoding.

This module provides configuration classes and utilities for managing
video encoding parameters, codec profiles, and hardware capabilities.

Key Features:
    - Codec profile management (H.264, H.265, AV1)
    - Quality preset configurations
    - Bitrate and GOP settings
    - Hardware capability detection
    - Multi-profile support for different use cases

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
import yaml
from pathlib import Path
import logging

from ..utils.logger import UnifiedLogger
from ..utils.exceptions import ConfigurationError


class Codec(Enum):
    """Supported video codecs."""
    H264_NVENC = "h264_nvenc"
    HEVC_NVENC = "hevc_nvenc"
    AV1_NVENC = "av1_nvenc"
    H264_CPU = "libx264"  # Fallback
    HEVC_CPU = "libx265"  # Fallback


class Preset(Enum):
    """NVENC encoding presets (p1-p7)."""
    P1_FASTEST = "p1"  # Fastest, lowest quality
    P2_FASTER = "p2"
    P3_FAST = "p3"
    P4_MEDIUM = "p4"  # Default balanced
    P5_SLOW = "p5"
    P6_SLOWER = "p6"
    P7_SLOWEST = "p7"  # Slowest, highest quality


class Profile(Enum):
    """Video codec profiles."""
    BASELINE = "baseline"
    MAIN = "main"
    HIGH = "high"
    HIGH444 = "high444"  # For professional use


class RateControlMode(Enum):
    """Rate control modes."""
    CBR = "cbr"  # Constant Bitrate
    VBR = "vbr"  # Variable Bitrate
    CQ = "cq"    # Constant Quality
    VBR_HQ = "vbr_hq"  # High Quality VBR


class TuneMode(Enum):
    """NVENC tuning modes."""
    HQ = "hq"    # High Quality
    LL = "ll"    # Low Latency
    ULL = "ull"  # Ultra Low Latency
    LOSSLESS = "lossless"  # Lossless encoding


@dataclass
class BitrateConfig:
    """Bitrate configuration."""
    target: int = 8_000_000      # Target bitrate (8 Mbps)
    max: int = 12_000_000        # Maximum bitrate (12 Mbps)
    min: int = 4_000_000         # Minimum bitrate (4 Mbps)
    buffer_size: int = 16_000_000  # VBV buffer size
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return asdict(self)
    
    def scale(self, factor: float) -> 'BitrateConfig':
        """Scale bitrate by factor."""
        return BitrateConfig(
            target=int(self.target * factor),
            max=int(self.max * factor),
            min=int(self.min * factor),
            buffer_size=int(self.buffer_size * factor)
        )


@dataclass
class GOPConfig:
    """Group of Pictures configuration."""
    size: int = 30           # GOP size (keyframe interval)
    b_frames: int = 2        # Number of B-frames
    ref_frames: int = 3      # Reference frames
    closed_gop: bool = False  # Closed GOP for editing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class QualityConfig:
    """Quality configuration."""
    crf: Optional[int] = None     # Constant Rate Factor (0-51)
    qp: Optional[int] = None      # Quantization Parameter
    qmin: int = 18                # Minimum QP
    qmax: int = 51                # Maximum QP
    aq_enabled: bool = True       # Adaptive Quantization
    aq_strength: int = 8          # AQ strength (1-15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gpu_id: int = 0               # GPU device ID
    max_sessions: int = -1        # Max encoding sessions (-1 = unlimited)
    lookahead: int = 20          # Lookahead frames
    multipass: bool = False       # Two-pass encoding
    weighted_pred: bool = False   # Weighted prediction (disabled by default for B-frame compatibility)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EncodingProfile:
    """Complete encoding profile configuration."""
    
    # Basic settings
    name: str = "default"
    codec: Codec = Codec.H264_NVENC
    preset: Preset = Preset.P4_MEDIUM
    profile: Profile = Profile.HIGH
    pixel_format: str = "yuv420p"
    
    # Rate control
    rate_control: RateControlMode = RateControlMode.VBR
    bitrate: BitrateConfig = field(default_factory=BitrateConfig)
    
    # GOP settings
    gop: GOPConfig = field(default_factory=GOPConfig)
    
    # Quality settings
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Hardware settings
    hardware: HardwareConfig = field(default_factory=lambda: HardwareConfig(weighted_pred=False))
    
    # Tuning
    tune: Optional[TuneMode] = None
    
    # Advanced flags
    spatial_aq: bool = True
    temporal_aq: bool = True
    non_ref_p: bool = False
    strict_gop: bool = False
    aud: bool = False  # Access Unit Delimiters
    
    def to_av_options(self) -> Dict[str, str]:
        """Convert to PyAV codec options."""
        options = {
            'preset': self.preset.value,
            'profile': self.profile.value,
            'gpu': str(self.hardware.gpu_id),
            'rc': self.rate_control.value,
            'b': str(self.bitrate.target),
            'maxrate': str(self.bitrate.max),
            'minrate': str(self.bitrate.min),
            'bufsize': str(self.bitrate.buffer_size),
            'g': str(self.gop.size),
            'bf': str(self.gop.b_frames),
            'refs': str(self.gop.ref_frames),
        }
        
        # Quality settings
        if self.quality.crf is not None:
            options['crf'] = str(self.quality.crf)
        if self.quality.qp is not None:
            options['qp'] = str(self.quality.qp)
        
        options['qmin'] = str(self.quality.qmin)
        options['qmax'] = str(self.quality.qmax)
        
        # Tuning
        if self.tune:
            options['tune'] = self.tune.value
        
        # Advanced settings
        if self.spatial_aq:
            options['spatial-aq'] = '1'
            options['aq-strength'] = str(self.quality.aq_strength)
        if self.temporal_aq:
            options['temporal-aq'] = '1'
        if self.hardware.lookahead > 0:
            options['rc-lookahead'] = str(self.hardware.lookahead)
        if self.hardware.multipass:
            options['multipass'] = 'fullres'
        if self.hardware.weighted_pred:
            options['weighted_pred'] = '1'
        if self.non_ref_p:
            options['nonref_p'] = '1'
        if self.strict_gop:
            options['strict_gop'] = '1'
        if self.aud:
            options['aud'] = '1'
        if self.gop.closed_gop:
            options['closed_gop'] = '1'
            
        return options
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'codec': self.codec.value,
            'preset': self.preset.value,
            'profile': self.profile.value,
            'pixel_format': self.pixel_format,
            'rate_control': self.rate_control.value,
            'bitrate': self.bitrate.to_dict(),
            'gop': self.gop.to_dict(),
            'quality': self.quality.to_dict(),
            'hardware': self.hardware.to_dict(),
            'tune': self.tune.value if self.tune else None,
            'spatial_aq': self.spatial_aq,
            'temporal_aq': self.temporal_aq,
            'non_ref_p': self.non_ref_p,
            'strict_gop': self.strict_gop,
            'aud': self.aud
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncodingProfile':
        """Create from dictionary."""
        # Parse enums
        if 'codec' in data and isinstance(data['codec'], str):
            data['codec'] = Codec(data['codec'])
        if 'preset' in data and isinstance(data['preset'], str):
            data['preset'] = Preset(data['preset'])
        if 'profile' in data and isinstance(data['profile'], str):
            data['profile'] = Profile(data['profile'])
        if 'rate_control' in data and isinstance(data['rate_control'], str):
            data['rate_control'] = RateControlMode(data['rate_control'])
        if 'tune' in data and data['tune'] and isinstance(data['tune'], str):
            data['tune'] = TuneMode(data['tune'])
        
        # Parse nested configs
        if 'bitrate' in data and isinstance(data['bitrate'], dict):
            data['bitrate'] = BitrateConfig(**data['bitrate'])
        if 'gop' in data and isinstance(data['gop'], dict):
            data['gop'] = GOPConfig(**data['gop'])
        if 'quality' in data and isinstance(data['quality'], dict):
            data['quality'] = QualityConfig(**data['quality'])
        if 'hardware' in data and isinstance(data['hardware'], dict):
            data['hardware'] = HardwareConfig(**data['hardware'])
        
        return cls(**data)


class EncodingProfileManager:
    """Manager for encoding profiles."""
    
    # Predefined profiles
    PROFILES = {
        'realtime': EncodingProfile(
            name='realtime',
            preset=Preset.P1_FASTEST,
            tune=TuneMode.ULL,
            bitrate=BitrateConfig(target=6_000_000),
            gop=GOPConfig(size=60, b_frames=0),
            hardware=HardwareConfig(lookahead=0, weighted_pred=False)
        ),
        'streaming': EncodingProfile(
            name='streaming',
            preset=Preset.P3_FAST,
            tune=TuneMode.LL,
            bitrate=BitrateConfig(target=8_000_000),
            gop=GOPConfig(size=60, b_frames=2)
        ),
        'balanced': EncodingProfile(
            name='balanced',
            preset=Preset.P4_MEDIUM,
            bitrate=BitrateConfig(target=10_000_000),
            gop=GOPConfig(size=30, b_frames=2)
        ),
        'quality': EncodingProfile(
            name='quality',
            preset=Preset.P6_SLOWER,
            tune=TuneMode.HQ,
            bitrate=BitrateConfig(target=15_000_000),
            gop=GOPConfig(size=30, b_frames=3),
            hardware=HardwareConfig(multipass=True, lookahead=30)
        ),
        'archival': EncodingProfile(
            name='archival',
            preset=Preset.P7_SLOWEST,
            tune=TuneMode.HQ,
            rate_control=RateControlMode.CQ,
            quality=QualityConfig(crf=18),
            bitrate=BitrateConfig(target=20_000_000, max=30_000_000),
            gop=GOPConfig(size=15, b_frames=3, closed_gop=True),
            hardware=HardwareConfig(multipass=True, lookahead=40)
        )
    }
    
    def __init__(self):
        """Initialize profile manager."""
        self.logger = UnifiedLogger("EncodingProfileManager")
        self.custom_profiles: Dict[str, EncodingProfile] = {}
    
    def get_profile(self, name: str) -> EncodingProfile:
        """
        Get encoding profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            EncodingProfile: Profile configuration
        """
        # Check custom profiles first
        if name in self.custom_profiles:
            return self.custom_profiles[name]
        
        # Check predefined profiles
        if name in self.PROFILES:
            return self.PROFILES[name]
        
        raise ConfigurationError(f"Profile '{name}' not found")
    
    def add_custom_profile(self, profile: EncodingProfile):
        """
        Add custom encoding profile.
        
        Args:
            profile: Custom profile to add
        """
        self.custom_profiles[profile.name] = profile
        self.logger.info(f"✅ Added custom profile: {profile.name}")
    
    def load_profile_from_file(self, path: Path) -> EncodingProfile:
        """
        Load profile from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            EncodingProfile: Loaded profile
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            profile = EncodingProfile.from_dict(data)
            self.add_custom_profile(profile)
            
            return profile
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load profile: {e}")
    
    def save_profile_to_file(self, profile: EncodingProfile, path: Path):
        """
        Save profile to YAML file.
        
        Args:
            profile: Profile to save
            path: Output path
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                yaml.dump(profile.to_dict(), f, default_flow_style=False)
            
            self.logger.info(f"✅ Saved profile to: {path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save profile: {e}")
    
    def list_profiles(self) -> List[str]:
        """Get list of available profile names."""
        all_profiles = list(self.PROFILES.keys()) + list(self.custom_profiles.keys())
        return sorted(set(all_profiles))
    
    def recommend_profile(
        self,
        use_case: str,
        resolution: Tuple[int, int],
        fps: float
    ) -> EncodingProfile:
        """
        Recommend profile based on use case.
        
        Args:
            use_case: Use case (realtime, streaming, quality, archival)
            resolution: Video resolution (width, height)
            fps: Frames per second
            
        Returns:
            EncodingProfile: Recommended profile
        """
        # Get base profile
        if use_case not in self.PROFILES:
            use_case = 'balanced'
        
        profile = self.PROFILES[use_case]
        
        # Adjust bitrate based on resolution
        width, height = resolution
        pixels = width * height
        
        # Base: 1080p = 1920*1080 = 2,073,600 pixels
        base_pixels = 1920 * 1080
        scale_factor = pixels / base_pixels
        
        # Scale bitrate
        if scale_factor != 1.0:
            profile.bitrate = profile.bitrate.scale(scale_factor)
        
        # Adjust GOP based on FPS
        if fps > 30:
            profile.gop.size = int(profile.gop.size * (fps / 30))
        
        return profile


# Convenience functions
def get_default_profile() -> EncodingProfile:
    """Get default encoding profile."""
    return EncodingProfile()


def get_streaming_profile() -> EncodingProfile:
    """Get streaming-optimized profile."""
    return EncodingProfileManager.PROFILES['streaming']


def get_quality_profile() -> EncodingProfile:
    """Get quality-optimized profile."""
    return EncodingProfileManager.PROFILES['quality']


def create_custom_profile(
    name: str,
    codec: str = 'h264_nvenc',
    bitrate: int = 8_000_000,
    preset: str = 'p4'
) -> EncodingProfile:
    """
    Create custom encoding profile.
    
    Args:
        name: Profile name
        codec: Codec name
        bitrate: Target bitrate
        preset: Encoding preset
        
    Returns:
        EncodingProfile: Custom profile
    """
    return EncodingProfile(
        name=name,
        codec=Codec(codec),
        preset=Preset(preset),
        bitrate=BitrateConfig(target=bitrate)
    )