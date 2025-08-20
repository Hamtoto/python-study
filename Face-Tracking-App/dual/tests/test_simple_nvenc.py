#!/usr/bin/env python3
"""
Simple NVENC test to verify basic functionality.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import av

from dual_face_tracker.encoders import (
    create_nvenc_encoder,
    EncodingProfile,
    Codec,
    Preset,
    BitrateConfig,
    GOPConfig,
    HardwareConfig
)

def create_simple_test_frames(count: int = 30) -> list:
    """Create simple test frames."""
    frames = []
    
    for i in range(count):
        # Simple gradient frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Fill with color gradient
        frame[:, :, 0] = (i * 8) % 256  # Red channel
        frame[:, :, 1] = 128  # Green channel
        frame[:, :, 2] = 255 - ((i * 8) % 256)  # Blue channel
        
        # Add frame number
        cv2.putText(
            frame,
            f"Frame {i:02d}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        frames.append(frame)
    
    return frames

def test_basic_nvenc():
    """Test basic NVENC functionality."""
    print("🎬 Testing basic NVENC encoding...")
    
    # Create simple profile with minimal settings
    profile = EncodingProfile(
        name='simple_test',
        codec=Codec.H264_NVENC,
        preset=Preset.P1_FASTEST,
        bitrate=BitrateConfig(target=2_000_000, max=3_000_000, min=1_000_000),
        gop=GOPConfig(size=30, b_frames=0),  # No B-frames to avoid weighted pred issues
        hardware=HardwareConfig(
            lookahead=0,  # Disable lookahead
            weighted_pred=False,  # Disable weighted prediction
            multipass=False
        )
    )
    
    # Create test frames
    frames = create_simple_test_frames(30)
    print(f"📋 Created {len(frames)} test frames (640x480)")
    
    # Test encoding
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        encoder = create_nvenc_encoder(
            output_path=output_path,
            width=640,
            height=480,
            fps=30.0,
            codec='h264_nvenc',
            bitrate=2_000_000,
            preset='p1'
        )
        
        # Override with our simple profile
        encoder.profile = profile
        
        print(f"📂 Output path: {output_path}")
        
        with encoder:
            print("🎯 Starting encoding...")
            for i, frame in enumerate(frames):
                success = encoder.encode_frame(frame)
                if not success:
                    print(f"❌ Failed to encode frame {i}")
                    return False
                print(f"✅ Encoded frame {i}")
        
        # Check output file
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✅ Output file created: {file_size / 1024:.1f} KB")
            
            # Try to open with PyAV to verify
            try:
                container = av.open(output_path)
                stream = container.streams.video[0]
                print(f"✅ Video verification: {stream.width}x{stream.height}, {stream.frames} frames")
                container.close()
                return True
            except Exception as e:
                print(f"❌ Video verification failed: {e}")
                return False
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        return False
    finally:
        # Clean up
        if Path(output_path).exists():
            os.unlink(output_path)

def main():
    """Main test function."""
    print("🧪 Simple NVENC Test")
    print("=" * 40)
    
    try:
        success = test_basic_nvenc()
        
        if success:
            print("\n🎉 Simple NVENC test PASSED!")
            return 0
        else:
            print("\n💥 Simple NVENC test FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())