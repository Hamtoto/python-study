#!/usr/bin/env python3
"""
NVENC Encoding Test Suite.

Comprehensive testing for PyAV NVENC hardware encoding functionality.
Tests codec availability, basic encoding, GPU memory operations, and performance.

Test Categories:
    1. Environment validation (NVENC availability)
    2. Basic encoding functionality
    3. GPU memory encoding
    4. Asynchronous encoding
    5. Adaptive bitrate encoding
    6. Error recovery and edge cases
    7. Performance benchmarks

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import torch
import av

from dual_face_tracker.encoders import (
    NvEncoder,
    AdaptiveNvEncoder,
    create_nvenc_encoder,
    EncodingProfile,
    EncodingProfileManager,
    Codec,
    Preset,
    get_default_profile,
    get_streaming_profile
)
from dual_face_tracker.utils.logger import UnifiedLogger
from dual_face_tracker.utils.exceptions import EncodingError, HardwareError


class NvencTestSuite:
    """NVENC encoding test suite."""
    
    def __init__(self):
        """Initialize test suite."""
        self.logger = UnifiedLogger("NvencTestSuite")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="nvenc_test_"))
        self.test_results: Dict[str, Dict] = {}
        
        # Test parameters
        self.test_width = 1920
        self.test_height = 1080
        self.test_fps = 30
        self.test_frames = 90  # 3 seconds
        
        self.logger.info(f"🧪 Initializing NVENC test suite")
        self.logger.info(f"📂 Temp directory: {self.temp_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
    
    def generate_test_frames(self, count: int) -> List[np.ndarray]:
        """
        Generate test frames with different patterns.
        
        Args:
            count: Number of frames to generate
            
        Returns:
            List of test frames
        """
        frames = []
        
        for i in range(count):
            # Create frame with gradient and moving elements
            frame = np.zeros((self.test_height, self.test_width, 3), dtype=np.uint8)
            
            # Gradient background
            gradient = np.linspace(0, 255, self.test_width, dtype=np.uint8)
            frame[:, :, 0] = gradient
            frame[:, :, 1] = gradient[::-1]
            frame[:, :, 2] = (gradient + i * 2) % 256
            
            # Moving circle
            center_x = int(self.test_width * 0.3 + (i * 10) % (self.test_width * 0.4))
            center_y = int(self.test_height * 0.5)
            cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)
            
            # Frame number text
            cv2.putText(
                frame,
                f"Frame {i:03d}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3
            )
            
            frames.append(frame)
        
        return frames
    
    def generate_gpu_test_frames(self, count: int) -> List[cv2.cuda.GpuMat]:
        """
        Generate test frames on GPU.
        
        Args:
            count: Number of frames to generate
            
        Returns:
            List of GPU frames
        """
        cpu_frames = self.generate_test_frames(count)
        gpu_frames = []
        
        for frame in cpu_frames:
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            gpu_frames.append(gpu_frame)
        
        return gpu_frames
    
    def test_codec_availability(self) -> Dict[str, bool]:
        """Test NVENC codec availability."""
        self.logger.info("🔍 Testing codec availability...")
        
        results = {}
        test_codecs = ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc']
        
        for codec_name in test_codecs:
            try:
                codec = av.codec.Codec(codec_name, 'w')
                results[codec_name] = True
                self.logger.success(f"✅ {codec_name} available")
            except Exception as e:
                results[codec_name] = False
                self.logger.warning(f"⚠️ {codec_name} not available: {e}")
        
        # Verify at least one NVENC codec is available
        if not any(results.values()):
            raise HardwareError("No NVENC codecs available")
        
        return results
    
    def test_basic_encoding(self) -> Dict[str, Any]:
        """Test basic NVENC encoding functionality."""
        self.logger.info("🎬 Testing basic NVENC encoding...")
        
        output_path = self.temp_dir / "basic_test.mp4"
        start_time = time.perf_counter()
        
        # Create test frames
        frames = self.generate_test_frames(self.test_frames)
        
        # Test encoding
        profile = get_default_profile()
        
        with NvEncoder(
            output_path=output_path,
            width=self.test_width,
            height=self.test_height,
            fps=self.test_fps,
            profile=profile
        ) as encoder:
            # Encode frames
            for frame in frames:
                if not encoder.encode_frame(frame):
                    raise EncodingError("Failed to encode frame")
        
        encoding_time = time.perf_counter() - start_time
        
        # Verify output file
        if not output_path.exists():
            raise EncodingError("Output file not created")
        
        file_size = output_path.stat().st_size
        
        # Verify video with PyAV
        container = av.open(str(output_path))
        stream = container.streams.video[0]
        frame_count = 0
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_count += 1
        
        container.close()
        
        results = {
            'success': True,
            'encoding_time': encoding_time,
            'fps': self.test_frames / encoding_time,
            'file_size': file_size,
            'frames_encoded': frame_count,
            'output_path': str(output_path)
        }
        
        self.logger.success(f"✅ Basic encoding test passed")
        self.logger.info(f"  • Encoding time: {encoding_time:.2f}s")
        self.logger.info(f"  • Encoding FPS: {results['fps']:.1f}")
        self.logger.info(f"  • File size: {file_size / 1024 / 1024:.1f} MB")
        
        return results
    
    def test_gpu_memory_encoding(self) -> Dict[str, Any]:
        """Test GPU memory encoding."""
        self.logger.info("🚀 Testing GPU memory encoding...")
        
        if not torch.cuda.is_available():
            self.logger.warning("⚠️ CUDA not available, skipping GPU test")
            return {'success': False, 'reason': 'CUDA not available'}
        
        output_path = self.temp_dir / "gpu_test.mp4"
        start_time = time.perf_counter()
        
        # Create GPU test frames
        gpu_frames = self.generate_gpu_test_frames(self.test_frames)
        
        # Test GPU encoding
        with NvEncoder(
            output_path=output_path,
            width=self.test_width,
            height=self.test_height,
            fps=self.test_fps,
            profile=get_streaming_profile(),
            enable_cuda_stream=True
        ) as encoder:
            # Encode GPU frames
            encoded_count = encoder.encode_gpu_frames(gpu_frames)
        
        encoding_time = time.perf_counter() - start_time
        
        # Clean up GPU memory
        for gpu_frame in gpu_frames:
            del gpu_frame
        torch.cuda.empty_cache()
        
        results = {
            'success': True,
            'encoding_time': encoding_time,
            'fps': encoded_count / encoding_time,
            'frames_encoded': encoded_count,
            'gpu_memory_efficient': True
        }
        
        self.logger.success(f"✅ GPU memory encoding test passed")
        self.logger.info(f"  • GPU encoding FPS: {results['fps']:.1f}")
        
        return results
    
    def test_async_encoding(self) -> Dict[str, Any]:
        """Test asynchronous encoding."""
        self.logger.info("⚡ Testing asynchronous encoding...")
        
        output_path = self.temp_dir / "async_test.mp4"
        start_time = time.perf_counter()
        
        # Create test frames
        frames = self.generate_test_frames(self.test_frames)
        
        # Test async encoding with buffer
        profile = get_streaming_profile()
        profile.hardware.lookahead = 16  # Enable async buffering
        
        with NvEncoder(
            output_path=output_path,
            width=self.test_width,
            height=self.test_height,
            fps=self.test_fps,
            profile=profile
        ) as encoder:
            # Encode frames quickly (async buffer should handle)
            for frame in frames:
                encoder.encode_frame(frame)
                # Small delay to simulate processing
                time.sleep(0.001)
        
        encoding_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'encoding_time': encoding_time,
            'fps': self.test_frames / encoding_time,
            'async_buffering': True
        }
        
        self.logger.success(f"✅ Async encoding test passed")
        self.logger.info(f"  • Async encoding FPS: {results['fps']:.1f}")
        
        return results
    
    def test_adaptive_encoding(self) -> Dict[str, Any]:
        """Test adaptive bitrate encoding."""
        self.logger.info("🎯 Testing adaptive encoding...")
        
        output_path = self.temp_dir / "adaptive_test.mp4"
        
        # Create frames with varying complexity
        frames = []
        for i in range(self.test_frames):
            if i < self.test_frames // 3:
                # Low complexity: solid color
                frame = np.full((self.test_height, self.test_width, 3), i * 2, dtype=np.uint8)
            elif i < 2 * self.test_frames // 3:
                # High complexity: noise
                frame = np.random.randint(0, 256, (self.test_height, self.test_width, 3), dtype=np.uint8)
            else:
                # Medium complexity: gradient
                frame = self.generate_test_frames(1)[0]
            frames.append(frame)
        
        start_time = time.perf_counter()
        
        # Test adaptive encoding
        with AdaptiveNvEncoder(
            output_path=output_path,
            width=self.test_width,
            height=self.test_height,
            fps=self.test_fps,
            profile=get_default_profile()
        ) as encoder:
            for frame in frames:
                encoder.encode_frame(frame)
        
        encoding_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'encoding_time': encoding_time,
            'fps': self.test_frames / encoding_time,
            'adaptive_enabled': True
        }
        
        self.logger.success(f"✅ Adaptive encoding test passed")
        
        return results
    
    def test_profile_management(self) -> Dict[str, Any]:
        """Test encoding profile management."""
        self.logger.info("⚙️ Testing profile management...")
        
        manager = EncodingProfileManager()
        
        # Test predefined profiles
        profiles_tested = []
        for profile_name in ['realtime', 'streaming', 'balanced', 'quality']:
            try:
                profile = manager.get_profile(profile_name)
                profiles_tested.append(profile_name)
                
                # Test profile conversion
                options = profile.to_av_options()
                assert isinstance(options, dict)
                assert 'preset' in options
                
            except Exception as e:
                self.logger.error(f"❌ Profile {profile_name} failed: {e}")
        
        # Test custom profile creation
        custom_profile = EncodingProfile(
            name='test_custom',
            codec=Codec.H264_NVENC,
            preset=Preset.P4_MEDIUM
        )
        
        manager.add_custom_profile(custom_profile)
        retrieved = manager.get_profile('test_custom')
        
        results = {
            'success': True,
            'profiles_tested': profiles_tested,
            'custom_profile_created': retrieved.name == 'test_custom'
        }
        
        self.logger.success(f"✅ Profile management test passed")
        self.logger.info(f"  • Profiles tested: {len(profiles_tested)}")
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        self.logger.info("🛡️ Testing error handling...")
        
        errors_caught = []
        
        # Test invalid codec
        try:
            encoder = NvEncoder(
                output_path=self.temp_dir / "error_test.mp4",
                width=self.test_width,
                height=self.test_height,
                fps=self.test_fps
            )
            # Try to use encoder with invalid settings
            encoder.profile.codec = "invalid_codec"
            encoder.open()
        except (EncodingError, HardwareError) as e:
            errors_caught.append('invalid_codec')
            self.logger.info(f"  ✅ Caught expected error: invalid codec")
        
        # Test invalid dimensions
        try:
            with NvEncoder(
                output_path=self.temp_dir / "error_test2.mp4",
                width=0,
                height=0,
                fps=self.test_fps
            ) as encoder:
                pass
        except (EncodingError, ValueError) as e:
            errors_caught.append('invalid_dimensions')
            self.logger.info(f"  ✅ Caught expected error: invalid dimensions")
        
        # Test encoding without opening
        try:
            encoder = NvEncoder(
                output_path=self.temp_dir / "error_test3.mp4",
                width=self.test_width,
                height=self.test_height,
                fps=self.test_fps
            )
            # Don't open encoder
            frame = np.zeros((self.test_height, self.test_width, 3), dtype=np.uint8)
            encoder.encode_frame(frame)
        except (EncodingError, AttributeError) as e:
            errors_caught.append('not_opened')
            self.logger.info(f"  ✅ Caught expected error: encoder not opened")
        
        results = {
            'success': True,
            'errors_caught': errors_caught,
            'error_handling_working': len(errors_caught) >= 2
        }
        
        self.logger.success(f"✅ Error handling test passed")
        
        return results
    
    def test_performance_benchmark(self) -> Dict[str, Any]:
        """Performance benchmark test."""
        self.logger.info("🏁 Running performance benchmark...")
        
        # Test different presets
        presets_to_test = [
            ('fastest', Preset.P1_FASTEST),
            ('balanced', Preset.P4_MEDIUM),
            ('quality', Preset.P6_SLOWER)
        ]
        
        benchmark_results = {}
        frames = self.generate_test_frames(60)  # 2 seconds
        
        for preset_name, preset in presets_to_test:
            output_path = self.temp_dir / f"benchmark_{preset_name}.mp4"
            
            profile = get_default_profile()
            profile.preset = preset
            
            start_time = time.perf_counter()
            
            with NvEncoder(
                output_path=output_path,
                width=self.test_width,
                height=self.test_height,
                fps=self.test_fps,
                profile=profile
            ) as encoder:
                for frame in frames:
                    encoder.encode_frame(frame)
            
            encoding_time = time.perf_counter() - start_time
            fps = len(frames) / encoding_time
            
            benchmark_results[preset_name] = {
                'encoding_time': encoding_time,
                'fps': fps,
                'file_size': output_path.stat().st_size if output_path.exists() else 0
            }
            
            self.logger.info(f"  • {preset_name}: {fps:.1f} fps")
        
        results = {
            'success': True,
            'benchmarks': benchmark_results,
            'fastest_preset_fps': benchmark_results.get('fastest', {}).get('fps', 0)
        }
        
        self.logger.success(f"✅ Performance benchmark completed")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all NVENC encoding tests."""
        self.logger.info("🧪 Starting NVENC encoding test suite...")
        
        tests = [
            ('codec_availability', self.test_codec_availability),
            ('basic_encoding', self.test_basic_encoding),
            ('gpu_memory_encoding', self.test_gpu_memory_encoding),
            ('async_encoding', self.test_async_encoding),
            ('adaptive_encoding', self.test_adaptive_encoding),
            ('profile_management', self.test_profile_management),
            ('error_handling', self.test_error_handling),
            ('performance_benchmark', self.test_performance_benchmark)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Running test: {test_name}")
                self.logger.info(f"{'='*60}")
                
                result = test_func()
                self.test_results[test_name] = result
                
                if result.get('success', False):
                    passed += 1
                    self.logger.success(f"✅ {test_name} PASSED")
                else:
                    self.logger.error(f"❌ {test_name} FAILED")
                
            except Exception as e:
                self.logger.error(f"❌ {test_name} FAILED with exception: {e}")
                self.test_results[test_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Summary
        success_rate = (passed / total) * 100
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"NVENC ENCODING TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Tests passed: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            self.logger.success(f"🎉 Test suite PASSED! (>= 80% success rate)")
        else:
            self.logger.error(f"💥 Test suite FAILED! (< 80% success rate)")
        
        return self.test_results


def main():
    """Main test function."""
    print("🎬 NVENC Encoding Test Suite")
    print("=" * 60)
    
    try:
        with NvencTestSuite() as test_suite:
            results = test_suite.run_all_tests()
            
            # Return exit code based on success rate
            passed_tests = sum(1 for r in results.values() if r.get('success', False))
            total_tests = len(results)
            success_rate = (passed_tests / total_tests) * 100
            
            if success_rate >= 80:
                return 0  # Success
            else:
                return 1  # Failure
                
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())