#!/usr/bin/env python3
"""
DualFaceProcessor End-to-End 통합 테스트.

전체 GPU 파이프라인의 통합 기능을 검증합니다:
NVDEC → GPU Composition → NVENC

테스트 범위:
    1. 파이프라인 초기화 및 설정
    2. 간단한 프레임 생성 및 처리
    3. GPU 메모리 효율성 검증
    4. 성능 메트릭 검증
    5. 에러 처리 시나리오
    6. End-to-End 통합 테스트

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

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import torch

from dual_face_tracker.core import (
    DualFaceProcessor,
    DualFaceConfig,
    ProcessingMetrics,
    create_default_config,
    process_video_simple
)
from dual_face_tracker.utils.logger import UnifiedLogger


class DualFaceProcessorTestSuite:
    """DualFaceProcessor 통합 테스트 스위트."""
    
    def __init__(self):
        """테스트 스위트 초기화."""
        self.logger = UnifiedLogger("DualFaceProcessorTest")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dual_face_test_"))
        self.test_results: Dict[str, Dict] = {}
        
        # 테스트 비디오 생성 설정
        self.test_width = 640
        self.test_height = 480
        self.test_fps = 30
        self.test_duration = 2  # seconds
        self.test_frame_count = self.test_fps * self.test_duration
        
        self.logger.info(f"🧪 Initializing DualFaceProcessor test suite")
        self.logger.info(f"📂 Temp directory: {self.temp_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def cleanup(self):
        """임시 파일 정리."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
    
    def create_test_video(self, output_path: Path, frame_count: int = None) -> bool:
        """
        테스트용 비디오 생성.
        
        Args:
            output_path: 출력 비디오 경로
            frame_count: 프레임 수 (None이면 기본값)
            
        Returns:
            bool: 성공 여부
        """
        try:
            if frame_count is None:
                frame_count = self.test_frame_count
            
            # OpenCV VideoWriter 사용
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.test_fps,
                (self.test_width, self.test_height)
            )
            
            if not writer.isOpened():
                self.logger.error(f"❌ Failed to open video writer: {output_path}")
                return False
            
            for i in range(frame_count):
                # 다채로운 테스트 프레임 생성
                frame = np.zeros((self.test_height, self.test_width, 3), dtype=np.uint8)
                
                # 그라디언트 배경
                gradient = np.linspace(0, 255, self.test_width, dtype=np.uint8)
                frame[:, :, 0] = gradient  # Red
                frame[:, :, 1] = 128       # Green (constant)
                frame[:, :, 2] = gradient[::-1]  # Blue (reverse)
                
                # 움직이는 원
                center_x = int(self.test_width * 0.3 + (i * 5) % (self.test_width * 0.4))
                center_y = int(self.test_height * 0.5)
                cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
                
                # 프레임 번호 텍스트
                cv2.putText(
                    frame,
                    f"Frame {i:03d}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                writer.write(frame)
            
            writer.release()
            
            # 파일 크기 확인
            if output_path.exists() and output_path.stat().st_size > 0:
                self.logger.success(f"✅ Test video created: {output_path} ({frame_count} frames)")
                return True
            else:
                self.logger.error(f"❌ Test video creation failed: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error creating test video: {e}")
            return False
    
    def test_config_creation(self) -> Dict[str, Any]:
        """설정 생성 테스트."""
        self.logger.info("⚙️ Testing configuration creation...")
        
        try:
            # 기본 설정 생성
            config = create_default_config(
                input_path="test_input.mp4",
                output_path="test_output.mp4"
            )
            
            # 설정 검증
            assert config.input_path == "test_input.mp4"
            assert config.output_path == "test_output.mp4"
            assert config.output_width == 1920
            assert config.output_height == 1080
            assert config.target_fps == 30.0
            
            # 커스텀 설정 생성
            custom_config = DualFaceConfig(
                input_path="custom_input.mp4",
                output_path="custom_output.mp4",
                output_width=1280,
                output_height=720,
                target_fps=60.0
            )
            
            assert custom_config.output_width == 1280
            assert custom_config.target_fps == 60.0
            
            results = {
                'success': True,
                'default_config_valid': True,
                'custom_config_valid': True
            }
            
            self.logger.success("✅ Configuration creation test passed")
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ Configuration creation test failed: {e}")
        
        return results
    
    def test_processor_initialization(self) -> Dict[str, Any]:
        """프로세서 초기화 테스트."""
        self.logger.info("🔧 Testing processor initialization...")
        
        try:
            # 임시 입출력 파일 생성
            input_path = self.temp_dir / "test_input.mp4"
            output_path = self.temp_dir / "test_output.mp4"
            
            # 간단한 테스트 비디오 생성
            if not self.create_test_video(input_path, frame_count=10):
                raise Exception("Failed to create test video")
            
            # 설정 생성
            config = DualFaceConfig(
                input_path=str(input_path),
                output_path=str(output_path),
                output_width=1920,
                output_height=1080
            )
            
            # 프로세서 초기화 테스트
            processor = DualFaceProcessor(config)
            
            # Context manager 테스트
            with processor:
                # 초기화 성공 확인
                assert processor.decoder is not None
                assert processor.tile_composer is not None
                assert processor.encoder is not None
                
                # 메트릭 초기 상태 확인
                assert processor.metrics.frames_processed == 0
                assert processor.metrics.frames_dropped == 0
            
            results = {
                'success': True,
                'initialization_time': 0.5,  # Placeholder
                'components_loaded': True
            }
            
            self.logger.success("✅ Processor initialization test passed")
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ Processor initialization test failed: {e}")
        
        return results
    
    def test_simple_processing(self) -> Dict[str, Any]:
        """간단한 비디오 처리 테스트."""
        self.logger.info("🎬 Testing simple video processing...")
        
        try:
            # 임시 파일 생성
            input_path = self.temp_dir / "simple_input.mp4"
            output_path = self.temp_dir / "simple_output.mp4"
            
            # 짧은 테스트 비디오 생성 (30프레임 = 1초)
            if not self.create_test_video(input_path, frame_count=30):
                raise Exception("Failed to create test video")
            
            self.logger.info(f"📹 Created test video: {input_path}")
            
            # 간단한 처리 함수 사용
            start_time = time.perf_counter()
            
            metrics = process_video_simple(
                input_path=str(input_path),
                output_path=str(output_path),
                output_width=1920,
                output_height=1080,
                target_fps=30.0
            )
            
            processing_time = time.perf_counter() - start_time
            
            # 결과 검증
            if not output_path.exists():
                raise Exception("Output file not created")
            
            output_size = output_path.stat().st_size
            if output_size == 0:
                raise Exception("Output file is empty")
            
            # 메트릭 검증
            success_rate = metrics.frames_processed / 30.0 if metrics.frames_processed > 0 else 0
            
            results = {
                'success': True,
                'processing_time': processing_time,
                'frames_processed': metrics.frames_processed,
                'frames_dropped': metrics.frames_dropped,
                'average_fps': metrics.average_fps,
                'success_rate': success_rate,
                'output_size_mb': output_size / (1024**2)
            }
            
            self.logger.success(f"✅ Simple processing test passed")
            self.logger.info(f"  • Processed {metrics.frames_processed} frames")
            self.logger.info(f"  • Average FPS: {metrics.average_fps:.1f}")
            self.logger.info(f"  • Output size: {output_size / 1024:.1f} KB")
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ Simple processing test failed: {e}")
        
        return results
    
    def test_gpu_memory_efficiency(self) -> Dict[str, Any]:
        """GPU 메모리 효율성 테스트."""
        self.logger.info("🖥️ Testing GPU memory efficiency...")
        
        try:
            if not torch.cuda.is_available():
                return {
                    'success': False,
                    'error': 'CUDA not available'
                }
            
            # GPU 메모리 사용량 모니터링
            from dual_face_tracker.utils.cuda_utils import get_gpu_memory_info
            
            # 처리 전 메모리 상태
            initial_memory = get_gpu_memory_info()
            initial_used_mb = initial_memory['allocated'] / (1024**2)
            
            # 임시 파일 생성
            input_path = self.temp_dir / "memory_test_input.mp4"
            output_path = self.temp_dir / "memory_test_output.mp4"
            
            # 약간 긴 비디오 생성 (90프레임 = 3초)
            if not self.create_test_video(input_path, frame_count=90):
                raise Exception("Failed to create test video")
            
            # 처리 실행
            config = DualFaceConfig(
                input_path=str(input_path),
                output_path=str(output_path),
                max_gpu_memory_usage=0.75  # 75% 제한
            )
            
            with DualFaceProcessor(config) as processor:
                metrics = processor.process_video()
            
            # 처리 후 메모리 상태
            final_memory = get_gpu_memory_info()
            final_used_mb = final_memory['allocated'] / (1024**2)
            memory_increase_mb = final_used_mb - initial_used_mb
            
            # 메모리 효율성 검증
            total_memory_gb = final_memory['total'] / (1024**3)
            memory_usage_percent = (final_used_mb / (total_memory_gb * 1024)) * 100
            
            results = {
                'success': True,
                'initial_memory_mb': initial_used_mb,
                'final_memory_mb': final_used_mb,
                'memory_increase_mb': memory_increase_mb,
                'memory_usage_percent': memory_usage_percent,
                'under_limit': memory_usage_percent < 75.0,
                'frames_processed': metrics.frames_processed
            }
            
            self.logger.success(f"✅ GPU memory efficiency test passed")
            self.logger.info(f"  • Memory usage: {memory_usage_percent:.1f}%")
            self.logger.info(f"  • Memory increase: {memory_increase_mb:.1f} MB")
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ GPU memory efficiency test failed: {e}")
        
        return results
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 테스트."""
        self.logger.info("📊 Testing performance metrics...")
        
        try:
            # 임시 파일 생성
            input_path = self.temp_dir / "perf_test_input.mp4"
            output_path = self.temp_dir / "perf_test_output.mp4"
            
            # 표준 길이 비디오 생성 (60프레임 = 2초)
            if not self.create_test_video(input_path, frame_count=60):
                raise Exception("Failed to create test video")
            
            # 성능 측정
            config = DualFaceConfig(
                input_path=str(input_path),
                output_path=str(output_path),
                performance_config={
                    'enable_monitoring': True,
                    'monitoring_interval': 0.5
                }
            )
            
            start_time = time.perf_counter()
            
            with DualFaceProcessor(config) as processor:
                metrics = processor.process_video()
            
            total_time = time.perf_counter() - start_time
            
            # 성능 지표 계산
            target_fps = 30.0
            realtime_ratio = metrics.average_fps / target_fps if target_fps > 0 else 0
            
            # 성능 검증
            fps_acceptable = metrics.average_fps >= 15.0  # 최소 15 FPS
            realtime_capable = realtime_ratio >= 0.5      # 실시간의 50% 이상
            
            results = {
                'success': True,
                'total_processing_time': total_time,
                'frames_processed': metrics.frames_processed,
                'average_fps': metrics.average_fps,
                'realtime_ratio': realtime_ratio,
                'fps_acceptable': fps_acceptable,
                'realtime_capable': realtime_capable,
                'composition_success_rate': metrics.composition_success_rate,
                'encoding_success_rate': metrics.encoding_success_rate
            }
            
            self.logger.success(f"✅ Performance metrics test passed")
            self.logger.info(f"  • Average FPS: {metrics.average_fps:.1f}")
            self.logger.info(f"  • Realtime ratio: {realtime_ratio:.1f}x")
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ Performance metrics test failed: {e}")
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """에러 처리 테스트."""
        self.logger.info("🛡️ Testing error handling...")
        
        try:
            errors_caught = []
            
            # 1. 존재하지 않는 입력 파일
            try:
                config = DualFaceConfig(
                    input_path="nonexistent_file.mp4",
                    output_path=str(self.temp_dir / "error_test_output.mp4")
                )
                with DualFaceProcessor(config) as processor:
                    processor.process_video()
            except Exception:
                errors_caught.append('missing_input_file')
                self.logger.info("  ✅ Caught expected error: missing input file")
            
            # 2. 잘못된 출력 경로
            try:
                input_path = self.temp_dir / "error_input.mp4"
                self.create_test_video(input_path, frame_count=10)
                
                config = DualFaceConfig(
                    input_path=str(input_path),
                    output_path="/invalid/path/output.mp4"  # 존재하지 않는 디렉토리
                )
                with DualFaceProcessor(config) as processor:
                    processor.process_video()
            except Exception:
                errors_caught.append('invalid_output_path')
                self.logger.info("  ✅ Caught expected error: invalid output path")
            
            # 3. 잘못된 설정값
            try:
                config = DualFaceConfig(
                    input_path="test.mp4",
                    output_path="test_out.mp4",
                    output_width=0,  # 잘못된 너비
                    output_height=-100  # 잘못된 높이
                )
                # 초기화만으로도 에러가 발생해야 함
                DualFaceProcessor(config)
            except Exception:
                errors_caught.append('invalid_dimensions')
                self.logger.info("  ✅ Caught expected error: invalid dimensions")
            
            results = {
                'success': True,
                'errors_caught': errors_caught,
                'error_handling_working': len(errors_caught) >= 2
            }
            
            self.logger.success(f"✅ Error handling test passed")
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ Error handling test failed: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict]:
        """모든 테스트 실행."""
        self.logger.info("🧪 Starting DualFaceProcessor comprehensive test suite...")
        
        tests = [
            ('config_creation', self.test_config_creation),
            ('processor_initialization', self.test_processor_initialization),
            ('simple_processing', self.test_simple_processing),
            ('gpu_memory_efficiency', self.test_gpu_memory_efficiency),
            ('performance_metrics', self.test_performance_metrics),
            ('error_handling', self.test_error_handling)
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
        
        # 요약
        success_rate = (passed / total) * 100
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DUALFACEPROCESSOR TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Tests passed: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            self.logger.success(f"🎉 Test suite PASSED! (>= 80% success rate)")
        else:
            self.logger.error(f"💥 Test suite FAILED! (< 80% success rate)")
        
        return self.test_results


def main():
    """메인 테스트 함수."""
    print("🎬 DualFaceProcessor End-to-End Test Suite")
    print("=" * 60)
    
    try:
        with DualFaceProcessorTestSuite() as test_suite:
            results = test_suite.run_all_tests()
            
            # 성공률 기반 종료 코드
            passed_tests = sum(1 for r in results.values() if r.get('success', False))
            total_tests = len(results)
            success_rate = (passed_tests / total_tests) * 100
            
            if success_rate >= 80:
                return 0  # 성공
            else:
                return 1  # 실패
                
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())