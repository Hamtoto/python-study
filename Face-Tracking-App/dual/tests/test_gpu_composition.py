#!/usr/bin/env python3
"""
GPU Composition 시스템 통합 테스트

TileComposer, GpuResizer, TileCompositionErrorPolicy의 통합 테스트를 수행합니다.
실제 이미지와 다양한 시나리오를 사용하여 GPU 가속 타일 합성 시스템을 검증합니다.
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Tuple, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_face_tracker.composers.tile_composer import TileComposer
from dual_face_tracker.composers.gpu_resizer import GpuResizer, ResizeStrategy, ResizeParams
from dual_face_tracker.composers.composition_policy import TileCompositionErrorPolicy, ErrorType
from dual_face_tracker.utils.logger import get_logger
from dual_face_tracker.utils.cuda_utils import check_cuda_memory

logger = get_logger(__name__)


class GPUCompositionTester:
    """
    GPU Composition 시스템 테스터
    
    Tests:
    1. TileComposer 기본 기능
    2. GpuResizer 다양한 전략
    3. TileCompositionErrorPolicy 에러 처리
    4. 성능 벤치마크
    5. 메모리 효율성
    """
    
    def __init__(self):
        """테스터 초기화"""
        self.tile_composer = None
        self.gpu_resizer = None
        self.error_policy = None
        
        # 테스트 결과
        self.test_results = {
            "tile_composer": {"passed": 0, "failed": 0, "details": []},
            "gpu_resizer": {"passed": 0, "failed": 0, "details": []},
            "error_policy": {"passed": 0, "failed": 0, "details": []},
            "performance": {"passed": 0, "failed": 0, "details": []},
            "integration": {"passed": 0, "failed": 0, "details": []}
        }
        
        logger.info("🚀 GPU Composition 테스터 초기화")
    
    def run_all_tests(self) -> bool:
        """
        모든 테스트 실행
        
        Returns:
            bool: 전체 테스트 성공 여부
        """
        try:
            logger.info("=" * 80)
            logger.info("🎯 GPU Composition 시스템 통합 테스트 시작")
            logger.info("=" * 80)
            
            # GPU 환경 확인
            if not self._check_gpu_environment():
                logger.error("❌ GPU 환경 요구사항 미충족")
                return False
            
            # 컴포넌트 초기화
            self._initialize_components()
            
            # 테스트 실행
            success = True
            
            success &= self._test_tile_composer()
            success &= self._test_gpu_resizer() 
            success &= self._test_error_policy()
            success &= self._test_performance()
            success &= self._test_integration()
            
            # 결과 출력
            self._print_test_results(success)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 치명적 오류: {e}")
            return False
        finally:
            self._cleanup()
    
    def _check_gpu_environment(self) -> bool:
        """GPU 환경 확인"""
        try:
            if not torch.cuda.is_available():
                logger.error("❌ CUDA 사용 불가")
                return False
            
            if not cv2.cuda.getCudaEnabledDeviceCount():
                logger.error("❌ OpenCV CUDA 지원 없음")
                return False
            
            check_cuda_memory()
            
            gpu_info = torch.cuda.get_device_properties(0)
            logger.info(f"✅ GPU 환경 확인: {gpu_info.name}, {gpu_info.total_memory / (1024**3):.1f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU 환경 확인 실패: {e}")
            return False
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.tile_composer = TileComposer(
                output_width=1920,
                output_height=1080,
                use_cuda_stream=True
            )
            
            self.gpu_resizer = GpuResizer(
                default_interpolation=cv2.INTER_LINEAR,
                use_cuda_stream=True,
                buffer_pool_size=5
            )
            
            self.error_policy = TileCompositionErrorPolicy(
                output_width=1920,
                output_height=1080,
                max_consecutive_errors=5,
                memory_threshold_percent=85.0,
                enable_quality_reduction=True
            )
            
            logger.info("✅ 컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            raise
    
    def _test_tile_composer(self) -> bool:
        """TileComposer 테스트"""
        try:
            logger.info("🧪 TileComposer 테스트 시작")
            
            test_success = True
            
            # 테스트 1: 기본 듀얼 프레임 합성
            test_success &= self._test_dual_frame_composition()
            
            # 테스트 2: 단일 프레임 합성
            test_success &= self._test_single_frame_composition()
            
            # 테스트 3: 다양한 해상도 처리
            test_success &= self._test_various_resolutions()
            
            # 테스트 4: CUDA 스트림 동작
            test_success &= self._test_cuda_stream_operation()
            
            # 테스트 5: 메모리 사용량 모니터링
            test_success &= self._test_memory_monitoring()
            
            logger.info(f"✅ TileComposer 테스트 완료: {'성공' if test_success else '실패'}")
            return test_success
            
        except Exception as e:
            logger.error(f"❌ TileComposer 테스트 실패: {e}")
            self.test_results["tile_composer"]["failed"] += 1
            return False
    
    def _test_dual_frame_composition(self) -> bool:
        """듀얼 프레임 합성 테스트"""
        try:
            # 테스트 프레임 생성
            left_frame = self._create_test_frame(640, 480, (255, 0, 0))  # 빨간색
            right_frame = self._create_test_frame(800, 600, (0, 255, 0))  # 초록색
            
            # GPU로 업로드
            left_gpu = cv2.cuda.GpuMat()
            right_gpu = cv2.cuda.GpuMat()
            left_gpu.upload(left_frame)
            right_gpu.upload(right_frame)
            
            # 합성 수행
            start_time = time.time()
            composed = self.tile_composer.compose_dual_frame(left_gpu, right_gpu)
            composition_time = (time.time() - start_time) * 1000
            
            # 결과 검증 (OpenCV size()는 (width, height) 순서)
            if composed.size() != (1920, 1080):
                raise ValueError(f"출력 크기 불일치: {composed.size()}")
            
            # 동기화
            self.tile_composer.synchronize()
            
            logger.info(f"✅ 듀얼 프레임 합성: {composition_time:.2f}ms")
            self.test_results["tile_composer"]["passed"] += 1
            self.test_results["tile_composer"]["details"].append(
                f"듀얼 프레임 합성: {composition_time:.2f}ms"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 듀얼 프레임 합성 테스트 실패: {e}")
            self.test_results["tile_composer"]["failed"] += 1
            return False
    
    def _test_single_frame_composition(self) -> bool:
        """단일 프레임 합성 테스트"""
        try:
            # 테스트 프레임 생성
            frame = self._create_test_frame(1280, 720, (0, 0, 255))  # 파란색
            
            positions = ["center", "left", "right"]
            
            for position in positions:
                # 합성 수행
                start_time = time.time()
                composed = self.tile_composer.compose_single_frame(frame, position)
                composition_time = (time.time() - start_time) * 1000
                
                # 결과 검증
                if composed.size() != (1920, 1080):
                    raise ValueError(f"출력 크기 불일치 ({position}): {composed.size()}")
                
                logger.info(f"✅ 단일 프레임 합성 ({position}): {composition_time:.2f}ms")
            
            self.test_results["tile_composer"]["passed"] += 1
            self.test_results["tile_composer"]["details"].append(
                f"단일 프레임 합성: {len(positions)}개 위치"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 단일 프레임 합성 테스트 실패: {e}")
            self.test_results["tile_composer"]["failed"] += 1
            return False
    
    def _test_various_resolutions(self) -> bool:
        """다양한 해상도 처리 테스트"""
        try:
            resolutions = [
                (320, 240),   # QVGA
                (640, 480),   # VGA
                (1280, 720),  # HD
                (1920, 1080), # FHD
                (2560, 1440)  # QHD
            ]
            
            for width, height in resolutions:
                # 테스트 프레임 생성
                frame = self._create_test_frame(width, height, (128, 128, 128))
                
                # 합성 수행
                start_time = time.time()
                composed = self.tile_composer.compose_single_frame(frame, "center")
                composition_time = (time.time() - start_time) * 1000
                
                # 결과 검증
                if composed.size() != (1920, 1080):
                    raise ValueError(f"출력 크기 불일치 ({width}x{height})")
                
                logger.info(f"✅ 해상도 {width}x{height}: {composition_time:.2f}ms")
            
            self.test_results["tile_composer"]["passed"] += 1
            self.test_results["tile_composer"]["details"].append(
                f"해상도 테스트: {len(resolutions)}개 해상도"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 해상도 테스트 실패: {e}")
            self.test_results["tile_composer"]["failed"] += 1
            return False
    
    def _test_cuda_stream_operation(self) -> bool:
        """CUDA 스트림 동작 테스트"""
        try:
            if not self.tile_composer.use_cuda_stream:
                logger.warning("⚠️ CUDA 스트림이 비활성화되어 있음")
                return True
            
            # 비동기 처리 테스트
            frames = []
            for i in range(5):
                frame = self._create_test_frame(800, 600, (i * 50, i * 40, i * 30))
                frames.append(frame)
            
            # 연속 처리
            start_time = time.time()
            composed_frames = []
            
            for i, frame in enumerate(frames):
                composed = self.tile_composer.compose_single_frame(frame, "center")
                composed_frames.append(composed)
            
            # 동기화
            self.tile_composer.synchronize()
            total_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ CUDA 스트림 처리: {len(frames)}프레임, {total_time:.2f}ms")
            self.test_results["tile_composer"]["passed"] += 1
            self.test_results["tile_composer"]["details"].append(
                f"CUDA 스트림: {len(frames)}프레임, {total_time:.2f}ms"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CUDA 스트림 테스트 실패: {e}")
            self.test_results["tile_composer"]["failed"] += 1
            return False
    
    def _test_memory_monitoring(self) -> bool:
        """메모리 사용량 모니터링 테스트"""
        try:
            # 초기 메모리 상태
            initial_memory = self.tile_composer.get_memory_usage()
            
            # 대용량 프레임으로 메모리 사용량 증가
            large_frame = self._create_test_frame(2560, 1440, (200, 100, 50))
            
            for i in range(3):
                composed = self.tile_composer.compose_single_frame(large_frame, "center")
                
            # 최종 메모리 상태
            final_memory = self.tile_composer.get_memory_usage()
            
            if initial_memory and final_memory:
                memory_diff = final_memory.get("used_vram_gb", 0) - initial_memory.get("used_vram_gb", 0)
                logger.info(f"✅ 메모리 모니터링: 사용량 변화 {memory_diff:.2f}GB")
            else:
                logger.info("✅ 메모리 모니터링: 정보 조회 성공")
            
            self.test_results["tile_composer"]["passed"] += 1
            self.test_results["tile_composer"]["details"].append("메모리 모니터링")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 메모리 모니터링 테스트 실패: {e}")
            self.test_results["tile_composer"]["failed"] += 1
            return False
    
    def _test_gpu_resizer(self) -> bool:
        """GpuResizer 테스트"""
        try:
            logger.info("🧪 GpuResizer 테스트 시작")
            
            test_success = True
            
            # 테스트 1: 다양한 리사이징 전략
            test_success &= self._test_resize_strategies()
            
            # 테스트 2: 배치 리사이징
            test_success &= self._test_batch_resize()
            
            # 테스트 3: 버퍼 풀 관리
            test_success &= self._test_buffer_pool()
            
            # 테스트 4: 성능 측정
            test_success &= self._test_resize_performance()
            
            logger.info(f"✅ GpuResizer 테스트 완료: {'성공' if test_success else '실패'}")
            return test_success
            
        except Exception as e:
            logger.error(f"❌ GpuResizer 테스트 실패: {e}")
            self.test_results["gpu_resizer"]["failed"] += 1
            return False
    
    def _test_resize_strategies(self) -> bool:
        """리사이징 전략 테스트"""
        try:
            # 테스트 프레임
            frame = self._create_test_frame(800, 600, (150, 100, 200))
            
            strategies = [
                ResizeStrategy.FIT_CONTAIN,
                ResizeStrategy.FIT_COVER,
                ResizeStrategy.STRETCH,
                ResizeStrategy.CENTER_CROP
            ]
            
            target_width, target_height = 960, 540
            
            for strategy in strategies:
                start_time = time.time()
                resized = self.gpu_resizer.resize_to_fit(
                    frame, target_width, target_height, strategy
                )
                resize_time = (time.time() - start_time) * 1000
                
                # 결과 검증
                result_height, result_width = resized.size()
                if result_width != target_width or result_height != target_height:
                    raise ValueError(f"리사이즈 실패 ({strategy.value}): {result_width}x{result_height}")
                
                logger.info(f"✅ 리사이징 전략 ({strategy.value}): {resize_time:.2f}ms")
            
            self.test_results["gpu_resizer"]["passed"] += 1
            self.test_results["gpu_resizer"]["details"].append(
                f"리사이징 전략: {len(strategies)}개"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 리사이징 전략 테스트 실패: {e}")
            self.test_results["gpu_resizer"]["failed"] += 1
            return False
    
    def _test_batch_resize(self) -> bool:
        """배치 리사이징 테스트"""
        try:
            # 테스트 프레임들 생성
            frames = []
            for i in range(10):
                frame = self._create_test_frame(
                    640 + i * 64, 480 + i * 48, 
                    (i * 25, i * 20, i * 15)
                )
                frames.append(frame)
            
            params = ResizeParams(
                target_width=960,
                target_height=540,
                strategy=ResizeStrategy.FIT_CONTAIN
            )
            
            # 배치 리사이징
            start_time = time.time()
            resized_frames = self.gpu_resizer.resize_batch(frames, params)
            batch_time = (time.time() - start_time) * 1000
            
            # 결과 검증
            if len(resized_frames) != len(frames):
                raise ValueError(f"배치 크기 불일치: {len(resized_frames)} != {len(frames)}")
            
            for i, resized in enumerate(resized_frames):
                result_height, result_width = resized.size()
                if result_width != 960 or result_height != 540:
                    raise ValueError(f"프레임 {i} 크기 불일치: {result_width}x{result_height}")
            
            logger.info(f"✅ 배치 리사이징: {len(frames)}프레임, {batch_time:.2f}ms")
            self.test_results["gpu_resizer"]["passed"] += 1
            self.test_results["gpu_resizer"]["details"].append(
                f"배치 리사이징: {len(frames)}프레임"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 배치 리사이징 테스트 실패: {e}")
            self.test_results["gpu_resizer"]["failed"] += 1
            return False
    
    def _test_buffer_pool(self) -> bool:
        """버퍼 풀 관리 테스트"""
        try:
            # 초기 버퍼 풀 상태
            initial_info = self.gpu_resizer.get_buffer_pool_info()
            
            # 다양한 크기로 리사이징하여 버퍼 풀 사용
            sizes = [(640, 480), (800, 600), (1024, 768), (640, 480), (800, 600)]  # 중복 포함
            frame = self._create_test_frame(1280, 720, (100, 150, 200))
            
            for width, height in sizes:
                resized = self.gpu_resizer.resize_to_fit(
                    frame, width, height, ResizeStrategy.STRETCH
                )
            
            # 최종 버퍼 풀 상태
            final_info = self.gpu_resizer.get_buffer_pool_info()
            
            buffer_count = final_info.get("buffer_count", 0)
            total_memory = final_info.get("total_memory_mb", 0)
            
            logger.info(f"✅ 버퍼 풀: {buffer_count}개 버퍼, {total_memory:.1f}MB")
            self.test_results["gpu_resizer"]["passed"] += 1
            self.test_results["gpu_resizer"]["details"].append(
                f"버퍼 풀: {buffer_count}개 버퍼"
            )
            
            # 버퍼 풀 정리 테스트
            self.gpu_resizer.clear_buffer_pool()
            cleared_info = self.gpu_resizer.get_buffer_pool_info()
            
            if cleared_info.get("buffer_count", -1) != 0:
                logger.warning("⚠️ 버퍼 풀 정리 후에도 버퍼가 남아있음")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 버퍼 풀 테스트 실패: {e}")
            self.test_results["gpu_resizer"]["failed"] += 1
            return False
    
    def _test_resize_performance(self) -> bool:
        """리사이징 성능 테스트"""
        try:
            # 다양한 크기의 프레임으로 성능 측정
            test_cases = [
                (640, 480, "VGA"),
                (1280, 720, "HD"),
                (1920, 1080, "FHD"),
                (2560, 1440, "QHD")
            ]
            
            target_width, target_height = 960, 540
            iterations = 10
            
            for width, height, name in test_cases:
                frame = self._create_test_frame(width, height, (128, 64, 192))
                
                # 성능 측정
                start_time = time.time()
                for _ in range(iterations):
                    resized = self.gpu_resizer.resize_to_fit(
                        frame, target_width, target_height, ResizeStrategy.FIT_CONTAIN
                    )
                
                # 동기화
                self.gpu_resizer.synchronize()
                total_time = (time.time() - start_time) * 1000
                avg_time = total_time / iterations
                
                logger.info(f"✅ 리사이징 성능 ({name}): {avg_time:.2f}ms/frame")
            
            self.test_results["gpu_resizer"]["passed"] += 1
            self.test_results["gpu_resizer"]["details"].append(
                f"성능 측정: {len(test_cases)}개 해상도"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 리사이징 성능 테스트 실패: {e}")
            self.test_results["gpu_resizer"]["failed"] += 1
            return False
    
    def _test_error_policy(self) -> bool:
        """TileCompositionErrorPolicy 테스트"""
        try:
            logger.info("🧪 TileCompositionErrorPolicy 테스트 시작")
            
            test_success = True
            
            # 테스트 1: 에러 처리 시뮬레이션
            test_success &= self._test_error_handling_simulation()
            
            # 테스트 2: 복구 프레임 생성
            test_success &= self._test_recovery_frame_generation()
            
            # 테스트 3: 에러 통계
            test_success &= self._test_error_statistics()
            
            # 테스트 4: 연속 에러 처리
            test_success &= self._test_consecutive_errors()
            
            logger.info(f"✅ TileCompositionErrorPolicy 테스트 완료: {'성공' if test_success else '실패'}")
            return test_success
            
        except Exception as e:
            logger.error(f"❌ TileCompositionErrorPolicy 테스트 실패: {e}")
            self.test_results["error_policy"]["failed"] += 1
            return False
    
    def _test_error_handling_simulation(self) -> bool:
        """에러 처리 시뮬레이션 테스트"""
        try:
            # 다양한 에러 시뮬레이션
            test_errors = [
                ("GPU 메모리 부족", Exception("CUDA out of memory")),
                ("프레임 처리 오류", Exception("Frame processing failed")),
                ("리사이징 실패", Exception("Resize operation failed")),
                ("CUDA 오류", Exception("CUDA error occurred"))
            ]
            
            available_frame = self._create_test_frame(800, 600, (100, 100, 100))
            
            for error_name, error in test_errors:
                recovery_frame, should_continue = self.error_policy.handle_error(
                    error, 
                    frame_number=1,
                    available_frames={"fallback": available_frame}
                )
                
                # 결과 검증
                if recovery_frame.size() != (1920, 1080):
                    raise ValueError(f"복구 프레임 크기 불일치 ({error_name})")
                
                logger.info(f"✅ 에러 처리 ({error_name}): 계속 처리={should_continue}")
            
            self.test_results["error_policy"]["passed"] += 1
            self.test_results["error_policy"]["details"].append(
                f"에러 처리: {len(test_errors)}개 시나리오"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 에러 처리 시뮬레이션 실패: {e}")
            self.test_results["error_policy"]["failed"] += 1
            return False
    
    def _test_recovery_frame_generation(self) -> bool:
        """복구 프레임 생성 테스트"""
        try:
            available_frame = self._create_test_frame(640, 480, (200, 150, 100))
            
            # 단일 얼굴 실패 처리
            recovery1 = self.error_policy.handle_single_face_failure(available_frame)
            if recovery1.size() != (1920, 1080):
                raise ValueError("단일 얼굴 실패 복구 프레임 크기 불일치")
            
            # 완전 실패 처리
            recovery2 = self.error_policy.handle_complete_failure(available_frame)
            if recovery2.size() != (1920, 1080):
                raise ValueError("완전 실패 복구 프레임 크기 불일치")
            
            # 성공 프레임 업데이트
            success_frame = self.tile_composer.compose_single_frame(available_frame, "center")
            self.error_policy.update_successful_frame(success_frame)
            
            logger.info("✅ 복구 프레임 생성: 단일 실패, 완전 실패, 성공 업데이트")
            self.test_results["error_policy"]["passed"] += 1
            self.test_results["error_policy"]["details"].append("복구 프레임 생성")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 복구 프레임 생성 테스트 실패: {e}")
            self.test_results["error_policy"]["failed"] += 1
            return False
    
    def _test_error_statistics(self) -> bool:
        """에러 통계 테스트"""
        try:
            # 초기 통계
            initial_stats = self.error_policy.get_error_statistics()
            
            # 여러 에러 시뮬레이션
            for i in range(3):
                error = Exception(f"Test error {i}")
                self.error_policy.handle_error(error, frame_number=i)
            
            # 최종 통계
            final_stats = self.error_policy.get_error_statistics()
            
            total_errors = final_stats.get("total_errors", 0)
            consecutive_errors = final_stats.get("consecutive_errors", 0)
            
            if total_errors < 3:
                raise ValueError(f"에러 카운트 불일치: {total_errors} < 3")
            
            logger.info(f"✅ 에러 통계: 총 {total_errors}회, 연속 {consecutive_errors}회")
            
            # 통계 리셋 테스트
            self.error_policy.reset_error_stats()
            reset_stats = self.error_policy.get_error_statistics()
            
            if reset_stats.get("total_errors", -1) != 0:
                logger.warning("⚠️ 에러 통계 리셋 후에도 카운트가 남아있음")
            
            self.test_results["error_policy"]["passed"] += 1
            self.test_results["error_policy"]["details"].append("에러 통계")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 에러 통계 테스트 실패: {e}")
            self.test_results["error_policy"]["failed"] += 1
            return False
    
    def _test_consecutive_errors(self) -> bool:
        """연속 에러 처리 테스트"""
        try:
            # 연속 에러 시뮬레이션 (임계치 근처까지)
            max_errors = self.error_policy.max_consecutive_errors
            
            should_continue = True
            for i in range(max_errors - 1):  # 임계치 직전까지
                error = Exception(f"Consecutive error {i}")
                recovery_frame, should_continue = self.error_policy.handle_error(error, frame_number=i)
                
                if not should_continue and i < max_errors - 2:
                    raise ValueError(f"너무 일찍 처리 중단: {i}/{max_errors}")
            
            # 임계치 도달 테스트
            final_error = Exception("Final threshold error")
            recovery_frame, should_continue = self.error_policy.handle_error(final_error, frame_number=max_errors)
            
            if should_continue:
                logger.warning("⚠️ 임계치 도달 후에도 계속 처리 지시됨")
            
            stats = self.error_policy.get_error_statistics()
            consecutive_count = stats.get("consecutive_errors", 0)
            
            logger.info(f"✅ 연속 에러 처리: {consecutive_count}회, 계속 처리={should_continue}")
            self.test_results["error_policy"]["passed"] += 1
            self.test_results["error_policy"]["details"].append(f"연속 에러: {consecutive_count}회")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 연속 에러 테스트 실패: {e}")
            self.test_results["error_policy"]["failed"] += 1
            return False
    
    def _test_performance(self) -> bool:
        """성능 테스트"""
        try:
            logger.info("🧪 성능 테스트 시작")
            
            test_success = True
            
            # 테스트 1: 전체 파이프라인 성능
            test_success &= self._test_end_to_end_performance()
            
            # 테스트 2: 메모리 효율성
            test_success &= self._test_memory_efficiency()
            
            # 테스트 3: 처리량 측정
            test_success &= self._test_throughput()
            
            logger.info(f"✅ 성능 테스트 완료: {'성공' if test_success else '실패'}")
            return test_success
            
        except Exception as e:
            logger.error(f"❌ 성능 테스트 실패: {e}")
            self.test_results["performance"]["failed"] += 1
            return False
    
    def _test_end_to_end_performance(self) -> bool:
        """전체 파이프라인 성능 테스트"""
        try:
            # 테스트 프레임들
            left_frames = []
            right_frames = []
            
            for i in range(30):  # 1초 분량 (30fps)
                left_frame = self._create_test_frame(
                    800 + i * 10, 600 + i * 8, 
                    (i * 8, i * 6, i * 4)
                )
                right_frame = self._create_test_frame(
                    1024 - i * 8, 768 - i * 6,
                    (255 - i * 8, 255 - i * 6, 255 - i * 4)
                )
                
                left_frames.append(left_frame)
                right_frames.append(right_frame)
            
            # 전체 처리 시간 측정
            start_time = time.time()
            
            for i, (left, right) in enumerate(zip(left_frames, right_frames)):
                composed = self.tile_composer.compose_dual_frame(left, right)
                
                if i % 10 == 0:  # 10프레임마다 동기화
                    self.tile_composer.synchronize()
            
            total_time = (time.time() - start_time) * 1000
            avg_frame_time = total_time / len(left_frames)
            estimated_fps = 1000 / avg_frame_time
            
            logger.info(f"✅ End-to-End 성능: {avg_frame_time:.2f}ms/프레임, {estimated_fps:.1f} FPS")
            
            # 실시간 처리 가능 여부 확인 (30 FPS 기준)
            if estimated_fps >= 30:
                logger.info("🎯 실시간 처리 가능 (30+ FPS)")
            else:
                logger.warning(f"⚠️ 실시간 처리 어려움 ({estimated_fps:.1f} FPS < 30 FPS)")
            
            self.test_results["performance"]["passed"] += 1
            self.test_results["performance"]["details"].append(
                f"End-to-End: {estimated_fps:.1f} FPS"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ End-to-End 성능 테스트 실패: {e}")
            self.test_results["performance"]["failed"] += 1
            return False
    
    def _test_memory_efficiency(self) -> bool:
        """메모리 효율성 테스트"""
        try:
            # 초기 메모리 상태
            initial_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            
            # 대용량 프레임으로 스트레스 테스트
            large_frames = []
            for i in range(10):
                frame = self._create_test_frame(1920, 1080, (i * 25, i * 20, i * 15))
                large_frames.append(frame)
            
            # 연속 처리
            for i, frame in enumerate(large_frames):
                composed = self.tile_composer.compose_single_frame(frame, "center")
                
                # 중간 메모리 체크
                if i == 4:  # 중간 지점
                    mid_memory = torch.cuda.memory_allocated(0) / (1024**2)
                    memory_increase = mid_memory - initial_memory
                    
                    logger.info(f"🔧 메모리 사용량 증가: {memory_increase:.1f}MB")
            
            # 최종 메모리 상태
            final_memory = torch.cuda.memory_allocated(0) / (1024**2)
            total_increase = final_memory - initial_memory
            
            # 메모리 효율성 검증 (임계치: 1GB)
            efficiency_threshold = 1024  # MB
            if total_increase > efficiency_threshold:
                logger.warning(f"⚠️ 메모리 사용량이 많음: {total_increase:.1f}MB > {efficiency_threshold}MB")
            else:
                logger.info(f"✅ 메모리 효율적: {total_increase:.1f}MB < {efficiency_threshold}MB")
            
            self.test_results["performance"]["passed"] += 1
            self.test_results["performance"]["details"].append(
                f"메모리 효율성: {total_increase:.1f}MB 증가"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 메모리 효율성 테스트 실패: {e}")
            self.test_results["performance"]["failed"] += 1
            return False
    
    def _test_throughput(self) -> bool:
        """처리량 테스트"""
        try:
            # 다양한 배치 크기로 처리량 측정
            batch_sizes = [1, 5, 10, 15, 20]
            
            for batch_size in batch_sizes:
                # 배치 생성
                frames = []
                for i in range(batch_size):
                    frame = self._create_test_frame(1280, 720, (i * 10, i * 8, i * 6))
                    frames.append(frame)
                
                # 배치 처리 시간 측정
                start_time = time.time()
                
                for frame in frames:
                    composed = self.tile_composer.compose_single_frame(frame, "center")
                
                self.tile_composer.synchronize()
                batch_time = (time.time() - start_time) * 1000
                
                throughput = batch_size / (batch_time / 1000)  # frames/sec
                avg_time = batch_time / batch_size
                
                logger.info(f"✅ 배치 크기 {batch_size}: {throughput:.1f} FPS, {avg_time:.2f}ms/프레임")
            
            self.test_results["performance"]["passed"] += 1
            self.test_results["performance"]["details"].append(
                f"처리량 테스트: {len(batch_sizes)}개 배치 크기"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 처리량 테스트 실패: {e}")
            self.test_results["performance"]["failed"] += 1
            return False
    
    def _test_integration(self) -> bool:
        """통합 테스트"""
        try:
            logger.info("🧪 통합 테스트 시작")
            
            test_success = True
            
            # 실제 사용 시나리오 시뮬레이션
            test_success &= self._test_real_world_scenario()
            
            logger.info(f"✅ 통합 테스트 완료: {'성공' if test_success else '실패'}")
            return test_success
            
        except Exception as e:
            logger.error(f"❌ 통합 테스트 실패: {e}")
            self.test_results["integration"]["failed"] += 1
            return False
    
    def _test_real_world_scenario(self) -> bool:
        """실제 사용 시나리오 테스트"""
        try:
            # 실제 듀얼 페이스 비디오 처리 시뮬레이션
            frame_count = 60  # 2초 분량 (30fps)
            
            scenarios = [
                "both_faces",    # 양쪽 얼굴 모두 있음
                "left_only",     # 좌측만
                "right_only",    # 우측만
                "no_faces",      # 얼굴 없음
                "error_recovery" # 에러 발생
            ]
            
            total_processing_time = 0
            successful_frames = 0
            
            for frame_idx in range(frame_count):
                scenario = scenarios[frame_idx % len(scenarios)]
                
                start_time = time.time()
                
                try:
                    if scenario == "both_faces":
                        # 정상 듀얼 프레임 처리
                        left_frame = self._create_test_frame(800, 600, (255, 100, 100))
                        right_frame = self._create_test_frame(1024, 768, (100, 255, 100))
                        composed = self.tile_composer.compose_dual_frame(left_frame, right_frame)
                        
                    elif scenario == "left_only":
                        # 좌측만 있는 경우
                        left_frame = self._create_test_frame(800, 600, (255, 100, 100))
                        composed = self.tile_composer.compose_single_frame(left_frame, "left")
                        
                    elif scenario == "right_only":
                        # 우측만 있는 경우
                        right_frame = self._create_test_frame(1024, 768, (100, 255, 100))
                        composed = self.tile_composer.compose_single_frame(right_frame, "right")
                        
                    elif scenario == "no_faces":
                        # 얼굴이 없는 경우 (에러 정책 사용)
                        composed = self.error_policy.handle_complete_failure()
                        
                    elif scenario == "error_recovery":
                        # 에러 발생 시뮬레이션
                        error = Exception("Simulated processing error")
                        composed, should_continue = self.error_policy.handle_error(
                            error, frame_number=frame_idx
                        )
                        
                        if not should_continue:
                            logger.warning(f"⚠️ 프레임 {frame_idx}에서 처리 중단 지시")
                    
                    # 성공한 경우
                    if composed is not None and composed.size() == (1920, 1080):
                        successful_frames += 1
                        # 성공 프레임을 에러 정책에 업데이트
                        self.error_policy.update_successful_frame(composed)
                    
                    frame_time = (time.time() - start_time) * 1000
                    total_processing_time += frame_time
                    
                    if frame_idx % 15 == 0:  # 0.5초마다 로그
                        logger.info(f"🔧 진행 상황: {frame_idx}/{frame_count} 프레임, "
                                  f"성공률: {successful_frames/(frame_idx+1)*100:.1f}%")
                
                except Exception as e:
                    logger.warning(f"⚠️ 프레임 {frame_idx} 처리 실패 ({scenario}): {e}")
                    # 에러 정책을 통한 복구
                    try:
                        composed, _ = self.error_policy.handle_error(e, frame_number=frame_idx)
                        if composed is not None:
                            successful_frames += 1
                    except Exception:
                        pass  # 복구도 실패하면 넘어감
            
            # 최종 동기화
            self.tile_composer.synchronize()
            
            # 결과 분석
            avg_frame_time = total_processing_time / frame_count
            success_rate = successful_frames / frame_count * 100
            estimated_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            
            logger.info("=" * 60)
            logger.info("📊 실제 시나리오 테스트 결과")
            logger.info("=" * 60)
            logger.info(f"🎬 총 프레임 수: {frame_count}")
            logger.info(f"✅ 성공 프레임: {successful_frames}")
            logger.info(f"📈 성공률: {success_rate:.1f}%")
            logger.info(f"⏱️ 평균 처리 시간: {avg_frame_time:.2f}ms/프레임")
            logger.info(f"🎯 예상 FPS: {estimated_fps:.1f}")
            logger.info(f"🚀 실시간 처리: {'가능' if estimated_fps >= 30 else '어려움'}")
            
            # 에러 통계
            error_stats = self.error_policy.get_error_statistics()
            logger.info(f"⚠️ 총 에러: {error_stats.get('total_errors', 0)}회")
            logger.info(f"🔄 연속 에러: {error_stats.get('consecutive_errors', 0)}회")
            logger.info("=" * 60)
            
            # 성공 기준 (성공률 80% 이상, FPS 20 이상)
            if success_rate >= 80 and estimated_fps >= 20:
                logger.info("🎉 실제 시나리오 테스트 성공!")
                self.test_results["integration"]["passed"] += 1
                self.test_results["integration"]["details"].append(
                    f"실제 시나리오: {success_rate:.1f}% 성공률, {estimated_fps:.1f} FPS"
                )
                return True
            else:
                logger.warning(f"⚠️ 실제 시나리오 테스트 기준 미달: "
                             f"성공률 {success_rate:.1f}% < 80% 또는 FPS {estimated_fps:.1f} < 20")
                self.test_results["integration"]["failed"] += 1
                return False
            
        except Exception as e:
            logger.error(f"❌ 실제 시나리오 테스트 실패: {e}")
            self.test_results["integration"]["failed"] += 1
            return False
    
    def _create_test_frame(self, width: int, height: int, color: Tuple[int, int, int]) -> np.ndarray:
        """테스트용 프레임 생성"""
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        
        # 간단한 패턴 추가 (시각적 구분용)
        cv2.rectangle(frame, (10, 10), (width-10, height-10), (255, 255, 255), 2)
        cv2.putText(frame, f"{width}x{height}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def _print_test_results(self, overall_success: bool):
        """테스트 결과 출력"""
        logger.info("=" * 80)
        logger.info("📊 GPU Composition 시스템 테스트 결과")
        logger.info("=" * 80)
        
        total_passed = 0
        total_failed = 0
        
        for component, results in self.test_results.items():
            passed = results["passed"]
            failed = results["failed"]
            details = results["details"]
            
            total_passed += passed
            total_failed += failed
            
            status = "✅" if failed == 0 else "⚠️" if passed > failed else "❌"
            logger.info(f"{status} {component.upper()}: {passed}개 성공, {failed}개 실패")
            
            for detail in details:
                logger.info(f"   • {detail}")
        
        logger.info("=" * 80)
        logger.info(f"📈 전체 결과: {total_passed}개 성공, {total_failed}개 실패")
        logger.info(f"🎯 성공률: {total_passed/(total_passed+total_failed)*100:.1f}%")
        
        if overall_success:
            logger.info("🎉 GPU Composition 시스템 테스트 전체 성공!")
        else:
            logger.error("❌ GPU Composition 시스템 테스트 실패")
        
        logger.info("=" * 80)
    
    def _cleanup(self):
        """리소스 정리"""
        try:
            if self.tile_composer:
                self.tile_composer.cleanup()
            
            if self.gpu_resizer:
                self.gpu_resizer.cleanup()
            
            if self.error_policy:
                self.error_policy.cleanup()
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🔧 테스터 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 테스터 리소스 정리 실패: {e}")


def main():
    """메인 함수"""
    try:
        tester = GPUCompositionTester()
        success = tester.run_all_tests()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("⚠️ 사용자에 의해 테스트 중단")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()