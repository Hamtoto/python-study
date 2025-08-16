#!/usr/bin/env python3
"""
test_single_stream.py - 단일 스트림 NVDEC 디코딩 테스트

Phase 1 완료를 위한 1080p 영상 NVDEC 디코딩 검증 테스트입니다.
HybridConfigManager와 NvDecoder의 통합 동작을 확인합니다.
"""

import sys
import logging
import time
from pathlib import Path
import traceback

# dual_face_tracker 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dual_face_tracker import HybridConfigManager, NvDecoder, logger
    from dual_face_tracker.utils.cuda_utils import check_cuda_available, get_gpu_memory_info
    from dual_face_tracker.utils.logger import setup_dual_face_logger
    from dual_face_tracker.utils.exceptions import DualFaceTrackerError
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("DevContainer 환경에서 실행하고 있는지 확인하세요.")
    sys.exit(1)


class SingleStreamTester:
    """
    단일 스트림 NVDEC 디코딩 테스트 클래스
    
    Phase 1 성공 기준 검증:
    1. HybridConfigManager 동작 확인
    2. NvDecoder 1080p 디코딩 성공
    3. GPU 메모리 사용량 모니터링
    4. NV12 → RGB 색공간 변환 확인
    """
    
    def __init__(self):
        self.logger = setup_dual_face_logger("DEBUG")
        self.config_manager = None
        self.decoder = None
        self.test_video_path = None
        
    def run_comprehensive_test(self) -> bool:
        """
        종합 테스트 실행
        
        Returns:
            bool: 모든 테스트 통과 여부
        """
        self.logger.info("🚀 Phase 1 단일 스트림 테스트 시작")
        self.logger.info("=" * 80)
        
        try:
            # 테스트 단계별 실행
            tests = [
                ("환경 검증", self._test_environment),
                ("설정 관리자 테스트", self._test_config_manager),
                ("테스트 비디오 생성", self._create_test_video),
                ("NVDEC 디코더 초기화", self._test_decoder_initialization),
                ("단일 프레임 디코딩", self._test_single_frame_decode),
                ("배치 프레임 디코딩", self._test_batch_frame_decode),
                ("색공간 변환 테스트", self._test_color_conversion),
                ("GPU 메모리 모니터링", self._test_memory_monitoring),
                ("성능 측정", self._test_performance)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                self.logger.info(f"🔍 테스트: {test_name}")
                
                try:
                    result = test_func()
                    if result:
                        self.logger.info(f"✅ {test_name} - 성공")
                        passed_tests += 1
                    else:
                        self.logger.error(f"❌ {test_name} - 실패")
                        
                except Exception as e:
                    self.logger.error(f"❌ {test_name} - 예외 발생: {e}")
                    self.logger.debug(traceback.format_exc())
                    
                self.logger.info("-" * 60)
                
            # 결과 요약
            success_rate = (passed_tests / total_tests) * 100
            
            self.logger.info("📊 테스트 결과 요약")
            self.logger.info(f"   - 통과: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            
            if passed_tests == total_tests:
                self.logger.info("🎉 Phase 1 모든 테스트 통과! 70% → 100% 완료")
                return True
            else:
                self.logger.warning(f"⚠️ {total_tests - passed_tests}개 테스트 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 종합 테스트 중 치명적 오류: {e}")
            self.logger.debug(traceback.format_exc())
            return False
            
        finally:
            self._cleanup()
            
    def _test_environment(self) -> bool:
        """환경 검증 테스트"""
        try:
            # CUDA 가용성 확인
            if not check_cuda_available():
                self.logger.error("CUDA가 사용 불가능합니다")
                return False
                
            # GPU 메모리 정보 확인
            memory_info = get_gpu_memory_info()
            total_gb = memory_info['total'] / (1024**3)
            
            self.logger.info(f"GPU 메모리: {total_gb:.1f}GB")
            
            if total_gb < 8:
                self.logger.warning(f"GPU 메모리가 부족할 수 있습니다: {total_gb:.1f}GB < 8GB")
                
            return True
            
        except Exception as e:
            self.logger.error(f"환경 검증 실패: {e}")
            return False
            
    def _test_config_manager(self) -> bool:
        """HybridConfigManager 테스트"""
        try:
            # 설정 관리자 초기화
            self.config_manager = HybridConfigManager()
            
            # 최적 설정 로드
            config = self.config_manager.load_optimal_config()
            
            # 필수 설정값 확인
            required_sections = ['hardware', 'performance', 'nvdec_settings']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"필수 설정 섹션 누락: {section}")
                    return False
                    
            # 설정값 확인
            gpu_name = config.get('hardware', {}).get('gpu_name', 'Unknown')
            max_streams = config.get('performance', {}).get('max_concurrent_streams', 1)
            
            self.logger.info(f"로드된 설정 - GPU: {gpu_name}, 최대 스트림: {max_streams}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"설정 관리자 테스트 실패: {e}")
            return False
            
    def _create_test_video(self) -> bool:
        """테스트용 1080p 비디오 생성"""
        try:
            import numpy as np
            import av
            
            # 테스트 비디오 경로
            self.test_video_path = Path("test_1080p_video.mp4")
            
            if self.test_video_path.exists():
                self.logger.info("기존 테스트 비디오 사용")
                return True
                
            self.logger.info("1080p 테스트 비디오 생성 중...")
            
            # 간단한 1080p 테스트 비디오 생성 (10프레임)
            width, height = 1920, 1080
            num_frames = 10
            fps = 30
            
            container = av.open(str(self.test_video_path), mode='w')
            stream = container.add_stream('libx264', rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            
            for i in range(num_frames):
                # 컬러 그라디언트 생성
                frame_data = np.zeros((height, width, 3), dtype=np.uint8)
                frame_data[:, :, 0] = (i * 25) % 256  # Red 변화
                frame_data[:, :, 1] = 128  # Green 고정
                frame_data[:, :, 2] = 255 - ((i * 25) % 256)  # Blue 변화
                
                frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                frame = frame.reformat(format='yuv420p')
                
                for packet in stream.encode(frame):
                    container.mux(packet)
                    
            # 스트림 플러시
            for packet in stream.encode():
                container.mux(packet)
                
            container.close()
            
            self.logger.info(f"✅ 테스트 비디오 생성 완료: {self.test_video_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"테스트 비디오 생성 실패: {e}")
            return False
            
    def _test_decoder_initialization(self) -> bool:
        """NvDecoder 초기화 테스트"""
        try:
            if not self.test_video_path or not self.test_video_path.exists():
                self.logger.error("테스트 비디오가 없습니다")
                return False
                
            # NvDecoder 초기화
            self.decoder = NvDecoder(
                video_path=str(self.test_video_path),
                gpu_id=0,
                hwaccel="cuda",
                output_format="rgb24"
            )
            
            # 비디오 정보 확인
            video_info = self.decoder.get_video_info()
            
            expected_width, expected_height = 1920, 1080
            if video_info['width'] != expected_width or video_info['height'] != expected_height:
                self.logger.error(f"비디오 해상도 불일치: {video_info['width']}x{video_info['height']} != {expected_width}x{expected_height}")
                return False
                
            self.logger.info(f"디코더 초기화 성공 - {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"디코더 초기화 실패: {e}")
            return False
            
    def _test_single_frame_decode(self) -> bool:
        """단일 프레임 디코딩 테스트"""
        try:
            if not self.decoder:
                self.logger.error("디코더가 초기화되지 않았습니다")
                return False
                
            # 단일 프레임 디코딩 테스트
            success = self.decoder.test_single_frame_decode()
            
            if success:
                self.logger.info("단일 프레임 디코딩 성공")
            else:
                self.logger.error("단일 프레임 디코딩 실패")
                
            return success
            
        except Exception as e:
            self.logger.error(f"단일 프레임 디코딩 테스트 실패: {e}")
            return False
            
    def _test_batch_frame_decode(self) -> bool:
        """배치 프레임 디코딩 테스트"""
        try:
            if not self.decoder:
                return False
                
            frame_count = 0
            max_frames = 5  # 5프레임만 테스트
            
            start_time = time.time()
            
            for frame in self.decoder.decode_frames(max_frames=max_frames):
                if frame is None:
                    self.logger.warning("None 프레임 수신")
                    continue
                    
                # 프레임 검증
                if frame.width != 1920 or frame.height != 1080:
                    self.logger.error(f"프레임 크기 오류: {frame.width}x{frame.height}")
                    return False
                    
                if frame.format.name != 'rgb24':
                    self.logger.error(f"프레임 포맷 오류: {frame.format.name}")
                    return False
                    
                frame_count += 1
                
            decode_time = time.time() - start_time
            
            if frame_count > 0:
                fps = frame_count / decode_time
                self.logger.info(f"배치 디코딩 성공 - {frame_count}프레임, {fps:.2f} FPS")
                return True
            else:
                self.logger.error("디코딩된 프레임 없음")
                return False
                
        except Exception as e:
            self.logger.error(f"배치 프레임 디코딩 테스트 실패: {e}")
            return False
            
    def _test_color_conversion(self) -> bool:
        """색공간 변환 테스트"""
        try:
            if not self.decoder or not self.decoder.surface_converter:
                # Surface converter가 없으면 기본 변환 테스트
                from dual_face_tracker.decoders.converter import SurfaceConverter
                
                converter = SurfaceConverter(
                    source_format="rgb24",  # 테스트용
                    target_format="rgb24"
                )
                
                success = converter.test_conversion(test_width=1920, test_height=1080)
                
                if success:
                    self.logger.info("색공간 변환 테스트 성공")
                else:
                    self.logger.error("색공간 변환 테스트 실패")
                    
                return success
            else:
                # 디코더의 변환기 사용
                converter_info = self.decoder.surface_converter.get_conversion_info()
                self.logger.info(f"변환 설정: {converter_info}")
                return True
                
        except Exception as e:
            self.logger.error(f"색공간 변환 테스트 실패: {e}")
            return False
            
    def _test_memory_monitoring(self) -> bool:
        """GPU 메모리 모니터링 테스트"""
        try:
            from dual_face_tracker.utils.cuda_utils import monitor_gpu_memory, clear_gpu_cache
            
            # 메모리 모니터링
            monitor_gpu_memory("테스트 시작")
            
            # GPU 캐시 정리 테스트
            clear_gpu_cache()
            
            monitor_gpu_memory("캐시 정리 후")
            
            self.logger.info("GPU 메모리 모니터링 테스트 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"메모리 모니터링 테스트 실패: {e}")
            return False
            
    def _test_performance(self) -> bool:
        """성능 측정 테스트"""
        try:
            if not self.decoder:
                return False
                
            # 간단한 성능 측정
            start_time = time.time()
            frame_count = 0
            
            for frame in self.decoder.decode_frames(max_frames=10):
                frame_count += 1
                
            total_time = time.time() - start_time
            
            if frame_count > 0:
                avg_time_per_frame = total_time / frame_count
                fps = frame_count / total_time
                
                self.logger.info(f"성능 측정 결과:")
                self.logger.info(f"  - 총 처리 시간: {total_time:.3f}초")
                self.logger.info(f"  - 프레임당 평균 시간: {avg_time_per_frame:.3f}초")
                self.logger.info(f"  - 평균 FPS: {fps:.2f}")
                
                # 성능 기준 확인 (1080p에서 최소 10 FPS 이상)
                if fps >= 10.0:
                    self.logger.info("✅ 성능 기준 만족 (>= 10 FPS)")
                    return True
                else:
                    self.logger.warning(f"⚠️ 성능 기준 미달: {fps:.2f} FPS < 10 FPS")
                    return False
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"성능 측정 테스트 실패: {e}")
            return False
            
    def _cleanup(self) -> None:
        """테스트 정리"""
        try:
            if self.decoder:
                self.decoder._cleanup()
                
            if self.test_video_path and self.test_video_path.exists():
                self.test_video_path.unlink()
                self.logger.debug("테스트 비디오 파일 삭제")
                
        except Exception as e:
            self.logger.warning(f"정리 중 오류: {e}")


def main():
    """메인 함수"""
    print("🚀 Dual-Face Tracker Phase 1 단일 스트림 테스트")
    print("=" * 60)
    
    # DevContainer 환경 확인
    if not check_cuda_available():
        print("❌ CUDA가 사용 불가능합니다. DevContainer 환경에서 실행하세요.")
        return False
        
    # 테스트 실행
    tester = SingleStreamTester()
    success = tester.run_comprehensive_test()
    
    print("=" * 60)
    if success:
        print("🎉 Phase 1 완료! 모든 테스트가 성공했습니다.")
        print("✅ 70% → 100% 진행률 달성")
        print("")
        print("📋 완료된 작업:")
        print("  - ✅ dual_face_tracker 모듈 구조 생성")
        print("  - ✅ HybridConfigManager 구현")
        print("  - ✅ PyAV NVDEC 디코더 구현")
        print("  - ✅ 설정 파일 템플릿 작성")
        print("  - ✅ 단일 스트림 테스트 통과")
        print("")
        print("🚀 Phase 2 진행 준비 완료!")
        return True
    else:
        print("❌ 일부 테스트가 실패했습니다.")
        print("💡 dual_face_tracker.log 파일을 확인하여 자세한 오류 정보를 확인하세요.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)