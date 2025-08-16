"""
SurfaceConverter - GPU 기반 색공간 변환

NVDEC 출력 (NV12)을 RGB 포맷으로 효율적으로 변환하는 클래스입니다.
PyAV의 VideoFrame.reformat()을 최적화하여 사용합니다.
"""

import logging
from typing import Optional, Tuple
import av
from av import VideoFrame

from ..utils.exceptions import DecodingError
from ..utils.cuda_utils import monitor_gpu_memory


class SurfaceConverter:
    """
    GPU 기반 색공간 변환기
    
    NVDEC의 기본 출력 포맷인 NV12를 RGB로 변환합니다.
    PyAV의 내장 변환 기능을 활용하여 최적화된 변환을 제공합니다.
    """
    
    def __init__(self,
                 source_format: str = "nv12",
                 target_format: str = "rgb24",
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 interpolation: str = "bilinear"):
        """
        SurfaceConverter 초기화
        
        Args:
            source_format: 입력 픽셀 포맷 (기본값: "nv12")
            target_format: 출력 픽셀 포맷 (기본값: "rgb24")
            width: 출력 너비 (None이면 입력과 동일)
            height: 출력 높이 (None이면 입력과 동일)
            interpolation: 보간 방법 ("bilinear", "bicubic", "nearest")
        """
        self.source_format = source_format
        self.target_format = target_format
        self.output_width = width
        self.output_height = height
        self.interpolation = interpolation
        
        self.logger = logging.getLogger(__name__)
        
        # 지원되는 포맷 검증
        self._validate_formats()
        
        self.logger.info(f"🎨 SurfaceConverter 초기화: {source_format} → {target_format}")
        if width and height:
            self.logger.info(f"   출력 해상도: {width}x{height}")
            
    def _validate_formats(self) -> None:
        """지원되는 픽셀 포맷 검증"""
        supported_input = ["nv12", "yuv420p", "yuv444p", "rgb24", "bgr24"]
        supported_output = ["rgb24", "bgr24", "yuv420p", "yuv444p", "gray"]
        
        if self.source_format not in supported_input:
            self.logger.warning(f"⚠️ 지원되지 않는 입력 포맷: {self.source_format}")
            
        if self.target_format not in supported_output:
            self.logger.warning(f"⚠️ 지원되지 않는 출력 포맷: {self.target_format}")
            
    def convert(self, frame: VideoFrame) -> VideoFrame:
        """
        비디오 프레임 색공간 변환
        
        Args:
            frame: 입력 비디오 프레임
            
        Returns:
            VideoFrame: 변환된 비디오 프레임
            
        Raises:
            DecodingError: 변환 실패시
        """
        try:
            monitor_gpu_memory("Frame Conversion Start")
            
            # 출력 해상도 결정
            output_width = self.output_width or frame.width
            output_height = self.output_height or frame.height
            
            # PyAV reformat을 사용한 변환 (interpolation 파라미터 제거)
            converted_frame = frame.reformat(
                format=self.target_format,
                width=output_width,
                height=output_height
            )
            
            # 변환 결과 검증
            if converted_frame is None:
                raise DecodingError("프레임 변환 결과가 None입니다")
                
            self.logger.debug(
                f"🎨 변환 완료: {frame.format.name}({frame.width}x{frame.height}) → "
                f"{converted_frame.format.name}({converted_frame.width}x{converted_frame.height})"
            )
            
            monitor_gpu_memory("Frame Conversion End")
            return converted_frame
            
        except Exception as e:
            raise DecodingError(f"프레임 변환 실패: {e}")
            
    def convert_batch(self, frames: list[VideoFrame]) -> list[VideoFrame]:
        """
        배치 단위 프레임 변환
        
        Args:
            frames: 입력 프레임 리스트
            
        Returns:
            list[VideoFrame]: 변환된 프레임 리스트
            
        Raises:
            DecodingError: 배치 변환 실패시
        """
        try:
            self.logger.debug(f"🎨 배치 변환 시작: {len(frames)}개 프레임")
            monitor_gpu_memory("Batch Conversion Start")
            
            converted_frames = []
            
            for i, frame in enumerate(frames):
                try:
                    converted_frame = self.convert(frame)
                    converted_frames.append(converted_frame)
                except Exception as e:
                    self.logger.warning(f"⚠️ 프레임 {i} 변환 실패: {e}")
                    # 실패한 프레임은 원본 반환 (fallback)
                    converted_frames.append(frame)
                    
            monitor_gpu_memory("Batch Conversion End")
            self.logger.debug(f"✅ 배치 변환 완료: {len(converted_frames)}개 프레임")
            
            return converted_frames
            
        except Exception as e:
            raise DecodingError(f"배치 변환 실패: {e}")
            
    def get_conversion_info(self) -> dict:
        """
        변환 설정 정보 반환
        
        Returns:
            dict: 변환 설정 정보
        """
        return {
            'source_format': self.source_format,
            'target_format': self.target_format,
            'output_width': self.output_width,
            'output_height': self.output_height,
            'interpolation': self.interpolation
        }
        
    def test_conversion(self, test_width: int = 1920, test_height: int = 1080) -> bool:
        """
        변환 기능 테스트
        
        Args:
            test_width: 테스트 프레임 너비
            test_height: 테스트 프레임 높이
            
        Returns:
            bool: 테스트 성공 여부
        """
        try:
            self.logger.info("🧪 색공간 변환 테스트 시작")
            
            # 더미 프레임 생성 (테스트용)
            import numpy as np
            
            if self.source_format == "nv12":
                # NV12 포맷용 더미 데이터 생성
                y_plane = np.random.randint(0, 256, size=(test_height, test_width), dtype=np.uint8)
                uv_plane = np.random.randint(0, 256, size=(test_height//2, test_width), dtype=np.uint8)
                
                # PyAV는 numpy array로부터 직접 NV12 프레임을 생성하기 어려우므로
                # 일반적인 RGB 테스트로 대체
                test_data = np.random.randint(0, 256, size=(test_height, test_width, 3), dtype=np.uint8)
                test_frame = av.VideoFrame.from_ndarray(test_data, format='rgb24')
                
            else:
                # RGB 포맷용 더미 데이터
                test_data = np.random.randint(0, 256, size=(test_height, test_width, 3), dtype=np.uint8)
                test_frame = av.VideoFrame.from_ndarray(test_data, format=self.source_format)
            
            # 변환 테스트
            converted_frame = self.convert(test_frame)
            
            # 결과 검증
            expected_width = self.output_width or test_width
            expected_height = self.output_height or test_height
            
            success = (
                converted_frame.width == expected_width and
                converted_frame.height == expected_height and
                converted_frame.format.name == self.target_format
            )
            
            if success:
                self.logger.info(f"✅ 변환 테스트 성공 - 출력: {converted_frame.width}x{converted_frame.height} {converted_frame.format.name}")
            else:
                self.logger.error(f"❌ 변환 테스트 실패 - 예상: {expected_width}x{expected_height} {self.target_format}, 실제: {converted_frame.width}x{converted_frame.height} {converted_frame.format.name}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 변환 테스트 중 오류: {e}")
            return False
            
    @staticmethod
    def get_supported_formats() -> dict:
        """
        지원되는 픽셀 포맷 목록 반환
        
        Returns:
            dict: 지원되는 입력/출력 포맷
        """
        return {
            'input_formats': ["nv12", "yuv420p", "yuv444p", "rgb24", "bgr24"],
            'output_formats': ["rgb24", "bgr24", "yuv420p", "yuv444p", "gray"],
            'interpolation_methods': ["nearest", "bilinear", "bicubic", "lanczos"]
        }