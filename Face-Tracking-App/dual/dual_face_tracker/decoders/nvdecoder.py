"""
NvDecoder - PyAV 기반 NVDEC 하드웨어 가속 비디오 디코더

PyAV hwaccel=cuda를 활용한 NVIDIA NVDEC 하드웨어 디코딩을 구현합니다.
1080p 영상 디코딩을 위한 최적화된 인터페이스를 제공합니다.
"""

import logging
from typing import Optional, Generator, Tuple, Any, Dict
from pathlib import Path
import av
from av import VideoFrame
import torch

from ..utils.exceptions import DecodingError, GPUMemoryError
from ..utils.cuda_utils import check_cuda_available, monitor_gpu_memory
from .converter import SurfaceConverter


class NvDecoder:
    """
    PyAV 기반 NVDEC 하드웨어 가속 비디오 디코더
    
    Features:
    - NVIDIA NVDEC 하드웨어 가속 디코딩
    - NV12 → RGB 색공간 변환
    - GPU 메모리 사용량 모니터링
    - 에러 처리 및 복구 메커니즘
    """
    
    def __init__(self, 
                 video_path: str,
                 gpu_id: int = 0,
                 hwaccel: str = "cuda",
                 output_format: str = "rgb24"):
        """
        NvDecoder 초기화
        
        Args:
            video_path: 디코딩할 비디오 파일 경로
            gpu_id: 사용할 GPU ID (기본값: 0)
            hwaccel: 하드웨어 가속 방식 ("cuda", "auto" 등)
            output_format: 출력 픽셀 포맷 (기본값: "rgb24")
            
        Raises:
            DecodingError: 디코더 초기화 실패시
        """
        self.video_path = Path(video_path)
        self.gpu_id = gpu_id
        self.hwaccel = hwaccel
        self.output_format = output_format
        
        self.container: Optional[av.InputContainer] = None
        self.video_stream: Optional[av.Stream] = None
        self.surface_converter: Optional[SurfaceConverter] = None
        
        self.logger = logging.getLogger(__name__)
        
        # 비디오 정보
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None
        self.total_frames: Optional[int] = None
        self.duration: Optional[float] = None
        
        self._validate_environment()
        self._initialize_decoder()
        
    def _validate_environment(self) -> None:
        """환경 검증 - CUDA 및 파일 존재 확인"""
        if not check_cuda_available():
            raise DecodingError("CUDA가 사용 불가능합니다")
            
        if not self.video_path.exists():
            raise DecodingError(f"비디오 파일이 존재하지 않습니다: {self.video_path}")
            
        # GPU 설정
        if torch.cuda.device_count() <= self.gpu_id:
            raise DecodingError(f"GPU {self.gpu_id}가 존재하지 않습니다")
            
        torch.cuda.set_device(self.gpu_id)
        self.logger.info(f"🎯 GPU {self.gpu_id} 설정 완료")
        
    def _initialize_decoder(self) -> None:
        """
        디코더 초기화 - PyAV 컨테이너 열기 및 하드웨어 가속 설정
        
        Raises:
            DecodingError: 디코더 초기화 실패시
        """
        try:
            self.logger.info(f"🎬 비디오 파일 열기: {self.video_path}")
            
            # PyAV 컨테이너 열기 (하드웨어 가속 옵션 포함)
            options = {
                'hwaccel': self.hwaccel,
                'hwaccel_device': str(self.gpu_id)
            }
            
            self.container = av.open(str(self.video_path), options=options)
            
            # 비디오 스트림 선택
            video_streams = [s for s in self.container.streams if s.type == 'video']
            if not video_streams:
                raise DecodingError("비디오 스트림을 찾을 수 없습니다")
                
            self.video_stream = video_streams[0]
            
            # 하드웨어 디코딩 설정 시도
            try:
                self.video_stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
                self.logger.info("✅ 멀티스레딩 설정 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 멀티스레딩 설정 실패: {e}")
            
            # 비디오 정보 추출
            self._extract_video_info()
            
            # Surface Converter 초기화
            self.surface_converter = SurfaceConverter(
                source_format="nv12",  # NVDEC 기본 출력 포맷
                target_format=self.output_format,
                width=self.width,
                height=self.height
            )
            
            self.logger.info(f"✅ 디코더 초기화 완료 - {self.width}x{self.height} @ {self.fps:.2f}fps")
            
        except Exception as e:
            self._cleanup()
            raise DecodingError(f"디코더 초기화 실패: {e}")
            
    def _extract_video_info(self) -> None:
        """비디오 스트림에서 메타데이터 정보 추출"""
        if not self.video_stream:
            return
            
        codec_context = self.video_stream.codec_context
        
        self.width = codec_context.width
        self.height = codec_context.height
        
        # FPS 계산
        if self.video_stream.average_rate is not None:
            self.fps = float(self.video_stream.average_rate)
        elif self.video_stream.base_rate is not None:
            self.fps = float(self.video_stream.base_rate)
        else:
            self.fps = 30.0  # 기본값
            
        # 총 프레임 수 및 지속시간
        if self.video_stream.frames > 0:
            self.total_frames = self.video_stream.frames
        else:
            # duration으로부터 추정
            if self.video_stream.duration is not None:
                duration_seconds = float(self.video_stream.duration * self.video_stream.time_base)
                self.total_frames = int(duration_seconds * self.fps)
                self.duration = duration_seconds
                
        self.logger.debug(f"🔍 비디오 정보: {self.width}x{self.height}, {self.fps}fps, {self.total_frames}프레임")
        
    def decode_frames(self, 
                     max_frames: Optional[int] = None,
                     start_time: Optional[float] = None) -> Generator[VideoFrame, None, None]:
        """
        비디오 프레임 디코딩 제너레이터
        
        Args:
            max_frames: 최대 디코딩할 프레임 수 (None이면 전체)
            start_time: 시작 시간 (초 단위, None이면 처음부터)
            
        Yields:
            VideoFrame: 디코딩된 비디오 프레임 (RGB 포맷)
            
        Raises:
            DecodingError: 디코딩 중 오류 발생시
        """
        if not self.container or not self.video_stream:
            raise DecodingError("디코더가 초기화되지 않았습니다")
            
        try:
            # 시작 위치 탐색
            if start_time is not None:
                self._seek_to_time(start_time)
                
            frame_count = 0
            monitor_gpu_memory("Frame Decoding Start")
            
            for packet in self.container.demux(self.video_stream):
                if packet.is_corrupt:
                    self.logger.warning("⚠️ 손상된 패킷 건너뛰기")
                    continue
                    
                try:
                    # 패킷 디코딩
                    frames = packet.decode()
                    
                    for frame in frames:
                        if max_frames and frame_count >= max_frames:
                            return
                            
                        # 색공간 변환 (NV12 → RGB)
                        converted_frame = self._convert_frame(frame)
                        
                        yield converted_frame
                        frame_count += 1
                        
                        # 주기적으로 GPU 메모리 모니터링
                        if frame_count % 100 == 0:
                            monitor_gpu_memory(f"Frame {frame_count}")
                            
                except av.InvalidDataError as e:
                    self.logger.warning(f"⚠️ 프레임 디코딩 실패: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"❌ 예상치 못한 디코딩 오류: {e}")
                    raise DecodingError(f"프레임 디코딩 중 오류: {e}")
                    
            # 남은 프레임 flush
            try:
                remaining_frames = self.video_stream.codec_context.decode()
                for frame in remaining_frames:
                    if max_frames and frame_count >= max_frames:
                        break
                    converted_frame = self._convert_frame(frame)
                    yield converted_frame
                    frame_count += 1
            except Exception as e:
                self.logger.debug(f"🔧 플러시 중 오류 (무시): {e}")
                
            self.logger.info(f"✅ 총 {frame_count}개 프레임 디코딩 완료")
            
        except Exception as e:
            raise DecodingError(f"프레임 디코딩 실패: {e}")
            
    def _seek_to_time(self, time_seconds: float) -> None:
        """
        지정된 시간으로 탐색
        
        Args:
            time_seconds: 탐색할 시간 (초 단위)
        """
        try:
            # PyAV seek 사용
            seek_target = int(time_seconds / self.video_stream.time_base)
            self.container.seek(seek_target, stream=self.video_stream)
            self.logger.debug(f"🔍 {time_seconds}초 위치로 탐색 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 탐색 실패: {e}")
            
    def _convert_frame(self, frame: VideoFrame) -> VideoFrame:
        """
        프레임 색공간 변환 (NV12 → RGB)
        
        Args:
            frame: 원본 프레임 (NV12 포맷)
            
        Returns:
            VideoFrame: 변환된 프레임 (RGB 포맷)
        """
        if not self.surface_converter:
            # Fallback: PyAV 내장 변환 사용
            return frame.reformat(format=self.output_format)
            
        return self.surface_converter.convert(frame)
        
    def get_video_info(self) -> Dict[str, Any]:
        """
        비디오 정보 반환
        
        Returns:
            Dict[str, Any]: 비디오 메타데이터
        """
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'codec': self.video_stream.codec_context.name if self.video_stream else None,
            'pixel_format': self.video_stream.codec_context.pix_fmt if self.video_stream else None
        }
        
    def test_single_frame_decode(self) -> bool:
        """
        단일 프레임 디코딩 테스트
        
        Returns:
            bool: 디코딩 성공 여부
        """
        try:
            self.logger.info("🧪 단일 프레임 디코딩 테스트 시작")
            
            for frame in self.decode_frames(max_frames=1):
                self.logger.info(f"✅ 테스트 성공 - 프레임 크기: {frame.width}x{frame.height}")
                return True
                
            self.logger.warning("⚠️ 프레임을 디코딩할 수 없음")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 단일 프레임 테스트 실패: {e}")
            return False
            
    def close(self) -> None:
        """디코더 닫기 (공개 메소드)"""
        self._cleanup()
        
    def _cleanup(self) -> None:
        """리소스 정리"""
        try:
            if self.container:
                self.container.close()
                self.container = None
                
            self.video_stream = None
            self.surface_converter = None
            
            self.logger.debug("🧹 디코더 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")
            
    def __enter__(self):
        """Context manager 진입"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self._cleanup()
        
    def __del__(self):
        """소멸자"""
        self._cleanup()