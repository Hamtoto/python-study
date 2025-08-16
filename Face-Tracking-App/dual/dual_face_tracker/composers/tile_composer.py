"""
CUDA 기반 듀얼 페이스 타일 합성기

실시간 처리를 위한 GPU 가속 스플릿 스크린 타일 합성 시스템.
좌우 960px씩 분할된 1920x1080 출력을 생성합니다.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
import torch

from ..utils.logger import get_logger
from ..utils.exceptions import CompositionError, GPUMemoryError
from ..utils.cuda_utils import check_cuda_memory, ensure_gpu_mat, safe_upload_to_gpu

logger = get_logger(__name__)


class TileComposer:
    """
    CUDA 기반 듀얼 페이스 스플릿 스크린 타일 합성기
    
    Features:
    - GPU 가속 리사이징 및 합성
    - 1920x1080 출력 (좌우 960px 분할)
    - 얼굴 중앙 정렬 및 자동 크롭
    - CUDA 스트림 지원
    - 메모리 효율적 버퍼 재사용
    """
    
    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1080,
        interpolation: int = cv2.INTER_LINEAR,
        use_cuda_stream: bool = True
    ):
        """
        타일 합성기 초기화
        
        Args:
            output_width: 출력 비디오 너비 (기본: 1920)
            output_height: 출력 비디오 높이 (기본: 1080)
            interpolation: 리사이징 보간법 (기본: INTER_LINEAR)
            use_cuda_stream: CUDA 스트림 사용 여부
        """
        self.output_width = output_width
        self.output_height = output_height
        self.interpolation = interpolation
        self.use_cuda_stream = use_cuda_stream
        
        # 분할 영역 설정
        self.left_width = output_width // 2  # 960px
        self.right_width = output_width // 2  # 960px
        self.tile_height = output_height  # 1080px
        
        # GPU 버퍼 미리 할당 (메모리 효율성)
        self._output_buffer = None
        self._left_buffer = None
        self._right_buffer = None
        self._temp_buffer = None
        
        # CUDA 스트림 초기화
        if self.use_cuda_stream:
            try:
                self.stream = cv2.cuda.Stream()
                logger.info("🔧 CUDA 스트림 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ CUDA 스트림 초기화 실패, CPU 모드로 전환: {e}")
                self.use_cuda_stream = False
                self.stream = None
        else:
            self.stream = None
            
        logger.info(f"🎨 TileComposer 초기화 완료 - {output_width}x{output_height}")
        logger.info(f"   • 분할 영역: 좌({self.left_width}px) + 우({self.right_width}px)")
        logger.info(f"   • CUDA 스트림: {'활성화' if self.use_cuda_stream else '비활성화'}")
        
    def compose_dual_frame(
        self, 
        left_frame: Union[np.ndarray, cv2.cuda.GpuMat], 
        right_frame: Union[np.ndarray, cv2.cuda.GpuMat]
    ) -> cv2.cuda.GpuMat:
        """
        좌우 프레임을 스플릿 스크린으로 합성
        
        Args:
            left_frame: 좌측 프레임 (CPU Mat 또는 GPU GpuMat)
            right_frame: 우측 프레임 (CPU Mat 또는 GPU GpuMat)
            
        Returns:
            cv2.cuda.GpuMat: 합성된 1920x1080 프레임
            
        Raises:
            CompositionError: 합성 실패 시
            GPUMemoryError: GPU 메모리 부족 시
        """
        try:
            # GPU 메모리 상태 확인
            check_cuda_memory()
            
            # 출력 버퍼 초기화 (필요시)
            self._ensure_output_buffer()
            
            # 프레임을 GPU로 변환 (필요시)
            gpu_left = ensure_gpu_mat(left_frame)
            gpu_right = ensure_gpu_mat(right_frame)
            
            # 좌측 영역 처리
            left_resized = self._resize_and_center(
                gpu_left, 
                self.left_width, 
                self.tile_height,
                target_region="left"
            )
            
            # 우측 영역 처리  
            right_resized = self._resize_and_center(
                gpu_right,
                self.right_width,
                self.tile_height, 
                target_region="right"
            )
            
            # 타일 합성
            composed_frame = self._combine_tiles(left_resized, right_resized)
            
            logger.debug(f"🔧 타일 합성 완료: {composed_frame.size()}")
            return composed_frame
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU 메모리 부족으로 타일 합성 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise GPUMemoryError(error_msg) from e
            
        except Exception as e:
            error_msg = f"타일 합성 중 오류 발생: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def compose_single_frame(
        self,
        frame: Union[np.ndarray, cv2.cuda.GpuMat],
        position: str = "center"
    ) -> cv2.cuda.GpuMat:
        """
        단일 프레임을 중앙 또는 지정 위치에 배치
        
        Args:
            frame: 입력 프레임
            position: 배치 위치 ("center", "left", "right")
            
        Returns:
            cv2.cuda.GpuMat: 합성된 프레임
        """
        try:
            check_cuda_memory()
            self._ensure_output_buffer()
            
            gpu_frame = ensure_gpu_mat(frame)
            
            if position == "center":
                # 전체 영역에 중앙 배치
                resized = self._resize_and_center(
                    gpu_frame,
                    self.output_width,
                    self.output_height,
                    target_region="center"
                )
                return resized
                
            elif position == "left":
                # 좌측 영역에 배치, 우측은 검은색
                left_resized = self._resize_and_center(
                    gpu_frame,
                    self.left_width,
                    self.tile_height,
                    target_region="left"
                )
                return self._combine_with_black(left_resized, "left")
                
            elif position == "right":
                # 우측 영역에 배치, 좌측은 검은색
                right_resized = self._resize_and_center(
                    gpu_frame,
                    self.right_width,
                    self.tile_height,
                    target_region="right"
                )
                return self._combine_with_black(right_resized, "right")
                
            else:
                raise ValueError(f"지원하지 않는 position: {position}")
                
        except Exception as e:
            error_msg = f"단일 프레임 합성 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _resize_and_center(
        self, 
        gpu_frame: cv2.cuda.GpuMat, 
        target_width: int, 
        target_height: int,
        target_region: str = "unknown"
    ) -> cv2.cuda.GpuMat:
        """
        프레임을 목표 크기로 리사이즈하고 중앙 정렬
        
        Args:
            gpu_frame: 입력 GPU 프레임
            target_width: 목표 너비
            target_height: 목표 높이
            target_region: 목표 영역 ("left", "right", "center")
            
        Returns:
            cv2.cuda.GpuMat: 리사이즈된 프레임
        """
        try:
            current_width, current_height = gpu_frame.size()
            
            # 종횡비 유지하면서 리사이즈
            scale = min(target_width / current_width, target_height / current_height)
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # GPU 리사이즈 수행
            if self.use_cuda_stream and self.stream:
                resized = cv2.cuda.resize(gpu_frame, (new_width, new_height), 
                                        interpolation=self.interpolation, stream=self.stream)
            else:
                resized = cv2.cuda.resize(gpu_frame, (new_width, new_height), 
                                        interpolation=self.interpolation)
            
            # 중앙 정렬을 위한 패딩 계산
            pad_x = (target_width - new_width) // 2
            pad_y = (target_height - new_height) // 2
            
            if pad_x > 0 or pad_y > 0:
                # 패딩 추가하여 중앙 정렬
                centered = self._add_centered_padding(
                    resized, target_width, target_height, pad_x, pad_y
                )
            else:
                centered = resized
                
            logger.debug(f"🔧 리사이즈 완료 ({target_region}): "
                        f"{current_width}x{current_height} → {target_width}x{target_height}")
            
            return centered
            
        except Exception as e:
            error_msg = f"프레임 리사이즈 실패 ({target_region}): {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _add_centered_padding(
        self,
        gpu_frame: cv2.cuda.GpuMat,
        target_width: int,
        target_height: int,
        pad_x: int,
        pad_y: int
    ) -> cv2.cuda.GpuMat:
        """
        프레임에 중앙 정렬을 위한 패딩 추가
        
        Args:
            gpu_frame: 입력 GPU 프레임
            target_width: 목표 너비
            target_height: 목표 높이
            pad_x: X축 패딩
            pad_y: Y축 패딩
            
        Returns:
            cv2.cuda.GpuMat: 패딩된 프레임
        """
        try:
            # 목표 크기의 검은 캔버스 생성
            padded = cv2.cuda.GpuMat(target_height, target_width, gpu_frame.type())
            padded.setTo((0, 0, 0))  # 검은색으로 초기화
            
            # 원본 프레임의 크기 (OpenCV size()는 (width, height) 순서!)
            frame_width, frame_height = gpu_frame.size()
            
            # OpenCV 4.13에서는 ROI를 다르게 처리
            # GPU에서 직접 복사하는 방식 사용
            temp_cpu = gpu_frame.download()
            padded_cpu = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            # temp_cpu는 (height, width, channels) = (frame_height, frame_width, 3)
            padded_cpu[pad_y:pad_y + frame_height, pad_x:pad_x + frame_width] = temp_cpu
            
            if not safe_upload_to_gpu(padded, padded_cpu):
                raise CompositionError("Tile padding upload failed")
                
            return padded
            
        except Exception as e:
            error_msg = f"패딩 추가 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _combine_tiles(
        self,
        left_tile: cv2.cuda.GpuMat,
        right_tile: cv2.cuda.GpuMat
    ) -> cv2.cuda.GpuMat:
        """
        좌우 타일을 수평으로 결합
        
        Args:
            left_tile: 좌측 타일
            right_tile: 우측 타일
            
        Returns:
            cv2.cuda.GpuMat: 결합된 프레임
        """
        try:
            # OpenCV 4.13 호환성: CPU에서 결합 후 GPU로 업로드
            left_cpu = left_tile.download()
            right_cpu = right_tile.download()
            
            # CPU에서 결합
            combined_cpu = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            combined_cpu[0:self.tile_height, 0:self.left_width] = left_cpu
            combined_cpu[0:self.tile_height, self.left_width:self.output_width] = right_cpu
            
            # GPU로 업로드
            if not safe_upload_to_gpu(self._output_buffer, combined_cpu):
                raise CompositionError("Combined tiles upload failed")
            
            return self._output_buffer
            
        except Exception as e:
            error_msg = f"타일 결합 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _combine_with_black(
        self,
        tile: cv2.cuda.GpuMat,
        position: str
    ) -> cv2.cuda.GpuMat:
        """
        타일을 검은 배경과 결합
        
        Args:
            tile: 입력 타일
            position: 배치 위치 ("left" 또는 "right")
            
        Returns:
            cv2.cuda.GpuMat: 결합된 프레임
        """
        try:
            # OpenCV 4.13 호환성: CPU에서 결합 후 GPU로 업로드
            tile_cpu = tile.download()
            
            # 검은 배경 생성
            combined_cpu = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            
            if position == "left":
                combined_cpu[0:self.tile_height, 0:self.left_width] = tile_cpu
            elif position == "right":
                combined_cpu[0:self.tile_height, self.left_width:self.output_width] = tile_cpu
            else:
                raise ValueError(f"지원하지 않는 position: {position}")
            
            # GPU로 업로드
            if not safe_upload_to_gpu(self._output_buffer, combined_cpu):
                raise CompositionError("Combined tiles upload failed")
                
            return self._output_buffer
            
        except Exception as e:
            error_msg = f"검은 배경 결합 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _ensure_output_buffer(self):
        """출력 버퍼가 할당되어 있는지 확인하고 필요시 생성"""
        if self._output_buffer is None:
            try:
                # BGR 3채널 버퍼 생성
                self._output_buffer = cv2.cuda.GpuMat(
                    self.output_height, 
                    self.output_width, 
                    cv2.CV_8UC3
                )
                self._output_buffer.setTo((0, 0, 0))  # 검은색으로 초기화
                
                logger.info(f"🔧 GPU 출력 버퍼 할당: {self.output_width}x{self.output_height}")
                
            except Exception as e:
                error_msg = f"GPU 버퍼 할당 실패: {e}"
                logger.error(f"❌ {error_msg}")
                raise GPUMemoryError(error_msg) from e
    
    def synchronize(self):
        """CUDA 스트림 동기화 (비동기 작업 완료 대기)"""
        if self.use_cuda_stream and self.stream:
            try:
                self.stream.waitForCompletion()
                logger.debug("🔧 CUDA 스트림 동기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ CUDA 스트림 동기화 실패: {e}")
    
    def get_memory_usage(self) -> dict:
        """
        현재 메모리 사용량 정보 반환
        
        Returns:
            dict: 메모리 사용량 정보
        """
        try:
            gpu_info = torch.cuda.get_device_properties(0)
            used_memory = torch.cuda.memory_allocated(0)
            total_memory = gpu_info.total_memory
            
            buffer_size = 0
            if self._output_buffer is not None:
                buffer_size = (self.output_width * self.output_height * 3)  # BGR
            
            return {
                "total_vram_gb": total_memory / (1024**3),
                "used_vram_gb": used_memory / (1024**3),
                "buffer_size_mb": buffer_size / (1024**2),
                "utilization_percent": (used_memory / total_memory) * 100
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 메모리 정보 조회 실패: {e}")
            return {}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self._output_buffer is not None:
                # GpuMat은 자동으로 해제되지만 명시적으로 정리
                self._output_buffer = None
                
            if self._left_buffer is not None:
                self._left_buffer = None
                
            if self._right_buffer is not None:
                self._right_buffer = None
                
            if self._temp_buffer is not None:
                self._temp_buffer = None
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("🔧 TileComposer 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except Exception:
            pass  # 소멸자에서는 예외를 발생시키지 않음