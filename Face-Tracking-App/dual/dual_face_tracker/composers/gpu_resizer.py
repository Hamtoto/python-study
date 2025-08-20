"""
GPU 가속 이미지 리사이징 유틸리티

OpenCV CUDA를 활용한 고성능 이미지 리사이징 및 변환 기능 제공.
다양한 리사이징 전략과 GPU 메모리 최적화 지원.
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional
import torch
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger
from ..utils.exceptions import CompositionError, GPUMemoryError
from ..utils.cuda_utils import check_cuda_memory, ensure_gpu_mat, safe_upload_to_gpu

logger = get_logger(__name__)


class ResizeStrategy(Enum):
    """리사이징 전략"""
    FIT_CONTAIN = "fit_contain"    # 종횡비 유지, 전체 포함
    FIT_COVER = "fit_cover"        # 종횡비 유지, 영역 채움 (크롭 가능)
    STRETCH = "stretch"            # 종횡비 무시, 영역에 맞춤
    CENTER_CROP = "center_crop"    # 중앙 크롭 후 리사이즈


@dataclass
class ResizeParams:
    """리사이징 파라미터"""
    target_width: int
    target_height: int
    strategy: ResizeStrategy = ResizeStrategy.FIT_CONTAIN
    interpolation: int = cv2.INTER_LINEAR
    padding_color: Tuple[int, int, int] = (0, 0, 0)  # BGR
    center_crop_ratio: float = 1.0  # CENTER_CROP 시 크롭 비율


class GpuResizer:
    """
    GPU 가속 이미지 리사이징 클래스
    
    Features:
    - 다양한 리사이징 전략 지원
    - CUDA 스트림 기반 비동기 처리
    - 메모리 효율적 버퍼 관리
    - 배치 처리 지원
    - 고성능 보간 알고리즘
    """
    
    def __init__(
        self,
        default_interpolation: int = cv2.INTER_LINEAR,
        use_cuda_stream: bool = True,
        buffer_pool_size: int = 5
    ):
        """
        GPU 리사이저 초기화
        
        Args:
            default_interpolation: 기본 보간법
            use_cuda_stream: CUDA 스트림 사용 여부
            buffer_pool_size: 버퍼 풀 크기
        """
        self.default_interpolation = default_interpolation
        self.use_cuda_stream = use_cuda_stream
        self.buffer_pool_size = buffer_pool_size
        
        # CUDA 스트림 초기화
        if self.use_cuda_stream:
            try:
                self.stream = cv2.cuda.Stream()
                logger.info("🔧 GpuResizer CUDA 스트림 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ CUDA 스트림 초기화 실패, CPU 모드로 전환: {e}")
                self.use_cuda_stream = False
                self.stream = None
        else:
            self.stream = None
        
        # 버퍼 풀 (자주 사용되는 크기의 버퍼 캐싱)
        self._buffer_pool = {}
        
        logger.info(f"🎨 GpuResizer 초기화 완료")
        logger.info(f"   • 기본 보간법: {self._interpolation_name(default_interpolation)}")
        logger.info(f"   • CUDA 스트림: {'활성화' if self.use_cuda_stream else '비활성화'}")
        logger.info(f"   • 버퍼 풀 크기: {buffer_pool_size}")
    
    def resize_to_fit(
        self,
        gpu_frame: Union[np.ndarray, cv2.cuda.GpuMat],
        target_width: int,
        target_height: int,
        strategy: ResizeStrategy = ResizeStrategy.FIT_CONTAIN,
        interpolation: Optional[int] = None
    ) -> cv2.cuda.GpuMat:
        """
        프레임을 지정된 크기에 맞게 리사이즈
        
        Args:
            gpu_frame: 입력 프레임
            target_width: 목표 너비
            target_height: 목표 높이
            strategy: 리사이징 전략
            interpolation: 보간법 (None이면 기본값 사용)
            
        Returns:
            cv2.cuda.GpuMat: 리사이즈된 프레임
        """
        resize_params = ResizeParams(
            target_width=target_width,
            target_height=target_height,
            strategy=strategy,
            interpolation=interpolation or self.default_interpolation
        )
        
        return self.resize_with_params(gpu_frame, resize_params)
    
    def resize_with_params(
        self,
        gpu_frame: Union[np.ndarray, cv2.cuda.GpuMat],
        params: ResizeParams
    ) -> cv2.cuda.GpuMat:
        """
        파라미터를 사용한 리사이징
        
        Args:
            gpu_frame: 입력 프레임
            params: 리사이징 파라미터
            
        Returns:
            cv2.cuda.GpuMat: 리사이즈된 프레임
        """
        try:
            check_cuda_memory()
            gpu_mat = ensure_gpu_mat(gpu_frame)
            
            current_width, current_height = gpu_mat.size()
            
            logger.debug(f"🔧 리사이징 시작: {current_width}x{current_height} → "
                        f"{params.target_width}x{params.target_height} "
                        f"({params.strategy.value})")
            
            if params.strategy == ResizeStrategy.FIT_CONTAIN:
                return self._resize_fit_contain(gpu_mat, params)
            elif params.strategy == ResizeStrategy.FIT_COVER:
                return self._resize_fit_cover(gpu_mat, params)
            elif params.strategy == ResizeStrategy.STRETCH:
                return self._resize_stretch(gpu_mat, params)
            elif params.strategy == ResizeStrategy.CENTER_CROP:
                return self._resize_center_crop(gpu_mat, params)
            else:
                raise ValueError(f"지원하지 않는 리사이징 전략: {params.strategy}")
                
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU 메모리 부족으로 리사이징 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise GPUMemoryError(error_msg) from e
            
        except Exception as e:
            error_msg = f"프레임 리사이징 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def resize_batch(
        self,
        gpu_frames: list,
        params: ResizeParams
    ) -> list:
        """
        여러 프레임을 배치로 리사이징
        
        Args:
            gpu_frames: 입력 프레임 리스트
            params: 리사이징 파라미터
            
        Returns:
            list: 리사이즈된 프레임 리스트
        """
        try:
            check_cuda_memory()
            resized_frames = []
            
            for i, frame in enumerate(gpu_frames):
                try:
                    resized = self.resize_with_params(frame, params)
                    resized_frames.append(resized)
                    
                    if (i + 1) % 10 == 0:
                        logger.debug(f"🔧 배치 리사이징 진행: {i + 1}/{len(gpu_frames)}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 프레임 {i} 리사이징 실패: {e}")
                    # 실패한 프레임은 빈 프레임으로 대체
                    empty_frame = self._create_empty_frame(params)
                    resized_frames.append(empty_frame)
            
            logger.info(f"✅ 배치 리사이징 완료: {len(resized_frames)}개 프레임")
            return resized_frames
            
        except Exception as e:
            error_msg = f"배치 리사이징 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _resize_fit_contain(
        self,
        gpu_mat: cv2.cuda.GpuMat,
        params: ResizeParams
    ) -> cv2.cuda.GpuMat:
        """
        종횡비 유지하며 전체 이미지를 포함하도록 리사이즈 (패딩 추가)
        """
        current_height, current_width = gpu_mat.size()
        
        # 종횡비 계산
        scale = min(
            params.target_width / current_width,
            params.target_height / current_height
        )
        
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # 리사이즈 수행
        resized = self._gpu_resize(gpu_mat, new_width, new_height, params.interpolation)
        
        # 패딩 추가하여 목표 크기로 확장
        if new_width != params.target_width or new_height != params.target_height:
            pad_x = (params.target_width - new_width) // 2
            pad_y = (params.target_height - new_height) // 2
            padded = self._add_padding(resized, params, pad_x, pad_y)
            return padded
        else:
            return resized
    
    def _resize_fit_cover(
        self,
        gpu_mat: cv2.cuda.GpuMat,
        params: ResizeParams
    ) -> cv2.cuda.GpuMat:
        """
        종횡비 유지하며 영역을 완전히 채우도록 리사이즈 (크롭 가능)
        """
        current_height, current_width = gpu_mat.size()
        
        # 종횡비 계산 (영역을 채우기 위해 큰 스케일 선택)
        scale = max(
            params.target_width / current_width,
            params.target_height / current_height
        )
        
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # 리사이즈 수행
        resized = self._gpu_resize(gpu_mat, new_width, new_height, params.interpolation)
        
        # 중앙 크롭하여 목표 크기로 조정
        if new_width != params.target_width or new_height != params.target_height:
            crop_x = (new_width - params.target_width) // 2
            crop_y = (new_height - params.target_height) // 2
            
            # OpenCV 4.13 호환성: CPU에서 크롭 후 GPU로 업로드
            resized_cpu = resized.download()
            cropped_cpu = resized_cpu[crop_y:crop_y + params.target_height, 
                                    crop_x:crop_x + params.target_width]
            
            result = cv2.cuda.GpuMat(params.target_height, params.target_width, gpu_mat.type())
            if not safe_upload_to_gpu(result, cropped_cpu):
                raise CompositionError("Cropped frame upload failed")
            
            return result
        else:
            return resized
    
    def _resize_stretch(
        self,
        gpu_mat: cv2.cuda.GpuMat,
        params: ResizeParams
    ) -> cv2.cuda.GpuMat:
        """
        종횡비 무시하고 목표 크기로 직접 리사이즈
        """
        return self._gpu_resize(gpu_mat, params.target_width, params.target_height, params.interpolation)
    
    def _resize_center_crop(
        self,
        gpu_mat: cv2.cuda.GpuMat,
        params: ResizeParams
    ) -> cv2.cuda.GpuMat:
        """
        중앙 크롭 후 리사이즈
        """
        current_height, current_width = gpu_mat.size()
        
        # 크롭 영역 계산
        crop_width = int(current_width * params.center_crop_ratio)
        crop_height = int(current_height * params.center_crop_ratio)
        
        crop_x = (current_width - crop_width) // 2
        crop_y = (current_height - crop_height) // 2
        
        # OpenCV 4.13 호환성: CPU에서 크롭 후 GPU로 업로드
        gpu_cpu = gpu_mat.download()
        cropped_cpu = gpu_cpu[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
        cropped_gpu = cv2.cuda.GpuMat()
        if not safe_upload_to_gpu(cropped_gpu, cropped_cpu):
            raise CompositionError("Center crop upload failed")
        
        # 크롭된 영역을 목표 크기로 리사이즈
        return self._gpu_resize(cropped_gpu, params.target_width, params.target_height, params.interpolation)
    
    def _gpu_resize(
        self,
        gpu_mat: cv2.cuda.GpuMat,
        new_width: int,
        new_height: int,
        interpolation: int
    ) -> cv2.cuda.GpuMat:
        """
        GPU에서 리사이즈 수행
        """
        try:
            # GPU 리사이즈 수행 (OpenCV 4.13+ 방식)
            if self.use_cuda_stream and self.stream:
                resized_mat = cv2.cuda.resize(gpu_mat, (new_width, new_height), 
                                            interpolation=interpolation, stream=self.stream)
            else:
                resized_mat = cv2.cuda.resize(gpu_mat, (new_width, new_height), 
                                            interpolation=interpolation)
            
            return resized_mat
            
        except Exception as e:
            error_msg = f"GPU 리사이즈 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _add_padding(
        self,
        gpu_mat: cv2.cuda.GpuMat,
        params: ResizeParams,
        pad_x: int,
        pad_y: int
    ) -> cv2.cuda.GpuMat:
        """
        프레임에 패딩 추가
        """
        try:
            # OpenCV 4.13 호환성: CPU에서 패딩 후 GPU로 업로드
            gpu_cpu = gpu_mat.download()
            img_height, img_width = gpu_cpu.shape[:2]
            
            # 패딩된 이미지 생성
            padded_cpu = np.full(
                (params.target_height, params.target_width, 3), 
                params.padding_color, 
                dtype=np.uint8
            )
            
            # 중앙에 원본 이미지 배치
            padded_cpu[pad_y:pad_y + img_height, pad_x:pad_x + img_width] = gpu_cpu
            
            # GPU로 업로드
            padded = cv2.cuda.GpuMat(params.target_height, params.target_width, gpu_mat.type())
            if not safe_upload_to_gpu(padded, padded_cpu):
                raise CompositionError("Padded frame upload failed")
            
            return padded
            
        except Exception as e:
            error_msg = f"패딩 추가 실패: {e}"
            logger.error(f"❌ {error_msg}")
            raise CompositionError(error_msg) from e
    
    def _get_buffer(self, height: int, width: int, mat_type: int) -> cv2.cuda.GpuMat:
        """
        버퍼 풀에서 적절한 크기의 버퍼 가져오기 또는 생성
        """
        buffer_key = (height, width, mat_type)
        
        if buffer_key not in self._buffer_pool:
            if len(self._buffer_pool) >= self.buffer_pool_size:
                # 풀이 가득 찬 경우, 새 버퍼를 직접 생성
                return cv2.cuda.GpuMat(height, width, mat_type)
            
            # 새 버퍼 생성하여 풀에 추가
            self._buffer_pool[buffer_key] = cv2.cuda.GpuMat(height, width, mat_type)
            logger.debug(f"🔧 버퍼 풀에 새 버퍼 추가: {width}x{height}")
        
        return self._buffer_pool[buffer_key]
    
    def _create_empty_frame(self, params: ResizeParams) -> cv2.cuda.GpuMat:
        """
        빈 프레임 생성 (오류 처리용)
        """
        empty_frame = cv2.cuda.GpuMat(params.target_height, params.target_width, cv2.CV_8UC3)
        empty_frame.setTo(params.padding_color)
        return empty_frame
    
    def _interpolation_name(self, interpolation: int) -> str:
        """보간법 이름 반환"""
        interpolation_names = {
            cv2.INTER_NEAREST: "NEAREST",
            cv2.INTER_LINEAR: "LINEAR", 
            cv2.INTER_CUBIC: "CUBIC",
            cv2.INTER_AREA: "AREA",
            cv2.INTER_LANCZOS4: "LANCZOS4"
        }
        return interpolation_names.get(interpolation, f"UNKNOWN({interpolation})")
    
    def synchronize(self):
        """CUDA 스트림 동기화"""
        if self.use_cuda_stream and self.stream:
            try:
                self.stream.waitForCompletion()
                logger.debug("🔧 GpuResizer CUDA 스트림 동기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ CUDA 스트림 동기화 실패: {e}")
    
    def clear_buffer_pool(self):
        """버퍼 풀 정리"""
        try:
            buffer_count = len(self._buffer_pool)
            self._buffer_pool.clear()
            
            if buffer_count > 0:
                logger.info(f"🔧 버퍼 풀 정리 완료: {buffer_count}개 버퍼 해제")
                
        except Exception as e:
            logger.warning(f"⚠️ 버퍼 풀 정리 실패: {e}")
    
    def get_buffer_pool_info(self) -> dict:
        """
        버퍼 풀 정보 반환
        
        Returns:
            dict: 버퍼 풀 정보
        """
        try:
            total_memory = 0
            buffer_info = []
            
            for (height, width, mat_type), buffer in self._buffer_pool.items():
                # BGR 3채널 기준 메모리 계산
                channels = 3 if mat_type == cv2.CV_8UC3 else 1
                memory_bytes = height * width * channels
                total_memory += memory_bytes
                
                buffer_info.append({
                    "size": f"{width}x{height}",
                    "type": mat_type,
                    "memory_mb": memory_bytes / (1024**2)
                })
            
            return {
                "buffer_count": len(self._buffer_pool),
                "total_memory_mb": total_memory / (1024**2),
                "buffers": buffer_info
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 버퍼 풀 정보 조회 실패: {e}")
            return {}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.clear_buffer_pool()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🔧 GpuResizer 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ GpuResizer 리소스 정리 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except Exception:
            pass  # 소멸자에서는 예외를 발생시키지 않음