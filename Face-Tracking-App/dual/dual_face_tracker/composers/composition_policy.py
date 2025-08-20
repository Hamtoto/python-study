"""
타일 합성 에러 처리 정책

다양한 실패 시나리오에 대한 복구 전략과 폴백 메커니즘 제공.
실시간 처리 연속성을 보장하기 위한 에러 처리 시스템.
"""

import cv2
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass
import time

from ..utils.logger import get_logger
from ..utils.exceptions import CompositionError, GPUMemoryError
from ..utils.cuda_utils import safe_upload_to_gpu

logger = get_logger(__name__)


class ErrorType(Enum):
    """에러 타입 분류"""
    GPU_MEMORY_ERROR = "gpu_memory_error"
    FRAME_PROCESSING_ERROR = "frame_processing_error"
    SINGLE_FACE_MISSING = "single_face_missing"
    DUAL_FACE_MISSING = "dual_face_missing"
    RESIZE_ERROR = "resize_error"
    COMPOSITION_ERROR = "composition_error"
    CUDA_ERROR = "cuda_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryAction(Enum):
    """복구 액션"""
    USE_FALLBACK_FRAME = "use_fallback_frame"
    USE_PREVIOUS_FRAME = "use_previous_frame"
    USE_BLACK_FRAME = "use_black_frame"
    USE_SINGLE_FRAME = "use_single_frame"
    RETRY_WITH_CPU = "retry_with_cpu"
    REDUCE_QUALITY = "reduce_quality"
    SKIP_FRAME = "skip_frame"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    error_type: ErrorType
    error_message: str
    timestamp: float
    frame_number: Optional[int] = None
    gpu_memory_used: Optional[float] = None
    retry_count: int = 0
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class RecoveryStrategy:
    """복구 전략"""
    primary_action: RecoveryAction
    fallback_actions: list = None
    max_retries: int = 3
    retry_delay: float = 0.1
    quality_reduction_factor: float = 0.7
    
    def __post_init__(self):
        if self.fallback_actions is None:
            self.fallback_actions = []


class TileCompositionErrorPolicy:
    """
    타일 합성 에러 처리 정책 클래스
    
    Features:
    - 다양한 에러 시나리오별 복구 전략
    - 실시간 처리 연속성 보장
    - 자동 품질 조정
    - 에러 통계 수집
    - 임계치 기반 자동 복구
    """
    
    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1080,
        max_consecutive_errors: int = 10,
        memory_threshold_percent: float = 85.0,
        enable_quality_reduction: bool = True,
        keep_error_history: bool = True
    ):
        """
        에러 정책 초기화
        
        Args:
            output_width: 출력 너비
            output_height: 출력 높이
            max_consecutive_errors: 최대 연속 에러 수
            memory_threshold_percent: 메모리 임계치 (%)
            enable_quality_reduction: 품질 자동 조정 활성화
            keep_error_history: 에러 히스토리 보관 여부
        """
        self.output_width = output_width
        self.output_height = output_height
        self.max_consecutive_errors = max_consecutive_errors
        self.memory_threshold_percent = memory_threshold_percent
        self.enable_quality_reduction = enable_quality_reduction
        self.keep_error_history = keep_error_history
        
        # 에러 통계
        self.consecutive_errors = 0
        self.total_errors = 0
        self.error_history = [] if keep_error_history else None
        self.last_successful_frame = None
        
        # 폴백 프레임 버퍼
        self._fallback_frame_cpu = None
        self._fallback_frame_gpu = None
        self._black_frame_cpu = None
        self._black_frame_gpu = None
        
        # 복구 전략 매핑
        self._recovery_strategies = self._initialize_recovery_strategies()
        
        logger.info(f"🛡️ TileCompositionErrorPolicy 초기화 완료")
        logger.info(f"   • 출력 크기: {output_width}x{output_height}")
        logger.info(f"   • 최대 연속 에러: {max_consecutive_errors}")
        logger.info(f"   • 메모리 임계치: {memory_threshold_percent}%")
        logger.info(f"   • 품질 자동 조정: {'활성화' if enable_quality_reduction else '비활성화'}")
    
    def handle_error(
        self,
        error: Exception,
        frame_number: Optional[int] = None,
        available_frames: Optional[Dict[str, Union[np.ndarray, cv2.cuda.GpuMat]]] = None,
        gpu_memory_info: Optional[Dict[str, float]] = None
    ) -> Tuple[cv2.cuda.GpuMat, bool]:
        """
        에러 처리 및 복구 프레임 생성
        
        Args:
            error: 발생한 예외
            frame_number: 프레임 번호
            available_frames: 사용 가능한 프레임들
            gpu_memory_info: GPU 메모리 정보
            
        Returns:
            Tuple[cv2.cuda.GpuMat, bool]: (복구된 프레임, 계속 처리 여부)
        """
        try:
            # 에러 분석
            error_context = self._analyze_error(error, frame_number, gpu_memory_info)
            
            # 에러 통계 업데이트
            self._update_error_stats(error_context)
            
            # 복구 전략 결정
            strategy = self._get_recovery_strategy(error_context)
            
            # 복구 프레임 생성
            recovery_frame = self._execute_recovery_strategy(
                strategy, error_context, available_frames
            )
            
            # 계속 처리할지 결정
            should_continue = self._should_continue_processing(error_context)
            
            logger.info(f"🛡️ 에러 복구 완료 (프레임 {frame_number}): "
                       f"{strategy.primary_action.value}, 계속 처리: {should_continue}")
            
            return recovery_frame, should_continue
            
        except Exception as recovery_error:
            logger.error(f"❌ 에러 복구 실패: {recovery_error}")
            # 최후의 수단: 검은 프레임 반환
            emergency_frame = self._create_emergency_frame()
            return emergency_frame, False
    
    def handle_single_face_failure(
        self,
        available_frame: Optional[Union[np.ndarray, cv2.cuda.GpuMat]] = None
    ) -> cv2.cuda.GpuMat:
        """
        단일 얼굴 감지 실패 처리
        
        Args:
            available_frame: 사용 가능한 프레임 (있는 경우)
            
        Returns:
            cv2.cuda.GpuMat: 복구된 프레임
        """
        try:
            if available_frame is not None:
                # 사용 가능한 프레임을 중앙에 배치
                gpu_frame = self._ensure_gpu_mat(available_frame)
                return self._create_centered_frame(gpu_frame)
            else:
                # 이전 성공 프레임 또는 폴백 프레임 사용
                if self.last_successful_frame is not None:
                    return self.last_successful_frame
                else:
                    return self._get_fallback_frame()
                    
        except Exception as e:
            logger.error(f"❌ 단일 얼굴 실패 처리 중 오류: {e}")
            return self._get_black_frame()
    
    def handle_complete_failure(
        self,
        fallback_frame: Optional[Union[np.ndarray, cv2.cuda.GpuMat]] = None
    ) -> cv2.cuda.GpuMat:
        """
        완전 실패 처리 (양쪽 얼굴 모두 감지 실패)
        
        Args:
            fallback_frame: 폴백 프레임
            
        Returns:
            cv2.cuda.GpuMat: 복구된 프레임
        """
        try:
            if fallback_frame is not None:
                gpu_frame = self._ensure_gpu_mat(fallback_frame)
                return self._create_centered_frame(gpu_frame)
            elif self.last_successful_frame is not None:
                logger.info("🛡️ 마지막 성공 프레임 사용")
                return self.last_successful_frame
            else:
                logger.info("🛡️ 폴백 프레임 사용")
                return self._get_fallback_frame()
                
        except Exception as e:
            logger.error(f"❌ 완전 실패 처리 중 오류: {e}")
            return self._get_black_frame()
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorType, RecoveryStrategy]:
        """복구 전략 초기화"""
        return {
            ErrorType.GPU_MEMORY_ERROR: RecoveryStrategy(
                primary_action=RecoveryAction.REDUCE_QUALITY,
                fallback_actions=[
                    RecoveryAction.USE_PREVIOUS_FRAME,
                    RecoveryAction.USE_BLACK_FRAME
                ],
                max_retries=2
            ),
            ErrorType.FRAME_PROCESSING_ERROR: RecoveryStrategy(
                primary_action=RecoveryAction.USE_PREVIOUS_FRAME,
                fallback_actions=[RecoveryAction.USE_FALLBACK_FRAME],
                max_retries=3
            ),
            ErrorType.SINGLE_FACE_MISSING: RecoveryStrategy(
                primary_action=RecoveryAction.USE_SINGLE_FRAME,
                fallback_actions=[RecoveryAction.USE_PREVIOUS_FRAME],
                max_retries=1
            ),
            ErrorType.DUAL_FACE_MISSING: RecoveryStrategy(
                primary_action=RecoveryAction.USE_FALLBACK_FRAME,
                fallback_actions=[RecoveryAction.USE_BLACK_FRAME],
                max_retries=1
            ),
            ErrorType.RESIZE_ERROR: RecoveryStrategy(
                primary_action=RecoveryAction.REDUCE_QUALITY,
                fallback_actions=[RecoveryAction.RETRY_WITH_CPU],
                max_retries=2
            ),
            ErrorType.COMPOSITION_ERROR: RecoveryStrategy(
                primary_action=RecoveryAction.USE_PREVIOUS_FRAME,
                fallback_actions=[RecoveryAction.USE_FALLBACK_FRAME],
                max_retries=2
            ),
            ErrorType.CUDA_ERROR: RecoveryStrategy(
                primary_action=RecoveryAction.RETRY_WITH_CPU,
                fallback_actions=[RecoveryAction.USE_BLACK_FRAME],
                max_retries=1
            ),
            ErrorType.UNKNOWN_ERROR: RecoveryStrategy(
                primary_action=RecoveryAction.USE_PREVIOUS_FRAME,
                fallback_actions=[RecoveryAction.USE_FALLBACK_FRAME, RecoveryAction.USE_BLACK_FRAME],
                max_retries=1
            )
        }
    
    def _analyze_error(
        self,
        error: Exception,
        frame_number: Optional[int],
        gpu_memory_info: Optional[Dict[str, float]]
    ) -> ErrorContext:
        """에러 분석 및 컨텍스트 생성"""
        error_type = ErrorType.UNKNOWN_ERROR
        
        if isinstance(error, GPUMemoryError) or "out of memory" in str(error).lower():
            error_type = ErrorType.GPU_MEMORY_ERROR
        elif isinstance(error, CompositionError):
            if "resize" in str(error).lower():
                error_type = ErrorType.RESIZE_ERROR
            else:
                error_type = ErrorType.COMPOSITION_ERROR
        elif "cuda" in str(error).lower():
            error_type = ErrorType.CUDA_ERROR
        elif "frame" in str(error).lower():
            error_type = ErrorType.FRAME_PROCESSING_ERROR
        
        gpu_memory_used = None
        if gpu_memory_info:
            gpu_memory_used = gpu_memory_info.get('utilization_percent', None)
        
        return ErrorContext(
            error_type=error_type,
            error_message=str(error),
            timestamp=time.time(),
            frame_number=frame_number,
            gpu_memory_used=gpu_memory_used
        )
    
    def _update_error_stats(self, error_context: ErrorContext):
        """에러 통계 업데이트"""
        self.consecutive_errors += 1
        self.total_errors += 1
        
        if self.keep_error_history:
            self.error_history.append(error_context)
            
            # 히스토리 크기 제한 (최대 100개)
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]
        
        logger.warning(f"⚠️ 에러 발생 ({error_context.error_type.value}): "
                      f"연속 {self.consecutive_errors}회, 총 {self.total_errors}회")
    
    def _get_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """에러 컨텍스트에 따른 복구 전략 반환"""
        strategy = self._recovery_strategies.get(
            error_context.error_type,
            self._recovery_strategies[ErrorType.UNKNOWN_ERROR]
        )
        
        # 연속 에러가 많으면 더 강력한 복구 전략 사용
        if self.consecutive_errors >= self.max_consecutive_errors // 2:
            # 보다 안전한 전략으로 변경
            if strategy.primary_action not in [RecoveryAction.USE_BLACK_FRAME, RecoveryAction.EMERGENCY_STOP]:
                strategy.primary_action = RecoveryAction.USE_BLACK_FRAME
        
        return strategy
    
    def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error_context: ErrorContext,
        available_frames: Optional[Dict[str, Union[np.ndarray, cv2.cuda.GpuMat]]]
    ) -> cv2.cuda.GpuMat:
        """복구 전략 실행"""
        try:
            if strategy.primary_action == RecoveryAction.USE_FALLBACK_FRAME:
                return self._get_fallback_frame()
            elif strategy.primary_action == RecoveryAction.USE_PREVIOUS_FRAME:
                return self._get_previous_frame()
            elif strategy.primary_action == RecoveryAction.USE_BLACK_FRAME:
                return self._get_black_frame()
            elif strategy.primary_action == RecoveryAction.USE_SINGLE_FRAME:
                return self._handle_single_frame_recovery(available_frames)
            elif strategy.primary_action == RecoveryAction.REDUCE_QUALITY:
                return self._create_reduced_quality_frame(available_frames)
            else:
                # 다른 전략들은 폴백 프레임으로 처리
                return self._get_fallback_frame()
                
        except Exception as e:
            logger.error(f"❌ 복구 전략 실행 실패: {e}")
            # 폴백 액션 시도
            for fallback_action in strategy.fallback_actions:
                try:
                    if fallback_action == RecoveryAction.USE_BLACK_FRAME:
                        return self._get_black_frame()
                    elif fallback_action == RecoveryAction.USE_FALLBACK_FRAME:
                        return self._get_fallback_frame()
                    elif fallback_action == RecoveryAction.USE_PREVIOUS_FRAME:
                        return self._get_previous_frame()
                except Exception:
                    continue
            
            # 모든 폴백 실패 시 응급 프레임
            return self._create_emergency_frame()
    
    def _should_continue_processing(self, error_context: ErrorContext) -> bool:
        """처리를 계속할지 결정"""
        # 연속 에러가 임계치를 넘으면 중단
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.error(f"❌ 연속 에러 임계치 초과: {self.consecutive_errors}/{self.max_consecutive_errors}")
            return False
        
        # GPU 메모리 부족이 계속되면 중단
        if (error_context.error_type == ErrorType.GPU_MEMORY_ERROR and 
            error_context.gpu_memory_used and 
            error_context.gpu_memory_used > self.memory_threshold_percent):
            logger.error(f"❌ GPU 메모리 부족 지속: {error_context.gpu_memory_used}%")
            return False
        
        return True
    
    def _handle_single_frame_recovery(
        self,
        available_frames: Optional[Dict[str, Union[np.ndarray, cv2.cuda.GpuMat]]]
    ) -> cv2.cuda.GpuMat:
        """단일 프레임 복구 처리"""
        if available_frames:
            # 사용 가능한 프레임 중 하나 선택
            for frame_name, frame in available_frames.items():
                if frame is not None:
                    gpu_frame = self._ensure_gpu_mat(frame)
                    return self._create_centered_frame(gpu_frame)
        
        return self._get_fallback_frame()
    
    def _create_reduced_quality_frame(
        self,
        available_frames: Optional[Dict[str, Union[np.ndarray, cv2.cuda.GpuMat]]]
    ) -> cv2.cuda.GpuMat:
        """품질 축소된 복구 프레임 생성"""
        if not self.enable_quality_reduction:
            return self._get_fallback_frame()
        
        # 해상도를 줄여서 처리
        reduced_width = int(self.output_width * 0.7)
        reduced_height = int(self.output_height * 0.7)
        
        # 저품질 버퍼 생성
        reduced_frame = cv2.cuda.GpuMat(reduced_height, reduced_width, cv2.CV_8UC3)
        reduced_frame.setTo((64, 64, 64))  # 회색으로 초기화
        
        # 원래 크기로 업스케일
        output_frame = cv2.cuda.resize(reduced_frame, (self.output_width, self.output_height))
        
        return output_frame
    
    def _create_centered_frame(self, gpu_frame: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
        """프레임을 중앙에 배치"""
        try:
            # 출력 크기에 맞게 리사이즈 및 중앙 배치
            output_frame = cv2.cuda.GpuMat(self.output_height, self.output_width, cv2.CV_8UC3)
            output_frame.setTo((0, 0, 0))  # 검은 배경
            
            frame_width, frame_height = gpu_frame.size()
            
            # 종횡비 유지하며 리사이즈
            scale = min(self.output_width / frame_width, self.output_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            resized = cv2.cuda.resize(gpu_frame, (new_width, new_height))
            
            # OpenCV 4.13 호환성: CPU에서 중앙 배치 후 GPU로 업로드
            resized_cpu = resized.download()
            output_cpu = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            
            # 중앙 배치
            x_offset = (self.output_width - new_width) // 2
            y_offset = (self.output_height - new_height) // 2
            
            output_cpu[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_cpu
            
            if not safe_upload_to_gpu(output_frame, output_cpu):
                raise CompositionError("Center frame upload failed")
            return output_frame
            
        except Exception as e:
            logger.error(f"❌ 중앙 프레임 생성 실패: {e}")
            return self._get_black_frame()
    
    def _get_fallback_frame(self) -> cv2.cuda.GpuMat:
        """폴백 프레임 반환"""
        if self._fallback_frame_gpu is None:
            self._create_fallback_frames()
        return self._fallback_frame_gpu
    
    def _get_black_frame(self) -> cv2.cuda.GpuMat:
        """검은 프레임 반환"""
        if self._black_frame_gpu is None:
            self._create_black_frames()
        return self._black_frame_gpu
    
    def _get_previous_frame(self) -> cv2.cuda.GpuMat:
        """이전 성공 프레임 반환"""
        if self.last_successful_frame is not None:
            return self.last_successful_frame
        else:
            return self._get_fallback_frame()
    
    def _create_fallback_frames(self):
        """폴백 프레임 생성"""
        try:
            # 간단한 그라디언트 패턴의 폴백 프레임
            self._fallback_frame_gpu = cv2.cuda.GpuMat(self.output_height, self.output_width, cv2.CV_8UC3)
            
            # CPU에서 그라디언트 생성
            fallback_cpu = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            for y in range(self.output_height):
                intensity = int((y / self.output_height) * 255)
                fallback_cpu[y, :] = [intensity // 4, intensity // 2, intensity // 4]  # 어두운 초록
            
            # GPU로 업로드
            if not safe_upload_to_gpu(self._fallback_frame_gpu, fallback_cpu):
                raise CompositionError("Fallback frame upload failed")
            
            logger.debug("🔧 폴백 프레임 생성 완료")
            
        except Exception as e:
            logger.error(f"❌ 폴백 프레임 생성 실패: {e}")
            self._create_black_frames()
    
    def _create_black_frames(self):
        """검은 프레임 생성"""
        try:
            self._black_frame_gpu = cv2.cuda.GpuMat(self.output_height, self.output_width, cv2.CV_8UC3)
            self._black_frame_gpu.setTo((0, 0, 0))
            
            logger.debug("🔧 검은 프레임 생성 완료")
            
        except Exception as e:
            logger.error(f"❌ 검은 프레임 생성 실패: {e}")
            # 최후의 수단으로 CPU에서 생성
            black_cpu = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            self._black_frame_gpu = cv2.cuda.GpuMat()
            if not safe_upload_to_gpu(self._black_frame_gpu, black_cpu):
                raise CompositionError("Black frame upload failed")
    
    def _create_emergency_frame(self) -> cv2.cuda.GpuMat:
        """응급 프레임 생성 (모든 복구 전략 실패 시)"""
        try:
            emergency_frame = cv2.cuda.GpuMat(self.output_height, self.output_width, cv2.CV_8UC3)
            emergency_frame.setTo((0, 0, 128))  # 어두운 빨강
            return emergency_frame
        except Exception:
            # 정말 마지막 수단
            emergency_cpu = np.full((self.output_height, self.output_width, 3), (0, 0, 128), dtype=np.uint8)
            emergency_gpu = cv2.cuda.GpuMat()
            if not safe_upload_to_gpu(emergency_gpu, emergency_cpu):
                raise CompositionError("Emergency frame upload failed")
            return emergency_gpu
    
    def _ensure_gpu_mat(self, frame: Union[np.ndarray, cv2.cuda.GpuMat]) -> cv2.cuda.GpuMat:
        """프레임을 GPU Mat으로 변환"""
        if isinstance(frame, cv2.cuda.GpuMat):
            return frame
        else:
            gpu_mat = cv2.cuda.GpuMat()
            if not safe_upload_to_gpu(gpu_mat, frame):
                raise CompositionError("Frame upload failed")
            return gpu_mat
    
    def update_successful_frame(self, frame: cv2.cuda.GpuMat):
        """성공한 프레임 업데이트"""
        try:
            self.last_successful_frame = frame
            self.consecutive_errors = 0  # 성공 시 연속 에러 카운트 리셋
            
        except Exception as e:
            logger.warning(f"⚠️ 성공 프레임 업데이트 실패: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        에러 통계 반환
        
        Returns:
            dict: 에러 통계 정보
        """
        try:
            error_type_counts = {}
            if self.keep_error_history and self.error_history:
                for error_context in self.error_history:
                    error_type = error_context.error_type.value
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            return {
                "total_errors": self.total_errors,
                "consecutive_errors": self.consecutive_errors,
                "max_consecutive_errors": self.max_consecutive_errors,
                "error_type_distribution": error_type_counts,
                "has_successful_frame": self.last_successful_frame is not None,
                "error_history_size": len(self.error_history) if self.error_history else 0
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 에러 통계 조회 실패: {e}")
            return {}
    
    def reset_error_stats(self):
        """에러 통계 리셋"""
        try:
            self.consecutive_errors = 0
            self.total_errors = 0
            if self.error_history:
                self.error_history.clear()
            
            logger.info("🔧 에러 통계 리셋 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 에러 통계 리셋 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self._fallback_frame_gpu = None
            self._fallback_frame_cpu = None
            self._black_frame_gpu = None
            self._black_frame_cpu = None
            self.last_successful_frame = None
            
            if self.error_history:
                self.error_history.clear()
            
            logger.info("🔧 TileCompositionErrorPolicy 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 에러 정책 리소스 정리 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except Exception:
            pass  # 소멸자에서는 예외를 발생시키지 않음