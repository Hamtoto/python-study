"""
에러 핸들러 레지스트리

다양한 에러 타입에 대한 특화된 핸들러를 관리하는 시스템
"""

from typing import Dict, Type, Callable, Any, Optional
from abc import ABC, abstractmethod
import traceback

from ..utils.logger import logger
from ..utils.exceptions import (
    GPUMemoryError, NVENCSessionError, VideoProcessingError,
    DecodingError, EncodingError, InferenceError
)


class ErrorHandler(ABC):
    """에러 핸들러 베이스 클래스"""
    
    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """에러 처리 가능 여부 확인"""
        pass
    
    @abstractmethod
    async def handle(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """에러 처리 실행"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """처리 우선순위 (낮을수록 높은 우선순위)"""
        pass


class GPUMemoryErrorHandler(ErrorHandler):
    """GPU 메모리 에러 핸들러"""
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, (GPUMemoryError, RuntimeError)) and "memory" in str(error).lower()
    
    async def handle(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("🧹 GPU 메모리 에러 처리 중...")
        
        try:
            import torch
            if torch.cuda.is_available():
                # 캐시 정리
                torch.cuda.empty_cache()
                
                # 메모리 사용량 확인
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                
                logger.info(f"🔍 정리 후 GPU 메모리: {allocated:.1f}MB (예약: {reserved:.1f}MB)")
                
                return {
                    "handled": True,
                    "action": "memory_cleanup",
                    "memory_freed_mb": context.get("initial_memory", 0) - allocated,
                    "retry_recommended": allocated < 20000  # 20GB 이하면 재시도 권장
                }
        except Exception as cleanup_error:
            logger.error(f"❌ GPU 메모리 정리 실패: {cleanup_error}")
        
        return {"handled": False, "action": "cleanup_failed"}
    
    @property
    def priority(self) -> int:
        return 1


class NVENCSessionErrorHandler(ErrorHandler):
    """NVENC 세션 에러 핸들러"""
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, NVENCSessionError) or "nvenc" in str(error).lower()
    
    async def handle(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("🔄 NVENC 세션 에러 처리 중...")
        
        # 소프트웨어 인코딩으로 폴백
        return {
            "handled": True,
            "action": "fallback_to_software",
            "fallback_encoder": "libx264",
            "retry_recommended": True,
            "session_limit_reached": True
        }
    
    @property
    def priority(self) -> int:
        return 2


class VideoProcessingErrorHandler(ErrorHandler):
    """비디오 처리 에러 핸들러"""
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, (VideoProcessingError, DecodingError, EncodingError))
    
    async def handle(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning(f"📹 비디오 처리 에러 처리 중: {type(error).__name__}")
        
        error_type = type(error).__name__
        
        if isinstance(error, DecodingError):
            return {
                "handled": True,
                "action": "retry_with_software_decoder",
                "retry_recommended": True,
                "max_retries": 2
            }
        
        elif isinstance(error, EncodingError):
            return {
                "handled": True,
                "action": "retry_with_different_codec",
                "fallback_codec": "libx264",
                "retry_recommended": True
            }
        
        else:
            return {
                "handled": True,
                "action": "skip_and_continue",
                "retry_recommended": False,
                "create_placeholder": True
            }
    
    @property
    def priority(self) -> int:
        return 3


class InferenceErrorHandler(ErrorHandler):
    """추론 에러 핸들러"""
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, InferenceError) or "inference" in str(error).lower()
    
    async def handle(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("🧠 추론 에러 처리 중...")
        
        # 배치 크기 감소
        current_batch = context.get("batch_size", 4)
        new_batch = max(1, current_batch // 2)
        
        return {
            "handled": True,
            "action": "reduce_batch_size",
            "old_batch_size": current_batch,
            "new_batch_size": new_batch,
            "retry_recommended": True
        }
    
    @property
    def priority(self) -> int:
        return 4


class GenericErrorHandler(ErrorHandler):
    """일반 에러 핸들러 (폴백)"""
    
    def can_handle(self, error: Exception) -> bool:
        return True  # 모든 에러 처리 가능
    
    async def handle(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning(f"⚠️ 일반 에러 처리: {type(error).__name__}")
        
        # 에러 로깅
        logger.error(f"에러 상세: {str(error)}")
        logger.debug(f"에러 트레이스백:\n{traceback.format_exc()}")
        
        return {
            "handled": True,
            "action": "log_and_continue",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "retry_recommended": False
        }
    
    @property
    def priority(self) -> int:
        return 999  # 가장 낮은 우선순위


class ErrorHandlerRegistry:
    """
    에러 핸들러 레지스트리
    
    기능:
    - 에러 타입별 핸들러 등록/관리
    - 우선순위 기반 핸들러 선택
    - 에러 처리 통계 수집
    - 처리 결과 로깅
    """
    
    def __init__(self):
        self.handlers: list[ErrorHandler] = []
        self.handling_stats = {
            "total_errors": 0,
            "handled_errors": 0,
            "errors_by_type": {},
            "handlers_by_type": {}
        }
        
        # 기본 핸들러 등록
        self._register_default_handlers()
        
        logger.info("🛡️ ErrorHandlerRegistry 초기화 완료")
    
    def _register_default_handlers(self):
        """기본 에러 핸들러들 등록"""
        default_handlers = [
            GPUMemoryErrorHandler(),
            NVENCSessionErrorHandler(),
            VideoProcessingErrorHandler(),
            InferenceErrorHandler(),
            GenericErrorHandler()  # 마지막에 등록 (폴백)
        ]
        
        for handler in default_handlers:
            self.register_handler(handler)
    
    def register_handler(self, handler: ErrorHandler):
        """에러 핸들러 등록"""
        self.handlers.append(handler)
        # 우선순위 순으로 정렬
        self.handlers.sort(key=lambda h: h.priority)
        
        logger.debug(f"📝 에러 핸들러 등록: {type(handler).__name__} (우선순위: {handler.priority})")
    
    def unregister_handler(self, handler_type: Type[ErrorHandler]):
        """에러 핸들러 등록 해제"""
        self.handlers = [h for h in self.handlers if not isinstance(h, handler_type)]
        logger.debug(f"🗑️ 에러 핸들러 해제: {handler_type.__name__}")
    
    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        에러 처리 실행
        
        Args:
            error: 처리할 에러
            context: 에러 발생 컨텍스트 정보
        
        Returns:
            처리 결과 딕셔너리
        """
        if context is None:
            context = {}
        
        error_type = type(error).__name__
        
        # 통계 업데이트
        self.handling_stats["total_errors"] += 1
        self.handling_stats["errors_by_type"][error_type] = (
            self.handling_stats["errors_by_type"].get(error_type, 0) + 1
        )
        
        logger.info(f"🛡️ 에러 처리 시작: {error_type}")
        
        # 적절한 핸들러 찾기
        for handler in self.handlers:
            if handler.can_handle(error):
                try:
                    # 에러 처리 실행
                    result = await handler.handle(error, context)
                    
                    if result.get("handled", False):
                        # 처리 성공
                        self.handling_stats["handled_errors"] += 1
                        self.handling_stats["handlers_by_type"][error_type] = type(handler).__name__
                        
                        logger.info(f"✅ 에러 처리 완료: {type(handler).__name__} -> {result.get('action', 'unknown')}")
                        
                        # 결과에 메타데이터 추가
                        result.update({
                            "handler_used": type(handler).__name__,
                            "error_type": error_type,
                            "timestamp": context.get("timestamp", "unknown")
                        })
                        
                        return result
                    
                except Exception as handling_error:
                    logger.error(f"❌ 에러 핸들러 실행 실패 ({type(handler).__name__}): {handling_error}")
                    continue
        
        # 모든 핸들러 실패
        logger.error(f"❌ 에러 처리 실패: {error_type} - 적절한 핸들러 없음")
        
        return {
            "handled": False,
            "error_type": error_type,
            "error_message": str(error),
            "action": "unhandled"
        }
    
    def get_handling_stats(self) -> Dict[str, Any]:
        """에러 처리 통계 반환"""
        success_rate = 0
        if self.handling_stats["total_errors"] > 0:
            success_rate = (self.handling_stats["handled_errors"] / 
                          self.handling_stats["total_errors"]) * 100
        
        return {
            **self.handling_stats,
            "success_rate": success_rate,
            "registered_handlers": len(self.handlers)
        }
    
    def print_handling_summary(self):
        """에러 처리 요약 출력"""
        stats = self.get_handling_stats()
        
        print(f"""
🛡️ 에러 처리 요약:
   • 총 에러: {stats['total_errors']}개
   • 처리 성공: {stats['handled_errors']}개
   • 성공률: {stats['success_rate']:.1f}%
   • 등록된 핸들러: {stats['registered_handlers']}개
        """)
        
        if stats['errors_by_type']:
            print("📊 에러 타입별 통계:")
            for error_type, count in stats['errors_by_type'].items():
                handler = stats['handlers_by_type'].get(error_type, 'N/A')
                print(f"   • {error_type}: {count}회 (핸들러: {handler})")
    
    def reset_stats(self):
        """통계 초기화"""
        for key in self.handling_stats:
            if isinstance(self.handling_stats[key], dict):
                self.handling_stats[key].clear()
            else:
                self.handling_stats[key] = 0
        
        logger.info("📊 에러 처리 통계 초기화")


if __name__ == "__main__":
    # 테스트 코드
    import asyncio
    
    async def test_error_handlers():
        print("🧪 ErrorHandlerRegistry 테스트 시작...")
        
        registry = ErrorHandlerRegistry()
        
        # 다양한 에러 테스트
        test_errors = [
            GPUMemoryError("GPU memory allocation failed"),
            NVENCSessionError("NVENC session limit exceeded"),
            DecodingError("Video decoding failed"),
            ValueError("General error for testing")
        ]
        
        for error in test_errors:
            context = {"timestamp": "test", "batch_size": 4}
            result = await registry.handle_error(error, context)
            print(f"에러: {type(error).__name__} -> 처리: {result.get('handled', False)}")
        
        registry.print_handling_summary()
        
        print("✅ 테스트 완료!")
    
    asyncio.run(test_error_handlers())