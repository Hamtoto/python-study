"""
스트림 복구 관리 시스템

비디오 처리 중 발생하는 다양한 에러를 자동으로 감지하고 복구하는 시스템
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import logger
from ..utils.exceptions import (
    GPUMemoryError, NVENCSessionError, VideoProcessingError,
    DecodingError, EncodingError, InferenceError
)


class RecoveryStrategy(Enum):
    """복구 전략 타입"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    REDUCE_BATCH = "reduce_batch"
    MEMORY_CLEANUP = "memory_cleanup"


@dataclass
class RecoveryAction:
    """복구 행동 정의"""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay_seconds: float = 1.0
    condition_check: Optional[Callable[[], bool]] = None
    success_callback: Optional[Callable[[], None]] = None
    failure_callback: Optional[Callable[[], None]] = None


@dataclass
class RecoveryRecord:
    """복구 기록"""
    timestamp: float
    error_type: str
    error_message: str
    strategy_used: RecoveryStrategy
    attempt_number: int
    success: bool
    recovery_time: float
    video_path: Optional[str] = None
    
    @property
    def formatted_timestamp(self) -> str:
        return time.strftime('%H:%M:%S', time.localtime(self.timestamp))


class StreamRecoveryManager:
    """
    스트림 복구 관리자
    
    기능:
    - 다양한 에러 타입별 자동 복구 전략
    - 재시도 로직 (지수 백오프)
    - 소프트웨어 폴백 (NVENC → CPU 인코딩)
    - GPU 메모리 자동 정리
    - 배치 크기 동적 조정
    - 복구 통계 및 로깅
    """
    
    def __init__(self):
        # 복구 전략 정의
        self.recovery_strategies = self._init_recovery_strategies()
        
        # 복구 통계
        self.recovery_stats = {
            'total_errors': 0,
            'total_recoveries': 0,
            'recovery_success_rate': 0.0,
            'errors_by_type': defaultdict(int),
            'recoveries_by_strategy': defaultdict(int)
        }
        
        # 복구 기록 (최근 100개)
        self.recovery_history = deque(maxlen=100)
        
        # 현재 처리 설정
        self.current_batch_size = 4
        self.min_batch_size = 1
        self.max_batch_size = 32
        
        # 폴백 상태
        self.software_fallback_enabled = True
        self.use_software_encoding = False
        
        logger.info("🛡️ StreamRecoveryManager 초기화 완료")
    
    def _init_recovery_strategies(self) -> Dict[type, RecoveryAction]:
        """복구 전략 초기화"""
        return {
            GPUMemoryError: RecoveryAction(
                strategy=RecoveryStrategy.MEMORY_CLEANUP,
                max_attempts=2,
                delay_seconds=2.0
            ),
            NVENCSessionError: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=1,
                delay_seconds=1.0
            ),
            DecodingError: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                delay_seconds=1.0
            ),
            EncodingError: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=2,
                delay_seconds=1.0
            ),
            InferenceError: RecoveryAction(
                strategy=RecoveryStrategy.REDUCE_BATCH,
                max_attempts=2,
                delay_seconds=0.5
            ),
            Exception: RecoveryAction(  # 일반 예외
                strategy=RecoveryStrategy.RETRY,
                max_attempts=1,
                delay_seconds=2.0
            )
        }
    
    async def process_with_recovery(self, 
                                  video_path: str,
                                  processor_func: Callable,
                                  stream_id: int = 0,
                                  **kwargs) -> Any:
        """
        복구 로직이 내장된 비디오 처리
        
        Args:
            video_path: 처리할 비디오 경로
            processor_func: 실제 처리 함수
            stream_id: 스트림 식별자
            **kwargs: 처리 함수에 전달할 추가 인자
        
        Returns:
            처리 결과 또는 에러 시 None
        """
        video_name = Path(video_path).name
        
        for attempt in range(3):  # 최대 3번 시도
            try:
                logger.info(f"🎬 비디오 처리 시작: {video_name} (시도 {attempt + 1})")
                
                result = await self._execute_with_monitoring(
                    processor_func, video_path, stream_id, **kwargs
                )
                
                logger.info(f"✅ 비디오 처리 성공: {video_name}")
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ 비디오 처리 실패: {video_name} - {str(e)}")
                
                # 복구 시도
                recovery_success = await self.attempt_recovery(
                    error=e, 
                    video_path=video_path,
                    attempt=attempt + 1
                )
                
                if not recovery_success and attempt == 2:
                    # 최종 실패 - 에러 출력 파일 생성
                    logger.error(f"❌ 최종 실패: {video_name} - 복구 불가")
                    return self._create_error_output(video_path, str(e))
                
                # 다음 시도 전 대기
                if attempt < 2:
                    await asyncio.sleep(2.0)
        
        return None
    
    async def _execute_with_monitoring(self, 
                                     processor_func: Callable,
                                     video_path: str,
                                     stream_id: int,
                                     **kwargs) -> Any:
        """모니터링이 포함된 처리 실행"""
        start_time = time.time()
        
        try:
            # GPU 메모리 사용량 체크
            if TORCH_AVAILABLE and torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"🔍 처리 시작 GPU 메모리: {initial_memory:.1f}MB")
            
            # 실제 처리 실행
            result = await processor_func(video_path, stream_id=stream_id, **kwargs)
            
            processing_time = time.time() - start_time
            logger.debug(f"⏱️ 처리 시간: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            # 에러 발생시 메모리 상태 로깅
            if TORCH_AVAILABLE and torch.cuda.is_available():
                error_memory = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"❌ 에러 시점 GPU 메모리: {error_memory:.1f}MB")
            
            raise e
    
    async def attempt_recovery(self, 
                             error: Exception, 
                             video_path: str,
                             attempt: int) -> bool:
        """
        에러 타입에 따른 복구 시도
        
        Returns:
            복구 성공 여부
        """
        error_type = type(error)
        error_message = str(error)
        
        # 통계 업데이트
        self.recovery_stats['total_errors'] += 1
        self.recovery_stats['errors_by_type'][error_type.__name__] += 1
        
        # 복구 전략 선택
        recovery_action = self._get_recovery_action(error_type)
        
        if not recovery_action:
            logger.warning(f"⚠️ 복구 전략 없음: {error_type.__name__}")
            return False
        
        logger.info(f"🔧 복구 시도: {recovery_action.strategy.value} "
                   f"({attempt}/{recovery_action.max_attempts})")
        
        recovery_start = time.time()
        success = False
        
        try:
            # 복구 전략 실행
            if recovery_action.strategy == RecoveryStrategy.MEMORY_CLEANUP:
                success = await self._memory_cleanup_recovery()
                
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                success = await self._fallback_recovery(error_type)
                
            elif recovery_action.strategy == RecoveryStrategy.REDUCE_BATCH:
                success = await self._reduce_batch_recovery()
                
            elif recovery_action.strategy == RecoveryStrategy.RETRY:
                # 단순 재시도 (대기 후)
                await asyncio.sleep(recovery_action.delay_seconds)
                success = True
                
            else:
                logger.warning(f"⚠️ 지원되지 않는 복구 전략: {recovery_action.strategy}")
                success = False
            
            # 복구 결과 기록
            recovery_time = time.time() - recovery_start
            
            self._record_recovery(
                error_type=error_type.__name__,
                error_message=error_message,
                strategy=recovery_action.strategy,
                attempt=attempt,
                success=success,
                recovery_time=recovery_time,
                video_path=video_path
            )
            
            if success:
                self.recovery_stats['total_recoveries'] += 1
                self.recovery_stats['recoveries_by_strategy'][recovery_action.strategy.value] += 1
                logger.info(f"✅ 복구 성공: {recovery_action.strategy.value}")
            else:
                logger.warning(f"❌ 복구 실패: {recovery_action.strategy.value}")
            
        except Exception as recovery_error:
            logger.error(f"❌ 복구 과정 에러: {recovery_error}")
            success = False
        
        # 성공률 업데이트
        if self.recovery_stats['total_errors'] > 0:
            self.recovery_stats['recovery_success_rate'] = (
                self.recovery_stats['total_recoveries'] / 
                self.recovery_stats['total_errors'] * 100
            )
        
        return success
    
    def _get_recovery_action(self, error_type: type) -> Optional[RecoveryAction]:
        """에러 타입에 따른 복구 액션 반환"""
        # 정확한 타입 매칭 시도
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # 상위 클래스 매칭 시도
        for registered_type, action in self.recovery_strategies.items():
            if issubclass(error_type, registered_type):
                return action
        
        # 기본 복구 전략
        return self.recovery_strategies.get(Exception)
    
    async def _memory_cleanup_recovery(self) -> bool:
        """GPU 메모리 정리 복구"""
        try:
            logger.info("🧹 GPU 메모리 정리 중...")
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # GPU 캐시 정리
                torch.cuda.empty_cache()
                
                # 메모리 사용량 확인
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                
                logger.info(f"🔍 정리 후 GPU 메모리: {allocated:.1f}MB (예약: {reserved:.1f}MB)")
                
                # 추가 정리가 필요한 경우
                if allocated > 20000:  # 20GB 이상
                    logger.warning("⚠️ 메모리 사용량 여전히 높음 - 배치 크기 감소")
                    await self._reduce_batch_recovery()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 메모리 정리 실패: {e}")
            return False
    
    async def _fallback_recovery(self, error_type: type) -> bool:
        """소프트웨어 폴백 복구"""
        try:
            if error_type == NVENCSessionError:
                logger.info("🔄 NVENC → 소프트웨어 인코딩 폴백")
                self.use_software_encoding = True
                
            elif error_type == DecodingError:
                logger.info("🔄 하드웨어 → 소프트웨어 디코딩 폴백")
                # TODO: 소프트웨어 디코딩 플래그 설정
                
            return True
            
        except Exception as e:
            logger.error(f"❌ 폴백 설정 실패: {e}")
            return False
    
    async def _reduce_batch_recovery(self) -> bool:
        """배치 크기 감소 복구"""
        try:
            old_batch_size = self.current_batch_size
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            
            logger.info(f"📉 배치 크기 감소: {old_batch_size} → {self.current_batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 배치 크기 조정 실패: {e}")
            return False
    
    def _record_recovery(self, 
                        error_type: str,
                        error_message: str,
                        strategy: RecoveryStrategy,
                        attempt: int,
                        success: bool,
                        recovery_time: float,
                        video_path: Optional[str] = None):
        """복구 기록 저장"""
        record = RecoveryRecord(
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            strategy_used=strategy,
            attempt_number=attempt,
            success=success,
            recovery_time=recovery_time,
            video_path=video_path
        )
        
        self.recovery_history.append(record)
    
    def _create_error_output(self, video_path: str, error_message: str) -> Dict[str, Any]:
        """에러 발생시 빈 출력 생성"""
        return {
            'video_path': video_path,
            'success': False,
            'error': error_message,
            'output_path': None,
            'recovery_attempted': True,
            'final_failure': True
        }
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """복구 통계 반환"""
        recent_errors = len([r for r in self.recovery_history if r.timestamp > time.time() - 3600])
        recent_successes = len([r for r in self.recovery_history 
                               if r.timestamp > time.time() - 3600 and r.success])
        
        return {
            **self.recovery_stats,
            'recent_hour_errors': recent_errors,
            'recent_hour_success_rate': (recent_successes / max(1, recent_errors)) * 100,
            'current_batch_size': self.current_batch_size,
            'software_fallback_active': self.use_software_encoding
        }
    
    def print_recovery_summary(self):
        """복구 요약 출력"""
        stats = self.get_recovery_stats()
        
        print(f"""
🛡️ 복구 시스템 요약:
   • 총 에러 수: {stats['total_errors']}
   • 복구 성공: {stats['total_recoveries']}
   • 복구 성공률: {stats['recovery_success_rate']:.1f}%
   • 현재 배치 크기: {stats['current_batch_size']}
   • 소프트웨어 폴백: {'활성' if stats['software_fallback_active'] else '비활성'}
        """)
        
        if stats['errors_by_type']:
            print("📊 에러 타입별 통계:")
            for error_type, count in stats['errors_by_type'].items():
                print(f"   • {error_type}: {count}회")
        
        if stats['recoveries_by_strategy']:
            print("🔧 복구 전략별 사용:")
            for strategy, count in stats['recoveries_by_strategy'].items():
                print(f"   • {strategy}: {count}회")
    
    def reset_to_optimal_settings(self):
        """최적 설정으로 초기화"""
        self.current_batch_size = 4  # 기본값 복원
        self.use_software_encoding = False
        
        logger.info("🔄 복구 설정 초기화 완료")


if __name__ == "__main__":
    # 테스트 코드
    async def test_recovery_manager():
        print("🧪 StreamRecoveryManager 테스트 시작...")
        
        manager = StreamRecoveryManager()
        
        # 가짜 처리 함수
        async def mock_processor(video_path, **kwargs):
            if "fail" in video_path:
                raise GPUMemoryError("Test GPU memory error")
            return {"success": True, "output": f"processed_{Path(video_path).name}"}
        
        # 성공 케이스
        result = await manager.process_with_recovery("success_video.mp4", mock_processor)
        print(f"성공 결과: {result}")
        
        # 실패 후 복구 케이스
        result = await manager.process_with_recovery("fail_video.mp4", mock_processor)
        print(f"복구 결과: {result}")
        
        # 통계 출력
        manager.print_recovery_summary()
        
        print("✅ 테스트 완료!")
    
    asyncio.run(test_recovery_manager())