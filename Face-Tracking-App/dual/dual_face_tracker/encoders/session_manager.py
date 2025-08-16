"""
NVENC 세션 관리자

RTX 5090 등 고성능 GPU에서 NVENC 동시 세션 수 제한을 관리합니다.
- 동시 세션 수 제한 (RTX 5090: 최대 2-3개)
- 세션 풀 관리 및 대기열
- 자동 폴백 메커니즘
- 세션 상태 모니터링

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0
"""

import asyncio
import threading
import time
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import logging
from enum import Enum

from ..utils.logger import UnifiedLogger
from ..utils.exceptions import (
    EncodingError,
    HardwareError,
    ResourceError
)


class SessionState(Enum):
    """세션 상태"""
    IDLE = "idle"
    ACTIVE = "active"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class NvencSession:
    """NVENC 세션 정보"""
    session_id: int
    state: SessionState = SessionState.IDLE
    created_at: float = field(default_factory=time.perf_counter)
    last_used: float = field(default_factory=time.perf_counter)
    use_count: int = 0
    error_count: int = 0
    encoder_instance: Optional[Any] = None
    
    def mark_used(self):
        """세션 사용 기록"""
        self.last_used = time.perf_counter()
        self.use_count += 1
        
    def mark_error(self):
        """에러 기록"""
        self.error_count += 1
        self.state = SessionState.ERROR


class NvencSessionManager:
    """
    NVENC 세션 관리자
    
    GPU별 동시 세션 수 제한을 관리하고, 세션 풀을 통해
    효율적인 리소스 사용을 보장합니다.
    """
    
    def __init__(self, max_concurrent_sessions: int = 2):
        """
        Args:
            max_concurrent_sessions: 최대 동시 세션 수 (RTX 5090: 2개 권장)
        """
        self.max_concurrent_sessions = max_concurrent_sessions
        self.logger = UnifiedLogger("NvencSessionManager")
        
        # 세션 관리
        self._sessions: Dict[int, NvencSession] = {}
        self._active_sessions: List[int] = []
        self._session_semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._session_lock = threading.Lock()
        self._next_session_id = 1
        
        # 대기열
        self._waiting_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        
        # 통계
        self._total_sessions_created = 0
        self._total_sessions_closed = 0
        self._total_errors = 0
        
        self.logger.info(f"🔧 NVENC 세션 관리자 초기화 (최대 동시 세션: {max_concurrent_sessions})")
    
    async def get_session(self, timeout: float = 30.0) -> int:
        """
        세션 할당 요청
        
        Args:
            timeout: 세션 대기 시간 제한 (초)
            
        Returns:
            int: 할당된 세션 ID
            
        Raises:
            ResourceError: 세션 할당 실패
        """
        start_time = time.perf_counter()
        
        try:
            # 세마포어로 동시 세션 수 제한
            await asyncio.wait_for(
                self._session_semaphore.acquire(), 
                timeout=timeout
            )
            
            # 세션 생성
            with self._session_lock:
                session_id = self._next_session_id
                self._next_session_id += 1
                
                session = NvencSession(
                    session_id=session_id,
                    state=SessionState.ACTIVE
                )
                
                self._sessions[session_id] = session
                self._active_sessions.append(session_id)
                self._total_sessions_created += 1
            
            wait_time = time.perf_counter() - start_time
            self.logger.debug(f"🎯 세션 할당: ID={session_id}, 대기시간={wait_time:.2f}s")
            
            return session_id
            
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            raise ResourceError(
                f"NVENC 세션 할당 타임아웃 (대기: {elapsed:.1f}s, "
                f"활성 세션: {len(self._active_sessions)}/{self.max_concurrent_sessions})"
            )
    
    async def release_session(self, session_id: int):
        """
        세션 해제
        
        Args:
            session_id: 해제할 세션 ID
        """
        with self._session_lock:
            if session_id not in self._sessions:
                self.logger.warning(f"⚠️ 존재하지 않는 세션 해제 시도: {session_id}")
                return
            
            session = self._sessions[session_id]
            session.state = SessionState.CLOSED
            
            if session_id in self._active_sessions:
                self._active_sessions.remove(session_id)
            
            del self._sessions[session_id]
            self._total_sessions_closed += 1
        
        # 세마포어 해제
        self._session_semaphore.release()
        
        self.logger.debug(f"🔓 세션 해제: ID={session_id}")
    
    def mark_session_error(self, session_id: int, error: Exception):
        """
        세션 에러 기록
        
        Args:
            session_id: 세션 ID
            error: 발생한 에러
        """
        with self._session_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.mark_error()
                self._total_errors += 1
                
                self.logger.error(f"❌ 세션 에러: ID={session_id}, Error={error}")
    
    @asynccontextmanager
    async def acquire_session(self, timeout: float = 30.0):
        """
        세션 컨텍스트 매니저
        
        사용 예:
            async with session_manager.acquire_session() as session_id:
                # NVENC 인코딩 작업
                pass
        """
        session_id = None
        try:
            session_id = await self.get_session(timeout=timeout)
            yield session_id
            
        except Exception as e:
            if session_id is not None:
                self.mark_session_error(session_id, e)
            raise
            
        finally:
            if session_id is not None:
                await self.release_session(session_id)
    
    def get_session_info(self, session_id: int) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        with self._session_lock:
            if session_id not in self._sessions:
                return None
            
            session = self._sessions[session_id]
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "created_at": session.created_at,
                "last_used": session.last_used,
                "use_count": session.use_count,
                "error_count": session.error_count,
                "age": time.perf_counter() - session.created_at
            }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """세션 관리자 통계"""
        with self._session_lock:
            return {
                "max_concurrent_sessions": self.max_concurrent_sessions,
                "active_sessions": len(self._active_sessions),
                "total_sessions": len(self._sessions),
                "total_created": self._total_sessions_created,
                "total_closed": self._total_sessions_closed,
                "total_errors": self._total_errors,
                "utilization": len(self._active_sessions) / self.max_concurrent_sessions,
                "error_rate": (
                    self._total_errors / max(1, self._total_sessions_created)
                ) * 100
            }
    
    async def wait_for_available_session(self, max_wait_time: float = 60.0) -> bool:
        """
        사용 가능한 세션이 생길 때까지 대기
        
        Args:
            max_wait_time: 최대 대기 시간 (초)
            
        Returns:
            bool: 세션 사용 가능 여부
        """
        start_time = time.perf_counter()
        
        while (time.perf_counter() - start_time) < max_wait_time:
            if len(self._active_sessions) < self.max_concurrent_sessions:
                return True
                
            await asyncio.sleep(0.1)  # 100ms 대기
        
        return False
    
    def force_cleanup_sessions(self):
        """강제로 모든 세션 정리 (비상용)"""
        self.logger.warning("🚨 강제 세션 정리 시작")
        
        with self._session_lock:
            for session_id in list(self._sessions.keys()):
                session = self._sessions[session_id]
                session.state = SessionState.CLOSED
                
                # 인코더 인스턴스가 있다면 정리
                if session.encoder_instance:
                    try:
                        session.encoder_instance.close()
                    except:
                        pass
            
            self._sessions.clear()
            self._active_sessions.clear()
        
        # 세마포어 초기화
        self._session_semaphore = asyncio.Semaphore(self.max_concurrent_sessions)
        
        self.logger.warning("🧹 강제 세션 정리 완료")
    
    async def shutdown(self):
        """세션 관리자 종료"""
        self.logger.info("🛑 NVENC 세션 관리자 종료 중...")
        
        self._shutdown_event.set()
        
        # 모든 활성 세션 해제
        with self._session_lock:
            for session_id in list(self._active_sessions):
                await self.release_session(session_id)
        
        self.logger.info("✅ NVENC 세션 관리자 종료 완료")


# 전역 세션 관리자 (싱글턴)
_global_session_manager: Optional[NvencSessionManager] = None


def get_global_session_manager(max_concurrent_sessions: int = 2) -> NvencSessionManager:
    """전역 NVENC 세션 관리자 반환"""
    global _global_session_manager
    
    if _global_session_manager is None:
        _global_session_manager = NvencSessionManager(max_concurrent_sessions)
    
    return _global_session_manager


def reset_global_session_manager():
    """전역 세션 관리자 초기화 (테스트용)"""
    global _global_session_manager
    
    if _global_session_manager:
        asyncio.create_task(_global_session_manager.shutdown())
    
    _global_session_manager = None