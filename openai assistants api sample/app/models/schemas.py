from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    message: str
    thread_id: str
    role: str = "assistant"
    created_at: Optional[int] = None


class ThreadCreate(BaseModel):
    """스레드 생성 요청 스키마"""
    pass


class ThreadResponse(BaseModel):
    """스레드 응답 스키마"""
    id: str
    object: str
    created_at: int
    metadata: Optional[dict] = None


class MessageCreate(BaseModel):
    """메시지 생성 요청 스키마"""
    content: str
    role: str = "user"


class MessageResponse(BaseModel):
    """메시지 응답 스키마"""
    id: str
    object: str
    created_at: int
    thread_id: str
    role: str
    content: List[dict]
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None


class RunResponse(BaseModel):
    """실행 응답 스키마"""
    id: str
    object: str
    created_at: int
    assistant_id: str
    thread_id: str
    status: str
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    expires_at: Optional[int] = None


class HealthCheck(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str = "healthy"
    timestamp: datetime
    version: str = "1.0.0"