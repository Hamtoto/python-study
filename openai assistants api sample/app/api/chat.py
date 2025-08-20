from fastapi import APIRouter, HTTPException
from ..models.schemas import ChatRequest, ChatResponse, HealthCheck
from ..core.openai_client import openai_client
from datetime import datetime

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    간단한 채팅 인터페이스
    thread_id가 없으면 새로운 스레드를 생성하고,
    있으면 기존 스레드를 사용
    """
    try:
        thread_id = request.thread_id
        
        if not thread_id:
            thread = await openai_client.create_thread()
            thread_id = thread.id
        
        response = await openai_client.get_assistant_response(
            thread_id=thread_id,
            message_content=request.message
        )
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to get response from assistant")
        
        return ChatResponse(
            message=response["content"],
            thread_id=thread_id,
            role=response["role"],
            created_at=response.get("created_at")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now()
    )