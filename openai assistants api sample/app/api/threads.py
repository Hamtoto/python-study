from fastapi import APIRouter, HTTPException
from ..models.schemas import ThreadCreate, ThreadResponse
from ..core.openai_client import openai_client

router = APIRouter()


@router.post("/threads", response_model=ThreadResponse)
async def create_thread(request: ThreadCreate = ThreadCreate()):
    """새로운 스레드 생성"""
    try:
        thread = await openai_client.create_thread()
        
        return ThreadResponse(
            id=thread.id,
            object=thread.object,
            created_at=thread.created_at,
            metadata=thread.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """스레드의 모든 메시지 조회"""
    try:
        messages = await openai_client.get_messages(thread_id)
        
        message_list = []
        for message in messages.data:
            message_list.append({
                "id": message.id,
                "object": message.object,
                "created_at": message.created_at,
                "thread_id": message.thread_id,
                "role": message.role,
                "content": [{"type": content.type, "text": content.text.value if hasattr(content, 'text') else str(content)} for content in message.content],
                "assistant_id": message.assistant_id,
                "run_id": message.run_id
            })
        
        return {
            "object": "list",
            "data": message_list,
            "first_id": messages.first_id,
            "last_id": messages.last_id,
            "has_more": messages.has_more
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))