from fastapi import APIRouter, HTTPException
from ..models.schemas import MessageCreate, MessageResponse, RunResponse
from ..core.openai_client import openai_client

router = APIRouter()


@router.post("/threads/{thread_id}/messages")
async def create_message(thread_id: str, request: MessageCreate):
    """스레드에 메시지 추가"""
    try:
        message = await openai_client.send_message(
            thread_id=thread_id,
            content=request.content
        )
        
        return {
            "id": message.id,
            "object": message.object,
            "created_at": message.created_at,
            "thread_id": message.thread_id,
            "role": message.role,
            "content": [{"type": content.type, "text": content.text.value if hasattr(content, 'text') else str(content)} for content in message.content],
            "assistant_id": message.assistant_id,
            "run_id": message.run_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, assistant_id: str = None):
    """스레드에서 어시스턴트 실행"""
    try:
        run = await openai_client.run_assistant(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        return {
            "id": run.id,
            "object": run.object,
            "created_at": run.created_at,
            "assistant_id": run.assistant_id,
            "thread_id": run.thread_id,
            "status": run.status,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "failed_at": run.failed_at,
            "cancelled_at": run.cancelled_at,
            "expires_at": run.expires_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str):
    """실행 상태 조회"""
    try:
        run = openai_client.client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        return {
            "id": run.id,
            "object": run.object,
            "created_at": run.created_at,
            "assistant_id": run.assistant_id,
            "thread_id": run.thread_id,
            "status": run.status,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "failed_at": run.failed_at,
            "cancelled_at": run.cancelled_at,
            "expires_at": run.expires_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))