from openai import OpenAI
from .config import settings


class OpenAIClient:
    """OpenAI API 클라이언트 래퍼 클래스"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.assistant_id = settings.assistant_id
    
    async def create_thread(self):
        """새로운 스레드 생성"""
        thread = self.client.beta.threads.create()
        return thread
    
    async def send_message(self, thread_id: str, content: str):
        """스레드에 메시지 전송"""
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        return message
    
    async def run_assistant(self, thread_id: str, assistant_id: str = None):
        """어시스턴트 실행"""
        if not assistant_id:
            assistant_id = self.assistant_id
            
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        return run
    
    async def wait_for_completion(self, thread_id: str, run_id: str):
        """실행 완료 대기"""
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                return run
            elif run.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run failed with status: {run.status}")
                
            import asyncio
            await asyncio.sleep(0.5)
    
    async def get_messages(self, thread_id: str):
        """스레드의 메시지 조회"""
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order="asc"
        )
        return messages
    
    async def get_assistant_response(self, thread_id: str, message_content: str):
        """메시지를 보내고 어시스턴트 응답을 받아옴"""
        await self.send_message(thread_id, message_content)
        run = await self.run_assistant(thread_id)
        await self.wait_for_completion(thread_id, run.id)
        messages = await self.get_messages(thread_id)
        
        # 최신 assistant 메시지를 찾기 위해 역순으로 탐색
        for message in reversed(messages.data):
            if message.role == "assistant":
                return {
                    "role": message.role,
                    "content": message.content[0].text.value,
                    "created_at": message.created_at
                }
        
        return None


openai_client = OpenAIClient()