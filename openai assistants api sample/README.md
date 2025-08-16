# OpenAI Assistants API Backend

FastAPI를 사용한 OpenAI Assistants API 백엔드 서버입니다.

## 설치 및 실행

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
`.env.example`을 `.env`로 복사하고 다음 값들을 설정하세요:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
ASSISTANT_ID=asst_your-assistant-id-here
```

3. 서버 실행:
```bash
python main.py
```

서버는 `https://hamtoto.kr:8010`에서 실행됩니다.

## API 엔드포인트

### 기본
- `GET /` - 루트 엔드포인트
- `GET /api/v1/health` - 헬스 체크

### 채팅
- `POST /api/v1/chat` - 간단한 채팅 인터페이스

### 스레드 관리
- `POST /api/v1/threads` - 새 스레드 생성
- `GET /api/v1/threads/{thread_id}/messages` - 스레드 메시지 조회

### 메시지 관리
- `POST /api/v1/threads/{thread_id}/messages` - 메시지 전송
- `POST /api/v1/threads/{thread_id}/runs` - 어시스턴트 실행
- `GET /api/v1/threads/{thread_id}/runs/{run_id}` - 실행 상태 조회

## API 문서

서버 실행 후 `https://hamtoto.kr:8010/docs`에서 Swagger UI를 통해 API 문서를 확인할 수 있습니다.

## 사용 예시

### 채팅 요청
```json
POST /api/v1/chat
{
  "message": "안녕하세요!",
  "thread_id": null
}
```

### 응답
```json
{
  "message": "안녕하세요! 무엇을 도와드릴까요?",
  "thread_id": "thread_abc123",
  "role": "assistant",
  "created_at": 1234567890
}
```