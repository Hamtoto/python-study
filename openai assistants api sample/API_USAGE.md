# OpenAI Assistants API 엔드포인트

**Base URL**: `https://hamtoto.kr/openai-api`

## 채팅 API

### 간단한 채팅 인터페이스
```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "사용자 메시지",
  "thread_id": "thread_abc123" // 선택사항, 없으면 새 스레드 생성
}
```

**응답:**
```json
{
  "message": "Assistant의 응답",
  "thread_id": "thread_abc123",
  "role": "assistant",
  "created_at": 1754481000
}
```

## 스레드 관리

### 새 스레드 생성
```http
POST /api/v1/threads
Content-Type: application/json

{}
```

**응답:**
```json
{
  "id": "thread_abc123",
  "object": "thread", 
  "created_at": 1754481000,
  "metadata": null
}
```

### 스레드 메시지 조회
```http
GET /api/v1/threads/{thread_id}/messages
```

**응답:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "msg_abc123",
      "object": "thread.message",
      "created_at": 1754481000,
      "thread_id": "thread_abc123",
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": {
            "value": "사용자 메시지 내용"
          }
        }
      ]
    }
  ]
}
```

## 메시지 관리

### 스레드에 메시지 추가
```http
POST /api/v1/threads/{thread_id}/messages
Content-Type: application/json

{
  "content": "메시지 내용",
  "role": "user"
}
```

### Assistant 실행
```http
POST /api/v1/threads/{thread_id}/runs
```

**응답:**
```json
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1754481000,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "queued"
}
```

### 실행 상태 조회
```http
GET /api/v1/threads/{thread_id}/runs/{run_id}
```

## 기본 엔드포인트

### 루트 엔드포인트
```http
GET /
```

### 헬스 체크
```http
GET /api/v1/health
```

**응답:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-11T10:30:00Z",
  "version": "1.0.0"
}
```