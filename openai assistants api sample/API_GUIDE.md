# OpenAI Assistants API 백엔드 가이드

## 📋 프로젝트 개요

FastAPI 기반의 OpenAI Assistants API 백엔드 서버입니다. 디지털 사이니지 데이터 분석 전문가로 설정된 Assistant를 통해 대화형 AI 서비스를 제공합니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
```

**.env 파일 설정:**
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
ASSISTANT_ID=asst_your-assistant-id-here
HOST=0.0.0.0
PORT=8010
DEBUG=true
CORS_ORIGINS=["http://localhost:3000","https://hamtoto.kr"]
```

### 2. 서버 실행

```bash
# 개발 모드
python main.py

# 또는 스크립트 사용
./run.sh
```

서버는 기본적으로 `https://hamtoto.kr:8010`에서 실행됩니다.

## 📊 API 엔드포인트

### 기본 엔드포인트

#### 루트 엔드포인트
```http
GET /
```
API 정보와 문서 링크를 반환합니다.

#### 헬스 체크
```http
GET /api/v1/health
```
서버 상태를 확인합니다.

### 채팅 API

#### 간단한 채팅 인터페이스
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

### 스레드 관리 API

#### 새 스레드 생성
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

#### 스레드 메시지 조회
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

### 메시지 관리 API

#### 스레드에 메시지 추가
```http
POST /api/v1/threads/{thread_id}/messages
Content-Type: application/json

{
  "content": "메시지 내용",
  "role": "user"
}
```

#### Assistant 실행
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

#### 실행 상태 조회
```http
GET /api/v1/threads/{thread_id}/runs/{run_id}
```

## 🏗️ 프로젝트 구조

```
openai assistants api sample/
├── app/
│   ├── __init__.py
│   ├── api/                    # API 라우터
│   │   ├── __init__.py
│   │   ├── chat.py            # 채팅 API
│   │   ├── messages.py        # 메시지 관리 API
│   │   └── threads.py         # 스레드 관리 API
│   ├── core/                  # 핵심 모듈
│   │   ├── __init__.py
│   │   ├── config.py          # 설정 관리
│   │   └── openai_client.py   # OpenAI 클라이언트
│   ├── models/                # 데이터 모델
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic 스키마
│   └── utils/                 # 유틸리티
│       └── __init__.py
├── main.py                    # FastAPI 앱 진입점
├── requirements.txt           # Python 의존성
├── run.sh                     # 실행 스크립트
└── activate.sh               # 가상환경 활성화
```

## 🔧 핵심 컴포넌트

### 1. OpenAI 클라이언트 (`app/core/openai_client.py`)
- OpenAI API와의 통신 담당
- Assistant 실행 및 응답 처리
- 스레드 생성 및 관리

### 2. 데이터 스키마 (`app/models/schemas.py`)
- **ChatRequest/ChatResponse**: 채팅 인터페이스용
- **ThreadResponse**: 스레드 생성 응답
- **MessageCreate/MessageResponse**: 메시지 관리용
- **RunResponse**: Assistant 실행 상태용
- **HealthCheck**: 헬스 체크용

### 3. API 라우터
- **chat.py**: 간단한 채팅 인터페이스 제공
- **threads.py**: 스레드 생성 및 조회
- **messages.py**: 메시지 추가 및 Assistant 실행

## 💡 사용 시나리오

### 시나리오 1: 새로운 대화 시작
1. `POST /api/v1/chat` (thread_id 없이)
2. 서버가 새 스레드 생성
3. Assistant 응답 반환

### 시나리오 2: 기존 대화 계속
1. `POST /api/v1/chat` (기존 thread_id 포함)
2. 기존 스레드에서 대화 계속
3. 컨텍스트를 유지한 Assistant 응답

### 시나리오 3: 고급 제어
1. `POST /api/v1/threads` - 스레드 생성
2. `POST /api/v1/threads/{thread_id}/messages` - 메시지 추가
3. `POST /api/v1/threads/{thread_id}/runs` - Assistant 실행
4. `GET /api/v1/threads/{thread_id}/runs/{run_id}` - 상태 확인

## 🌐 클라이언트 연동 예시

### JavaScript/React 연동

```javascript
// 새로운 대화 시작
const startNewChat = async (message) => {
  const response = await fetch('https://hamtoto.kr/openai-api/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  
  const data = await response.json();
  return data; // { message, thread_id, role, created_at }
};

// 기존 대화 계속
const continueChat = async (message, threadId) => {
  const response = await fetch('https://hamtoto.kr/openai-api/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      message, 
      thread_id: threadId 
    })
  });
  
  return await response.json();
};

// 시스템 데이터와 함께 전송 (디지털 사이니지 전용)
const chatWithSystemData = async (userMessage, systemData, threadId = null) => {
  const messageWithData = `
사용자 질문: ${userMessage}

현재 시스템 데이터:
- 타겟광고 수: ${systemData.targetedAds.length}개
- 디바이스 수: ${systemData.signages.length}개  
- 최근 시청 이벤트: ${systemData.viewEvents.length}개

타겟광고 데이터: ${JSON.stringify(systemData.targetedAds.slice(0, 5))}
디바이스 데이터: ${JSON.stringify(systemData.signages.slice(0, 5))}
시청 이벤트 데이터: ${JSON.stringify(systemData.viewEvents.slice(0, 10))}
  `;

  const response = await fetch('https://hamtoto.kr/openai-api/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: messageWithData,
      thread_id: threadId
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.message;
};
```

### Python 클라이언트 예시

```python
import requests
import json

class OpenAIAssistantClient:
    def __init__(self, base_url="https://hamtoto.kr/openai-api"):
        self.base_url = base_url
        
    def chat(self, message, thread_id=None):
        """간단한 채팅 인터페이스"""
        payload = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id
            
        response = requests.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
        
    def create_thread(self):
        """새 스레드 생성"""
        response = requests.post(f"{self.base_url}/api/v1/threads", json={})
        response.raise_for_status()
        return response.json()
        
    def get_messages(self, thread_id):
        """스레드 메시지 조회"""
        response = requests.get(f"{self.base_url}/api/v1/threads/{thread_id}/messages")
        response.raise_for_status()
        return response.json()

# 사용 예시
client = OpenAIAssistantClient()

# 새 대화 시작
result = client.chat("안녕하세요!")
print(f"응답: {result['message']}")
print(f"스레드 ID: {result['thread_id']}")

# 기존 대화 계속
thread_id = result['thread_id']
result2 = client.chat("이전에 뭘 말했는지 기억하나요?", thread_id)
print(f"응답: {result2['message']}")
```

## 🔒 보안 및 인증

현재는 인증 없이 동작하지만, 프로덕션 환경에서는 다음을 고려해야 합니다:

- API 키 기반 인증
- 요청 속도 제한 (Rate Limiting)
- CORS 설정 최적화
- HTTPS 강제 사용

## 🧪 테스트

### Swagger UI
서버 실행 후 `https://hamtoto.kr:8010/docs`에서 API 문서 및 테스트 가능

### Postman 컬렉션
`OpenAI_Assistants_API.postman_collection.json` 파일을 Postman에 import하여 테스트

### cURL 예시
```bash
# 헬스 체크
curl -X GET "https://hamtoto.kr/openai-api/api/v1/health"

# 새 대화 시작
curl -X POST "https://hamtoto.kr/openai-api/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕하세요!"}'

# 기존 대화 계속
curl -X POST "https://hamtoto.kr/openai-api/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "이전 대화를 기억하세요?", "thread_id": "thread_abc123"}'
```

## 📝 환경 설정 상세

### 개발 환경
- Python 3.8+
- FastAPI
- OpenAI Python SDK
- Uvicorn (ASGI 서버)

### 프로덕션 환경
- Docker 컨테이너화 권장
- Nginx 리버스 프록시
- SSL/TLS 인증서 적용
- 로그 모니터링 시스템

## 🚨 문제 해결

### 일반적인 오류

1. **OpenAI API 키 오류**
   - `.env` 파일에 올바른 API 키 설정 확인

2. **Assistant ID 오류**  
   - OpenAI 플랫폼에서 생성한 Assistant ID 확인

3. **CORS 오류**
   - `config.py`에서 허용할 origin 추가

4. **포트 충돌**
   - `config.py`에서 다른 포트로 변경

### 로그 확인
```bash
# 서버 로그 실시간 확인
tail -f logs/app.log

# 특정 에러 검색
grep "ERROR" logs/app.log
```

## 🔄 업데이트 및 배포

### 의존성 업데이트
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

### 서버 재시작
```bash
# 개발 환경
python main.py

# 프로덕션 환경 (PM2 사용 시)
pm2 restart openai-assistant-api
```

## 📞 지원

- **API 문서**: `https://hamtoto.kr:8010/docs`
- **GitHub Issues**: 프로젝트 저장소의 Issues 탭
- **이메일 지원**: 관리자에게 문의

---

이 가이드를 통해 OpenAI Assistants API 백엔드를 효과적으로 사용할 수 있습니다. 추가 질문이나 기능 요청이 있으면 언제든 문의해 주세요.