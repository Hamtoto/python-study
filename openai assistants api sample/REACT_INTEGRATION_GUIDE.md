# OpenAI Assistants API 연동 가이드

## 🔗 우리 백엔드 API 정보
- **Base URL**: `https://hamtoto.kr/openai-api`
- **메인 엔드포인트**: `POST /api/v1/chat`
- **테스트 완료**: 연속 대화, 스레드 관리 모두 정상 작동
- **Assistant 설정**: 백엔드에서 디지털 사이니지 데이터 분석 전문가로 설정 완료

## 📡 API 엔드포인트

### 1. 채팅 API (메인 기능)
```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "사용자 메시지 + 시스템 데이터",
  "thread_id": "thread_abc123" // 선택사항, 없으면 새 스레드 생성
}
```

**응답:**
```json
{
  "message": "Assistant의 분석 응답",
  "thread_id": "thread_abc123",
  "role": "assistant", 
  "created_at": 1754481000
}
```

### 2. 기타 엔드포인트
```http
GET /api/v1/health                              // 헬스 체크
POST /api/v1/threads                            // 새 스레드 생성
GET /api/v1/threads/{thread_id}/messages        // 메시지 히스토리
```

## 📝 기존 코드 수정 방법

### 1. OpenAI API 함수 추가

기존 코드 상단에 이 함수를 추가하세요:

```javascript
const invokeOpenAIAssistant = async (message, systemData, threadId = null) => {
  // 시스템 데이터를 포함한 사용자 메시지 구성
  const messageWithData = `
사용자 질문: ${message}

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
  return data.message; // OpenAI Assistant 응답
};
```

### 2. handleSendMessage 함수 수정

기존 `handleSendMessage` 함수에서 이 부분만 찾아서 교체하세요:

**기존 코드 (약 52-60줄):**
```javascript
const systemData = await collectSystemData();
const prompt = `
당신은 디지털 사이니지 광고 플랫폼의 데이터 분석 전문가입니다. 
... (긴 프롬프트) ...
사용자 질문: ${messageContent}
`;

const response = await InvokeLLM({
  prompt,
  add_context_from_internet: false
});
```

**새로운 코드:**
```javascript
const systemData = await collectSystemData();
const response = await invokeOpenAIAssistant(messageContent, systemData, currentConversationId);
```

### 3. 완료!

- ✅ 기존 UI/UX 그대로 유지
- ✅ 실시간 시스템 데이터 포함
- ✅ 연속 대화 기능 (thread_id 자동 관리)
- ✅ 기존 localStorage 저장 로직 그대로 사용
- ✅ OpenAI Assistant가 백엔드에서 전문가 역할 수행

## 🔍 변경사항 요약

1. **API 함수 1개 추가**: `invokeOpenAIAssistant`
2. **코드 10줄 → 2줄로 단축**: 프롬프트 구성 + InvokeLLM 호출 → 간단한 API 호출
3. **나머지는 모두 그대로**: 데이터 수집, UI, 저장 로직 등

## 💡 작동 원리

1. **프론트엔드**: 사용자 메시지 + 실시간 시스템 데이터를 우리 백엔드로 전송
2. **우리 백엔드**: 받은 데이터를 OpenAI Assistant에게 전달
3. **OpenAI Assistant**: 미리 설정된 "디지털 사이니지 데이터 분석 전문가" 역할로 분석 응답
4. **프론트엔드**: 분석 결과 받아서 화면에 표시

## 🧪 테스트 확인사항

- 메시지 전송 시 OpenAI Assistant 응답이 오는지 확인
- 연속 대화가 이전 내용을 기억하는지 확인  
- 실시간 시스템 데이터가 분석에 반영되는지 확인
- 응답 시간이 3-10초 정도 소요되는 것이 정상

## 🚨 주의사항

- **프롬프트 중복 제거**: 기존 긴 프롬프트는 백엔드 Assistant에 이미 설정되어 있음
- **스레드 관리**: `currentConversationId`를 `thread_id`로 그대로 사용 가능
- **에러 처리**: 기존 에러 처리 로직 그대로 사용