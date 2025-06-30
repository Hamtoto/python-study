import os, sys
import requests
from dotenv import load_dotenv
import time
from flask import session
# OpenAI API와 텔레그램 봇을 사용하여 챗봇 기능을 구현하는 코드입니다.
from openai import OpenAI 

# 봇 토큰 로드 (기존과 동일)
if getattr(sys, 'frozen', False):
    # 실행 파일로 빌드된 경우 .env 파일 예외처리
    dotenv_path = os.path.join(sys._MEIPASS, ".env")
    load_dotenv(dotenv_path)
else:
    load_dotenv()  

# OPENAI_API_KEY가 환경 변수에 설정되어 있다고 가정합니다.
# 만약 환경 변수에 없다면, client = OpenAI(api_key="YOUR_API_KEY_HERE") 와 같이 직접 키를 넣어주세요.
client = OpenAI()

def generate_response(messages):
    """
    OpenAI API를 사용하여 챗봇 응답을 생성합니다.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages,
            temperature=1,
            max_tokens=16384
        )
        return completion.choices[0].message.content.strip() 
    except Exception as e:
        print("OpenAI API 오류:", e)
        return "현재 응답을 생성할 수 없습니다."

def send_message(token, chat_id, text):
    # 텔레그램 봇을 사용하여 메시지를 전송합니다.
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text}
    response = requests.post(url, data=payload)
    return response

def get_updates(token, offset=None):
    # 텔레그램 봇의 업데이트를 가져옵니다.
    # offset은 이전에 처리된 업데이트의 ID를 지정하여 중복 처리를 방지합니다.
    # 기본적으로 None으로 설정되어 있으며, 이후에 처리된 업데이트의 ID를 사용하여 업데이트를 가져옵니다.
    # 이 함수는 텔레그램 API의 getUpdates 메서드를 사용합니다.
    # offset이 None이면 가장 최근 업데이트부터 가져오고, 
    # 이전에 처리된 업데이트의 ID를 지정하면 해당 ID 이후의 업데이트만 가져옵니다.
    # 이 방식은 폴링(polling) 방식으로,
    # 주기적으로 새로운 메시지를 확인하고 처리하는 데 사용됩니다.
    url = f'https://api.telegram.org/bot{token}/getUpdates'
    params = {'timeout': 100, 'offset': offset}
    response = requests.get(url, params=params)
    result = response.json()
    return result.get('result', [])

# 환경 변수에서 텔레그램 봇 토큰과 챗 ID 로드
token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID") # 이 값은 get_updates를 통해 동적으로 받아오므로, 실제 사용에서는 초기값으로만 쓰이거나 제거될 수 있습니다.

# Flask 웹 애플리케이션 초기화
from flask import Flask, request, jsonify
app = Flask(__name__)
# 세션 데이터 암호화용 키
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_in_production")

@app.route('/chat', methods=['POST'])
def chat():
    # 사용자의 메시지를 받아 챗봇 응답을 생성하고 세션에 저장합니다.
    history = session.get('history', [])
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"reply": "메시지를 입력해주세요."})
    history.append({"role": "user", "content": user_input})
    assistant_reply = generate_response(history)
    history.append({"role": "assistant", "content": assistant_reply})
    session['history'] = history
    return jsonify({"reply": assistant_reply})

def handle_updates(updates):
    # 텔레그램 봇의 업데이트를 처리합니다.
    # 이 함수는 get_updates로 가져온 업데이트 리스트를 순회하며,
    # 각 메시지에 대해 챗봇 응답을 생성하고 텔레그램으로 응답을 전송합니다.
    # 메시지의 길이가 텔레그램의 최대 메시지 길이 제한(4000자)을 초과하는 경우,
    # 메시지를 4000자 단위로 분할하여 전송합니다.
    # 이 방식은 텔레그램 API의 sendMessage 메서드를 사용하여 메시지를 전송합니다.
    MAX_MESSAGE_LENGTH = 4000 

    for update in updates:
        # 메시지가 있고 텍스트 내용이 있는 경우만 처리
        if 'message' in update and 'text' in update['message']:
            chat_id = update['message']['chat']['id'] # 메시지를 보낸 채팅방 ID
            user_text = update['message']['text'] # 사용자 입력 텍스트
            
            # 사용자 입력으로 챗봇 응답 생성
            bot_reply = generate_response([{"role": "user", "content": user_text}])
            
            # 응답 텍스트가 텔레그램 메시지 길이 제한을 초과하는지 확인 (수정됨)
            if len(bot_reply) > MAX_MESSAGE_LENGTH:
                # 텍스트를 MAX_MESSAGE_LENGTH 단위로 분할하여 전송
                for i in range(0, len(bot_reply), MAX_MESSAGE_LENGTH):
                    chunk = bot_reply[i:i + MAX_MESSAGE_LENGTH]
                    send_message(token, chat_id, chunk)
                    time.sleep(0.5) # 텔레그램 API 제한 방지를 위해 짧은 지연 추가
            else:
                # 응답이 길지 않으면 한 번에 전송
                send_message(token, chat_id, bot_reply)

def main():
    # 텔레그램 봇의 메인 루프를 시작합니다.
    # 이 함수는 get_updates를 사용하여 새로운 업데이트를 주기적으로 확인하고,
    # 새로운 메시지가 있는 경우 handle_updates 함수를 호출하여 처리합니다.
    # offset을 사용하여 이전에 처리된 업데이트를 건너뛰고,
    # 새로운 업데이트만 처리합니다.
    # 이 방식은 폴링(polling) 방식으로,
    # 주기적으로 텔레그램 서버에 새로운 메시지를 요청하여 처리합니다.
    offset = None # 이전에 처리된 업데이트의 ID를 저장
    while True:
        updates = get_updates(token, offset) # 새로운 업데이트 가져오기
        if updates:
            handle_updates(updates) # 업데이트 처리
            # 마지막 처리된 update_id 이후로 오프셋 설정
            offset = updates[-1]['update_id'] + 1
        time.sleep(1) # 1초 대기 후 다시 요청

if __name__ == "__main__":
    main()