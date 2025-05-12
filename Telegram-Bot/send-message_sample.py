import os, sys
import requests
from dotenv import load_dotenv

"""
참조 Link
https://core.telegram.org/bots/api


.env 파일에 토큰 정보 및 챗 ID 기입
TELEGRAM_BOT_TOKEN='Your Bot TOKEN'
TELEGRAM_CHAT_ID='ChatID'
"""
# Bot Token 로드
if getattr(sys, 'frozen', False):
    # 실행 파일로 빌드된 경우 sys._MEIPASS 경로 참조해 분기 처리 
    dotenv_path = os.path.join(sys._MEIPASS, ".env")
    load_dotenv(dotenv_path)
else:
    load_dotenv()  

token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")

# 메시지 전송 함수
def send_message(token, chat_id, text):
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text}
    # sendMessage 엔드 포인트에 HTTP POST
    response = requests.post(url, data=payload, timeout=5)
    return response


if __name__ == "__main__":
    if not token or not chat_id:
        sys.exit("환경 변수 TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
    send_message(token, chat_id, 'Test')
