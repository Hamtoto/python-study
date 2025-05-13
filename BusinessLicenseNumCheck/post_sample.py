import os
import requests

API_KEY = os.getenv("DATA_GO_KR_API_KEY")
URL = f"https://api.odcloud.kr/api/nts-businessman/v1/status?serviceKey={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

data = {
    "b_no" : [
        "7801702311"
    ]
}

response = requests.post(URL, headers=headers , json=data)
print(response.json())