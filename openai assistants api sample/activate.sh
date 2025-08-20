#!/bin/bash

# 가상환경 활성화 스크립트
cd "$(dirname "$0")"
source venv/bin/activate
echo "가상환경이 활성화되었습니다."
echo "서버 실행: python main.py"
echo "비활성화: deactivate"
exec bash