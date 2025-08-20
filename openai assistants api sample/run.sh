#!/bin/bash

# 가상환경 활성화 및 서버 실행 스크립트
cd "$(dirname "$0")"
source venv/bin/activate
python main.py