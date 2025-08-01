#!/bin/bash

# Face-Tracking-App SINGLE 모드 실행 스크립트
echo "🎯 Face-Tracking-App SINGLE 모드 시작 (1인 화자)..."

# 가상환경 활성화
source .venv/bin/activate

# Python 스크립트 실행 (SINGLE 모드)
python src/face_tracker/main.py --mode single