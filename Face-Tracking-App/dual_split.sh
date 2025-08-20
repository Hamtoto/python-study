#!/bin/bash

# Face-Tracking-App DUAL_SPLIT 모드 실행 스크립트
echo "🎯 Face-Tracking-App DUAL_SPLIT 모드 시작 (화면 분할 2인 추적)..."

# 가상환경 활성화
source .venv/bin/activate

# Python 스크립트 실행 (DUAL_SPLIT 모드)
python src/face_tracker/main.py --mode dual_split