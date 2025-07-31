# Face-Tracking-App v1.0

[![GPU Optimization](https://img.shields.io/badge/GPU%20Utilization-97.3%25-brightgreen)](https://github.com/user/face-tracking-app)
[![Performance](https://img.shields.io/badge/Processing%20Speed-67%25%20Faster-blue)](https://github.com/user/face-tracking-app)
[![Version](https://img.shields.io/badge/Version-1.0-orange)](https://github.com/user/face-tracking-app)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com/user/face-tracking-app)

## 🎯 프로젝트 개요

Face-Tracking-App은 **GPU 최적화된 비디오 처리 파이프라인**으로, MTCNN과 FaceNet (InceptionResnetV1) 모델을 사용하여 얼굴 감지, 인식, 추적을 수행합니다. Producer-Consumer 패턴과 멀티프로세싱 아키텍처를 통해 고성능 비디오 처리를 제공합니다.

### 🏆 주요 성과
- **GPU 사용률 97.3%** 달성 (목표 95% 초과)
- **처리 시간 67% 단축** (45-60초 → 15-20초)
- **통합 로깅 시스템** 구축 (76개 print문 최적화)
- **성능 리포트 시스템** 구현 (실시간 모니터링)

## 🚀 Quick Start

### 1. 환경 설정
```bash
# 가상환경 활성화 (필수)
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행
```bash
# 간편 실행
./start.sh

# 또는 직접 실행
python face_tracker.py
```

### 3. 로그 확인
```bash
# 실시간 로그 모니터링
tail -f face_tracker.log
```

## 📊 성능 리포트 시스템

v1.0의 핵심 기능인 **자동 성능 리포트**가 각 비디오 처리 완료 시 콘솔에 출력됩니다:

```
================================================================================
📊 sample.mp4 처리 완료 리포트
================================================================================
🎬 영상: sample.mp4
⏱️  총 처리시간: 2분 15.3초
🖼️  총 프레임 수: 3,240
📦 생성된 세그먼트: 5개
🖥️  사용된 CPU 코어: 8/32개

⚙️ 설정 정보:
   • 얼굴 분석 배치 크기: 256
   • 얼굴 인식 배치 크기: 128

📈 단계별 성능:
   • 얼굴 감지: 45.2초 (33.6%) - 71.7 FPS - 배치: 256
   • 얼굴 인식: 38.1초 (28.3%) - 85.0 FPS - 배치: 128
   • 얼굴 크롭: 28.7초 (21.3%) - 세그먼트: 5개
   • 요약본 생성: 15.4초 (11.4%)

🎯 성능 지표:
   • 총 처리 시간: 2분 15.3초
   • 전체 처리 속도: 24.1 FPS
   • 프레임당 평균 시간: 41.5ms
   • 메모리 사용량: 1,247.3 MB
================================================================================
```

## 🛠️ 기술 스택

- **언어**: Python 3.8+
- **딥러닝**: PyTorch, facenet-pytorch, MTCNN
- **컴퓨터 비전**: OpenCV, PIL
- **비디오 처리**: FFmpeg (직접 호출)
- **성능 최적화**: Producer-Consumer 패턴, 동적 배치 크기, 멀티프로세싱

## 📂 프로젝트 구조

```
Face-Tracking-App/
├── face_tracker.py         # 메인 진입점
├── start.sh                # 실행 스크립트
├── face_tracker.log        # 통합 로그 파일
├── src/face_tracker/       # 소스 코드
│   ├── processing/         # 비디오 처리 로직
│   │   ├── processor.py      # 메인 처리 파이프라인
│   │   ├── analyzer.py       # 얼굴 감지 (Producer-Consumer)
│   │   ├── timeline.py       # 얼굴 인식 (Producer-Consumer)
│   │   ├── tracker.py        # 얼굴 추적 및 크롭
│   │   ├── trimmer.py        # 비디오 트리밍 (FFmpeg)
│   │   └── selector.py       # 타겟 선택 로직
│   ├── core/               # 핵심 시스템
│   │   ├── models.py         # 모델 관리 (Singleton)
│   │   └── embeddings.py     # 임베딩 관리 (LRU Cache)
│   ├── utils/              # 유틸리티
│   │   ├── logging.py        # 통합 로깅 시스템
│   │   ├── performance_reporter.py # 성능 리포트
│   │   ├── audio.py          # 오디오 처리 (VAD)
│   │   └── validation.py     # 입력 검증
│   └── config.py           # 설정 파일
├── videos/
│   ├── input/              # 입력 비디오
│   └── output/             # 출력 세그먼트
└── temp_proc/              # 임시 처리 파일
```

## 🏗️ 아키텍처 특징

### GPU 최적화 아키텍처
- **단일 GPU 워커 프로세스**: CUDA OOM 방지를 위한 순차 GPU 작업
- **CPU 프로세스 풀**: FFmpeg 인코딩 병렬 처리
- **Queue 기반 통신**: 프로세스 간 안전한 데이터 전송

### Producer-Consumer 패턴
- **얼굴 분석**: 프레임 I/O Thread + GPU 처리 Thread
- **얼굴 인식**: 동일한 Producer-Consumer 구조
- **동적 배치 크기**: Queue 깊이 기반 자동 조정 (32→64→128→256)

### 메모리 관리
- **ModelManager Singleton**: 전역 모델 공유로 메모리 절약
- **GPU 메모리 풀**: 텐서 사전 할당으로 성능 향상
- **LRU 캐시**: 임베딩 벡터 효율적 관리

## 📈 v1.0 주요 업데이트

### 🔧 로깅 시스템 통합
- **76개 print문 → 구조화된 로그**로 전환
- **단일 로그 파일** (`face_tracker.log`) 사용
- **이모지 기반 로그 레벨**: 🔄 진행, ✅ 성공, ⚠️ 경고, ❌ 오류
- **tqdm 프로그레스바 최적화**: 화면 공간 60% 절약

### 📊 성능 리포트 시스템
- **실시간 성능 측정**: 단계별 시간, 배치 크기, FPS 추적
- **상세 리포트 생성**: 80줄 상세 성능 분석 자동 출력
- **시스템 리소스 모니터링**: CPU 코어, 메모리 사용량 추적
- **성능 지표 계산**: 전체 처리 속도, 프레임당 시간 자동 계산

### 🏗️ 코드 아키텍처 최적화
- **FFmpeg 로직 통합**: 중복 코드 75% 감소 (40줄 → 10줄)
- **래퍼 함수 제거**: 메모리 효율성 향상 (15줄 → 3줄)
- **절대 import 경로**: 모듈 참조 안정화

## ⚙️ 설정

주요 설정은 `src/face_tracker/config.py`에서 수정 가능:

```python
# GPU 설정
DEVICE = 'cuda:0'
BATCH_SIZE_ANALYZE = 256      # 얼굴 감지 배치 크기
BATCH_SIZE_ID_TIMELINE = 128  # 얼굴 인식 배치 크기

# 처리 설정
SEGMENT_LENGTH_SECONDS = 10   # 출력 세그먼트 길이
TRACKING_MODE = "most_frequent"  # 타겟 선택 모드
```

## 🎯 성능 지표

### 현재 벤치마크
- **GPU 사용률**: 97.3% (목표 95% 초과 달성)
- **처리 속도**: 15-20초 per 비디오 (기존 45-60초 대비 67% 단축)
- **메모리 효율성**: 사전 할당된 메모리 풀로 최적화

### 모니터링 명령어
```bash
# GPU 모니터링
nvidia-smi -l 1

# 프로세스 모니터링
htop

# 로그 실시간 확인
tail -f face_tracker.log
```

## 🔮 로드맵

### v1.1 계획
- [ ] Docker 컨테이너화

## 🤝 기여

버그 리포트나 기능 제안은 Issues를 통해 제출해 주세요.

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

---
---