# Face-Tracking-App 설치 가이드

## 데브컨테이너 설치 순서

### 1. PyTorch 설치 (CUDA 128 지원)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. OpenCV 설치 (사용자 빌드)
```bash
# .deb 패키지 설치
dpkg -i /path/to/build_20250710-1_amd64.deb
# 의존성 문제 해결 (필요시)
apt-get install -f -y
```

### 3. 나머지 Python 패키지 설치
```bash
pip install -r requirements_clean.txt
```

## Dockerfile 예시

```dockerfile
FROM python:3.10-slim

# 시스템 패키지 업데이트
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 설치 (CUDA 128)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# OpenCV 설치 (사용자 빌드)
COPY .devcontainer/build_20250710-1_amd64.deb /tmp/
RUN dpkg -i /tmp/build_20250710-1_amd64.deb || apt-get install -f -y

# Python 의존성 설치
COPY requirements_clean.txt .
RUN pip install -r requirements_clean.txt

# 앱 복사
COPY . /app
WORKDIR /app
```

## 로컬 개발 환경 설정

```bash
# 1. 파이썬 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 2. PyTorch 설치
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. OpenCV 설치 (빌드된 버전)
# 시스템에 맞게 OpenCV 설치

# 4. 나머지 패키지 설치
pip install -r requirements_clean.txt
```