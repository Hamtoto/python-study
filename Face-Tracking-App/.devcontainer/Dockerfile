# Face-Tracking-App Development Container
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 시스템 패키지 업데이트 및 기본 도구 설치
RUN apt-get update && apt-get install -y \
    # 기본 개발 도구
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    # Python 관련
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # 멀티미디어 처리
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    # 오디오 처리
    portaudio19-dev \
    libasound2-dev \
    # GPU 모니터링
    nvidia-utils-535 \
    # 기타 유틸리티
    htop \
    vim \
    nano \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 설정
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 12.8 지원)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 사용자 빌드 OpenCV 설치
COPY build_20250710-1_amd64.deb /tmp/
RUN dpkg -i /tmp/build_20250710-1_amd64.deb || apt-get install -f -y \
    && rm /tmp/build_20250710-1_amd64.deb

# Python 의존성 설치
COPY requirements_clean.txt /tmp/
RUN pip install -r /tmp/requirements_clean.txt \
    && rm /tmp/requirements_clean.txt

# GPU 메모리 최적화를 위한 환경 변수
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_LAUNCH_BLOCKING=0

# 작업 디렉토리 설정
WORKDIR /workspace

# 비root 사용자 생성 (선택사항)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# 사용자 변경
USER $USERNAME

# Python 경로 설정
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# GPU 테스트 스크립트 생성
RUN echo '#!/bin/bash\n\
echo "=== GPU 환경 테스트 ==="\n\
echo "CUDA Version:"\n\
nvcc --version\n\
echo -e "\nGPU 정보:"\n\
nvidia-smi\n\
echo -e "\nPyTorch GPU 지원:"\n\
python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")\"\n\
echo -e "\nOpenCV 버전:"\n\
python3 -c "import cv2; print(f\"OpenCV: {cv2.__version__}\")\"\n\
echo -e "\nFaceNet PyTorch:"\n\
python3 -c "import facenet_pytorch; from facenet_pytorch import MTCNN; print(\"FaceNet PyTorch 설치 확인 완료\")"' > /home/$USERNAME/test_gpu.sh \
    && chmod +x /home/$USERNAME/test_gpu.sh