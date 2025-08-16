# 🚀 Clean venv-based DevContainer for Dual-Face GPU Pipeline
# Ubuntu 24.04 + Python 3.10 + venv 격리 환경
# PEP 668 우회, 즉시 개발 가능한 구성

FROM ubuntu:24.04

# =============================================================================
# 🔧 기본 환경 설정
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64
ENV PYTHONPATH=/workspace

# venv 경로 설정 (볼륨 마운트 영향 없는 위치)
ENV VENV_PATH=/opt/venv
ENV PATH=${VENV_PATH}/bin:${PATH}
ENV VIRTUAL_ENV=${VENV_PATH}

# =============================================================================
# 📦 시스템 패키지 설치 (최소한으로 축소)
# =============================================================================

# Python 3.10 저장소 추가 (Ubuntu 24.04 기본은 3.12)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update

# 핵심 시스템 패키지만 설치
RUN apt-get install -y \
    # Python 3.10 전체 스택
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    python3-numpy \
    # 빌드 도구
    build-essential \
    cmake \
    pkg-config \
    # 개발 도구
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    unzip \
    # 미디어 라이브러리 (최소한)
    libfreetype6 \
    libpng16-16 \
    libtiff6 \
    libtiff-dev \
    # GStreamer (PyAV 의존성)
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-libav \
    # FFmpeg (기본 패키지 + 실행 파일)
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev \
    # X11 (GUI 백엔드용)
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 🎯 CUDA 12.8 설치 (RTX 5090 지원)
# =============================================================================
RUN apt-get update && \
    apt-get install -y ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub -o /etc/apt/keyrings/cuda.asc && \
    chmod a+r /etc/apt/keyrings/cuda.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/cuda.asc] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    # 핵심 CUDA 패키지만 설치 (용량 최적화)
    apt-get -y install \
        cuda-toolkit-12-8 \
        cuda-cudart-12-8 \
        libcublas-12-8 \
        libcublas-dev-12-8 \
        libcufft-12-8 \
        libcufft-dev-12-8 \
        cudnn9-cuda-12 \
        && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 🧠 TensorRT 설치 (시스템 레벨)
# =============================================================================
RUN apt-get update && \
    apt-get install -y \
        tensorrt \
        python3-libnvinfer-dev \
        libnvinfer-plugin-dev \
        || echo "⚠️ TensorRT 시스템 설치 실패 - venv에서 pip으로 재시도" && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# 🎬 OpenCV CUDA 설치 (검증된 DEB 패키지)
# =============================================================================
# 호스트에서 빌드된 OpenCV CUDA 4.13.0 DEB 패키지 복사
COPY opencv-cuda-4.13.0_amd64.deb /tmp/
RUN dpkg -i /tmp/opencv-cuda-4.13.0_amd64.deb && \
    apt-get update && \
    apt-get install -f -y && \
    rm /tmp/opencv-cuda-4.13.0_amd64.deb

# =============================================================================
# 🐍 Python 환경 구성 (venv 기반)
# =============================================================================

# Python 3.10을 기본으로 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 작업 디렉토리 생성
WORKDIR /workspace

# venv 생성 (시스템 패키지와 완전 격리)
RUN python3.10 -m venv ${VENV_PATH} && \
    # venv 내에서 pip 최신화 (venv 내부이므로 PEP 668 적용 안됨)
    ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel

# =============================================================================
# 🔥 PyTorch 설치 (venv 내, RTX 5090 최적화)
# =============================================================================
RUN ${VENV_PATH}/bin/pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# TensorRT 재설치 (CUDA 12.8 호환 버전)
RUN ${VENV_PATH}/bin/pip uninstall -y tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs && \
    ${VENV_PATH}/bin/pip install --no-cache-dir tensorrt-cu12==10.5.0

# =============================================================================
# 📹 PyNvVideoCodec 설치 (VPF 대체)
# =============================================================================
RUN ${VENV_PATH}/bin/pip install --no-cache-dir PyNvVideoCodec

# =============================================================================
# 📋 Python 의존성 설치 (venv 내, 경량화된 패키지 목록)
# =============================================================================
COPY requirements.txt /tmp/

# 1단계: 기본 패키지 설치 (PyAV 제외)
RUN ${VENV_PATH}/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# PyAV를 소스에서 빌드 (NVENC/NVDEC 지원)
RUN ${VENV_PATH}/bin/pip install av==11.0.0 --no-binary av --no-cache-dir

# 2단계: 의존성 충돌 방지를 위해 --no-deps로 AI 라이브러리 설치
RUN ${VENV_PATH}/bin/pip install --no-deps \
    ultralytics==8.0.200 \
    facenet-pytorch==2.5.3

# 정리
RUN rm /tmp/requirements.txt

# OpenCV를 venv에서 사용할 수 있도록 심볼릭 링크 생성
RUN ln -s /usr/local/lib/python3.10/site-packages/cv2.so ${VENV_PATH}/lib/python3.10/site-packages/cv2.so || \
    ln -s /usr/lib/python3.10/dist-packages/cv2.so ${VENV_PATH}/lib/python3.10/site-packages/cv2.so || \
    echo "⚠️ OpenCV 심볼릭 링크는 런타임에서 설정"

# =============================================================================
# 🚀 venv 자동 활성화 설정
# =============================================================================

# .bashrc에 venv 자동 활성화 추가
RUN echo "# Auto-activate venv for DevContainer" >> /root/.bashrc && \
    echo "if [ -d \"/opt/venv\" ] && [ -z \"\$VIRTUAL_ENV\" ]; then" >> /root/.bashrc && \
    echo "    source /opt/venv/bin/activate" >> /root/.bashrc && \
    echo "    echo '✅ venv activated: /opt/venv'" >> /root/.bashrc && \
    echo "fi" >> /root/.bashrc

# =============================================================================
# 🧪 빌드 시점 검증 (GPU 없이 가능한 것들만)
# =============================================================================
RUN echo "🧪 빌드 시점 검증 시작..." && \
    # PyTorch 확인
    ${VENV_PATH}/bin/python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    # PyAV 확인 
    ${VENV_PATH}/bin/python -c "import av; print(f'✅ PyAV: {av.__version__}')" && \
    # OpenCV 확인
    ${VENV_PATH}/bin/python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" && \
    # TensorRT 확인 (optional)
    ${VENV_PATH}/bin/python -c "import tensorrt; print(f'✅ TensorRT: {tensorrt.__version__}')" || echo "⚠️ TensorRT는 런타임에서 확인" && \
    # PyNvVideoCodec 확인 (optional)
    ${VENV_PATH}/bin/python -c "import PyNvVideoCodec; print('✅ PyNvVideoCodec: OK')" || echo "⚠️ PyNvVideoCodec는 런타임에서 확인" && \
    echo "🎉 빌드 시점 검증 완료!"

# =============================================================================
# 🔧 런타임 설정
# =============================================================================

# GPU 접근 권한
RUN usermod -aG video root

# 포트 노출 (개발용)
EXPOSE 8000 8888 5000

# 단순한 bash 시작 (venv는 .bashrc에서 자동 활성화)
CMD ["/bin/bash"]