# Dual-Face GPU 파이프라인 환경 설정 가이드

## 🎯 환경 설정 개요

**목표**: PyNvCodec + TensorRT + 완전 GPU 파이프라인 실행 환경 구축  
**방식**: Docker 기반 재현 가능한 환경 + 자동화된 검증 시스템  
**플랫폼**: Ubuntu 22.04 + NVIDIA Driver 525+ + CUDA 12.1

---

## 🔧 1단계: 시스템 요구사항 확인

### **하드웨어 요구사항**

| 구분 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| **GPU** | RTX 4090 24GB | RTX 5090 32GB |
| **CPU** | Intel i7-12700K | Intel i9-13900K |
| **RAM** | 64GB DDR4 | 128GB DDR5 |
| **저장공간** | NVMe SSD 1TB | NVMe SSD 2TB |
| **네트워크** | 1Gbps | 10Gbps |

### **소프트웨어 요구사항**

```bash
# 시스템 버전 확인
lsb_release -a
# Ubuntu 22.04.3 LTS 이상

# NVIDIA 드라이버 확인  
nvidia-smi
# Driver Version: 525.0 이상 필요

# Docker 확인
docker --version
# Docker version 24.0.0 이상

# nvidia-container-toolkit 확인
nvidia-ctk --version
```

---

## 🐳 2단계: DevContainer 환경 구축

### **DevContainer 아키텍처**

**핵심 장점**:
- ✅ **VS Code 완전 통합** (디버깅, Git, 확장 자동 설치)
- ✅ **빌드타임 설치** (컨테이너 시작 즉시 사용 가능)
- ✅ **재현 가능한 환경** (Dockerfile 기반)
- ✅ **개발자 경험 극대화** (포트 포워딩, 볼륨 자동 마운트)

### **.devcontainer/Dockerfile**

```dockerfile
FROM nvidia/cuda:13.0.0-tensorrt-devel-ubuntu24.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 기본 패키지 설치 (Ubuntu 24.04 최신 패키지)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python 링크 설정
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# OpenCV CUDA deb 설치 (빌드타임)
COPY build_20250710-1_amd64.deb /tmp/
RUN dpkg -i /tmp/build_20250710-1_amd64.deb && \
    apt-get update && \
    apt-get install -f -y && \
    rm /tmp/build_20250710-1_amd64.deb

# PyTorch CUDA 12.8 설치 (RTX 5090 최적화)
RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Python 의존성 설치
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 환경 검증 (빌드타임)
RUN python3 -c "import torch, cv2, tensorrt; \
    print(f'✅ PyTorch: {torch.__version__}'); \
    print(f'✅ CUDA Available: {torch.cuda.is_available()}'); \
    print(f'✅ OpenCV: {cv2.__version__}'); \
    print(f'✅ TensorRT: {tensorrt.__version__}'); \
    print('🎉 모든 패키지 로드 성공!')"

WORKDIR /workspace
CMD ["bash"]
```

### **requirements.txt** (업데이트됨)

```txt
# PyTorch 생태계 (CUDA 12.8 호환, RTX 5090 최적화)
# 주의: PyTorch는 Dockerfile에서 별도 설치됨
# torch==2.1.0+cu128  # Dockerfile에서 설치
# torchvision==0.16.0+cu128
# torchaudio==2.1.0+cu128

# TensorRT (베이스 이미지에 포함, Python 바인딩만 설치)
tensorrt>=10.0.0  # Python 바인딩만

# PyNvCodec (Video Codec SDK)
pynvcodec-cu12==0.0.1

# OpenCV with CUDA support
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78

# 영상 처리
av==11.0.0
moviepy==1.0.3
ffmpeg-python==0.2.0

# 수치 계산
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0

# 얼굴 인식/추적
ultralytics==8.0.200
facenet-pytorch==2.5.3

# 유틸리티
tqdm==4.66.1
PyYAML==6.0.1
loguru==0.7.2
typer==0.9.0
rich==13.5.2

# 모니터링
psutil==5.9.5
pynvml==11.5.0
wandb==0.15.10

# 개발/테스트
pytest==7.4.2
pytest-asyncio==0.21.1
black==23.7.0
isort==5.12.0
mypy==1.5.1
```

### **.devcontainer/devcontainer.json**

```json
{
  "name": "Dual-Face GPU Pipeline",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },
  
  "runArgs": [
    "--runtime=nvidia",
    "--gpus=all"
  ],
  
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.debugpy", 
        "ms-toolsai.jupyter",
        "nvidia.nsight-vscode-edition",
        "ms-vscode.cmake-tools"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3.12",
        "python.terminal.activateEnvironment": false
      }
    }
  },
  
  "forwardPorts": [8888, 6006, 8050],
  "postCreateCommand": "echo '🎉 DevContainer 준비 완료! 컨테이너 시작 시간: 5-10초'"
}
```

**DevContainer vs Docker Compose 비교**:

| 항목 | DevContainer | Docker Compose |
|------|--------------|----------------|
| **시작 시간** | 5-10초 | 2-5분 (스크립트 실행) |
| **VS Code 통합** | ✅ 완벽 | ❌ 수동 설정 |
| **개발 경험** | ✅ 최적화됨 | ⚠️ 수동 관리 |
| **재현성** | ✅ 100% | ⚠️ 네트워크 의존적 |

---

## ⚙️ 3단계: PyNvCodec 통합 (Dockerfile에서 설치)

### **PyNvCodec Dockerfile 통합**

**기존 스크립트 방식의 문제**:
- ❌ 컨테이너 시작할 때마다 5-10분 빌드
- ❌ 네트워크 오류 시 실패
- ❌ GitHub 접근 필요

**개선된 Dockerfile 방식**:
- ✅ 빌드타임에 한 번만 설치
- ✅ 즉시 사용 가능
- ✅ 네트워크 무관

### **Dockerfile에 PyNvCodec 추가**

```dockerfile
# .devcontainer/Dockerfile에 추가할 섹션

# PyNvCodec (Video Processing Framework) 설치
RUN apt-get update && apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Video Codec SDK 다운로드 및 설치 (빌드타임)
RUN CODEC_SDK_VERSION="12.1.14" && \
    wget -q "https://developer.download.nvidia.com/compute/nvcodec/redist/Video_Codec_SDK_${CODEC_SDK_VERSION}.zip" -O /tmp/sdk.zip && \
    cd /tmp && unzip -q sdk.zip && \
    cp -r Video_Codec_SDK_${CODEC_SDK_VERSION}/Interface/* /usr/local/cuda/include/ && \
    cp Video_Codec_SDK_${CODEC_SDK_VERSION}/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/ && \
    rm -rf /tmp/sdk.zip /tmp/Video_Codec_SDK_${CODEC_SDK_VERSION}

# PyNvCodec (VPF) 빌드 및 설치 (빌드타임)
RUN git clone https://github.com/NVIDIA/VideoProcessingFramework.git /tmp/VPF && \
    cd /tmp/VPF && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGEN_PYTORCH_EXTENSION=ON \
        -DUSE_NVTX=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda && \
    make -j$(nproc) && \
    cd .. && pip install . && \
    rm -rf /tmp/VPF

# PyNvCodec 설치 검증 (빌드타임)
RUN python3 -c "import PyNvCodec as nvc; print(f'✅ PyNvCodec 설치 성공')"
```

---

## 🔍 4단계: 완전 통합 Dockerfile

### **최종 통합 Dockerfile**

모든 의존성을 빌드타임에 설치하는 완전한 Dockerfile:

```dockerfile
# .devcontainer/Dockerfile (완전 통합 버전)
FROM nvidia/cuda:13.0.0-tensorrt-devel-ubuntu24.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    build-essential cmake git wget curl vim htop \
    libavformat-dev libavcodec-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 링크
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# OpenCV CUDA deb 설치
COPY build_20250710-1_amd64.deb /tmp/
RUN dpkg -i /tmp/build_20250710-1_amd64.deb && \
    apt-get update && apt-get install -f -y && \
    rm /tmp/build_20250710-1_amd64.deb

# PyTorch CUDA 12.8 설치 
RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Video Codec SDK 설치
RUN CODEC_SDK_VERSION="12.1.14" && \
    wget -q "https://developer.download.nvidia.com/compute/nvcodec/redist/Video_Codec_SDK_${CODEC_SDK_VERSION}.zip" -O /tmp/sdk.zip && \
    cd /tmp && unzip -q sdk.zip && \
    cp -r Video_Codec_SDK_${CODEC_SDK_VERSION}/Interface/* /usr/local/cuda/include/ && \
    cp Video_Codec_SDK_${CODEC_SDK_VERSION}/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/ && \
    rm -rf /tmp/sdk.zip /tmp/Video_Codec_SDK_${CODEC_SDK_VERSION}

# PyNvCodec 빌드 및 설치
RUN git clone https://github.com/NVIDIA/VideoProcessingFramework.git /tmp/VPF && \
    cd /tmp/VPF && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGEN_PYTORCH_EXTENSION=ON -DUSE_NVTX=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda && \
    make -j$(nproc) && cd .. && pip install . && rm -rf /tmp/VPF

# Python 의존성 설치
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 종합 검증 (빌드타임)
RUN python3 -c "
import torch, cv2, tensorrt, PyNvCodec as nvc
print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✅ OpenCV: {cv2.__version__} (CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()})')
print(f'✅ TensorRT: {tensorrt.__version__}')
print(f'✅ PyNvCodec: 설치됨')
print('🎉 모든 패키지 검증 완료!')
"

WORKDIR /workspace
CMD ["bash"]
```

**주요 개선사항**:
- ✅ **단일 Dockerfile**: 모든 설치를 한 곳에서 관리
- ✅ **빌드타임 검증**: 이미지 빌드 시 모든 패키지 검증
- ✅ **레이어 최적화**: 불필요한 중간 파일 자동 정리
- ✅ **빠른 시작**: 컨테이너 시작 즉시 사용 가능

# NVIDIA 저장소 키 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2단계: TensorRT 패키지 설치
sudo apt-get update
sudo apt-get install -y \
    tensorrt=${TENSORRT_VERSION}* \
    tensorrt-libs=${TENSORRT_VERSION}* \
    tensorrt-dev=${TENSORRT_VERSION}* \
    python3-libnvinfer=${TENSORRT_VERSION}* \
    python3-libnvinfer-dev=${TENSORRT_VERSION}*

# 3단계: Python 바인딩 설치
pip install \
    tensorrt==${TENSORRT_VERSION} \
    tensorrt-libs==${TENSORRT_VERSION} \
    tensorrt-bindings==${TENSORRT_VERSION}

# 4단계: 설치 검증
echo "🧪 TensorRT 설치 검증 중..."

python3 -c "
import tensorrt as trt
import torch

print('✓ TensorRT 임포트 성공')
print(f'  TensorRT 버전: {trt.__version__}')
print(f'  PyTorch 버전: {torch.__version__}')
print(f'  CUDA 사용 가능: {torch.cuda.is_available()}')

# TensorRT 로거 및 빌더 테스트
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
print('✓ TensorRT Builder 생성 성공')

# GPU 정보
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {gpu_name}')
    print(f'  VRAM: {gpu_memory:.1f} GB')
"

echo "🎉 TensorRT 설치 및 검증 완료!"
```

---

## 🔍 4단계: 종합 환경 검증 시스템

### **자동화된 검증 스크립트**

```bash
#!/bin/bash
# validate_environment.sh - 완전 자동화된 환경 검증

set -e

echo "=================================================="
echo "🔧 Dual-Face GPU 파이프라인 환경 검증 시스템 v2.0"
echo "=================================================="

# 전역 변수
VALIDATION_LOG="/tmp/environment_validation.log"
SUCCESS_COUNT=0
TOTAL_TESTS=0

# 로깅 함수
log_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo "[TEST $TOTAL_TESTS] $1" | tee -a $VALIDATION_LOG
}

log_success() {
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo "✅ $1" | tee -a $VALIDATION_LOG
}

log_warning() {
    echo "⚠️ $1" | tee -a $VALIDATION_LOG
}

log_error() {
    echo "❌ $1" | tee -a $VALIDATION_LOG
    exit 1
}

# 1단계: 시스템 기본 환경 검증
echo -e "\n1️⃣ 시스템 기본 환경 검증"
echo "----------------------------"

log_test "운영체제 버전 확인"
OS_VERSION=$(lsb_release -r -s)
if [[ $(echo "$OS_VERSION >= 22.04" | bc -l) -eq 1 ]]; then
    log_success "Ubuntu $OS_VERSION (요구사항: 22.04+)"
else
    log_error "Ubuntu 버전 부족: $OS_VERSION (최소 22.04 필요)"
fi

log_test "NVIDIA 드라이버 검증"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    if [[ $DRIVER_MAJOR -ge 525 ]]; then
        log_success "NVIDIA 드라이버: $DRIVER_VERSION"
    else
        log_error "드라이버 버전 부족: $DRIVER_VERSION (최소 525.0 필요)"
    fi
else
    log_error "nvidia-smi 명령 없음 - NVIDIA 드라이버 미설치"
fi

log_test "CUDA 버전 확인"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
    log_success "CUDA 버전: $CUDA_VERSION"
else
    log_warning "nvcc 명령 없음 - CUDA 개발 도구 확인 필요"
fi

# 2단계: GPU 하드웨어 상세 검증
echo -e "\n2️⃣ GPU 하드웨어 상세 검증"
echo "----------------------------"

log_test "GPU 모델 및 메모리 확인"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | sed 's/ MiB//')
GPU_MEMORY_GB=$((GPU_MEMORY / 1024))

log_success "GPU 모델: $GPU_NAME"
log_success "VRAM 총량: ${GPU_MEMORY_GB}GB"

if [[ $GPU_MEMORY_GB -ge 24 ]]; then
    log_success "VRAM 요구사항 충족 (24GB+)"
else
    log_error "VRAM 부족: ${GPU_MEMORY_GB}GB (최소 24GB 필요)"
fi

log_test "NVDEC/NVENC 하드웨어 확인"
# ffmpeg를 통한 하드웨어 디코더/인코더 확인
if command -v ffmpeg &> /dev/null; then
    NVDEC_SUPPORT=$(ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -c "cuda" || echo "0")
    NVENC_SUPPORT=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep -c "nvenc" || echo "0")
    
    if [[ $NVDEC_SUPPORT -gt 0 ]]; then
        log_success "NVDEC 하드웨어 디코더 지원"
    else
        log_warning "NVDEC 지원 확인 필요"
    fi
    
    if [[ $NVENC_SUPPORT -gt 0 ]]; then
        log_success "NVENC 하드웨어 인코더 지원"
    else
        log_warning "NVENC 지원 확인 필요"
    fi
else
    log_warning "FFmpeg 설치 확인 필요"
fi

# 3단계: Python 환경 검증
echo -e "\n3️⃣ Python 환경 검증"
echo "----------------------------"

log_test "Python 버전 확인"
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
    log_success "Python $PYTHON_VERSION (요구사항: 3.9+)"
else
    log_error "Python 버전 부족: $PYTHON_VERSION (최소 3.9 필요)"
fi

log_test "필수 Python 패키지 검증"
REQUIRED_PACKAGES=(
    "torch"
    "torchvision" 
    "tensorrt"
    "PyNvCodec"
    "cv2"
    "numpy"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" &>/dev/null; then
        # 패키지 버전 정보 수집
        if [[ $package == "torch" ]]; then
            VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
            CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
            log_success "$package $VERSION (CUDA: $CUDA_AVAILABLE)"
        elif [[ $package == "tensorrt" ]]; then
            VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
            log_success "$package $VERSION"
        elif [[ $package == "PyNvCodec" ]]; then
            log_success "$package (PyNvCodec/VPF)"
        else
            log_success "$package"
        fi
    else
        log_error "$package 패키지 누락"
    fi
done

# 4단계: PyNvCodec 실제 동작 테스트
echo -e "\n4️⃣ PyNvCodec 실제 동작 테스트"
echo "----------------------------"

log_test "테스트 비디오 생성"
TEST_VIDEO="/tmp/dual_test_sample.mp4"
if [[ ! -f "$TEST_VIDEO" ]]; then
    ffmpeg -f lavfi -i testsrc=duration=3:size=1920x1080:rate=30 \
           -c:v libx264 -preset fast -y "$TEST_VIDEO" &>/dev/null
    log_success "테스트 비디오 생성: $TEST_VIDEO"
fi

log_test "PyNvCodec 디코딩 테스트"
DECODE_TEST_RESULT=$(python3 << 'EOF'
import PyNvCodec as nvc
import sys
import traceback

try:
    # NVDEC 디코더 생성
    decoder = nvc.PyDecodeHW("/tmp/dual_test_sample.mp4", nvc.PixelFormat.NV12, 0)
    print(f"디코더 생성 성공: {decoder.Width()}x{decoder.Height()}")
    
    # 색공간 변환기 생성
    converter = nvc.PySurfaceConverter(
        decoder.Width(), decoder.Height(),
        nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, 0
    )
    print("색공간 변환기 생성 성공")
    
    # 실제 프레임 디코딩 테스트 (10프레임)
    decode_success = 0
    for i in range(10):
        surface = decoder.DecodeSurface()
        if surface:
            rgb_surface = converter.Execute(surface)
            if rgb_surface:
                decode_success += 1
            else:
                print(f"변환 실패: 프레임 {i}")
                break
        else:
            print(f"디코딩 완료: {decode_success}프레임 처리")
            break
            
    if decode_success >= 5:  # 최소 5프레임 성공
        print("SUCCESS")
    else:
        print("FAILED")
        
except Exception as e:
    print("FAILED")
    print(f"오류: {e}")
    traceback.print_exc()
EOF
)

if [[ "$DECODE_TEST_RESULT" == *"SUCCESS"* ]]; then
    log_success "PyNvCodec 디코딩 테스트 통과"
else
    log_error "PyNvCodec 디코딩 테스트 실패"
fi

# 5단계: TensorRT 엔진 테스트
echo -e "\n5️⃣ TensorRT 엔진 테스트"
echo "----------------------------"

log_test "TensorRT 빌더 및 런타임 테스트"
TENSORRT_TEST_RESULT=$(python3 << 'EOF'
import tensorrt as trt
import torch
import numpy as np

try:
    # TensorRT 로거 및 빌더
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 간단한 네트워크 생성 (더미 테스트)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 입력 레이어
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
    
    # 간단한 Convolution 레이어
    conv_w = np.random.randn(64, 3, 3, 3).astype(np.float32)
    conv_b = np.random.randn(64).astype(np.float32)
    conv_layer = network.add_convolution(input_tensor, 64, (3, 3), conv_w, conv_b)
    
    # 출력 마크
    network.mark_output(conv_layer.get_output(0))
    
    # 빌더 구성
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # 엔진 빌드 시도
    engine = builder.build_engine(network, config)
    
    if engine:
        print("TensorRT 엔진 빌드 성공")
        print(f"입력 바인딩 수: {engine.num_bindings}")
        print("SUCCESS")
    else:
        print("TensorRT 엔진 빌드 실패")
        print("FAILED")
        
except Exception as e:
    print("FAILED")
    print(f"오류: {e}")
EOF
)

if [[ "$TENSORRT_TEST_RESULT" == *"SUCCESS"* ]]; then
    log_success "TensorRT 엔진 테스트 통과"
else
    log_warning "TensorRT 엔진 테스트 실패 (비치명적)"
fi

# 6단계: 통합 파이프라인 테스트
echo -e "\n6️⃣ 통합 파이프라인 프리뷰 테스트"
echo "----------------------------"

log_test "GPU 메모리 할당 및 관리 테스트"
GPU_MEMORY_TEST=$(python3 << 'EOF'
import torch
import gc

try:
    # 현재 GPU 메모리 상태
    if torch.cuda.is_available():
        gpu_id = 0
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        
        # 대용량 텐서 할당 테스트 (4GB)
        test_tensor = torch.randn(1024, 1024, 1024, dtype=torch.float32, device='cuda')
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        
        print(f"총 VRAM: {total_memory / 1024**3:.1f} GB")
        print(f"할당된 메모리: {allocated_memory / 1024**3:.1f} GB")
        
        # 메모리 해제
        del test_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        freed_memory = torch.cuda.memory_allocated(gpu_id)
        print(f"해제 후 메모리: {freed_memory / 1024**3:.1f} GB")
        
        print("SUCCESS")
    else:
        print("CUDA 사용 불가")
        print("FAILED")
        
except Exception as e:
    print("FAILED")
    print(f"오류: {e}")
EOF
)

if [[ "$GPU_MEMORY_TEST" == *"SUCCESS"* ]]; then
    log_success "GPU 메모리 관리 테스트 통과"
else
    log_error "GPU 메모리 관리 테스트 실패"
fi

# 7단계: 최종 검증 결과 정리
echo -e "\n7️⃣ 최종 검증 결과"
echo "----------------------------"

# JSON 리포트 생성
cat > environment_validation_report.json << EOF
{
    "validation_timestamp": "$(date -Iseconds)",
    "system_info": {
        "os_version": "$(lsb_release -d -s)",
        "gpu_name": "$GPU_NAME", 
        "gpu_memory_gb": $GPU_MEMORY_GB,
        "driver_version": "$DRIVER_VERSION",
        "cuda_version": "$CUDA_VERSION",
        "python_version": "$PYTHON_VERSION"
    },
    "test_results": {
        "total_tests": $TOTAL_TESTS,
        "passed_tests": $SUCCESS_COUNT,
        "success_rate": $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_TESTS" | bc -l)
    },
    "validation_status": "$([ $SUCCESS_COUNT -eq $TOTAL_TESTS ] && echo 'READY' || echo 'NEEDS_ATTENTION')",
    "critical_components": {
        "pynvcodec": "$([ "$DECODE_TEST_RESULT" == *"SUCCESS"* ] && echo 'READY' || echo 'FAILED')",
        "tensorrt": "$([ "$TENSORRT_TEST_RESULT" == *"SUCCESS"* ] && echo 'READY' || echo 'WARNING')",
        "gpu_memory": "$([ "$GPU_MEMORY_TEST" == *"SUCCESS"* ] && echo 'READY' || echo 'FAILED')"
    },
    "recommendations": [
        $([ $GPU_MEMORY_GB -lt 32 ] && echo '"GPU 메모리 32GB 권장",' || echo '')
        $([ "$DECODE_TEST_RESULT" != *"SUCCESS"* ] && echo '"PyNvCodec 재설치 필요",' || echo '')
        "정기적인 환경 검증 실행"
    ]
}
EOF

# 최종 결과 출력
echo "📊 검증 완료: $SUCCESS_COUNT/$TOTAL_TESTS 테스트 통과"
echo "📄 상세 리포트: environment_validation_report.json"

if [[ $SUCCESS_COUNT -eq $TOTAL_TESTS ]]; then
    echo -e "\n🎉 모든 테스트 통과! Dual-Face GPU 파이프라인 개발 환경이 완벽히 준비되었습니다!"
    exit 0
else
    echo -e "\n⚠️ $((TOTAL_TESTS - SUCCESS_COUNT))개 테스트 실패. 환경 설정을 점검해주세요."
    echo "📋 실패 상세: $VALIDATION_LOG"
    exit 1
fi

# 임시 파일 정리
rm -f "$TEST_VIDEO"

echo "=================================================="
```

---

## 🚀 5단계: 실행 및 개발 환경

### **개발 환경 시작**

```bash
# 1. 저장소 클론
git clone https://github.com/your-org/dual-face-tracker.git
cd dual-face-tracker

# 2. Docker 환경 구축
docker-compose up -d --build

# 3. 컨테이너 접속
docker-compose exec dual-face-tracker bash

# 4. 환경 검증 실행
bash scripts/validate_environment.sh

# 5. 개발 서버 시작 (Jupyter + 모니터링)
bash scripts/start_dev_services.sh
```

### **개발 도구 설정**

```bash
#!/bin/bash
# start_dev_services.sh - 개발 도구 일괄 시작

echo "🚀 개발 환경 서비스 시작..."

# Jupyter Lab 시작 (백그라운드)
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    &

# TensorBoard 시작 (로그 모니터링)
tensorboard \
    --logdir=./logs \
    --port=6006 \
    --host=0.0.0.0 \
    &

# GPU 모니터링 대시보드
python scripts/gpu_monitor_dashboard.py \
    --port=8050 \
    --host=0.0.0.0 \
    &

echo "✅ 개발 서비스 시작 완료!"
echo "📊 Jupyter Lab: http://localhost:8888"
echo "📈 TensorBoard: http://localhost:6006"
echo "🖥️ GPU 모니터: http://localhost:8050"

# 서비스 상태 확인
sleep 5
netstat -tulpn | grep -E "(8888|6006|8050)"
```

### **디버깅 도구**

```python
# scripts/debug_pipeline.py - 파이프라인 디버깅 도구

import torch
import psutil
import pynvml
from typing import Dict, Any
import time

class PipelineDebugger:
    """파이프라인 디버깅 및 프로파일링 도구"""
    
    def __init__(self):
        # NVML 초기화
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # 성능 메트릭
        self.metrics = {
            'frame_processing_times': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'decode_times': [],
            'inference_times': [],
            'encode_times': []
        }
        
    def profile_decode_performance(self, video_path: str):
        """디코딩 성능 프로파일링"""
        import PyNvCodec as nvc
        
        print(f"🔍 디코딩 성능 분석: {video_path}")
        
        decoder = nvc.PyDecodeHW(video_path, nvc.PixelFormat.NV12, 0)
        converter = nvc.PySurfaceConverter(
            decoder.Width(), decoder.Height(),
            nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, 0
        )
        
        decode_times = []
        total_frames = 0
        
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            
            # 디코딩
            surface = decoder.DecodeSurface()
            if not surface:
                break
                
            # 색공간 변환
            rgb_surface = converter.Execute(surface)
            
            frame_end = time.time()
            decode_times.append((frame_end - frame_start) * 1000)  # ms
            total_frames += 1
            
            # GPU 사용률 모니터링
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            self.metrics['gpu_utilization'].append(gpu_util.gpu)
            
            if total_frames % 100 == 0:
                avg_time = sum(decode_times[-100:]) / min(100, len(decode_times))
                print(f"  프레임 {total_frames}: 평균 디코딩 시간 {avg_time:.2f}ms")
                
        total_time = time.time() - start_time
        
        # 결과 분석
        avg_decode_time = sum(decode_times) / len(decode_times)
        fps = total_frames / total_time
        
        print(f"📊 디코딩 성능 결과:")
        print(f"  총 프레임: {total_frames}")
        print(f"  평균 디코딩 시간: {avg_decode_time:.2f}ms")
        print(f"  실제 FPS: {fps:.1f}")
        print(f"  GPU 평균 사용률: {sum(self.metrics['gpu_utilization'])/len(self.metrics['gpu_utilization']):.1f}%")
        
        return {
            'total_frames': total_frames,
            'avg_decode_time_ms': avg_decode_time,
            'fps': fps,
            'decode_times': decode_times
        }
        
    def profile_memory_usage(self):
        """메모리 사용량 모니터링"""
        
        # GPU 메모리
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_total = gpu_info.total / 1024**3  # GB
        gpu_used = gpu_info.used / 1024**3   # GB
        gpu_free = gpu_info.free / 1024**3   # GB
        
        # 시스템 메모리
        system_memory = psutil.virtual_memory()
        system_total = system_memory.total / 1024**3  # GB
        system_used = system_memory.used / 1024**3    # GB
        system_available = system_memory.available / 1024**3  # GB
        
        # PyTorch 메모리
        if torch.cuda.is_available():
            torch_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            torch_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        else:
            torch_allocated = torch_reserved = 0
            
        memory_status = {
            'gpu_memory': {
                'total_gb': gpu_total,
                'used_gb': gpu_used,
                'free_gb': gpu_free,
                'utilization_pct': (gpu_used / gpu_total) * 100
            },
            'system_memory': {
                'total_gb': system_total,
                'used_gb': system_used,
                'available_gb': system_available,
                'utilization_pct': (system_used / system_total) * 100
            },
            'torch_memory': {
                'allocated_gb': torch_allocated,
                'reserved_gb': torch_reserved
            }
        }
        
        print(f"💾 현재 메모리 상태:")
        print(f"  GPU: {gpu_used:.1f}/{gpu_total:.1f}GB ({(gpu_used/gpu_total)*100:.1f}%)")
        print(f"  시스템: {system_used:.1f}/{system_total:.1f}GB ({(system_used/system_total)*100:.1f}%)")
        print(f"  PyTorch: {torch_allocated:.1f}GB 할당, {torch_reserved:.1f}GB 예약")
        
        return memory_status
        
    def check_tensorrt_engines(self, model_dir: str = "./models"):
        """TensorRT 엔진 파일 확인 및 로드 테스트"""
        import os
        import tensorrt as trt
        
        print(f"🔧 TensorRT 엔진 검증: {model_dir}")
        
        engine_files = [f for f in os.listdir(model_dir) if f.endswith('.trt')]
        
        if not engine_files:
            print("⚠️ TensorRT 엔진 파일 없음")
            return False
            
        logger = trt.Logger(trt.Logger.WARNING)
        
        for engine_file in engine_files:
            engine_path = os.path.join(model_dir, engine_file)
            print(f"  테스트: {engine_file}")
            
            try:
                with open(engine_path, 'rb') as f:
                    engine_data = f.read()
                    
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(engine_data)
                
                if engine:
                    print(f"    ✅ 로드 성공 - 바인딩: {engine.num_bindings}")
                    
                    # 입출력 정보
                    for i in range(engine.num_bindings):
                        binding_name = engine.get_binding_name(i)
                        binding_shape = engine.get_binding_shape(i)
                        is_input = engine.binding_is_input(i)
                        print(f"      {'입력' if is_input else '출력'}: {binding_name} {binding_shape}")
                        
                else:
                    print(f"    ❌ 로드 실패")
                    
            except Exception as e:
                print(f"    ❌ 오류: {e}")
                
        return True

# 실행 예제
if __name__ == "__main__":
    debugger = PipelineDebugger()
    
    # 메모리 상태 확인
    debugger.profile_memory_usage()
    
    # TensorRT 엔진 확인
    debugger.check_tensorrt_engines()
    
    # 디코딩 성능 테스트 (테스트 비디오 필요)
    # debugger.profile_decode_performance("/path/to/test_video.mp4")
```

---

## 📋 6단계: 환경 유지보수 가이드

### **정기 환경 점검**

```bash
#!/bin/bash
# maintenance_check.sh - 주간 환경 점검

echo "🔧 주간 환경 점검 시작..."

# 1. GPU 상태 점검
echo "1️⃣ GPU 상태 점검"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv

# 2. 디스크 공간 점검  
echo -e "\n2️⃣ 디스크 공간 점검"
df -h /workspace /tmp

# 3. Docker 컨테이너 상태
echo -e "\n3️⃣ Docker 상태 점검"
docker ps -a
docker system df

# 4. 로그 파일 크기 점검
echo -e "\n4️⃣ 로그 파일 점검"
find /workspace/logs -name "*.log" -exec du -h {} \; | sort -hr

# 5. 환경 검증 재실행
echo -e "\n5️⃣ 환경 검증 재실행"
bash scripts/validate_environment.sh --quiet

echo "✅ 주간 점검 완료"
```

### **문제 해결 가이드**

| 문제 상황 | 증상 | 해결 방법 |
|-----------|------|-----------|
| **PyNvCodec 임포트 실패** | `ImportError: No module named 'PyNvCodec'` | VPF 재빌드 및 설치 |
| **NVDEC 디코딩 실패** | "No decoder available" 오류 | nvidia-smi로 GPU 상태 확인 |
| **TensorRT 엔진 로드 실패** | 엔진 deserialization 오류 | CUDA/TensorRT 버전 호환성 확인 |
| **GPU 메모리 부족** | CUDA OOM 오류 | 배치 크기 감소 또는 GPU 메모리 정리 |
| **Docker 권한 오류** | Permission denied | nvidia-container-toolkit 재설치 |

---

이 환경 설정 가이드를 따르면 완전 자동화된 검증 시스템과 함께 안정적인 개발 환경을 구축할 수 있습니다. 모든 단계는 스크립트로 자동화되어 재현 가능한 환경을 보장합니다.