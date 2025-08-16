#!/bin/bash
# 🚀 Dual-Face GPU Pipeline 개발 환경 실행 스크립트
# 가장 편한 방법: Docker CLI로 빌드된 이미지 바로 실행

set -e

echo "🚀 Dual-Face GPU Pipeline 개발 환경 시작..."
echo "========================================"

# 현재 디렉토리 확인 (호스트 또는 컨테이너 환경 모두 지원)
CURRENT_DIR=$(pwd)
HOST_DIR="/home/hamtoto/work/python-study/Face-Tracking-App/dual"
CONTAINER_DIR="/workspace"

# 컨테이너 내부에서 실행하는 경우
if [ "$CURRENT_DIR" = "$CONTAINER_DIR" ]; then
    echo "📦 DevContainer 환경에서 실행 중..."
    echo "⚠️ 이미 DevContainer 안에 있습니다!"
    echo ""
    echo "🎯 바로 개발을 시작하세요:"
    echo "   python test_pipeline.py     # 환경 검증"
    echo "   nvidia-smi                  # GPU 확인"
    echo "   python                      # Python REPL 시작"
    echo ""
    exit 0
fi

# 호스트에서 실행하는 경우
if [ "$CURRENT_DIR" != "$HOST_DIR" ]; then
    echo "⚠️ 경고: dual 디렉토리에서 실행해주세요"
    echo "   현재 위치: $CURRENT_DIR"
    echo "   예상 위치: $HOST_DIR"
    echo ""
    echo "올바른 실행 방법:"
    echo "   cd $HOST_DIR"
    echo "   ./run_dev.sh"
    exit 1
fi

# Docker 이미지 존재 확인
if ! docker image inspect dual-face-gpu-pipeline:latest >/dev/null 2>&1; then
    echo "❌ Docker 이미지 'dual-face-gpu-pipeline:latest'가 존재하지 않습니다"
    echo ""
    echo "먼저 이미지를 빌드해주세요:"
    echo "   docker build .devcontainer -t dual-face-gpu-pipeline:latest"
    exit 1
fi

# GPU 사용 가능 여부 확인
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA GPU 감지됨"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_ARGS="--gpus all"
else
    echo "⚠️ GPU 미감지 - CPU 모드로 실행"
    GPU_ARGS=""
fi

echo ""
echo "🐳 Docker 컨테이너 시작 중..."

# 기존 컨테이너 정리 (있다면)
if docker ps -a --format "table {{.Names}}" | grep -q "^dual-dev$"; then
    echo "🗑️ 기존 dual-dev 컨테이너 제거 중..."
    docker rm -f dual-dev >/dev/null 2>&1
fi

echo ""
echo "🔧 컨테이너 실행 설정:"
echo "   - 이미지: dual-face-gpu-pipeline:latest"
echo "   - GPU 지원: ${GPU_ARGS:-"비활성화"}"
echo "   - 워크스페이스: $(pwd) → /workspace"
echo "   - venv 경로: /workspace/.venv"
echo ""

# Docker 컨테이너 실행
echo "🚀 컨테이너 시작..."
docker run $GPU_ARGS -it --rm \
    --name dual-dev \
    --privileged \
    -v $(pwd):/workspace \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e DISPLAY=${DISPLAY:-:1} \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH=/workspace \
    -e PYTHONUNBUFFERED=1 \
    --shm-size=2g \
    -p 8000:8000 \
    -p 8888:8888 \
    -p 5000:5000 \
    --workdir /workspace \
    dual-face-gpu-pipeline:latest \
    /bin/bash -c '
        echo ""
        echo "🎉 Dual-Face GPU Pipeline 개발 환경 시작!"
        echo "=========================================="
        echo ""
        echo "📍 현재 위치: $(pwd)"
        echo "🐍 Python 버전: $(python --version)"
        echo ""
        
        # venv 자동 활성화
        echo "🔧 venv 활성화 중..."
        if [ -f "/opt/venv/bin/activate" ]; then
            source /opt/venv/bin/activate
            echo "✅ venv 활성화 완료: $VIRTUAL_ENV"
        else
            echo "⚠️ venv를 찾을 수 없습니다: /opt/venv"
        fi
        
        # OpenCV 자동 설정 (영구 해결)
        echo "🔧 OpenCV 환경 자동 설정 중..."
        if [ ! -f "/opt/venv/lib/python3.10/site-packages/cv2.so" ] || [ ! -s "/opt/venv/lib/python3.10/site-packages/cv2.so" ]; then
            echo "   • OpenCV venv 복사 중..."
            # 시스템 OpenCV를 venv로 직접 복사 (심링크 대신)
            if [ -f "/usr/lib/python3.10/dist-packages/cv2.so" ]; then
                cp -r /usr/lib/python3.10/dist-packages/cv2* /opt/venv/lib/python3.10/site-packages/ 2>/dev/null
                echo "   • ✅ OpenCV 복사 완료 (system → venv)"
            elif [ -f "/usr/local/lib/python3.10/site-packages/cv2.so" ]; then
                cp -r /usr/local/lib/python3.10/site-packages/cv2* /opt/venv/lib/python3.10/site-packages/ 2>/dev/null
                echo "   • ✅ OpenCV 복사 완료 (local → venv)"
            else
                echo "   • ⚠️ 시스템 OpenCV를 찾을 수 없음"
            fi
        else
            echo "   • ✅ OpenCV 이미 설정됨"
        fi
        
        # cuDNN 라이브러리 경로 설정
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
        echo "   • ✅ cuDNN 라이브러리 경로 설정"
        echo ""
        
        # 기본 환경 확인
        echo "🔍 환경 확인:"
        echo "   Python: $(python --version)"
        echo "   Pip: $(pip --version)"
        echo "   워크스페이스: $(pwd)"
        
        # OpenCV 작동 확인
        echo "   OpenCV: $(python -c 'import cv2; print(f\"v{cv2.__version__} CUDA:{cv2.cuda.getCudaEnabledDeviceCount()}\")'  2>/dev/null || echo '❌ Import 실패')"
        echo "   PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())'  2>/dev/null || echo '❌ Import 실패')"
        echo ""
        
        # GPU 확인
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "🎯 GPU 상태:"
            nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1
            echo ""
        fi
        
        echo "🚀 개발 환경 준비 완료!"
        echo ""
        echo "💡 유용한 명령어들:"
        echo "   python test_pipeline.py    # 파이프라인 검증"
        echo "   nvidia-smi                 # GPU 모니터링"
        echo "   pip list                   # 패키지 확인"
        echo "   exit                       # 컨테이너 종료"
        echo ""
        echo "🎯 dual_face_tracker_plan.md 구현을 시작하세요!"
        echo ""
        
        # .bashrc에도 venv 활성화 추가 (보험용)
        if ! grep -q "source /opt/venv/bin/activate" ~/.bashrc 2>/dev/null; then
            echo "source /opt/venv/bin/activate 2>/dev/null || true" >> ~/.bashrc
        fi
        
        exec /bin/bash
    '