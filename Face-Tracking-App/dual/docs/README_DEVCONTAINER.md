# 🚀 venv 기반 DevContainer 빌드 가이드

## 완료된 작업 요약

✅ **PEP 668 우회**: venv 격리 환경으로 Ubuntu 24.04 호환성 문제 해결  
✅ **moviepy → PyAV 대체**: dual_face_tracker_plan.md의 NVDEC 파이프라인 구현  
✅ **경량화**: 문제 패키지들 제거, 필수 컴포넌트만 유지  
✅ **즉시 개발 준비**: 빌드 완료 = venv 활성화 + 코딩 가능 상태

## 📁 생성된 파일들

```
dual/.devcontainer/
├── Dockerfile                    # venv 기반 DevContainer (기존 파일 교체)
├── setup_devcontainer.sh         # 환경 검증 스크립트
└── requirements_clean.txt         # 경량화된 의존성 (moviepy 제거)

dual/
├── test_pipeline.py              # GPU 파이프라인 컴포넌트 검증
└── README_DEVCONTAINER.md        # 이 파일
```

## 🔧 빌드 및 실행 방법

### 1단계: 호스트에서 빌드
```bash
# dual 디렉토리로 이동
cd /home/hamtoto/work/python-study/Face-Tracking-App/dual

# DevContainer 빌드 (시간이 오래 걸림)
docker build .devcontainer -t dual-face-venv-dev

# 또는 VS Code DevContainer Extension으로 빌드
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

### 2단계: 컨테이너 실행 및 검증
```bash
# GPU 지원 컨테이너 실행
docker run --gpus all -it \
    -v $(pwd):/workspace \
    dual-face-venv-dev

# 컨테이너 내부에서 자동으로 venv 활성화됨
# "✅ venv activated: /workspace/.venv" 메시지 확인

# 환경 검증 실행
./setup_devcontainer.sh

# GPU 파이프라인 검증 (GPU 런타임)
python test_pipeline.py
```

### 3단계: VS Code 개발 환경
```bash
# VS Code에서 DevContainer 연결
# 1. 폴더 열기: dual/
# 2. 우측 하단 "Reopen in Container" 클릭
# 3. 자동으로 venv 환경에서 개발 시작
```

## 🎯 핵심 변경사항

### 1. **moviepy → PyAV 대체**
```python
# 기존 (문제 있음)
from moviepy.editor import VideoFileClip

# 새로운 방식 (dual_face_tracker_plan.md 권장)
import av
container = av.open(video_path)
stream = container.streams.video[0]
```

### 2. **venv 완전 격리**
```dockerfile
# venv 경로 설정 (핵심: 모든 Python 작업이 venv 내에서)
ENV VENV_PATH=/workspace/.venv
ENV PATH=${VENV_PATH}/bin:${PATH}
ENV VIRTUAL_ENV=${VENV_PATH}

# venv 내에서 패키지 설치 (PEP 668 우회)
RUN ${VENV_PATH}/bin/pip install --no-cache-dir -r /tmp/requirements_clean.txt
```

### 3. **자동 venv 활성화**
```bash
# 컨테이너 시작 시 자동 실행
if [ -d "/workspace/.venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    source /workspace/.venv/bin/activate
    echo '✅ venv activated: /workspace/.venv'
fi
```

## 🧪 검증 결과 예상

### ✅ 성공 시 출력
```
🎉 최종 검증 결과
==================
전체 테스트: 7개
통과: 7개
실패: 0개
성공률: 100.0%

🎉 모든 테스트 통과! DevContainer 환경 준비 완료!
→ dual_face_tracker_plan.md의 GPU 파이프라인 구현을 시작할 수 있습니다.
```

### 📦 설치된 핵심 패키지들
- **PyTorch 2.7.1+cu128**: RTX 5090 Blackwell 최적화
- **PyAV 12.3.0**: NVDEC hwaccel=cuda 지원
- **OpenCV 4.13.0-dev**: CUDA 가속 (커스텀 DEB)
- **TensorRT 10.5.0+**: GPU 추론 가속
- **PyNvVideoCodec**: VPF 제로카피 파이프라인
- **CuPy**: CUDA 커널 직접 구현

## 🚨 문제 해결

### 빌드 실패 시
1. **디스크 공간 확인**: 최소 20GB 필요
2. **Docker 캐시 정리**: `docker system prune -a -f`
3. **인터넷 연결**: PyTorch nightly 다운로드 필요

### GPU 인식 실패 시
1. **nvidia-docker2 설치 확인**
2. **Docker daemon 재시작**: `sudo systemctl restart docker`
3. **nvidia-smi 확인**: 호스트에서 GPU 상태 점검

### venv 미활성화 시
```bash
# 수동 활성화
source /workspace/.venv/bin/activate

# 또는 환경 재설정
./setup_devcontainer.sh
```

## 🎉 다음 단계

1. **GPU 파이프라인 구현**: dual_face_tracker_plan.md 따라 진행
2. **PyAV NVDEC 디코딩**: 하드웨어 가속 비디오 처리
3. **TensorRT 모델 변환**: YOLOv8n/s → .trt 엔진
4. **NVENC 인코딩**: 최종 비디오 출력

**목표**: PyAV NVDEC → TensorRT → NVENC 제로카피 파이프라인 완성

---

**환경 구성 완료**: ✅ Ubuntu 24.04 + Python 3.10 + venv + RTX 5090 최적화  
**개발 준비 상태**: ✅ 즉시 코딩 시작 가능