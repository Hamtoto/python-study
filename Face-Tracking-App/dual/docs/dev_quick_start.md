# 🚀 Dual-Face GPU Pipeline 개발 가이드

## 🎯 **권장 방법: Shell 스크립트 실행**

### 1️⃣ **한 번만 빌드** (업데이트됨: 2025년 1월)
```bash
cd /home/hamtoto/work/python-study/Face-Tracking-App/dual
docker build .devcontainer -t dual-face-gpu-pipeline:latest --no-cache
```
> ⚠️ **중요**: `--no-cache` 옵션 추가로 깨끗한 빌드 보장

### 2️⃣ **개발 환경 시작 (매번)**
```bash
./run_dev.sh
```

### 3️⃣ **컨테이너 내에서 즉시 개발**
```bash
# venv 자동 활성화 완료 상태
python test_pipeline.py     # 환경 검증
nvidia-smi                  # GPU 확인
python                      # 개발 시작!
```

## ✅ **이 방법의 장점**
- 🚫 **DevContainer 복잡성 완전 제거**: VS Code 설정 오류 무시
- ✅ **확실한 동작**: 빌드된 이미지 직접 활용
- ✅ **즉시 시작**: 복잡한 설정 없이 바로 개발
- ✅ **완전 자동화**: venv, GPU, 포트 모든 설정 자동
- ✅ **재현 가능**: 어디서나 동일한 환경

## 🔧 **자동 구성되는 환경** (2025년 1월 업데이트)
- **워크스페이스**: `/workspace` (dual 디렉토리 마운트)
- **가상환경**: `/opt/venv` (자동 활성화, 볼륨 마운트 영향 없음)
- **Python**: 3.10.18 + CUDA 12.8 + RTX 5090 최적화
- **GPU 지원**: nvidia-docker + 전체 GPU 접근
- **포트**: 8000, 8888, 5000 자동 포워딩
- **패키지**: 
  - PyAV 11.0.0 (NVENC/NVDEC 하드웨어 가속)
  - TensorRT 10.5.0 (CUDA 12.8 호환)
  - PyTorch 2.9.0+cu128 (nightly)
  - OpenCV 4.13.0 (CUDA + cuDNN)

## 💻 **개발 워크플로우**
```bash
# 1. 한 번 빌드 (11분 소요)
docker build .devcontainer -t dual-face-gpu-pipeline:latest

# 2. 매일 개발 (5초 시작)
./run_dev.sh

# 3. 컨테이너 안에서
python test_pipeline.py        # 파이프라인 검증
python -c "import torch; print(torch.cuda.is_available())"  # CUDA 확인
# dual_face_tracker_plan.md 구현 시작!
```

## 🛠️ **유용한 명령어들**
```bash
# 환경 확인
echo $VIRTUAL_ENV              # venv 경로
python --version               # Python 버전
pip list | grep torch          # PyTorch 확인
pip list | grep av             # PyAV 확인

# GPU 모니터링
nvidia-smi                     # GPU 상태
nvidia-smi -l 1               # 실시간 모니터링
nvtop                         # GPU 사용률 (설치됨)

# 개발 도구
python test_pipeline.py       # GPU 파이프라인 전체 검증
./setup_devcontainer.sh       # 상세 환경 검증

# 컨테이너 관리
exit                          # 컨테이너 종료
docker ps                     # 실행 중인 컨테이너 확인
```

## 🎉 **DevContainer 문제 완전 해결!**

**VS Code DevContainer 설정 오류는 무시하고, Shell 스크립트로 안정적인 개발 환경 사용하세요!**

### ✅ **검증된 GPU 하드웨어 가속**
- PyAV NVENC/NVDEC: h264_nvenc, hevc_nvenc, h264_cuvid 등 13개 코덱
- OpenCV CUDA: GPU 이미지 처리 가속
- TensorRT: AI 추론 최적화
- PyTorch: RTX 5090 31GB 완벽 지원

자세한 설정 과정은 [`GPU_DEVCONTAINER_SETUP_GUIDE.md`](./GPU_DEVCONTAINER_SETUP_GUIDE.md)를 참고하세요.

이제 **dual_face_tracker_plan.md**의 **PyAV NVDEC → TensorRT → NVENC** 파이프라인 구현을 시작할 수 있습니다! 🚀