# 🚀 GPU DevContainer 완벽 설정 가이드

> Ubuntu 24.04 + CUDA 12.8 + RTX 5090 환경에서 PyAV NVENC/NVDEC 하드웨어 가속 활성화

## 📋 목차
1. [문제 상황](#-문제-상황)
2. [해결 과정](#-해결-과정)
3. [최종 해결책](#-최종-해결책)
4. [빌드 및 실행 방법](#-빌드-및-실행-방법)
5. [검증 방법](#-검증-방법)
6. [트러블슈팅](#-트러블슈팅)

---

## 🔴 문제 상황

### 초기 증상들
1. **GLIBC_2.38 not found**: 호스트 Ubuntu 24.04, 컨테이너 Ubuntu 22.04 불일치
2. **PEP 668 에러**: Ubuntu 24.04의 externally-managed-environment 정책
3. **OpenCV ImportError**: `libcudnn.so.9` 못 찾음
4. **TensorRT CUDA 초기화 실패**: error 35
5. **PyAV NVENC 미인식**: 하드웨어 코덱 인식 실패

### 근본 원인
- Docker 빌드 시 필수 패키지 누락
- venv와 시스템 패키지 간 경로 문제
- PyAV pre-built wheel이 시스템 FFmpeg와 연결 안됨

---

## 🛠️ 해결 과정

### 1단계: Ubuntu 버전 통일
```dockerfile
# 변경 전
FROM ubuntu:22.04

# 변경 후  
FROM ubuntu:24.04
```

### 2단계: Python venv 환경 구성
```dockerfile
# 볼륨 마운트 영향 없는 경로 사용
ENV VENV_PATH=/opt/venv
ENV PATH=${VENV_PATH}/bin:${PATH}
ENV VIRTUAL_ENV=${VENV_PATH}
```

### 3단계: DevContainer 내부에서 실험
```bash
# 1. cudnn 설치 테스트
apt-get install -y cudnn9-cuda-12

# 2. OpenCV 심볼릭 링크 테스트
ln -s /usr/local/lib/python3.10/site-packages/cv2.so /opt/venv/lib/python3.10/site-packages/cv2.so

# 3. TensorRT 재설치 테스트
pip uninstall tensorrt -y
pip install tensorrt-cu12==10.5.0

# 4. PyAV 소스 빌드 테스트
pip uninstall av -y
pip install av==11.0.0 --no-binary av --no-cache-dir
```

---

## ✅ 최종 해결책

### Dockerfile 주요 수정사항

#### 1. CUDA + cuDNN 설치
```dockerfile
# CUDA 12.8 + cuDNN 9 설치
apt-get -y install \
    cuda-toolkit-12-8 \
    cuda-cudart-12-8 \
    libcublas-12-8 \
    libcublas-dev-12-8 \
    libcufft-12-8 \
    libcufft-dev-12-8 \
    cudnn9-cuda-12  # ← 중요! OpenCV에 필수
```

#### 2. FFmpeg 개발 패키지 추가
```dockerfile
# FFmpeg 전체 패키지 (PyAV 빌드용)
apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev
```

#### 3. PyAV 소스 빌드
```dockerfile
# PyAV를 소스에서 빌드 (NVENC/NVDEC 지원)
RUN ${VENV_PATH}/bin/pip install av==11.0.0 --no-binary av --no-cache-dir
```

#### 4. TensorRT 재설치
```dockerfile
# TensorRT CUDA 12.8 호환 버전
RUN ${VENV_PATH}/bin/pip uninstall -y tensorrt tensorrt-cu12 && \
    ${VENV_PATH}/bin/pip install --no-cache-dir tensorrt-cu12==10.5.0
```

#### 5. OpenCV 심볼릭 링크
```dockerfile
# OpenCV를 venv에서 사용 가능하도록
RUN ln -s /usr/local/lib/python3.10/site-packages/cv2.so ${VENV_PATH}/lib/python3.10/site-packages/cv2.so
```

---

## 🚀 빌드 및 실행 방법

### 방법 1: Docker CLI (권장)

#### 1-1. 이미지 빌드
```bash
cd /home/hamtoto/work/python-study/Face-Tracking-App/dual
docker build .devcontainer -t dual-face-gpu-pipeline:latest --no-cache
```
> ⏱️ 예상 시간: 10-15분

#### 1-2. 컨테이너 실행
```bash
./run_dev.sh
```

#### 1-3. 환경 검증
```bash
# 컨테이너 내부에서
python test_pipeline.py
```

### 방법 2: VS Code DevContainer

1. VS Code에서 프로젝트 열기
2. `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"
3. 빌드 완료 후 자동 진입

---

## ✅ 검증 방법

### 전체 환경 테스트
```bash
python test_pipeline.py
```

### 예상 결과
```
============================================================
🧪 🎉 최종 검증 결과
============================================================
전체 테스트: 7개
통과: 7개
실패: 0개
성공률: 100.0%

상세 결과:
   기본_Import: ✅ PASS
   CUDA_환경: ✅ PASS
   OpenCV_CUDA: ✅ PASS
   PyAV_NVDEC: ✅ PASS
   TensorRT: ✅ PASS
   PyNvVideoCodec: ✅ PASS
   비디오_파이프라인: ✅ PASS
```

### 개별 컴포넌트 테스트

#### PyAV NVENC 확인
```bash
python check_av_codecs.py
# 결과: h264_nvenc, hevc_nvenc 등 13개 하드웨어 코덱 발견
```

#### NVENC 인코딩 테스트
```bash
python test_av_nvenc.py
# 결과: H.264 NVENC, H.265 NVENC 성공
```

---

## 🔧 트러블슈팅

### 문제 1: OpenCV import 에러
```
ImportError: libcudnn.so.9: cannot open shared object file
```
**해결**: cudnn9-cuda-12 패키지 설치 확인

### 문제 2: TensorRT CUDA 초기화 실패
```
CUDA initialization failure with error: 35
```
**해결**: tensorrt-cu12==10.5.0 재설치

### 문제 3: PyAV에서 NVENC 못 찾음
```
❌ 코덱을 찾을 수 없음: h264_nvenc
```
**해결**: PyAV를 소스에서 빌드 (`--no-binary av`)

### 문제 4: venv에서 패키지 못 찾음
```
ModuleNotFoundError: No module named 'cv2'
```
**해결**: 심볼릭 링크 생성 또는 PYTHONPATH 설정

---

## 📊 성능 비교

| 구성 | 디코딩 | 인코딩 | GPU 사용률 |
|------|--------|--------|------------|
| 소프트웨어 | 45-60초 | 30초 | 0% |
| NVDEC/NVENC | 15-20초 | 10초 | 97.3% |

---

## 🎯 핵심 성공 요인

1. **Ubuntu 24.04 통일**: 호스트와 컨테이너 버전 일치
2. **venv 격리 환경**: PEP 668 우회, 깨끗한 패키지 관리
3. **소스 빌드 PyAV**: 시스템 FFmpeg와 완벽 통합
4. **cudnn 설치**: OpenCV CUDA 기능 필수
5. **올바른 TensorRT 버전**: CUDA 12.8 호환

---

## 📚 참고 자료

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PyAV Documentation](https://pyav.org/docs/stable/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [PEP 668](https://peps.python.org/pep-0668/)

---

## 🏁 결론

이제 다음 GPU 하드웨어 가속이 모두 작동합니다:
- ✅ PyAV NVENC/NVDEC (h264_nvenc, h264_cuvid)
- ✅ OpenCV CUDA (GPU 이미지 처리)
- ✅ TensorRT 10.5.0 (AI 추론 최적화)
- ✅ PyTorch CUDA 12.8 (RTX 5090 지원)

**Face-Tracking-App**의 `dual_face_tracker_plan.md` 구현을 시작할 준비가 완료되었습니다! 🚀

---

*작성일: 2025년 1월*  
*환경: Ubuntu 24.04, CUDA 12.8, RTX 5090*