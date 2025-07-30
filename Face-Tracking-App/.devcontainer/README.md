# Face-Tracking-App 데브컨테이너 가이드

## 🚀 빠른 시작

### 1. VS Code에서 열기
1. VS Code에서 Face-Tracking-App 폴더 열기
2. "Reopen in Container" 선택
3. 컨테이너 빌드 및 시작 대기 (최초 5-10분)

### 2. 환경 확인
```bash
# GPU 환경 테스트
~/test_gpu.sh

# 앱 실행 테스트
python main.py
```

## 📁 파일 구조

```
.devcontainer/
├── Dockerfile              # 컨테이너 이미지 정의
├── docker-compose.yml      # 서비스 및 볼륨 설정
├── devcontainer.json       # VS Code 개발 환경 설정
├── build_20250710-1_amd64.deb  # 사용자 빌드 OpenCV
└── README.md               # 이 파일
```

## 🔧 주요 기능

### GPU 지원
- NVIDIA CUDA 12.8 지원
- PyTorch GPU 가속
- 멀티프로세싱 GPU 세마포어

### 개발 도구
- Python 3.10 + pip
- VS Code 확장 프로그램 자동 설치
- Git 설정 동기화
- Jupyter Lab 포트 포워딩

### 최적화된 환경
- 사용자 빌드 OpenCV
- GPU 메모리 최적화 설정
- 멀티프로세싱 IPC 지원

## 🎯 사용법

### 개발 작업
```bash
# 메인 앱 실행
python main.py

# 테스트 실행  
pytest test/

# GPU 메모리 모니터링
watch -n 1 nvidia-smi
```

### 디버깅
```bash
# GPU 상태 확인
~/test_gpu.sh

# 컨테이너 로그 확인
docker-compose -f .devcontainer/docker-compose.yml logs

# 컨테이너 상태 확인
docker-compose -f .devcontainer/docker-compose.yml ps
```

### 데이터 관리
- `/workspace/videos/input/` - 입력 비디오
- `/workspace/videos/output/` - 출력 결과
- `/workspace/temp_proc/` - 임시 처리 파일

## 🔄 컨테이너 관리

### 재빌드
```bash
# VS Code에서: Ctrl+Shift+P > "Dev Containers: Rebuild Container"
# 또는 터미널에서:
docker-compose -f .devcontainer/docker-compose.yml build --no-cache
```

### 정리
```bash
# 컨테이너 정지 및 삭제
docker-compose -f .devcontainer/docker-compose.yml down

# 볼륨까지 삭제 (주의: 데이터 손실)
docker-compose -f .devcontainer/docker-compose.yml down -v
```

## ⚠️ 주의사항

1. **GPU 드라이버**: 호스트에 NVIDIA 드라이버 설치 필요
2. **메모리**: 최소 16GB RAM + 8GB VRAM 권장
3. **저장공간**: 컨테이너 + 모델 + 데이터로 최소 50GB 필요
4. **권한**: Docker GPU 접근 권한 필요

## 🐛 트러블슈팅

### GPU 인식 안됨
```bash
# 호스트에서 확인
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
```

### 빌드 실패
```bash
# 캐시 없이 재빌드
docker-compose -f .devcontainer/docker-compose.yml build --no-cache --pull
```

### 권한 문제
```bash
# 컨테이너 내에서
sudo chown -R developer:developer /workspace
```