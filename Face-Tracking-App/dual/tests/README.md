# Tests Directory

이 디렉토리는 Dual-Face High-Speed Video Processing System의 모든 테스트 파일들을 포함합니다.

## 📁 디렉토리 구조

```
tests/
├── README.md                    # 이 파일
├── logs/                        # 테스트 로그 파일들
│   ├── phase1_test_log.txt      # Phase 1 테스트 로그
│   ├── phase3_fixed_test_log.txt # Phase 3 Fixed 테스트 로그
│   └── gpu_monitor.log          # GPU 모니터링 로그
│
├── # Phase 별 테스트 파일들
├── test_phase3_fixed.py         # Phase 3 수정된 멀티스트림 테스트
├── test_pipeline.py             # 전체 파이프라인 테스트
├── test_single_stream.py        # 단일 스트림 테스트
├── test_multi_stream.py         # 멀티스트림 테스트
├── test_multi_stream_minimal.py # 최소 멀티스트림 테스트
│
├── # 컴포넌트별 테스트 파일들
├── test_gpu_composition.py      # GPU 합성 테스트
├── test_dual_face_processor.py  # DualFaceProcessor 테스트
├── test_nvenc_encoding.py       # NVENC 인코딩 테스트
├── test_av_nvenc.py             # PyAV NVENC 테스트
├── test_simple_nvenc.py         # 간단한 NVENC 테스트
├── test_software_fallback.py    # 소프트웨어 폴백 테스트
├── test_memory_fix.py           # 메모리 수정 테스트
│
├── # 통합 및 고급 테스트
├── test_real_video_pipeline.py  # 실제 비디오 파이프라인 테스트
├── test_bytetrack_integration.py # ByteTrack 통합 테스트
├── test_conditional_reid_integration.py # ConditionalReID 통합 테스트
│
└── # 실행 스크립트들
├── run_phase1_test.sh           # Phase 1 테스트 실행
├── run_phase3_test.sh           # Phase 3 테스트 실행
├── run_phase3_test_with_log.sh  # Phase 3 로그포함 테스트 실행
└── run_phase3_fixed_test.sh     # Phase 3 Fixed 테스트 실행
```

## 🚀 주요 테스트들

### Phase 별 테스트
- **test_phase3_fixed.py**: NVENC 세션 제한 문제를 해결한 Phase 3 테스트
- **test_pipeline.py**: 전체 GPU 파이프라인 기능 검증
- **test_single_stream.py**: Phase 1 단일 스트림 처리 검증

### 핵심 컴포넌트 테스트
- **test_gpu_composition.py**: TileComposer, GpuResizer 등 GPU 합성 기능
- **test_dual_face_processor.py**: 메인 프로세서 통합 테스트
- **test_nvenc_encoding.py**: NVENC 하드웨어 인코딩 테스트

### 통합 테스트
- **test_multi_stream.py**: 4-Stream 병렬 처리 테스트
- **test_real_video_pipeline.py**: 실제 비디오 파일로 End-to-End 테스트

## ⚙️ 테스트 실행 방법

### DevContainer에서 실행 (권장)
```bash
# DevContainer 시작
./run_dev.sh

# 개별 테스트 실행
python3 tests/test_phase3_fixed.py

# 스크립트로 실행
./tests/run_phase3_fixed_test.sh
```

### 주요 테스트 스크립트
```bash
# Phase 1 기본 기능 테스트
./tests/run_phase1_test.sh

# Phase 3 최신 테스트 (NVENC 세션 제한 해결)
./tests/run_phase3_fixed_test.sh

# 상세 로그와 함께 테스트
./tests/run_phase3_test_with_log.sh
```

## 📊 테스트 결과 확인

### 로그 파일들
```bash
# 실시간 로그 모니터링
tail -f tests/logs/phase3_fixed_test_log.txt

# GPU 사용률 확인
tail -f tests/logs/gpu_monitor.log

# 에러만 필터링
grep -i error tests/logs/*.txt
```

### 성공 기준
- **Phase 1**: 환경 검증 + 기본 파이프라인 7/7 테스트 통과
- **Phase 3**: 멀티스트림 4/4 테스트 통과, 세션 제한 준수, 폴백 동작

## 🔧 개발자 노트

### Import 경로 설정
모든 테스트 파일들은 다음과 같이 경로를 설정합니다:
```python
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent  # tests/ 에서 dual/ 로
sys.path.insert(0, str(current_dir))
```

### 테스트 환경
- **필수**: DevContainer 환경에서 실행
- **GPU**: RTX 5090 또는 호환 NVIDIA GPU
- **CUDA**: 12.8, PyAV 11.0.0, OpenCV 4.13.0-dev

### 주의사항
- 모든 개발과 테스트는 DevContainer 내부에서만 수행
- 호스트 시스템에서 직접 실행하면 환경 문제 발생 가능
- GPU 메모리 부족 시 DevContainer 재시작 권장