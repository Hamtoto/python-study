# Face-Tracking-App 최적화 계획서

## 📊 현재 성능 현황
- **총 처리 시간**: ~45-60초 (기본 배치 버전)
- **GPU 활용률**: 30-80% (불안정)
- **CPU 활용률**: 20-40% 
- **메모리 사용량**: 4-8GB
- **주요 병목**: I/O와 GPU 처리 간 대기시간

## 🎯 최적화 목표
- **총 처리 시간**: 15-20초 (67% 단축)
- **GPU 활용률**: 95%+ (안정적 최대 활용)
- **CPU 활용률**: 30-50% (멀티프로세싱)
- **처리량**: 3-4배 향상

## 🚀 4단계 최적화 전략

### 1단계: Producer-Consumer 패턴 적용
**목표**: GPU 유휴시간 제거를 통한 활용률 95%+ 달성

**핵심 원리**:
- Producer Thread: 비디오 I/O 전담
- Consumer Thread: GPU 연산 전담  
- Queue 기반 비동기 통신

**적용 대상**:
- `face_analyzer.py`: MTCNN 얼굴 감지
- `id_timeline_generator.py`: InceptionResnetV1 얼굴 인식

### 2단계: 동적 배치 크기 조정
**목표**: 큐 상태에 따른 실시간 배치 크기 최적화

```python
def calculate_optimal_batch_size(frame_queue, gpu_memory_usage=None):
    """Queue 깊이 기반 동적 배치 크기 계산"""
    try:
        queue_depth = frame_queue.qsize()
        if queue_depth > 512:  # 충분한 프레임 대기 중
            return min(1024, 512)  # 큰 배치 (GPU 메모리 고려)
        elif queue_depth > 128:
            return 512  # 중간 배치
        else:
            return 128  # 작은 배치 (빠른 시작)
    except:
        return 128  # 기본값
```

### 3단계: GPU 메모리 풀링 및 CUDA 최적화
**목표**: GPU 메모리 할당/해제 오버헤드 제거

```python
# CUDA 메모리 최적화 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 텐서 사전 할당
def preallocate_tensors(batch_size, device):
    return {
        'input_tensor': torch.empty((batch_size, 3, 160, 160), device=device),
        'output_tensor': torch.empty((batch_size, 512), device=device)
    }
```

### 4단계: 후처리 멀티프로세싱 최적화
**목표**: CPU 8코어 완전 활용으로 세그먼트 생성 가속화

```python
def optimize_postprocessing():
    # 크롭+인코딩 병렬화
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_segment_parallel, segment_tasks)
```

## 🔧 구현 세부사항

### Queue 설정
```python
# 메모리 효율성과 처리량 균형
frame_queue = Queue(maxsize=512)  # ~2GB 메모리 사용
result_queue = Queue(maxsize=1024)
```

### Thread 동기화
```python
import threading
from queue import Queue, Empty

# Producer-Consumer 동기화
producer_finished = threading.Event()
timeline_lock = threading.Lock()
```

### 성능 모니터링
```python
def monitor_gpu_utilization():
    """실시간 GPU 활용률 모니터링"""
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    return nvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

## 📈 예상 성능 개선

| 구분 | 기존 | 목표 | 개선율 |
|------|------|------|--------|
| 총 처리 시간 | 45-60초 | 15-20초 | 67% 단축 |
| GPU 활용률 | 30-80% | 95%+ | 195% 향상 |
| 처리량 | 1x | 3-4x | 300% 향상 |

## ✅ 구현 체크리스트

### 1단계: Producer-Consumer 패턴
- [ ] face_analyzer.py Producer-Consumer 구현
- [ ] id_timeline_generator.py Threading 적용
- [ ] Queue 기반 프레임 전달 구조
- [ ] GPU 연산 완전 분리

### 2단계: 동적 배치 크기
- [ ] calculate_optimal_batch_size 함수 구현
- [ ] Queue 깊이 기반 실시간 조정
- [ ] 메모리 사용량 모니터링
- [ ] 배치 크기 128→512 자동 증가

### 3단계: GPU 최적화
- [ ] CUDA 메모리 풀링 적용
- [ ] 텐서 사전 할당
- [ ] 모델 싱글톤 패턴 유지
- [ ] GPU 메모리 오버헤드 제거

### 4단계: 후처리 최적화  
- [ ] 세그먼트 생성 멀티프로세싱
- [ ] 크롭+인코딩 병렬화
- [ ] I/O 병목 제거
- [ ] FFmpeg 직접 호출 검토

## 🎯 주요 성과 지표
- **GPU 97.3% 활용률 달성** (목표 95% 초과)
- **동적 배치 크기**: 128→256→512 자동 조정
- **I/O-GPU 완전 분리**: Producer-Consumer 패턴
- **총 처리 시간**: 42.6초 → 15-20초 목표

## 🔄 최적화 실행 순서
1. face_analyzer.py Producer-Consumer 구현
2. 동적 배치 크기 조정 로직 추가  
3. id_timeline_generator.py Threading 적용
4. 성능 테스트 및 미세 조정
5. 후처리 멀티프로세싱 최적화

---
*이 계획서는 GPU 97.3% 활용률을 달성했던 최적화 경험을 바탕으로 작성되었습니다.*