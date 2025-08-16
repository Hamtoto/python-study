# Dual-Face High-Speed Video Processing System (Full GPU Pipeline)

## 🎯 프로젝트 개요

**프로젝트명**: Dual-Face High-Speed Video Processing System  
**핵심 목표**: **PyAV NVDEC → TensorRT → NVENC 풀 GPU 파이프라인**

**주요 목표**:  
- 입력된 영상에서 2명의 얼굴을 검출하여 좌우 분기 처리
- **CUDA Stream 기반 병렬 처리**로 여러 영상 동시 처리
- **PyAV(hwaccel=cuda) → TensorRT → NVENC(H.264)** 제로카피 파이프라인 구축
- 기존 대비 **5-8배 처리량 향상** 달성 (디코딩+인코딩 병목 완전 제거)
- 클라이언트 요구사항: 최대한 많은 영상을 최대한 빠르게 처리하여 납품

**처리 방식 혁신**:
```
기존 CPU 파이프라인: 영상1(23분) → 영상2(23분) → 영상3(23분) = 69분
신규 Full GPU 파이프라인: [영상1,2,3,4] CUDA Stream 병렬 = 12-15분
```

---

## 🚀 Full GPU 제로카피 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    단일 GPU 프로세스 (RTX 5090)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │CUDA Stream 1│    │CUDA Stream 2│    │CUDA Stream 3│    │CUDA Stream 4│    │
│  │             │    │             │    │             │    │             │    │
│  │PyAV NVDEC   │    │PyAV NVDEC   │    │PyAV NVDEC   │    │PyAV NVDEC   │    │
│  │(hwaccel=cuda)│   │(hwaccel=cuda)│   │(hwaccel=cuda)│   │(hwaccel=cuda)│   │
│  │    ↓        │    │    ↓        │    │    ↓        │    │    ↓        │    │
│  │GPU 전처리     │    │GPU 전처리     │    │GPU 전처리    │    │GPU 전처리     │    │
│  │    ↓        │    │    ↓        │    │    ↓        │    │    ↓        │    │
│  │TensorRT     │    │TensorRT     │    │TensorRT     │    │TensorRT     │    │
│  │YOLO/SCRFD   │    │YOLO/SCRFD   │    │YOLO/SCRFD   │    │YOLO/SCRFD   │    │
│  │    ↓        │    │    ↓        │    │    ↓        │    │    ↓        │    │
│  │조건부 ReID   │    │조건부 ReID   │    │조건부 ReID   │    │조건부 ReID   │    │
│  │(ByteTrack+) │    │(ByteTrack+) │    │(ByteTrack+) │    │(ByteTrack+) │    │
│  │    ↓        │    │    ↓        │    │    ↓        │    │    ↓        │    │
│  │GPU 리사이즈+  │     │GPU 리사이즈+  │    │GPU 리사이즈+  │    │GPU 리사이즈+  │    │
│  │타일 합성      │     │타일 합성      │    │타일 합성      │    │타일 합성      │    │
│  │    ↓        │    │    ↓        │    │    ↓        │    │    ↓        │    │
│  │NVENC H.264  │    │NVENC H.264  │    │NVENC H.264  │    │NVENC H.264  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          배치 처리 최적화                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  동일 해상도 그룹핑 → 소배치(4-8) → TensorRT 병렬 추론 → GPU 효율 극대화               │
│  CPU↔GPU 메모리 복사 완전 제거 → 제로카피 파이프라인                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          지능형 관리 시스템                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  🔧 HybridConfigManager  │  ⚙️ ConditionalReID  │  🛡️ TileErrorPolicy      │
│  - 사용자 수동 설정       │  - ID 스왑 감지        │  - 스트림 실패 처리         │
│  - 자동 하드웨어 프로빙   │  - 경량 ReID 활성화    │  - 대체 프레임 생성         │
│  - 안전한 기본값 폴백     │  - 성능 최적화         │  - 실패 분석 및 복구        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 시스템 환경 및 리소스 할당 (런타임 프로빙 기반)

### ⚙️ **하이브리드 설정 관리 시스템** (유연성 + 안정성)

**핵심 혁신**: **사용자 수동 설정** → **자동 프로빙** → **기본값** 3단계 우선순위 시스템

**ConfigManager 아키텍처**:

```python
class HybridConfigManager:
    def __init__(self):
        self.config_priority = [
            'manual_config.yaml',      # 1순위: 사용자 수동 설정
            'auto_detected.yaml',      # 2순위: 자동 프로빙 결과
            'fallback_config.yaml'     # 3순위: 안전한 기본값
        ]
        self.hardware_prober = HardwareProber()
        self.current_config = None
        
    def load_optimal_config(self):
        """최적 설정 로드 (우선순위 기반)"""
        print("🔧 하이브리드 설정 관리 시작...")
        
        # 1단계: 수동 설정 파일 확인
        if self.exists_and_valid('manual_config.yaml'):
            print("✅ 사용자 수동 설정 발견 - 최우선 적용")
            self.current_config = self.load_yaml('manual_config.yaml')
            return self.current_config
            
        # 2단계: 자동 프로빙 실행
        print("🔍 하드웨어 자동 프로빙 실행 중...")
        try:
            auto_config = self.hardware_prober.generate_optimal_config()
            self.save_yaml('auto_detected.yaml', auto_config)
            print("✅ 자동 프로빙 성공 - 감지된 설정 적용")
            self.current_config = auto_config
            return self.current_config
        except Exception as e:
            print(f"⚠️ 자동 프로빙 실패: {e}")
            
        # 3단계: 안전한 기본값 사용
        print("🛡️ 기본 안전 설정 적용")
        self.current_config = self.load_yaml('fallback_config.yaml')
        return self.current_config
        
    def allow_user_override(self, section, key, value):
        """사용자 설정 재정의 허용"""
        override_config = {
            section: {key: value},
            'override_timestamp': datetime.now().isoformat(),
            'override_reason': f'User manual override for {section}.{key}'
        }
        
        # manual_config.yaml에 추가
        if os.path.exists('manual_config.yaml'):
            existing_config = self.load_yaml('manual_config.yaml')
            existing_config.update(override_config)
        else:
            existing_config = override_config
            
        self.save_yaml('manual_config.yaml', existing_config)
        print(f"✅ 사용자 재정의 저장: {section}.{key} = {value}")
        
    def get_setting(self, section, key, default=None):
        """설정값 조회 (우선순위 적용)"""
        if self.current_config is None:
            self.load_optimal_config()
            
        return self.current_config.get(section, {}).get(key, default)

class HardwareProber:
    """하드웨어 자동 프로빙 (필요시에만 실행)"""
    def __init__(self):
        self.probe_results = None
        
    def generate_optimal_config(self):
        """최적 설정 생성"""
        # GPU 능력 측정
        gpu_info = self.probe_gpu_capabilities()
        
        # 최적화된 설정 생성
        optimal_config = {
            'hardware': gpu_info,
            'performance': {
                'max_concurrent_streams': self.calculate_optimal_streams(gpu_info),
                'batch_size_analyze': self.calculate_optimal_batch_size(gpu_info),
                'vram_safety_margin': 0.15,  # 15% 안전 마진
                'target_gpu_utilization': 0.85  # 85% 목표
            },
            'nvdec_settings': {
                'max_sessions': gpu_info['nvdec_max_sessions'],
                'preferred_format': 'nv12'
            },
            'nvenc_settings': {
                'max_sessions': gpu_info['nvenc_max_sessions'],
                'preset': 'medium',
                'rc_mode': 'cbr'
            },
            'generated_timestamp': datetime.now().isoformat(),
            'gpu_driver_version': gpu_info.get('driver_version'),
            'cuda_version': gpu_info.get('cuda_version')
        }
        
        return optimal_config
        
    def probe_gpu_capabilities(self):
        """GPU 하드웨어 능력 측정 (기존 코드 유지)"""
        import pynvml
        pynvml.nvmlInit()
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
        
        # NVDEC/NVENC 세션 한도 테스트
        nvdec_sessions = self.test_concurrent_decoders()
        nvenc_sessions = self.test_concurrent_encoders()
        vram_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024**3)
        
        # 드라이버 정보
        driver_version = pynvml.nvmlSystemGetDriverVersion().decode()
        
        return {
            'gpu_name': gpu_name,
            'nvdec_max_sessions': nvdec_sessions,
            'nvenc_max_sessions': nvenc_sessions,
            'vram_gb': vram_total,
            'driver_version': driver_version,
            'compute_capability': self.get_compute_capability(handle)
        }
        
    def calculate_optimal_streams(self, gpu_info):
        """최적 동시 스트림 수 계산 (보수적 접근)"""
        # 하드웨어 제약 고려
        nvdec_limit = gpu_info['nvdec_max_sessions']
        nvenc_limit = gpu_info['nvenc_max_sessions']
        vram_limit = max(1, (gpu_info['vram_gb'] * 0.75) // 8)  # VRAM의 75%만 사용
        
        # 가장 제한적인 요소에 맞춤
        optimal_streams = min(nvdec_limit, nvenc_limit * 2, vram_limit, 4)
        
        print(f"📊 최적 스트림 수: {optimal_streams}개")
        print(f"   - NVDEC 제한: {nvdec_limit}개")
        print(f"   - NVENC 제한: {nvenc_limit}개 (×2 = {nvenc_limit*2})")
        print(f"   - VRAM 제한: {vram_limit}개")
        
        return optimal_streams
```

**설정 파일 구조 예시**:

```yaml
# manual_config.yaml (사용자 수동 설정 - 최우선)
hardware:
  gpu_name: "RTX 5090"
  nvdec_max_sessions: 4
  nvenc_max_sessions: 2

performance:
  max_concurrent_streams: 3  # 사용자 강제 제한
  batch_size_analyze: 128    # 성능 vs 메모리 trade-off
  vram_safety_margin: 0.2    # 보수적 20% 마진

override_reason: "Production environment - conservative settings"
```

**사용자 재정의 예시**:

```python
# 런타임에서 사용자가 설정 변경 가능
config_manager = HybridConfigManager()

# 배치 크기 수동 조정
config_manager.allow_user_override('performance', 'batch_size_analyze', 64)

# 안전 마진 증가
config_manager.allow_user_override('performance', 'vram_safety_margin', 0.25)

# 현재 설정 확인
current_batch_size = config_manager.get_setting('performance', 'batch_size_analyze', 256)
```

### 🎦 **동적 리소스 할당표**

| 구성요소 | 동적 감지 방식 | 활용 방식 | 예상 성능 (예시) |
|----------|----------------|-----------|----------------------|
| **NVDEC** | 런타임 세션 테스트 | PyNvCodec/PyAV | **2-6개 세션** (모델별 가변) |
| **NVENC** | 런타임 인코더 테스트 | 하드웨어 인코딩 | **1-3개 세션** (모델별 가변) |
| **CUDA Cores** | 컴퓨트 능력 측정 | TensorRT 병렬 추론 | **SM 활용률 90%+** |
| **VRAM** | pynvml 메모리 조회 | 동적 배치 크기 | **전체 VRAM의 87.5%** |
| **CPU** | 코어수 자동 감지 | 스트림 제어, 오디오 mux | **25% 이하 활용률** |
| **System RAM** | psutil 메모리 조회 | 호스트 버퍼, 메타데이터 | **전체 RAM의 44%** |

### 📊 **VRAM 사용량 정확한 산정**

| 항목 | 스트림당 사용량 | 4개 스트림 총합 | 비고 |
|------|----------------|----------------|------|
| **TensorRT 엔진** | 1.5-2GB | **6-8GB** | YOLOv8n/s 기준 (x-face는 3GB+) |
| **디코드 버퍼** | 1-1.5GB | **4-6GB** | 1080p 프레임 버퍼링 |
| **전처리 워크스페이스** | 0.5GB | **2GB** | GPU 리사이즈/정규화 |
| **추론 워크스페이스** | 1GB | **4GB** | TensorRT 실행 공간 |
| **합성 워크스페이스** | 0.5GB | **2GB** | 타일 합성 버퍼 |
| **NVENC 버퍼** | 0.5GB | **2GB** | 인코딩 버퍼 |
| **여유분/시스템** | - | **4GB** | CUDA 컨텍스트, 기타 |
| **총 VRAM 사용량** | - | **24-28GB** | **32GB 중 87.5% 이하** |

### 📊 **System RAM 사용량**

| 항목 | 예상 사용량 | 비고 |
|------|-------------|------|
| **Python 프로세스** | 2-4GB | 메인 애플리케이션 |
| **PyAV 버퍼** | 4-6GB | CPU 측 비디오 메타데이터 |
| **오디오 스트림** | 1-2GB | 원본 오디오 보존용 |
| **모니터링/로깅** | 1GB | 성능 지표, 로그 버퍼 |
| **시스템 여유분** | 8GB | OS, 기타 프로세스 |
| **총 사용량** | **16-21GB** | **48GB 중 44% 이하** |

---

## 🧠 핵심 기능

### 1. 비디오 디코딩 (**PyAV 제로카피 실전 설정**)

**PyNvCodec(VPF) 표준 제로카피 구현** (권장 솔루션):
```python
import PyNvCodec as nvc
import torch
import cupy as cp

def setup_pynvcodec_decoder(video_path, device_id=0):
    """PyNvCodec 기반 진짜 제로카피 디코더 설정"""
    try:
        # PyNvCodec 디코더 생성
        decoder = nvc.PyDecodeHW(
            video_path,
            nvc.PixelFormat.NV12,  # GPU 네이티브 포맷
            device_id
        )
        
        # 색공간 변환기 (GPU 내부 처리)
        converter = nvc.PySurfaceConverter(
            decoder.Width(), decoder.Height(),
            nvc.PixelFormat.NV12, nvc.PixelFormat.RGB,
            device_id
        )
        
        return decoder, converter
        
    except Exception as e:
        print(f"PyNvCodec 초기화 실패: {e}")
        return None, None

def decode_gpu_frames_zerocopy(decoder, converter):
    """진짜 제로카피 GPU 프레임 디코딩"""
    while True:
        # GPU 메모리에서 직접 NV12 디코딩
        nv12_surface = decoder.DecodeSurface()
        if not nv12_surface:
            break
            
        # GPU 내부 색공간 변환 (NV12→RGB)
        rgb_surface = converter.Execute(nv12_surface)
        
        # DLPack을 통한 제로카피 텐서 변환
        dlpack_tensor = rgb_surface.GetDLPackTensor()
        gpu_tensor = torch.from_dlpack(dlpack_tensor)
        
        yield gpu_tensor

# PyAV 백업 구현 (hw_frames_ctx 명시적 설정)
def setup_pyav_backup(video_path, device_id=0):
    """PyAV 백업 구현 - hw_frames_ctx 명시적 설정"""
    import av
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    # 하드웨어 디바이스 컨텍스트 생성
    hw_device = av.cuda.Device(device_id)
    
    # 코덱 컨텍스트에 하드웨어 설정
    stream.codec_context.hw_device_ctx = hw_device
    stream.codec_context.options['hwaccel'] = 'cuda'
    stream.codec_context.options['hwaccel_output_format'] = 'cuda'
    
    # hw_frames_ctx 명시적 설정
    stream.codec_context.hw_frames_ctx = hw_device.create_hwframes_ctx(
        format='nv12',
        width=stream.width,
        height=stream.height
    )
    
    return container, stream
```

**디코더 선택 전략** (우선순위 기반):
```python
class DecoderSelector:
    def __init__(self):
        self.decoder_priority = [
            ('pynvcodec', self.try_pynvcodec),
            ('pyav_hwframes', self.try_pyav_hwframes),
            ('pyav_basic', self.try_pyav_basic),
            ('cpu_fallback', self.try_cpu_fallback)
        ]
    
    def select_best_decoder(self, video_path, device_id=0):
        """최적 디코더 자동 선택"""
        video_info = self.probe_video(video_path)
        
        for decoder_name, decoder_func in self.decoder_priority:
            try:
                decoder = decoder_func(video_path, device_id)
                if self.validate_decoder(decoder, video_info):
                    print(f"✅ {decoder_name} 디코더 선택됨")
                    return decoder
            except Exception as e:
                print(f"⚠️ {decoder_name} 실패: {e}")
                continue
        
        raise RuntimeError("모든 디코더 초기화 실패")
    
    def should_prefer_pynvcodec(self, video_info):
        """PyNvCodec 우선 선택 조건"""
        conditions = [
            video_info.get('codec') in ['h264', 'hevc'],  # 지원 코덱
            video_info.get('width', 0) * video_info.get('height', 0) > 2073600,  # 1080p 이상
            video_info.get('bitrate', 0) > 10_000_000,  # 10Mbps 이상
        ]
        return sum(conditions) >= 2  # 2개 이상 조건 만족
```

**전환 전략**:
- **1순위**: PyNvCodec (VPF) - 고성능, 완전 제로카피
- **2순위**: PyAV + hw_frames_ctx - 호환성 좋음
- **3순위**: PyAV 기본 hwaccel - 최소 기능
- **4순위**: CPU 디코딩 - 최후 백업

**런타임 하드웨어 프로빙 통합**:

위 `HardwareProber` 클래스가 모든 하드웨어 감지를 담당하며, 다음과 같은 장점을 제공합니다:

- ✅ **모델별 적응**: RTX 4090, RTX 5090, A100 등 자동 최적화
- ✅ **드라이버 버전 대응**: NVDEC/NVENC 세션 한도 실시간 측정  
- ✅ **안전한 백업**: 측정 실패 시 보수적 기본값 사용
- ✅ **동적 큐 설정**: 측정된 한도에 맞춰 배치 크기 자동 조정

### 2. 얼굴 검출 (**모델 후보군 확대 + 벤치마킹**)

**모델 후보군 및 벤치마킹 계획**:

| 모델 | 크기 | 예상 FPS | mAP | VRAM | TensorRT 지원 | 우선순위 |
|------|------|---------|-----|------|---------------|----------|
| **YOLOv8n-face** | 6MB | **120+ FPS** | 85% | **1GB** | ✅ | **1순위** |
| **YOLOv8s-face** | 22MB | **80+ FPS** | 88% | **1.5GB** | ✅ | **2순위** |
| **SCRFD-2.5G** | 10MB | **100+ FPS** | 87% | **1.2GB** | ✅ | **3순위** |
| **YOLOv8x-face** | 136MB | **40+ FPS** | 92% | **3GB** | ✅ | 백업용 |
| **RT-DETR-S** | 20MB | **60+ FPS** | 89% | **1.8GB** | ✅ | 실험용 |

**TensorRT 최적화**:
- **FP16 우선**: mAP 손실 < 3%, 속도 1.5-2배 향상
#### 🎯 **INT8 채택 기준 구체화** (등급 개념)

```python
# INT8 채택 등급 시스템
class INT8AdoptionCriteria:
    def __init__(self):
        # 필수 성능 임계값
        self.precision_threshold = 1.5      # mAP 손실 ≤ 1.5%p
        self.miss_rate_threshold = 0.5      # 미스율 증가 ≤ 0.5%p
        
        # 교정 데이터셋 구성
        self.calibration_spec = {
            'total_videos': 10,
            'duration_per_video': 600,  # 10분
            'scene_conditions': [
                'normal_lighting',    # 일반 조명
                'low_lighting',      # 저조도
                'backlight',         # 역광
                'face_occlusion',    # 가림
                'motion_blur',       # 동작 블러
                'side_profile'       # 측면 프로필
            ]
        }
        
        # 아티팩트 및 재현성 보장
        self.artifact_config = {
            'engine_hash_logging': True,        # TensorRT 엔진 해시 보관
            'calibration_log_retention': 90,    # 교정 로그 90일 보관
            'reproducibility_seed': 42,         # 재현 가능성 시드
            'benchmark_report_archive': True    # 벤치마크 보고서 아카이브
        }
    
    def evaluate_int8_model(self, fp16_model, int8_model, test_dataset):
        """종합적 INT8 모델 평가"""
        # 1. 성능 비교 빤치마크
        fp16_metrics = self.comprehensive_benchmark(fp16_model, test_dataset)
        int8_metrics = self.comprehensive_benchmark(int8_model, test_dataset)
        
        # 2. 주요 지표 비교
        precision_loss = fp16_metrics['mAP'] - int8_metrics['mAP']
        miss_rate_increase = int8_metrics['miss_rate'] - fp16_metrics['miss_rate']
        speed_improvement = int8_metrics['fps'] / fp16_metrics['fps']
        
        # 3. 종합 평가
        evaluation_result = {
            'precision_loss_pct': precision_loss,
            'miss_rate_increase_pct': miss_rate_increase,
            'speed_improvement_ratio': speed_improvement,
            'meets_criteria': self.check_adoption_criteria(precision_loss, miss_rate_increase),
            'recommendation': self.generate_recommendation(precision_loss, miss_rate_increase, speed_improvement),
            'detailed_breakdown': self.analyze_by_scene_condition(int8_metrics, fp16_metrics)
        }
        
        # 4. 아티팩트 생성
        self.generate_artifacts(evaluation_result, fp16_model, int8_model)
        
        return evaluation_result
    
    def check_adoption_criteria(self, precision_loss, miss_rate_increase):
        """기본 채택 기준 검증"""
        return (precision_loss <= self.precision_threshold and 
                miss_rate_increase <= self.miss_rate_threshold)
    
    def generate_recommendation(self, precision_loss, miss_rate_increase, speed_improvement):
        """채택 권고사항 생성"""
        if precision_loss <= 0.8 and miss_rate_increase <= 0.2:  # 우수
            return {
                'grade': 'EXCELLENT',
                'action': 'ADOPT_IMMEDIATELY',
                'reason': f'성능 손실 최소({precision_loss:.2f}%p), 속도 향상 {speed_improvement:.1f}배'
            }
        elif self.check_adoption_criteria(precision_loss, miss_rate_increase):
            return {
                'grade': 'ACCEPTABLE',
                'action': 'ADOPT_WITH_MONITORING',
                'reason': f'기준 충족, 속도 향상 {speed_improvement:.1f}배, 성능 모니터링 필요'
            }
        else:
            return {
                'grade': 'NOT_RECOMMENDED',
                'action': 'USE_FP16',
                'reason': f'성능 손실 과다({precision_loss:.2f}%p > {self.precision_threshold}%p)'
            }
```

#### 📋 **INT8 교정 프로세스**

```bash
# INT8 교정 실행 스크립트
#!/bin/bash
echo "🎯 INT8 교정 및 검증 시작"

# 1. 교정 데이터 준비
python prepare_calibration_dataset.py \
    --videos 10 \
    --duration 600 \
    --conditions "normal,low_light,backlight,occlusion,blur,profile"

# 2. FP16 베이스라인 벤치마크
python benchmark_fp16.py \
    --model yolov8n-face \
    --dataset calibration_set \
    --output fp16_baseline.json

# 3. INT8 교정 및 벤치마크
python calibrate_int8.py \
    --fp16-model yolov8n-face.onnx \
    --calibration-data calibration_set \
    --output yolov8n-face-int8.trt

python benchmark_int8.py \
    --model yolov8n-face-int8.trt \
    --dataset calibration_set \
    --output int8_results.json

# 4. 검증 및 보고서 생성
python evaluate_int8_adoption.py \
    --fp16-results fp16_baseline.json \
    --int8-results int8_results.json \
    --output-report int8_evaluation_report.html

echo "✅ INT8 검증 완료: int8_evaluation_report.html 확인"
```
#### ⚡ **TensorRT 최적화 옵션**

- **FP16 기본 전략**: mAP 손실 < 3%, 속도 1.5-2배 향상
- **INT8 선택적 적용**: 위 기준 출족 시에만 사용
- **CUDA Graphs**: Launch overhead 90% 절감
- **Dynamic Shape 최적화**: 배치 크기 가변성 지원

**배치 처리 최적화** (Tail Latency 방지):
```python
class BatchFlusher:
    def __init__(self):
        self.max_batch = 8          # 최대 배치 크기
        self.max_wait_ms = 6        # 최대 대기 시간
        self.tick_ms = 4            # 주기적 flush 간격
        self.max_jitter_ms = 2      # 지터 허용 범위
        
        self.pending_frames = []
        self.stream_fairness = {}   # WFQ 공정성 관리
        
    def add_frame(self, frame, stream_id, timestamp):
        """프레임을 배치에 추가"""
        self.pending_frames.append({
            'frame': frame,
            'stream_id': stream_id, 
            'timestamp': timestamp,
            'wait_time': 0
        })
        
        # WFQ: 느린 스트림 우선순위 증가
        if stream_id not in self.stream_fairness:
            self.stream_fairness[stream_id] = {'priority': 1.0, 'processed': 0}
            
        # Flush 조건 체크
        if self.should_flush():
            return self.flush_batch()
        return None
        
    def should_flush(self):
        """배치 flush 조건"""
        if len(self.pending_frames) >= self.max_batch:
            return True
            
        if self.pending_frames:
            oldest_frame = min(self.pending_frames, key=lambda x: x['timestamp'])
            wait_time = time.time() - oldest_frame['timestamp']
            if wait_time >= self.max_wait_ms / 1000:
                return True
                
        return False
        
    def flush_batch(self):
        """WFQ 기반 공정한 배치 구성"""
        if not self.pending_frames:
            return None
            
        # 스트림별 우선순위 기반 정렬
        sorted_frames = sorted(self.pending_frames, 
                              key=lambda x: -self.stream_fairness[x['stream_id']]['priority'])
        
        # 배치 구성 (공정성 고려)
        batch = sorted_frames[:self.max_batch]
        self.pending_frames = sorted_frames[self.max_batch:]
        
        # 우선순위 업데이트 (처리된 스트림은 우선순위 감소)
        for frame in batch:
            stream_id = frame['stream_id']
            self.stream_fairness[stream_id]['processed'] += 1
            self.stream_fairness[stream_id]['priority'] *= 0.9  # 처리 후 우선순위 감소
            
        # 처리되지 않은 스트림은 우선순위 증가
        for frame in self.pending_frames:
            stream_id = frame['stream_id']
            self.stream_fairness[stream_id]['priority'] *= 1.1
            
        return [f['frame'] for f in batch]
```

### 3. 조건부 Re-ID 얼굴 추적 시스템 (**지능형 ID 안정성**)

**핵심 혁신**: ID 스왑이 의심되는 **결정적 순간에만** 경량 Re-ID 모델을 호출하여 검증

**ConditionalReID 아키텍처**:

```python
class ConditionalReID:
    def __init__(self):
        self.base_tracker = ByteTrack()           # 기본: 경량 IoU 추적
        self.reid_model = None                    # 128-D 경량 ReID (필요시 로드)
        self.id_swap_threshold = 0.3              # ID 스왑 의심 임계값
        self.confidence_history = {}              # ID별 신뢰도 히스토리
        self.position_jump_threshold = 200        # 급격한 위치 변화 임계값
        
    def should_activate_reid(self, track_id, current_detection):
        """ID 스왑 의심 상황 감지"""
        # 1. 추적 신뢰도 급락
        confidence = self.base_tracker.get_confidence(track_id)
        if confidence < self.id_swap_threshold:
            return True
            
        # 2. 급격한 위치 점프
        if track_id in self.last_positions:
            position_jump = self.calculate_position_jump(
                track_id, current_detection.bbox_center
            )
            if position_jump > self.position_jump_threshold:
                return True
                
        # 3. 얼굴 특성 급변 (크기, 각도)
        if self.detect_face_feature_anomaly(track_id, current_detection):
            return True
            
        return False
        
    def conditional_track(self, detections):
        """조건부 Re-ID 추적"""
        # 1. 기본 ByteTrack 수행
        base_tracks = self.base_tracker.update(detections)
        
        # 2. ID 스왑 의심 케이스 검출
        suspicious_tracks = []
        for track in base_tracks:
            if self.should_activate_reid(track.track_id, track):
                suspicious_tracks.append(track)
                
        # 3. 의심 케이스에만 ReID 적용
        if suspicious_tracks and self.reid_model is None:
            self.reid_model = self.load_lightweight_reid()  # 필요시에만 로드
            
        verified_tracks = []
        for track in base_tracks:
            if track in suspicious_tracks:
                # ReID로 재검증
                verified_track = self.verify_with_reid(track, detections)
                verified_tracks.append(verified_track)
            else:
                # 의심 없으면 그대로 사용
                verified_tracks.append(track)
                
        return verified_tracks
        
    def verify_with_reid(self, suspicious_track, all_detections):
        """ReID 기반 ID 재검증"""
        track_embedding = self.extract_reid_embedding(suspicious_track.detection)
        
        # 기존 ID 히스토리와 유사도 계산
        best_match_id = None
        best_similarity = 0
        
        for existing_id in self.embedding_history:
            similarity = self.cosine_similarity(
                track_embedding, 
                self.embedding_history[existing_id]
            )
            if similarity > best_similarity and similarity > 0.7:
                best_match_id = existing_id
                best_similarity = similarity
                
        # ID 재할당 또는 유지
        if best_match_id and best_match_id != suspicious_track.track_id:
            print(f"🔄 ID 수정: {suspicious_track.track_id} → {best_match_id}")
            suspicious_track.track_id = best_match_id
            
        return suspicious_track
        
    def load_lightweight_reid(self):
        """경량 128-D ReID 모델 로드 (필요시에만)"""
        # MobileNet 기반 < 50MB 경량 모델
        import torchvision.models as models
        reid_model = models.mobilenet_v3_small(pretrained=True)
        reid_model.classifier = torch.nn.Linear(reid_model.classifier[0].in_features, 128)
        return reid_model.eval()
```

**성능 최적화 전략**:

1. **기본 모드**: ByteTrack만 사용 (GPU 사용량 최소)
2. **의심 감지**: 특정 상황에서만 ReID 활성화
3. **메모리 효율**: ReID 모델을 필요시에만 로드
4. **임베딩 캐시**: 최근 N프레임의 임베딩만 GPU 메모리에 보관

**ID 안정성 보장 메커니즘**:

```python
class IDStabilityManager:
    def __init__(self):
        self.id_confidence_buffer = {}   # ID별 신뢰도 버퍼
        self.min_stable_frames = 10      # 안정화 최소 프레임
        self.max_unstable_frames = 5     # 불안정 허용 최대 프레임
        
    def is_id_stable(self, track_id):
        """ID 안정성 판단"""
        if track_id not in self.id_confidence_buffer:
            return False
            
        recent_confidences = self.id_confidence_buffer[track_id][-10:]
        stable_count = sum(1 for conf in recent_confidences if conf > 0.7)
        
        return stable_count >= self.min_stable_frames
        
    def update_id_confidence(self, track_id, confidence):
        """ID 신뢰도 업데이트"""
        if track_id not in self.id_confidence_buffer:
            self.id_confidence_buffer[track_id] = []
            
        self.id_confidence_buffer[track_id].append(confidence)
        
        # 최대 30프레임만 유지
        if len(self.id_confidence_buffer[track_id]) > 30:
            self.id_confidence_buffer[track_id].pop(0)
```

**트래킹 정확도 vs 성능 trade-off**:
- **일반 상황**: ByteTrack만 사용 → **높은 성능**
- **의심 상황**: ReID 추가 활성화 → **높은 정확도**
- **메모리 사용**: 평상시 최소, 필요시에만 증가

### 4. 좌우 분기 로직 (**안정성 대폭 강화**)

**개선된 분기 알고리즘** (EMA + 체류시간 + 히스테리시스):

```python
class StablePersonAssigner:
    def __init__(self):
        self.position_history = {}  # ID별 위치 히스토리
        self.residence_time = {}    # ID별 체류시간
        self.assignment = {}        # ID별 현재 할당 (left/right)
        
    def assign_person(self, track_id, bbox_center_x, frame_width):
        # 1. 위치 히스토리 업데이트 (EMA)
        if track_id not in self.position_history:
            self.position_history[track_id] = []
        
        self.position_history[track_id].append(bbox_center_x)
        if len(self.position_history[track_id]) > 30:  # 최근 30프레임
            self.position_history[track_id].pop(0)
            
        # 2. EMA 계산 (지수 이동 평균)
        ema_position = self.calculate_ema(self.position_history[track_id])
        
        # 3. 체류시간 우선 규칙
        frame_center = frame_width / 2
        current_side = "left" if ema_position < frame_center else "right"
        
        # 4. 히스테리시스 (최소 N프레임 유지)
        if track_id in self.assignment:
            if self.residence_time[track_id] < 15:  # 15프레임 최소 유지
                current_side = self.assignment[track_id]  # 기존 할당 유지
            else:
                # 충분한 체류시간 후 재할당 허용
                margin = frame_width * 0.1  # 10% 여유분
                if abs(ema_position - frame_center) > margin:
                    self.residence_time[track_id] = 0  # 재할당 시 시간 리셋
                    
        self.assignment[track_id] = current_side
        self.residence_time[track_id] = self.residence_time.get(track_id, 0) + 1
        
        return current_side
```

### 5. 영상 합성 및 출력 (**CPU → GPU 완전 전환**)

**Full GPU 합성 파이프라인**:
```python
# 기존 (병목): GPU → CPU → GPU
gpu_frame → cpu_resize() → cpu_tile_compose() → gpu_encode()

# 신규 (제로카피): GPU 내에서 완결  
gpu_frame → cuda_resize() → cuda_tile_compose() → nvenc_encode()
```

**CUDA 합성 구현**:
- **GPU 리사이즈**: `cv2.cuda.resize()` 사용
- **타일 합성**: CUDA kernel 직접 구현 or `cv2.cuda.copyMakeBorder()`
- **NVENC 직접 연결**: 합성된 프레임을 바로 NVENC로

### 🛡️ **타일 합성 에러 처리 정책** (단일 실패점 방지)

**핵심 문제**: 4개 스트림 중 하나라도 실패하면 **전체 타일 합성이 불가능**

**TileCompositionErrorPolicy 아키텍처**:

```python
class TileCompositionErrorPolicy:
    def __init__(self):
        self.failure_strategies = {
            'critical_stream': self.handle_critical_stream_failure,
            'normal_stream': self.handle_normal_stream_failure,
            'multiple_streams': self.handle_multiple_stream_failures
        }
        self.black_frame_cache = {}  # 해상도별 블랙 프레임 캐시
        self.last_good_frames = {}   # 스트림별 마지막 정상 프레임
        
    def handle_stream_failure(self, failed_streams, all_streams, batch_id):
        """스트림 실패 시 처리 정책"""
        failure_count = len(failed_streams)
        total_streams = len(all_streams)
        
        print(f"⚠️ 배치 {batch_id}: {failure_count}/{total_streams} 스트림 실패")
        
        # 1. 실패 유형 분류
        if failure_count >= total_streams * 0.5:  # 50% 이상 실패
            return self.handle_multiple_stream_failures(failed_streams, all_streams, batch_id)
        elif self.is_critical_stream(failed_streams[0]):
            return self.handle_critical_stream_failure(failed_streams[0], all_streams, batch_id)
        else:
            return self.handle_normal_stream_failure(failed_streams[0], all_streams, batch_id)
            
    def handle_critical_stream_failure(self, failed_stream_id, all_streams, batch_id):
        """중요 스트림 실패 시 - 전체 배치 스킵"""
        print(f"❌ 중요 스트림 {failed_stream_id} 실패 - 배치 {batch_id} 스킵")
        
        # 실패 원인 로깅
        failure_reason = self.diagnose_failure(failed_stream_id)
        self.log_failure_analytics(failed_stream_id, failure_reason, batch_id)
        
        # 전체 배치 스킵 처리
        self.increment_skip_counter(batch_id)
        return {
            'action': 'skip_batch',
            'reason': f'Critical stream {failed_stream_id} failure: {failure_reason}',
            'affected_streams': all_streams,
            'recovery_strategy': 'wait_for_next_batch'
        }
        
    def handle_normal_stream_failure(self, failed_stream_id, all_streams, batch_id):
        """일반 스트림 실패 시 - 대체 프레임 사용"""
        print(f"🔄 일반 스트림 {failed_stream_id} 실패 - 대체 프레임 적용")
        
        replacement_strategy = self.select_replacement_strategy(failed_stream_id)
        
        if replacement_strategy == 'last_good_frame':
            replacement_frame = self.get_last_good_frame(failed_stream_id)
        elif replacement_strategy == 'black_frame':
            replacement_frame = self.generate_black_frame(failed_stream_id)
        else:  # 'interpolated_frame'
            replacement_frame = self.interpolate_frame(failed_stream_id)
            
        # 타일 합성 계속 진행
        return {
            'action': 'continue_with_replacement',
            'failed_stream_id': failed_stream_id,
            'replacement_frame': replacement_frame,
            'replacement_type': replacement_strategy,
            'warning_logged': True
        }
        
    def handle_multiple_stream_failures(self, failed_streams, all_streams, batch_id):
        """다중 스트림 실패 시 - 긴급 처리"""
        print(f"🚨 다중 스트림 실패 ({len(failed_streams)}/{len(all_streams)}) - 긴급 처리")
        
        # 시스템 안정성 우선
        if len(failed_streams) >= 3:
            return {
                'action': 'emergency_shutdown',
                'reason': 'Multiple critical failures detected',
                'failed_streams': failed_streams,
                'recovery_strategy': 'restart_pipeline'
            }
        else:
            # 부분 복구 시도
            return self.attempt_partial_recovery(failed_streams, all_streams, batch_id)
            
    def select_replacement_strategy(self, failed_stream_id):
        """대체 전략 선택"""
        # 실패 빈도에 따른 전략
        failure_history = self.get_failure_history(failed_stream_id)
        
        if failure_history['consecutive_failures'] < 3:
            return 'last_good_frame'  # 최근 정상 프레임 사용
        elif failure_history['total_failures'] < 10:
            return 'interpolated_frame'  # 보간 프레임 생성
        else:
            return 'black_frame'  # 블랙 프레임 (최후 수단)
            
    def generate_black_frame(self, stream_id):
        """해상도별 블랙 프레임 생성 (캐시 활용)"""
        stream_resolution = self.get_stream_resolution(stream_id)
        
        cache_key = f"{stream_resolution['width']}x{stream_resolution['height']}"
        if cache_key not in self.black_frame_cache:
            # GPU에서 블랙 프레임 생성
            black_frame = torch.zeros(
                (3, stream_resolution['height'], stream_resolution['width']),
                dtype=torch.uint8,
                device='cuda'
            )
            self.black_frame_cache[cache_key] = black_frame
            
        return self.black_frame_cache[cache_key].clone()
        
    def get_last_good_frame(self, stream_id):
        """마지막 정상 프레임 반환"""
        if stream_id in self.last_good_frames:
            return self.last_good_frames[stream_id].clone()
        else:
            # 정상 프레임이 없으면 블랙 프레임
            return self.generate_black_frame(stream_id)
            
    def update_last_good_frame(self, stream_id, frame):
        """정상 프레임 업데이트"""
        self.last_good_frames[stream_id] = frame.clone()
        
    def diagnose_failure(self, stream_id):
        """실패 원인 진단"""
        possible_causes = []
        
        # GPU 메모리 확인
        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
            possible_causes.append('GPU_MEMORY_PRESSURE')
            
        # 디코더 상태 확인
        decoder_status = self.check_decoder_health(stream_id)
        if not decoder_status['healthy']:
            possible_causes.append(f'DECODER_ERROR: {decoder_status["error"]}')
            
        # 네트워크/파일 I/O 확인
        io_status = self.check_io_health(stream_id)
        if not io_status['healthy']:
            possible_causes.append(f'IO_ERROR: {io_status["error"]}')
            
        return possible_causes if possible_causes else ['UNKNOWN_ERROR']
        
    def is_critical_stream(self, stream_id):
        """중요 스트림 판단 기준"""
        # 예시: stream_0, stream_1은 중요, stream_2, stream_3은 일반
        return stream_id in ['stream_0', 'stream_1']
```

**에러 처리 정책 매트릭스**:

| 실패 상황 | 실패 스트림 수 | 처리 방식 | 타일 합성 여부 |
|-----------|---------------|-----------|----------------|
| **단일 일반 스트림** | 1개 | 대체 프레임 사용 | ✅ 계속 |
| **단일 중요 스트림** | 1개 | 전체 배치 스킵 | ❌ 스킵 |
| **다중 스트림 (2개)** | 2개 | 부분 복구 시도 | 🔄 조건부 |
| **다중 스트림 (3개+)** | 3개+ | 긴급 처리 모드 | 🚨 중단 |

**실패 분석 및 복구 지표**:

```python
class FailureAnalytics:
    def __init__(self):
        self.failure_metrics = {
            'total_batches': 0,
            'failed_batches': 0,
            'skipped_batches': 0,
            'recovered_batches': 0,
            'failure_rate': 0.0
        }
        
    def update_failure_metrics(self, batch_result):
        """실패 지표 업데이트"""
        self.failure_metrics['total_batches'] += 1
        
        if batch_result['action'] == 'skip_batch':
            self.failure_metrics['skipped_batches'] += 1
        elif batch_result['action'] == 'continue_with_replacement':
            self.failure_metrics['recovered_batches'] += 1
        elif batch_result['action'] == 'emergency_shutdown':
            self.failure_metrics['failed_batches'] += 1
            
        # 실패율 계산
        self.failure_metrics['failure_rate'] = (
            (self.failure_metrics['failed_batches'] + self.failure_metrics['skipped_batches']) /
            max(self.failure_metrics['total_batches'], 1)
        )
        
    def get_health_report(self):
        """시스템 건강성 보고"""
        return {
            'overall_health': 'HEALTHY' if self.failure_metrics['failure_rate'] < 0.05 else 'DEGRADED',
            'failure_rate_percent': self.failure_metrics['failure_rate'] * 100,
            'recovery_rate_percent': (
                self.failure_metrics['recovered_batches'] / 
                max(self.failure_metrics['total_batches'], 1)
            ) * 100,
            'recommendations': self.generate_recommendations()
        }
```

**NVENC 동시 세션 한도 검증 및 우회책**:
```python
class NVENCManager:
    def __init__(self):
        self.max_sessions = self.detect_nvenc_limit()
        self.active_sessions = []
        self.encoding_queue = []
        
    def detect_nvenc_limit(self):
        """NVENC 실제 세션 한도 측정"""
        import subprocess
        
        # nvidia-smi로 기본 정보 확인
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=encoder.max_sessions', 
                '--format=csv,noheader'
            ], capture_output=True, text=True, timeout=5)
            theoretical_max = int(result.stdout.strip())
        except:
            theoretical_max = 2  # RTX 5090 기본값
            
        # 실제 동시 인코딩 테스트
        actual_max = 0
        test_sessions = []
        
        for i in range(theoretical_max + 1):
            try:
                # 더미 NVENC 세션 생성
                session = self.create_test_nvenc_session()
                test_sessions.append(session)
                actual_max += 1
            except Exception as e:
                print(f"NVENC 세션 한도: {actual_max}개 (이론값: {theoretical_max})")
                break
                
        # 테스트 세션 정리
        for session in test_sessions:
            session.close()
            
        return min(actual_max, 4)  # 안전 마진 고려
        
    def request_encoding_slot(self, stream_id, priority='normal'):
        """인코딩 슬롯 요청 (대기열 관리)"""
        if len(self.active_sessions) < self.max_sessions:
            session = self.create_nvenc_session(stream_id)
            self.active_sessions.append(session)
            return session
        else:
            # 대기열에 추가 (우선순위 기반)
            self.encoding_queue.append({
                'stream_id': stream_id,
                'priority': priority,
                'timestamp': time.time()
            })
            return None
            
    def apply_bitrate_adjustment(self, session, quality_level):
        """비트레이트 동적 조정 (세션 수에 따라)"""
        base_bitrate = 8_000_000  # 8Mbps
        
        if len(self.active_sessions) >= 3:
            # 3개 이상 시 CBR → VBR 전환, 비트레이트 감소
            adjusted_bitrate = int(base_bitrate * 0.7)
            session.set_bitrate_mode('VBR')
            session.set_target_bitrate(adjusted_bitrate)
        else:
            # 여유 있을 때는 CBR 고품질
            session.set_bitrate_mode('CBR') 
            session.set_target_bitrate(base_bitrate)
```

**오디오 동기화 드리프트 대응** (VFR 입력 지원):
```python
class AudioSyncManager:
    def __init__(self):
        self.original_audio = None
        self.video_pts_history = []
        self.audio_pts_history = []
        self.sync_drift_threshold = 40  # 40ms 이상 드리프트 시 보정
        
    def extract_original_audio(self, video_path):
        """원본 오디오 스트림 추출 및 분석"""
        container = av.open(video_path)
        
        if container.streams.audio:
            audio_stream = container.streams.audio[0]
            self.original_audio = {
                'stream': audio_stream,
                'sample_rate': audio_stream.sample_rate,
                'channels': audio_stream.channels,
                'duration': audio_stream.duration,
                'time_base': audio_stream.time_base
            }
            
            # VFR 비디오 감지
            video_stream = container.streams.video[0]
            if hasattr(video_stream, 'average_rate') and hasattr(video_stream, 'base_rate'):
                if video_stream.average_rate != video_stream.base_rate:
                    print("VFR 비디오 감지 - 고급 동기화 모드 활성화")
                    self.vfr_mode = True
                    
    def track_av_sync(self, video_pts, audio_pts):
        """A/V 동기화 드리프트 추적"""
        self.video_pts_history.append(video_pts)
        self.audio_pts_history.append(audio_pts)
        
        # 최근 100프레임만 유지
        if len(self.video_pts_history) > 100:
            self.video_pts_history.pop(0)
            self.audio_pts_history.pop(0)
            
        # 드리프트 계산
        if len(self.video_pts_history) >= 10:
            video_duration = self.video_pts_history[-1] - self.video_pts_history[0]
            audio_duration = self.audio_pts_history[-1] - self.audio_pts_history[0]
            drift = abs(video_duration - audio_duration) * 1000  # ms
            
            if drift > self.sync_drift_threshold:
                return self.suggest_sync_correction(drift, video_duration, audio_duration)
                
        return None
        
    def suggest_sync_correction(self, drift_ms, video_dur, audio_dur):
        """동기화 보정 방법 제안"""
        if video_dur > audio_dur:
            # 비디오가 더 길음 - 오디오 늘리기 또는 비디오 자르기
            return {
                'type': 'audio_stretch',
                'factor': video_dur / audio_dur,
                'method': 'time_stretch' if drift_ms < 100 else 'frame_drop'
            }
        else:
            # 오디오가 더 길음 - 오디오 자르기 또는 비디오 늘리기
            return {
                'type': 'audio_trim',
                'trim_ms': drift_ms,
                'method': 'precise_cut'
            }
            
    def apply_sync_correction(self, correction):
        """PTS 기반 정밀 동기화 적용"""
        if correction['type'] == 'audio_stretch':
            if correction['method'] == 'time_stretch' and correction['factor'] < 1.05:
                # 5% 이내는 타임 스트레치 (품질 유지)
                return self.apply_time_stretch(correction['factor'])
            else:
                # 큰 차이는 프레임 드롭/복제
                return self.apply_frame_adjustment(correction['factor'])
                
        elif correction['type'] == 'audio_trim':
            return self.apply_precise_audio_trim(correction['trim_ms'])
```

---

## 📊 성능 목표 및 병목 해결 (VRAM/RAM 분리)

### 🎯 Full GPU 파이프라인 성능 목표 (일관성 확보)

#### 📊 **처리량 비교표** (혼동 방지)

| 처리 방식 | 동시 스트림 | 처리량(개/시간) | 4개 영상 총시간 | 단일 영상 시간 | VRAM 사용량 |
|----------|-------------|----------------|----------------|----------------|-------------|
| **기존 CPU** | 1개 | 2.6개/시간 | 92분 (순차) | 23분/개 | ~8GB |
| **보수적 GPU** | 2개 | 12개/시간 | 20분 (병렬) | 10분/개 | **12-14GB** |
| **목표 GPU** | 3개 | 15개/시간 | 16분 (병렬) | 5.3분/개 | **18-21GB** |
| **최적화 GPU** | 4개 | 16개/시간 | 15분 (병렬) | 3.75분/개 | **24-28GB** |

#### 🚀 **성능 향상 지표** (명확한 단위 구분)

| 지표 유형 | 기존 → 최적화 | 향상 배수 | 비고 |
|----------|---------------|-----------|------|
| **처리량** | 2.6 → 16개/시간 | **6.2배** | 시간당 처리 가능 영상 수 |
| **단일 영상** | 23분 → 3.75분 | **6.1배** | 개별 영상 처리 시간 |
| **4개 동시** | 92분 → 15분 | **6.1배** | 병렬 처리 총 시간 |

### ⚡ 주요 병목 완전 해결

| 병목 지점 | 기존 문제 | Full GPU 해결책 | 성능 향상 |
|-----------|----------|-----------------|-----------|
| **디코딩** | CPU 디코딩 | PyAV NVDEC | **15-20배** |
| **추론** | PyTorch FP32 | TensorRT FP16 | **2-3배** |
| **메모리 복사** | CPU↔GPU 전송 | Zero-Copy 파이프라인 | **지연시간 80% 감소** |
| **합성** | CPU 리사이즈/타일링 | CUDA 합성 | **10-15배** |
| **인코딩** | FFmpeg CPU | NVENC H.264 | **8-12배** |

### 📈 상세 성능 측정 지표

**NVIDIA 하드웨어 활용률**:
- **NVDEC 활용률**: 85% 이상 목표 (4개 엔진 개별 추적)
- **NVENC 활용률**: 75% 이상 목표 (2개 엔진 개별 추적)
- **GPU Compute**: 92% 이상 목표 (SM 단위 측정)
- **VRAM 효율성**: 32GB 중 **87.5% 이하** 사용
- **PCIe 대역폭**: < 20% 사용 (제로카피 효과)

**파이프라인 세부 지표**:
- **Per-Stage Latency**: 디코드(5ms), 추론(15ms), 합성(3ms), 인코딩(7ms)
- **Per-Stream FPS**: 스트림당 실시간 FPS (30+ 목표)
- **Frame Queue Depth**: 대기 프레임 < 5개
- **ID Switch Rate**: 분당 < 0.05회 (히스테리시스 효과)
- **Batch Efficiency**: TensorRT 배치 활용률 > 90%

**시스템 리소스 지표**:
- **System RAM**: 48GB 중 **44% 이하** 사용
- **CPU 활용률**: 32스레드 중 25% 이하 (제어 로직만)
- **디스크 I/O**: 순차 읽기 위주, 랜덤 I/O 최소화

---

## 🚀 단계별 개발 계획

### Phase 1: PyAV NVDEC + TensorRT 파이프라인 (4일)

| 단계 | 작업 내용 | 완료 조건 | 시간 |
|------|-----------|-----------|------|
| **1.1** | PyAV NVDEC 환경 구축 | hwaccel=cuda 디코딩 성공 | 1일 |
| **1.2** | 모델 후보군 TensorRT 변환 | YOLOv8n/s, SCRFD 엔진 생성 | 1일 |
| **1.3** | 단일 스트림 GPU 파이프라인 | NVDEC→TensorRT→NVENC 연결 | 1.5일 |
| **1.4** | ByteTrack 통합 + 성능 측정 | 단일 영상 처리 + 벤치마킹 | 0.5일 |

**완료 기준**: 단일 영상을 **5분 이내** 처리 (기존 23분 → 78% 단축)

### Phase 2: 멀티 스트림 + GPU 합성 (4일)

| 단계 | 작업 내용 | 완료 조건 | 시간 |
|------|-----------|-----------|------|
| **2.1** | 4x CUDA Stream 구현 | 독립적인 4개 스트림 동작 | 1.5일 |
| **2.2** | CUDA 합성 파이프라인 | GPU 리사이즈+타일링 구현 | 1.5일 |
| **2.3** | 좌우 분기 안정화 | EMA+히스테리시스 적용 | 0.5일 |
| **2.4** | VRAM 관리 시스템 | 메모리 모니터링 + 백프레셔 | 0.5일 |

**완료 기준**: 3개 영상을 **12분 이내** 동시 처리

### Phase 3: 고급 최적화 + 모니터링 (2일)

| 단계 | 작업 내용 | 완료 조건 | 시간 |
|------|-----------|-----------|------|
| **3.1** | 오디오 처리 통합 | PyAV 오디오 스트림 보존 | 0.5일 |
| **3.2** | 상세 모니터링 시스템 | per-stage latency 추적 | 0.5일 |
| **3.3** | 장애 복구 시스템 | OOM/크래시 자동 재시작 | 0.5일 |
| **3.4** | 최종 성능 테스트 | 4개 스트림 안정성 검증 | 0.5일 |

**최종 목표**: 4개 영상을 **15분 이내** 처리 (**8-10배 처리량 향상**)

> **총 예상 기간: 10일**

---

## 🗂️ 프로젝트 구조 (Full GPU 파이프라인)

```
dual_face_tracker/
├── core/                          # 핵심 모듈
│   ├── __init__.py
│   ├── pyav_decoder.py           # PyAV NVDEC wrapper
│   ├── tensorrt_detector.py       # TensorRT 다중 모델 엔진
│   ├── cuda_tracker.py            # ByteTrack/StrongSORT
│   ├── cuda_compositor.py         # GPU 합성 파이프라인
│   ├── nvenc_encoder.py           # NVENC wrapper
│   └── stream_manager.py          # 멀티 스트림 관리
├── utils/                         # 유틸리티
│   ├── __init__.py
│   ├── logger.py                  # 통합 로깅
│   ├── hw_monitor.py              # NVDEC/NVENC/VRAM 모니터링
│   ├── memory_manager.py          # VRAM/RAM 관리
│   ├── performance_tracker.py     # per-stage latency 추적
│   └── audio_processor.py         # 오디오 보존/동기화
├── models/                        # 모델 관련
│   ├── tensorrt_engines/          # TensorRT 엔진들
│   │   ├── yolov8n_face.trt      # 경량 모델
│   │   ├── yolov8s_face.trt      # 중간 모델  
│   │   └── scrfd_2.5g.trt        # 대안 모델
│   └── model_converter.py         # 모델 → TensorRT 변환기
├── config/                        # 설정 파일
│   ├── hardware_config.yaml       # NVDEC/NVENC 설정
│   ├── model_config.yaml          # 모델별 파라미터
│   ├── stream_config.yaml         # 스트림별 설정
│   └── performance_config.yaml    # 성능 임계값
├── monitoring/                    # 모니터링
│   ├── grafana_dashboard.json     # 성능 대시보드
│   ├── prometheus_config.yml      # 메트릭 수집
│   └── alerting_rules.yml         # 장애 알림
├── recovery/                      # 장애 복구
│   ├── checkpoint_manager.py      # 체크포인트 관리
│   ├── stream_recovery.py         # 스트림 재시작
│   └── health_checker.py          # 헬스체크
├── tests/                         # 테스트
│   ├── test_pyav_nvdec.py        # NVDEC 테스트
│   ├── test_tensorrt_models.py   # 모델별 성능 테스트
│   ├── test_cuda_composition.py  # GPU 합성 테스트
│   └── test_full_pipeline.py     # 통합 테스트
├── benchmark/                     # 벤치마킹
│   ├── model_comparison.py        # 모델 성능 비교
│   ├── hardware_profiling.py      # 하드웨어 프로파일링
│   └── scalability_test.py        # 확장성 테스트
├── logs/                          # 로그
│   └── full_gpu_tracker.log       # 통합 로그
├── main.py                        # 메인 실행 파일
├── requirements.txt               # 의존성
└── README.md                      # 가이드
```

---

## 🔧 환경 설정 및 의존성 (PyAV 중심)

### 📦 기존 패키지 (재사용)

| 패키지명 | 버전 | 용도 | 상태 |
|----------|------|------|------|
| **torch** | 2.7.1+cu128 | TensorRT 백엔드 | ✅ 활용 |
| **numpy** | 1.26.4 | 수치 연산 | ✅ 활용 |
| **opencv** | 4.13.0-dev | CUDA 합성 보조 | ✅ 제한적 활용 |
| **psutil** | 6.1.1 | 시스템 모니터링 | ✅ 활용 |

### 📥 Full GPU 파이프라인 전용 패키지

```bash
# 가상환경 활성화
source /home/hamtoto/work/python-study/Face-Tracking-App/.venv/bin/activate

# PyNvCodec (1순위 - 진짜 제로카피)
pip install pynvcodec               # PyNvCodec (VPF) 
pip install av                     # PyAV (백업용)

# TensorRT 최적화
pip install tensorrt               # TensorRT Python API
pip install torch2trt              # PyTorch → TensorRT

# 다중 모델 지원
pip install ultralytics            # YOLOv8 variants
pip install insightface            # SCRFD models

# 트래커 (선택적)
pip install deep-sort-realtime     # ByteTrack
pip install motpy                  # 대안 트래커

# 모니터링 & 복구
pip install py3nvml                # NVIDIA GPU 모니터링
pip install prometheus-client      # 메트릭 수집
pip install watchdog               # 파일 시스템 모니터링

# 고성능 I/O
pip install aiofiles               # 비동기 I/O
pip install uvloop                 # 고성능 이벤트 루프
```

### 🔍 **PyNvCodec 환경 검증 스크립트** (리스크 조기 차단)

**핵심 혁신**: 개발 착수 **전**에 실행하여 PyNvCodec 의존성 문제를 완전 차단

**validate_environment.sh** (필수 실행 스크립트):

```bash
#!/bin/bash
# validate_environment.sh - PyNvCodec 환경 완전 검증
set -e

echo "🔍 PyNvCodec 환경 검증 시작..."
echo "=================================================="

# 1단계: 시스템 기본 요구사항 확인
echo "1️⃣ 시스템 기본 요구사항 확인"
echo "----------------------------"

# NVIDIA 드라이버 확인
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "✓ NVIDIA 드라이버: $DRIVER_VERSION"

if [[ $(echo "$DRIVER_VERSION >= 525.0" | bc -l) -eq 0 ]]; then
    echo "❌ 드라이버 버전 부족 (최소 525.0 필요)"
    exit 1
fi

# CUDA 버전 확인
CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
echo "✓ CUDA 버전: $CUDA_VERSION"

# GPU 모델 확인
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo "✓ GPU 모델: $GPU_NAME"

# 2단계: PyNvCodec 라이브러리 검증
echo -e "\n2️⃣ PyNvCodec 라이브러리 검증"
echo "----------------------------"

# PyNvCodec 임포트 테스트
python3 -c "
try:
    import PyNvCodec as nvc
    print('✓ PyNvCodec 임포트 성공')
    print(f'  버전: {nvc.__version__ if hasattr(nvc, \"__version__\") else \"Unknown\"}')
except ImportError as e:
    print(f'❌ PyNvCodec 임포트 실패: {e}')
    exit(1)
"

# 3단계: 실제 디코딩 테스트
echo -e "\n3️⃣ 실제 NVDEC 디코딩 테스트"
echo "----------------------------"

# 테스트 비디오 생성 (필요시)
if [[ ! -f "test_sample.mp4" ]]; then
    echo "📹 테스트 비디오 생성 중..."
    ffmpeg -f lavfi -i testsrc=duration=5:size=1920x1080:rate=30 -c:v libx264 -y test_sample.mp4 &>/dev/null
fi

# PyNvCodec 디코딩 테스트
python3 << 'EOF'
import PyNvCodec as nvc
import sys

try:
    # NVDEC 디코더 생성 테스트
    decoder = nvc.PyDecodeHW("test_sample.mp4", nvc.PixelFormat.NV12, 0)
    print(f"✓ NVDEC 디코더 생성 성공")
    print(f"  해상도: {decoder.Width()}x{decoder.Height()}")
    
    # 색공간 변환기 테스트
    converter = nvc.PySurfaceConverter(
        decoder.Width(), decoder.Height(),
        nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, 0
    )
    print("✓ 색공간 변환기 생성 성공")
    
    # 실제 프레임 디코딩 테스트 (5프레임만)
    for i in range(5):
        surface = decoder.DecodeSurface()
        if surface:
            rgb_surface = converter.Execute(surface)
            if rgb_surface:
                print(f"✓ 프레임 {i+1} 디코딩/변환 성공")
            else:
                print(f"❌ 프레임 {i+1} 변환 실패")
                sys.exit(1)
        else:
            print(f"✓ 디코딩 완료 ({i}프레임 처리)")
            break
            
    print("🎉 NVDEC 디코딩 테스트 완전 성공!")
    
except Exception as e:
    print(f"❌ NVDEC 디코딩 테스트 실패: {e}")
    sys.exit(1)
EOF

# 4단계: TensorRT 연동 테스트
echo -e "\n4️⃣ TensorRT 연동 테스트"
echo "----------------------------"

python3 -c "
import tensorrt as trt
import torch

try:
    print(f'✓ TensorRT 버전: {trt.__version__}')
    print(f'✓ PyTorch 버전: {torch.__version__}')
    print(f'✓ CUDA 사용 가능: {torch.cuda.is_available()}')
    print(f'✓ GPU 개수: {torch.cuda.device_count()}')
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f'✓ GPU 0: {device_name}')
    
except Exception as e:
    print(f'❌ TensorRT/PyTorch 테스트 실패: {e}')
    exit(1)
"

# 5단계: NVENC 인코딩 테스트 (선택적)
echo -e "\n5️⃣ NVENC 인코딩 테스트"
echo "----------------------------"

# FFmpeg NVENC 테스트
ffmpeg -f lavfi -i testsrc=duration=2:size=640x480:rate=30 \
    -c:v h264_nvenc -preset fast -b:v 2M -f null - &>/dev/null

if [[ $? -eq 0 ]]; then
    echo "✓ NVENC H.264 인코딩 테스트 성공"
else
    echo "⚠️ NVENC 인코딩 테스트 실패 (비치명적)"
fi

# 6단계: 최종 검증 결과 저장
echo -e "\n6️⃣ 검증 결과 저장"
echo "----------------------------"

cat > environment_validation_report.json << EOF
{
    "validation_timestamp": "$(date -Iseconds)",
    "system_info": {
        "gpu_name": "$GPU_NAME",
        "driver_version": "$DRIVER_VERSION",
        "cuda_version": "$CUDA_VERSION"
    },
    "validation_results": {
        "pynvcodec_import": "PASSED",
        "nvdec_decoding": "PASSED",
        "tensorrt_integration": "PASSED",
        "nvenc_encoding": "$([ $? -eq 0 ] && echo 'PASSED' || echo 'WARNING')"
    },
    "environment_status": "READY_FOR_DEVELOPMENT"
}
EOF

echo "✅ 환경 검증 완료!"
echo "📄 상세 결과: environment_validation_report.json"
echo -e "\n🚀 PyNvCodec 개발 환경이 완벽히 준비되었습니다!"

# 임시 파일 정리
rm -f test_sample.mp4

echo "=================================================="
```

**자동 실행 통합**:

```python
# main.py에서 자동 실행
import subprocess
import sys
import os

def validate_environment_on_startup():
    """애플리케이션 시작 시 환경 검증"""
    print("🔧 시작 전 환경 검증 실행...")
    
    if not os.path.exists("environment_validation_report.json"):
        print("⚠️ 환경 검증 기록 없음 - 검증 실행 중...")
        
        try:
            result = subprocess.run(
                ["bash", "validate_environment.sh"],
                capture_output=True,
                text=True,
                timeout=120  # 2분 제한
            )
            
            if result.returncode != 0:
                print("❌ 환경 검증 실패:")
                print(result.stderr)
                sys.exit(1)
                
        except subprocess.TimeoutExpired:
            print("❌ 환경 검증 시간 초과")
            sys.exit(1)
            
    else:
        print("✅ 환경 검증 기록 확인 - 검증 통과")

# 메인 애플리케이션 시작 전 실행
if __name__ == "__main__":
    validate_environment_on_startup()
    # 메인 애플리케이션 시작...
```

**검증 스크립트 장점**:

1. **조기 차단**: 개발 시작 전에 모든 의존성 확인
2. **구체적 테스트**: 실제 디코딩/인코딩 동작 검증
3. **자동화**: 스크립트 한 번 실행으로 완전 검증
4. **결과 저장**: JSON 형태로 검증 결과 기록
5. **재현성**: 동일한 환경에서 반복 가능한 검증

### 🐳 **환경 고정 및 재현성 보장**

**Docker 베이스 환경** (완전 재현 가능):
```dockerfile
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# 정확한 버전 매트릭스 (검증된 조합)
ARG CUDA_VERSION=12.8
ARG TENSORRT_VERSION=10.7.0
ARG FFMPEG_VERSION=6.1.1
ARG PYAV_VERSION=12.3.0

ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# 필수 시스템 의존성 (버전 고정)
RUN apt-get update && apt-get install -y \
    python3.10=3.10.12-1~22.04.1 \
    python3.10-dev=3.10.12-1~22.04.1 \
    ffmpeg=${FFMPEG_VERSION}-* \
    && apt-get clean

# Python 패키지 (정확한 버전)
COPY requirements_frozen.txt /tmp/
RUN pip install -r /tmp/requirements_frozen.txt

# TensorRT 설치 (CUDA 버전과 매칭)
RUN pip install tensorrt==${TENSORRT_VERSION}

# PyAV NVDEC 지원 확인
RUN python3 -c "import av; print('NVDEC codecs:', [c for c in av.codec.codecs_available if 'nvdec' in c])"
```

### 🎨 **버전 매트릭스 현실화** (검증된 조합)

#### 🟢 **Production 안정 조합** (검증 완료)

```yaml
stable_production_2024:
  cuda: "12.4.1"                    # LTS 버전
  tensorrt: "10.0.1.6"              # 안정 릴리즈
  torch: "2.4.1+cu124"              # 안정 릴리즈
  pyav: "12.0.0"                    # NVDEC 안정 지원
  pynvcodec: "12.2.0"               # VPF 지원
  status: "PRODUCTION_READY"
  tested_gpus: ["RTX 4090", "RTX 4080", "A6000"]
  
stable_production_2025:
  cuda: "12.6.0"                    # 최신 안정
  tensorrt: "10.5.0"                # 학상 지원 개선
  torch: "2.5.1+cu126"              # 안정 릴리즈
  pyav: "12.1.0"                    # 비디오 개선
  pynvcodec: "12.3.0"               # VPF 개선
  status: "PRODUCTION_READY"
  tested_gpus: ["RTX 4090", "RTX 5090"]
```

#### 🔶 **Experimental 최신 조합** (개발용)

```yaml  
cutting_edge_2025:
  cuda: "12.8.0"                    # 최신 릴리즈
  tensorrt: "10.7.0"                # 최신 기능
  torch: "2.7.1+cu128"              # Nightly/RC 버전
  pyav: "12.3.0"                    # 최신 기능
  pynvcodec: "12.4.0"               # 최신 VPF
  status: "EXPERIMENTAL"
  tested_gpus: ["RTX 5090"]
  caution: "RTX 5090 전용, 안정성 미보장"
```

#### 📄 **requirements_production.txt** (안정 조합)

```txt
# 핵심 라이브러리 (2025 안정 조합)
torch==2.5.1+cu126
torchvision==0.20.1+cu126  
numpy==1.26.4
opencv-python==4.10.0.84

# 제로카피 디코더 (1순위: PyNvCodec)
pynvcodec==12.3.0
av==12.1.0                     # 백업

# TensorRT 최적화  
tensorrt==10.5.0
torch2trt==0.4.0

# 얼굴 검출
ultralytics==8.2.0
insightface==0.7.3

# 트래커 & 모니터링
deep-sort-realtime==1.3.2
py3nvml==0.2.8
psutil==6.1.1

# 고성능 I/O
cupy-cuda12x==12.3.0           # DLPack 지원
aiofiles==24.1.0
```

#### ⚙️ **하드웨어 호환성 매트릭스**

| GPU 모델 | VRAM | NVDEC | NVENC | 추천 조합 | 동시 스트림 |
|----------|------|-------|-------|------------|---------------|
| **RTX 4090** | 24GB | 2-3개 | 2개 | Production 2025 | **2-3개** |
| **RTX 5090** | 32GB | 4-6개 | 2-3개 | Experimental 2025 | **3-4개** |
| **RTX 4080** | 16GB | 2개 | 1개 | Production 2024 | **1-2개** |
| **A6000** | 48GB | 3-4개 | 1개 | Production 2024 | **3-4개** |

#### 🔧 **자동 환경 검증 스크립트**

```bash
#!/bin/bash
# validate_environment.sh - 자동화된 호환성 검증

echo "🔍 하드웨어 호확성 검증 시작..."

# 1. GPU 모델 감지
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo "GPU: $GPU_NAME"

# 2. 동시 세션 한도 측정 (HardwareProber 사용)
python3 -c "
from hardware_prober import HardwareProber
prober = HardwareProber()
print(f'확인된 세션: NVDEC={prober.gpu_info[\"nvdec_max_sessions\"]}, NVENC={prober.gpu_info[\"nvenc_max_sessions\"]}')
print(f'최적 동시 스트림: {prober.optimal_streams}개')
"

# 3. 조합 호환성 확인
if [[ "$GPU_NAME" == *"RTX 5090"* ]]; then
    echo "✅ RTX 5090 감지: Experimental 2025 조합 사용"
    export REQUIREMENTS_FILE="requirements_experimental.txt"
elif [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
    echo "✅ RTX 4090 감지: Production 2025 조합 사용"
    export REQUIREMENTS_FILE="requirements_production.txt"
else
    echo "⚠️ 미지원 GPU: Production 2024 조합 사용"
    export REQUIREMENTS_FILE="requirements_stable.txt"
fi

echo "✅ 환경 검증 완료: $REQUIREMENTS_FILE 사용 바랍니다"
```

**환경 재현 스크립트**:
```bash
#!/bin/bash
# setup_environment.sh - 완전 자동화된 환경 구성

set -e

echo "🚀 Dual-Face Tracker 환경 설정 시작"

# 1. 시스템 요구사항 검증
echo "1️⃣  시스템 요구사항 검증 중..."
./scripts/validate_hardware.sh

# 2. NVIDIA 드라이버 확인
echo "2️⃣  NVIDIA 드라이버 확인 중..."
if ! nvidia-smi | grep -q "565.57"; then
    echo "❌ NVIDIA 드라이버 업데이트 필요 (565.57.01+)"
    exit 1
fi

# 3. Docker 환경 구축
echo "3️⃣  Docker 환경 구축 중..."
docker build -t dual-face-tracker:latest \
    --build-arg CUDA_VERSION=12.8 \
    --build-arg TENSORRT_VERSION=10.7.0 \
    --build-arg FFMPEG_VERSION=6.1.1 \
    -f Dockerfile .

# 4. 기능 검증 테스트
echo "4️⃣  기능 검증 테스트 중..."
docker run --rm --gpus all dual-face-tracker:latest \
    python3 /app/tests/test_environment.py

# 5. 성능 벤치마크
echo "5️⃣  성능 벤치마크 실행 중..."
docker run --rm --gpus all -v $(pwd)/benchmark:/benchmark \
    dual-face-tracker:latest python3 /app/benchmark/system_benchmark.py

echo "✅ 환경 설정 완료!"
echo "📊 벤치마크 결과: benchmark/results.json"
```

### ⚠️ **버전 호환성 매트릭스**

| 구성요소 | 검증된 버전 | 호환 범위 | 알려진 이슈 |
|----------|-------------|-----------|-------------|
| **CUDA** | 12.8 | 12.6-13.0 | 12.5 이하는 RTX 5090 미지원 |
| **TensorRT** | 10.7.0 | 10.5-11.0 | 10.4 이하는 CUDA 12.8 비호환 |
| **PyAV** | 12.3.0 | 12.0-13.0 | 11.x는 NVDEC hwaccel 불안정 |
| **FFmpeg** | 6.1.1 | 6.0-7.0 | 5.x는 NVDEC 성능 저하 |
| **PyTorch** | 2.7.1+cu128 | 2.5-3.0 | 2.4 이하는 TensorRT 변환 오류 |

**알려진 버전 충돌**:
```python
# 피해야 할 조합
INCOMPATIBLE_COMBINATIONS = [
    ("tensorrt==8.x", "cuda==12.8"),     # TensorRT 8.x는 CUDA 12.8 미지원  
    ("av==11.x", "hwaccel=cuda"),        # PyAV 11.x NVDEC 불안정
    ("torch==2.3.x", "tensorrt==10.x"),  # PyTorch 2.3.x TensorRT 호환성 이슈
]
```

---

## 📊 성능 모니터링 및 장애 복구 (구체화)

### 🔍 실시간 하드웨어 모니터링

```python
class HardwareMonitor:
    def track_nvdec_usage(self):
        """NVDEC 4개 엔진별 활용률 추적"""
        return {
            'nvdec_0': self.get_decoder_utilization(0),
            'nvdec_1': self.get_decoder_utilization(1), 
            'nvdec_2': self.get_decoder_utilization(2),
            'nvdec_3': self.get_decoder_utilization(3),
        }
        
    def track_nvenc_usage(self):
        """NVENC 2개 엔진별 활용률 추적"""
        return {
            'nvenc_0': self.get_encoder_utilization(0),
            'nvenc_1': self.get_encoder_utilization(1),
        }
        
    def track_per_stage_latency(self):
        """파이프라인 단계별 지연시간"""
        return {
            'decode_latency': self.measure_decode_time(),      # 목표: <5ms
            'inference_latency': self.measure_inference_time(), # 목표: <15ms
            'compose_latency': self.measure_compose_time(),     # 목표: <3ms
            'encode_latency': self.measure_encode_time(),       # 목표: <7ms
        }
```

### 📈 파이프라인 성능 추적

```python
class PerformanceTracker:
    def __init__(self):
        self.stream_fps = {}        # 스트림별 실시간 FPS
        self.queue_depth = {}       # 프레임 큐 깊이
        self.id_switch_count = {}   # ID 스위치 발생률
        self.batch_efficiency = 0   # 배치 활용률
        
    def track_stream_performance(self, stream_id):
        return {
            'fps': self.calculate_fps(stream_id),
            'queue_depth': len(self.frame_queues[stream_id]),
            'processing_latency': self.get_end_to_end_latency(stream_id),
            'id_switches_per_minute': self.calculate_id_switch_rate(stream_id),
        }
        
    def detect_bottlenecks(self):
        """자동 병목 감지"""
        if self.queue_depth > 10:
            return "processing_bottleneck"
        elif self.get_vram_usage() > 0.9:
            return "memory_bottleneck" 
        elif self.get_nvdec_utilization() < 0.5:
            return "decode_underutilization"
```

### 🚨 장애 복구 시스템 (구체적 시나리오)

```python
class StreamRecoveryManager:
    def handle_oom_error(self, stream_id):
        """GPU 메모리 부족 시 복구"""
        # 1. 즉시 대응
        self.pause_stream(stream_id)
        torch.cuda.empty_cache()
        
        # 2. 배치 크기 감소
        self.reduce_batch_size(stream_id, factor=0.7)
        
        # 3. 체크포인트에서 재시작
        last_checkpoint = self.get_last_checkpoint(stream_id)
        self.restart_stream_from_checkpoint(stream_id, last_checkpoint)
        
        # 4. 실패 시 스트림 비활성화
        if self.retry_count[stream_id] > 3:
            self.disable_stream(stream_id)
            self.alert_admin(f"Stream {stream_id} disabled due to repeated OOM")
    
    def handle_nvdec_failure(self, stream_id):
        """NVDEC 하드웨어 실패 시 fallback"""
        # 1. CPU 디코딩으로 임시 전환
        self.enable_cpu_fallback(stream_id)
        
        # 2. 다른 NVDEC 엔진으로 재할당 시도
        available_decoder = self.find_available_nvdec_engine()
        if available_decoder:
            self.reassign_decoder(stream_id, available_decoder)
            self.disable_cpu_fallback(stream_id)
            
    def health_check_pipeline(self):
        """전체 파이프라인 헬스체크"""
        checks = {
            'nvdec_responsive': self.check_nvdec_health(),
            'tensorrt_engines': self.check_tensorrt_health(),
            'nvenc_responsive': self.check_nvenc_health(),
            'vram_available': self.get_vram_usage() < 0.9,
            'streams_active': len(self.active_streams) > 0,
        }
        
        if not all(checks.values()):
            self.trigger_recovery_procedure(checks)
            
    def handle_conditional_reid_failure(self, stream_id):
        """조건부 ReID 실패 시 복구"""
        print(f"🔄 조건부 ReID 실패: 스트림 {stream_id}")
        
        # 1. ReID 모델 재로드 시도
        try:
            self.conditional_reid.reload_reid_model()
            print(f"✅ ReID 모델 재로드 성공: 스트림 {stream_id}")
        except Exception as e:
            print(f"❌ ReID 모델 재로드 실패: {e}")
            # 2. ByteTrack 전용 모드로 전환
            self.conditional_reid.disable_reid_for_stream(stream_id)
            print(f"🔄 스트림 {stream_id} ByteTrack 전용 모드 전환")
            
    def handle_config_manager_failure(self):
        """설정 관리자 실패 시 복구"""
        print("🔧 설정 관리자 실패 감지")
        
        # 1. 설정 파일 유효성 재검증
        config_status = self.config_manager.validate_all_configs()
        
        if not config_status['manual_config_valid']:
            print("⚠️ 수동 설정 오류 - 자동 프로빙으로 전환")
            self.config_manager.fallback_to_auto_probing()
            
        if not config_status['auto_config_valid']:
            print("⚠️ 자동 설정 오류 - 기본값으로 전환")
            self.config_manager.fallback_to_defaults()
            
    def handle_environment_validation_failure(self):
        """환경 검증 실패 시 복구"""
        print("🔍 환경 검증 실패 - 자동 복구 시도")
        
        # 1. PyNvCodec 상태 재확인
        pynvcodec_status = self.check_pynvcodec_health()
        if not pynvcodec_status['available']:
            print("❌ PyNvCodec 사용 불가 - PyAV 백업으로 전환")
            self.switch_to_pyav_fallback()
            
        # 2. TensorRT 엔진 상태 확인
        tensorrt_status = self.check_tensorrt_engines()
        if not tensorrt_status['all_engines_loaded']:
            print("🔄 TensorRT 엔진 재로드 중...")
            self.reload_tensorrt_engines()
            
        # 3. 환경 검증 재실행
        try:
            subprocess.run(["bash", "validate_environment.sh"], check=True, timeout=60)
            print("✅ 환경 검증 복구 성공")
        except subprocess.CalledProcessError:
            print("❌ 환경 검증 복구 실패 - 수동 개입 필요")
            self.notify_admin_intervention_required()
            
    def handle_tile_composition_system_failure(self, failure_analytics):
        """타일 합성 시스템 전체 실패 시 복구"""
        print("🛡️ 타일 합성 시스템 실패 - 시스템 레벨 복구")
        
        # 실패율이 50% 이상이면 시스템 재시작
        if failure_analytics.failure_metrics['failure_rate'] > 0.5:
            print("🚨 실패율 50% 초과 - 파이프라인 재시작")
            return self.restart_entire_pipeline()
            
        # 실패율이 20-50%면 부분 복구
        elif failure_analytics.failure_metrics['failure_rate'] > 0.2:
            print("⚠️ 실패율 20% 초과 - 부분 복구 모드")
            return self.enable_degraded_mode()
            
        # 실패율이 20% 이하면 개별 스트림 복구
        else:
            print("🔄 개별 스트림 복구 모드")
            return self.recover_individual_streams()
            
    def integrated_recovery_procedure(self, failure_type, context):
        """통합 복구 프로시저"""
        recovery_strategy = {
            'conditional_reid_failure': self.handle_conditional_reid_failure,
            'config_manager_failure': self.handle_config_manager_failure,
            'environment_validation_failure': self.handle_environment_validation_failure,
            'tile_composition_failure': self.handle_tile_composition_system_failure,
            'oom_error': self.handle_oom_error,
            'nvdec_failure': self.handle_nvdec_failure
        }
        
        if failure_type in recovery_strategy:
            print(f"🔧 통합 복구 시작: {failure_type}")
            recovery_strategy[failure_type](context)
        else:
            print(f"❌ 알 수 없는 실패 유형: {failure_type}")
            self.handle_unknown_failure(failure_type, context)
```

### 📊 모니터링 지표 단순화 (6가지 핵심)

#### ✅ **필수 지표** (과세분화 제거)

```yaml
# 6가지 핵심 지표 (엔진별 분리 없음)
essential_metrics:
  1. gpu_utilization_percent           # GPU 전체 활용률
  2. vram_used_gb                      # VRAM 사용량 (총량만)
  3. nvdec_utilization_percent         # NVDEC 통합 활용률
  4. nvenc_utilization_percent         # NVENC 통합 활용률
  5. processing_latency_p95_ms         # 전체 파이프라인 p95 레이턴시
  6. queue_depth_p99_frames            # 대기 큐 p99 깊이

# 선택적 디버깅 지표 (개발 시에만)
debug_metrics:
  - h2d_copy_bytes_per_sec            # CPU→GPU 복사 (제로카피 검증)
  - d2h_copy_bytes_per_sec            # GPU→CPU 복사 (제로카피 검증)
  - model_inference_fps               # 모델 추론 FPS
```

#### 🚨 **알람 기준** (단순화)

```python
# 간단한 알람 기준
ALERT_THRESHOLDS = {
    'gpu_utilization_percent': 95,      # 95% 이상시 알람
    'vram_used_gb': 28,                 # 28GB 이상시 알람 (32GB 중)
    'processing_latency_p95_ms': 100,   # 100ms 이상시 알람
    'queue_depth_p99_frames': 20        # 20프레임 이상시 알람
}
```

---

## 🎯 성공 기준 및 검증 (정확한 기준)

### 정량적 성공 기준
1. **처리량 향상**: 기존 대비 **8배 이상** (Full GPU 효과)
2. **NVDEC 활용률**: **85% 이상** 유지 (4개 엔진 평균)
3. **VRAM 효율성**: **87.5% 이하** 사용 (28GB/32GB)  
4. **End-to-End 지연**: 프레임당 **30ms 이하**
5. **ID 안정성**: 분당 스위치 **0.05회 이하**

### 검증 시나리오
1. **4개 영상 동시 처리**: 1080p x 4스트림 15분 연속
2. **메모리 안정성**: 8시간 연속 처리 without OOM
3. **장애 복구**: NVDEC 실패 시 자동 복구 확인
4. **모델 성능 비교**: YOLOv8n vs YOLOv8s vs SCRFD 벤치마킹

---

## 🚨 리스크 관리 (구체적 대응책)

### 고위험 요소 및 대응
1. **PyAV NVDEC 설치 실패**: → PyNvCodec 백업, 최종 CPU 디코딩
2. **TensorRT 변환 실패**: → PyTorch FP16 백업 경로  
3. **VRAM 부족**: → 동적 배치 크기 조정, 스트림 수 감소
4. **NVENC 동시 한계**: → 2개 엔진 로드 밸런싱, 대기열 관리

### 성능 백업 시나리오
- **Plan A**: 4스트림 Full GPU (목표)
- **Plan B**: 3스트림 Full GPU (안정)
- **Plan C**: 2스트림 + CPU 백업 (최소)

---

**프로젝트 상태**: Full GPU 파이프라인 상세 설계 완료  
**예상 완료**: 10일 이내  
**핵심 혁신**: PyAV NVDEC → TensorRT → NVENC 제로카피 파이프라인  
**성능 목표**: 6.2배 처리량 향상 (2.6 → 16개/시간) | 단일 영상: 6.1배 단축 (23분 → 3.75분)  
**기술 스택**: PyNvCodec(VPF) + TensorRT + CUDA Composition + NVENC H.264
