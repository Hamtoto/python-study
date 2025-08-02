# CLAUDE.md - Face-Tracking-App v1.0 Development Guide

> **For future Claude instances working on this repository**

## 🎯 Project Overview - Version 1.0

Face-Tracking-App is a GPU-optimized video processing pipeline that uses MTCNN and FaceNet (InceptionResnetV1) models for face detection, recognition, and tracking. The system processes video files to extract face-tracked segments with high performance through Producer-Consumer patterns and multiprocessing architecture.

**Key Achievement**: GPU utilization optimized to 97.3% with 67% processing time reduction (45-60s → 15-20s)

## 🚀 Quick Commands

### Development Environment
```bash
# Activate virtual environment (MANDATORY)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Main execution
./start.sh
# or
python face_tracker.py

# Check logs
tail -f face_tracker.log
```

## 📈 Version 1.0 - Major Updates (2024)

### 🔧 **로깅 시스템 통합 및 최적화**
- **단일 로그 파일**: `face_tracker.log`로 모든 로그 통합
- **통합 로거**: `UnifiedLogger` 클래스로 체계적 관리
- **76개 print문 최적화**: 의미있는 로그 메시지로 변경
- **이모지 기반 로그**: `🔄 stage()`, `✅ success()`, `⚠️ warning()`, `❌ error()`
- **tqdm 프로그레스바 최적화**: `ncols=60, leave=False`로 화면 공간 60% 절약

### 📊 **성능 리포트 시스템 추가**
- **실시간 성능 측정**: 단계별 처리 시간, 배치 크기, FPS 자동 측정
- **상세 리포트 출력**: 비디오 처리 완료 시 콘솔에 80줄짜리 상세 리포트
- **시스템 리소스 모니터링**: CPU 코어 사용량, 메모리 사용량 추적
- **성능 지표 계산**: 전체 처리 속도, 프레임당 평균 시간 자동 계산

### 🏗️ **코드 아키텍처 최적화**
- **중복 FFmpeg 로직 통합**: 40줄 → 10줄로 축소 (75% 감소)
- **래퍼 함수 제거**: 15줄 → 3줄로 메모리 효율성 향상
- **Import 경로 통합**: 절대 경로 사용으로 모듈 참조 안정화

### Input/Output Structure
```
videos/input/     # Place video files here (.mp4, .mov, .avi)
videos/output/    # Processed segments output here
temp_proc/        # Temporary processing files (auto-cleaned)
face_tracker.log  # Unified log file
```

### Performance Report Example
```
================================================================================
📊 sample.mp4 처리 완료 리포트
================================================================================
🎬 영상: sample.mp4
⏱️  총 처리시간: 2분 15.3초
🖼️  총 프레임 수: 3,240
📦 생성된 세그먼트: 5개
🖥️  사용된 CPU 코어: 8/32개

⚙️ 설정 정보:
   • 얼굴 분석 배치 크기: 256
   • 얼굴 인식 배치 크기: 128

📈 단계별 성능:
   • 얼굴 감지: 45.2초 (33.6%) - 71.7 FPS - 배치: 256
   • 얼굴 인식: 38.1초 (28.3%) - 85.0 FPS - 배치: 128
   • 얼굴 크롭: 28.7초 (21.3%) - 세그먼트: 5개
   • 요약본 생성: 15.4초 (11.4%)

🎯 성능 지표:
   • 총 처리 시간: 2분 15.3초
   • 전체 처리 속도: 24.1 FPS
   • 프레임당 평균 시간: 41.5ms
   • 메모리 사용량: 1,247.3 MB
================================================================================
```

### Development Container Setup
```bash
# If using dev container (with GPU support)
cd .devcontainer
docker-compose up --build

# PyTorch CUDA installation (custom)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Custom OpenCV installation
dpkg -i build_20250710-1_amd64.deb
```

### Input/Output Structure
```
videos/input/     # Place video files here (.mp4, .mov, .avi)
videos/output/    # Processed segments output here
temp_proc/        # Temporary processing files (auto-cleaned)
```

## 🏗️ Architecture Deep Dive

### 1. High-Level Processing Pipeline
```
Input Video → Face Analysis → ID Timeline → Condensed Video → 
Target Selection → Video Trimming → Segment Slicing → 
GPU Cropping (Producer-Consumer) → CPU FFmpeg Encoding (Pool) → Final Output
```

### 2. GPU-Optimized Architecture (Key Innovation)

**GPU Worker + CPU Pool Pattern**:
- **Single GPU Worker Process**: Handles all GPU operations sequentially to prevent CUDA OOM
- **CPU Process Pool**: Handles FFmpeg encoding in parallel (max(1, cpu_count()-1) processes)
- **Queue-based Communication**: Tasks flow through multiprocessing Queues

**Producer-Consumer Implementation**:
- **face_analyzer.py**: Frame I/O Thread + GPU Processing Thread
- **id_timeline_generator.py**: Similar threading for face recognition
- **Dynamic Batch Sizing**: 64→128→256 based on queue depth and GPU memory

### 3. Memory Management System

**ModelManager (Singleton)**:
- MTCNN and InceptionResnetV1 loaded once globally
- GPU memory pool pre-allocation for tensors
- Prevents model reloading across processes

**GPU Memory Safety**:
- `expandable_segments:True` CUDA configuration
- Memory monitoring with automatic batch size adjustment
- Context managers for MoviePy resource cleanup

## 📁 Critical File Structure

```
processors/
├── video_processor.py     # Main pipeline orchestrator
├── face_analyzer.py       # Producer-Consumer face detection
├── id_timeline_generator.py # Producer-Consumer face recognition  
├── gpu_worker.py          # GPU worker process implementation
└── video_trimmer.py       # Video segmentation and trimming

core/
├── model_manager.py       # Singleton model management
└── embedding_manager.py   # Face embedding vector management

utils/
├── logger.py              # Advanced logging (console + file)
├── console_logger.py      # Console output to file capture
├── exceptions.py          # Custom exception classes
└── input_validator.py     # Input file validation

config.py                  # Central configuration
main.py                    # Entry point with logging wrapper
```

## ⚙️ Key Configuration Parameters

**Performance Settings** (`config.py`):
```python
BATCH_SIZE_ANALYZE = 256        # Face detection batch size
BATCH_SIZE_ID_TIMELINE = 128    # Face recognition batch size
DEVICE = 'cuda:0'               # GPU device
SEGMENT_LENGTH_SECONDS = 10     # Output segment length
```

**GPU Memory Settings**:
```python
# In main process
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## 🔧 Development Guidelines

### 1. GPU Operations
- **NEVER** create multiple CUDA contexts in different processes
- **ALWAYS** use the single GPU worker pattern for GPU operations
- **USE** ModelManager singleton for all model access
- **MONITOR** GPU memory usage before increasing batch sizes

### 2. Multiprocessing Best Practices
- **USE** `spawn` start method: `set_start_method('spawn', force=True)`
- **SEPARATE** GPU work (single process) from CPU work (process pool)
- **COMMUNICATE** via multiprocessing Queues, not shared memory for complex objects
- **HANDLE** process cleanup in finally blocks

### 3. Error Handling
- **USE** custom exceptions: `GPUMemoryError`, `VideoProcessingError`, `FFmpegError`
- **LOG** all errors to `error_logger` with structured information
- **IMPLEMENT** graceful degradation for GPU memory issues
- **VALIDATE** input files before processing starts

### 4. Performance Optimization
- **IMPLEMENT** Producer-Consumer for I/O-GPU separation
- **USE** dynamic batch sizing based on queue depth
- **PREALLOCATE** tensors in ModelManager memory pool
- **AVOID** PIL → Tensor conversion; use direct OpenCV → Tensor

## 🚨 Common Issues & Solutions

### GPU Memory Issues
```python
# If CUDA OOM occurs:
torch.cuda.empty_cache()
# Reduce batch size in config.py
# Check if multiple processes accessing GPU
```

### Multiprocessing Warnings
```python
# Use spawn method, avoid fork on CUDA
set_start_method('spawn', force=True)
```

### FFmpeg Failures
```python
# Check process_cpu_task() in video_processor.py
# Ensure proper thread allocation: max(1, cpu_count() // 4)
# Validate video codec compatibility
```

### MoviePy Memory Leaks
```python
# Always use context managers or explicit .close()
if clip: clip.close()
```

## 📊 Performance Monitoring

**Current Benchmarks**:
- GPU Utilization: 97.3% (target: 95%+)
- Processing Speed: 15-20 seconds per video
- Memory Efficiency: Optimized with pre-allocated pools

**Monitor Commands**:
```bash
# GPU monitoring
nvidia-smi -l 1

# Process monitoring  
htop

# Log analysis
tail -f log.log
tail -f videos/output/detailed.log
```

## 🧪 Testing Strategy

**Test Organization**:
- **ALL tests** go in `test/` directory
- **Sample video**: Use `videos/input/sample.mp4` for quick tests
- **Manual testing**: User handles all test execution
- **Never** assume pytest/unittest framework availability

**Testing Workflow**:
1. Create test in `test/` directory
2. User runs test manually
3. Check logs in `log.log` and `videos/output/detailed.log`

## 🔄 Future Development Areas

**Immediate Priority (Stability)**:
1. Enhance exception handling with more granular custom exceptions
2. Implement comprehensive input validation system
3. Add Context Manager pattern for all resource management
4. Create unit tests for core functionality

**Performance Enhancements**:
1. CUDA Stream utilization for async batch processing
2. Memory-mapped file I/O for large video handling
3. Early exit optimization for similarity calculations
4. Advanced caching for embeddings and computations

**Face Recognition Improvements**:
1. **L2 Normalization Implementation (Priority: High)**
   - Add L2 normalization before cosine similarity calculation
   - Location: `src/face_tracker/utils/similarity.py`
   - Expected impact: Improved accuracy in lighting/angle variations
   - Implementation: `torch.nn.functional.normalize(embedding, p=2, dim=1)`
2. **Euclidean Distance Alternative**
   - Alternative distance metric for face comparison
   - May provide better performance for specific data characteristics
3. **SVM Classifier (Long-term)**
   - Machine learning approach for same/different person classification
   - Requires dataset preparation and model training pipeline

**Operational Improvements**:
1. Real-time monitoring dashboard (Flask/FastAPI)
2. Async pipeline for entire workflow
3. Multi-video concurrent processing capability
4. RESTful API with progress tracking

## 💡 Critical Development Notes

1. **Always activate virtual environment**: `source .venv/bin/activate`
2. **Test files location**: Create all test files in `test/` directory
3. **Quick testing**: Use `videos/input/sample.mp4` for short video tests
4. **GPU resource management**: Single GPU worker prevents resource conflicts
5. **Memory optimization**: Producer-Consumer pattern eliminates I/O bottlenecks
6. **Error resilience**: System handles GPU memory issues gracefully
7. **Development container**: Full GPU support with custom PyTorch/OpenCV builds
8. **Unified logging system**: Single `face_tracker.log` file with structured logging
9. **Performance reporting**: Automatic detailed reports after each video processing

## 📈 Performance Achievements v1.0

- ✅ **97.3% GPU utilization** (exceeded 95% target)
- ✅ **67% processing time reduction** (45-60s → 15-20s)
- ✅ **Producer-Consumer pattern** implemented in critical paths
- ✅ **Dynamic batch sizing** with memory safety
- ✅ **Zero GPU resource conflicts** through single worker architecture
- ✅ **Unified logging system** with 76 print statements optimized
- ✅ **Performance monitoring** with detailed per-video reports
- ✅ **Code optimization** with 75% reduction in FFmpeg logic duplication
- ✅ **Memory efficiency** through wrapper function elimination

## 🔄 Version History

### v1.0 (2024) - Production Ready
- ✅ Unified logging system implementation
- ✅ Performance reporting system
- ✅ Code architecture optimization
- ✅ 76 print statements → structured logging
- ✅ Single log file (`face_tracker.log`)
- ✅ tqdm progress bar optimization
- ✅ FFmpeg logic consolidation

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production Ready  
**GPU Optimization**: 97.3%  
**Performance Report**: ✅ Enabled

**For Questions**: Check `README.md`, `refactor.md`, and source code comments