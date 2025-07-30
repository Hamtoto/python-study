# Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

## File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the
  gemini command:

### Examples:

**Single file analysis:**
gemini -p "@src/main.py Explain this file's purpose and structure"

Multiple files:
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

Entire directory:
gemini -p "@src/ Summarize the architecture of this codebase"

Multiple directories:
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

Current directory and subdirectories:
gemini -p "@./ Give me an overview of this entire project"

# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"

Implementation Verification Examples

Check if a feature is implemented:
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

Verify authentication implementation:
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

Check for specific patterns:
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

Verify error handling:
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

Check for rate limiting:
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

Verify caching strategy:
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

Check for specific security measures:
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

Verify test coverage for features:
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

When to Use Gemini CLI

Use gemini -p when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

Important Notes

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results

# CLAUDE.md - Face-Tracking-App Development Guide

> **For future Claude instances working on this repository**

## 🎯 Project Overview

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
python main.py

# Test execution (use test folder for all tests)
# Tests are in test/ directory - user handles testing manually
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
8. **Logging architecture**: Dual console/file logging with structured error tracking

## 📈 Performance Achievements

- ✅ **97.3% GPU utilization** (exceeded 95% target)
- ✅ **67% processing time reduction** (45-60s → 15-20s)
- ✅ **Producer-Consumer pattern** implemented in critical paths
- ✅ **Dynamic batch sizing** with memory safety
- ✅ **Zero GPU resource conflicts** through single worker architecture
- ✅ **Comprehensive logging system** with console/file separation

---

**Last Updated**: 2024 (refer to refactor.md for detailed optimization history)

**For Questions**: Check `README.md`, `refactor.md`, and source code comments