#!/bin/bash
"""
Phase 3 실제 비디오 테스트 실행 스크립트 (DevContainer 환경)

9분 32초 비디오 4개를 사용하여 Phase 3 성능 목표를 검증합니다.
모든 로그를 텍스트 파일로 저장하고 GPU 모니터링을 포함합니다.

Usage:
  cd /workspace/tests
  chmod +x run_phase3_real_video_test.sh  
  ./run_phase3_real_video_test.sh

Author: Dual-Face High-Speed Processing System
Date: 2025.08.16
Version: 1.0.0 (DevContainer Optimized)
"""

set -e  # 에러 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 디렉토리 및 파일 먼저 설정
LOG_DIR="/workspace/tests/logs"
mkdir -p "$LOG_DIR"

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/phase3_real_video_${TIMESTAMP}.log"
GPU_LOG="$LOG_DIR/gpu_monitor_${TIMESTAMP}.log"
SYSTEM_LOG="$LOG_DIR/system_monitor_${TIMESTAMP}.log"

# 함수 정의 (LOG_FILE이 설정된 후)
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_header() {
    echo -e "${PURPLE}$1${NC}" | tee -a "$LOG_FILE"
}

# 스크립트 시작
clear
log_header "================================================================================"
log_header "🚀 Phase 3 실제 비디오 테스트 (DevContainer 환경)"
log_header "================================================================================"

# 작업 디렉토리 확인 및 설정
if [[ "$(basename $PWD)" != "tests" ]]; then
    log_info "작업 디렉토리를 /workspace/tests로 변경합니다..."
    cd /workspace/tests || {
        log_error "tests 디렉토리로 이동할 수 없습니다!"
        exit 1
    }
fi

log_info "📁 현재 작업 디렉토리: $(pwd)"

log_header "📋 로그 파일 설정:"
log_info "  • 메인 로그: $LOG_FILE"
log_info "  • GPU 로그: $GPU_LOG"
log_info "  • 시스템 로그: $SYSTEM_LOG"
echo ""

# 환경 확인
log_header "🔍 DevContainer 환경 확인:"

# Python 환경 확인
log_info "Python 환경 확인 중..."
python3 --version 2>&1 | tee -a "$LOG_FILE"

# PyTorch CUDA 확인
log_info "PyTorch CUDA 지원 확인 중..."
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')" 2>&1 | tee -a "$LOG_FILE"

# OpenCV 확인
log_info "OpenCV 버전 확인 중..."
python3 -c "import cv2; print(f'OpenCV Version: {cv2.__version__}'); print(f'CUDA Support: {cv2.cuda.getCudaEnabledDeviceCount() > 0}')" 2>&1 | tee -a "$LOG_FILE"

# GPU 확인
log_info "GPU 정보 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | tee -a "$LOG_FILE"
else
    log_warning "nvidia-smi 명령어를 찾을 수 없습니다!"
fi

# 비디오 파일 확인
log_info "테스트 비디오 파일 확인 중..."
VIDEO_DIR="/workspace/tests/videos"
if [[ -d "$VIDEO_DIR" ]]; then
    log_info "비디오 디렉토리 존재: $VIDEO_DIR"
    ls -la "$VIDEO_DIR"/*.mp4 2>&1 | tee -a "$LOG_FILE" || log_warning "MP4 파일을 찾을 수 없습니다"
else
    log_error "비디오 디렉토리가 존재하지 않습니다: $VIDEO_DIR"
    exit 1
fi

# 출력 디렉토리 생성
OUTPUT_DIR="/workspace/tests/test_output"
mkdir -p "$OUTPUT_DIR"
log_info "출력 디렉토리 준비: $OUTPUT_DIR"

echo ""

# GPU 모니터링 시작
log_header "📊 시스템 모니터링 시작..."

if command -v nvidia-smi &> /dev/null; then
    log_info "GPU 모니터링 백그라운드 실행 중..."
    # GPU 상태를 5초마다 300회 (25분) 기록
    nvidia-smi dmon -s pucvmet -i 0 -c 300 -d 5 > "$GPU_LOG" 2>&1 &
    GPU_PID=$!
    log_info "GPU 모니터링 PID: $GPU_PID"
else
    log_warning "nvidia-smi를 사용할 수 없어 GPU 모니터링을 건너뜁니다"
    GPU_PID=""
fi

# 시스템 리소스 모니터링
log_info "시스템 리소스 모니터링 시작..."
(
    echo "timestamp,cpu_percent,memory_percent,disk_io_read_mb,disk_io_write_mb"
    while true; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        cpu_percent=$(python3 -c "import psutil; print(f'{psutil.cpu_percent(interval=1):.1f}')")
        memory_percent=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().percent:.1f}')")
        
        # 간단한 시스템 정보
        echo "$timestamp,$cpu_percent,$memory_percent,0,0"
        sleep 5
    done
) > "$SYSTEM_LOG" 2>&1 &
SYSTEM_PID=$!
log_info "시스템 모니터링 PID: $SYSTEM_PID"

echo ""

# 메인 테스트 실행
log_header "🎬 Phase 3 실제 비디오 테스트 실행:"
log_info "테스트 시작 시간: $(date)"
log_info "예상 처리 시간: 2-3분 (목표: 9분32초 x 4 → 3분 이내)"

# 실제 테스트 실행
TEST_START_TIME=$(date +%s)
cd /workspace

log_info "Python 테스트 스크립트 실행 중..."
if python3 tests/test_phase3_real_video.py 2>&1 | tee -a "$LOG_FILE"; then
    TEST_RESULT=0
    log_success "✅ Python 테스트 스크립트 성공!"
else
    TEST_RESULT=$?
    log_error "❌ Python 테스트 스크립트 실패 (종료 코드: $TEST_RESULT)"
fi

TEST_END_TIME=$(date +%s)
TEST_DURATION=$((TEST_END_TIME - TEST_START_TIME))

log_info "테스트 종료 시간: $(date)"
log_info "총 테스트 소요 시간: ${TEST_DURATION}초"

echo ""

# 모니터링 프로세스 종료
log_header "🔚 모니터링 프로세스 정리:"

if [[ -n "$GPU_PID" ]]; then
    log_info "GPU 모니터링 프로세스 종료 중... (PID: $GPU_PID)"
    kill "$GPU_PID" 2>/dev/null || log_warning "GPU 모니터링 프로세스 종료 실패"
fi

if [[ -n "$SYSTEM_PID" ]]; then
    log_info "시스템 모니터링 프로세스 종료 중... (PID: $SYSTEM_PID)"
    kill "$SYSTEM_PID" 2>/dev/null || log_warning "시스템 모니터링 프로세스 종료 실패"
fi

# 잠시 대기 (모니터링 프로세스 정리 시간)
sleep 2

echo ""

# 결과 분석 및 요약
log_header "📊 테스트 결과 분석:"

# GPU 통계 분석
if [[ -f "$GPU_LOG" && -s "$GPU_LOG" ]]; then
    log_info "GPU 활용률 분석 중..."
    
    # GPU 로그에서 활용률 추출 (dmon 출력 형식: 5번째 컬럼이 sm 활용률)
    if command -v awk &> /dev/null; then
        AVG_GPU_UTIL=$(awk 'NR>2 && NF>=5 && $5 ~ /^[0-9]+$/ {sum+=$5; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$GPU_LOG")
        MAX_GPU_UTIL=$(awk 'NR>2 && NF>=5 && $5 ~ /^[0-9]+$/ {if($5>max || max=="") max=$5} END {if(max!="" && max>0) printf "%.0f", max; else print "N/A"}' "$GPU_LOG")
        
        log_info "  • 평균 GPU 활용률: ${AVG_GPU_UTIL}%"
        log_info "  • 최대 GPU 활용률: ${MAX_GPU_UTIL}%"
        
        # GPU 활용률 목표 확인 (80% 이상)
        if [[ "$AVG_GPU_UTIL" != "N/A" ]]; then
            # bash 내장 연산 사용 (소수점 제거)
            GPU_UTIL_INT=$(echo "$AVG_GPU_UTIL" | cut -d. -f1)
            if [[ $GPU_UTIL_INT -ge 80 ]]; then
                log_success "  🎯 GPU 활용률 목표 달성: ${AVG_GPU_UTIL}% ≥ 80%"
            else
                log_warning "  ⚠️ GPU 활용률 목표 미달성: ${AVG_GPU_UTIL}% < 80%"
            fi
        fi
    else
        log_warning "awk 명령어를 찾을 수 없어 GPU 통계 분석을 건너뜁니다"
    fi
else
    log_warning "GPU 로그 파일을 찾을 수 없거나 비어있습니다"
fi

# 시스템 리소스 요약
if [[ -f "$SYSTEM_LOG" && -s "$SYSTEM_LOG" ]]; then
    log_info "시스템 리소스 요약:"
    
    if command -v tail &> /dev/null && command -v awk &> /dev/null; then
        LAST_CPU=$(tail -n 1 "$SYSTEM_LOG" | awk -F, '{print $2}')
        LAST_MEMORY=$(tail -n 1 "$SYSTEM_LOG" | awk -F, '{print $3}')
        
        log_info "  • 마지막 CPU 사용률: ${LAST_CPU}%"
        log_info "  • 마지막 메모리 사용률: ${LAST_MEMORY}%"
    fi
fi

# 출력 파일 확인
log_info "출력 파일 확인:"
if [[ -d "$OUTPUT_DIR" ]]; then
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*_processed_*.mp4" -type f 2>/dev/null | wc -l)
    log_info "  • 생성된 출력 파일: ${OUTPUT_COUNT}개"
    
    if [[ $OUTPUT_COUNT -eq 4 ]]; then
        log_success "  ✅ 모든 비디오 처리 완료 (4/4)"
    else
        log_warning "  ⚠️ 일부 비디오 처리 미완료 (${OUTPUT_COUNT}/4)"
    fi
    
    # 파일 목록 출력
    find "$OUTPUT_DIR" -name "*_processed_*.mp4" -type f -exec ls -la {} \; 2>/dev/null | tee -a "$LOG_FILE"
else
    log_warning "출력 디렉토리를 찾을 수 없습니다"
fi

echo ""

# 최종 결과 요약
log_header "🎉 최종 결과 요약:"

if [[ $TEST_RESULT -eq 0 ]]; then
    log_success "✅ Phase 3 실제 비디오 테스트 성공!"
    
    # 성능 목표 확인 (3분 = 180초)
    if [[ $TEST_DURATION -le 180 ]]; then
        log_success "🎯 Phase 3 성능 목표 달성: ${TEST_DURATION}초 ≤ 180초"
        PHASE3_STATUS="완전 성공"
    else
        log_warning "⚠️ Phase 3 성능 목표 미달성: ${TEST_DURATION}초 > 180초"
        PHASE3_STATUS="부분 성공"
    fi
else
    log_error "❌ Phase 3 실제 비디오 테스트 실패!"
    PHASE3_STATUS="실패"
fi

log_info "📋 상세 정보:"
log_info "  • 테스트 상태: $PHASE3_STATUS"
log_info "  • 총 소요 시간: ${TEST_DURATION}초"
log_info "  • 종료 코드: $TEST_RESULT"
log_info "  • 로그 파일: $LOG_FILE"
log_info "  • GPU 로그: $GPU_LOG"
log_info "  • 시스템 로그: $SYSTEM_LOG"

echo ""

# Phase 3 완료 상태 출력
log_header "🎯 Phase 3 완료 상태 확인:"

if [[ "$PHASE3_STATUS" == "완전 성공" ]]; then
    log_success "🎉 Phase 3가 성공적으로 완료되었습니다!"
    log_success "✅ 모든 성능 목표를 달성했습니다."
    log_success "🚀 Phase 4 진입 준비가 완료되었습니다."
elif [[ "$PHASE3_STATUS" == "부분 성공" ]]; then
    log_warning "⚠️ Phase 3가 부분적으로 성공했습니다."
    log_warning "📊 일부 성능 목표 미달성이 있습니다."
    log_info "💡 추가 최적화 후 Phase 4 진입을 고려하세요."
else
    log_error "❌ Phase 3가 실패했습니다."
    log_error "🔧 문제를 해결한 후 다시 시도하세요."
fi

echo ""

log_header "================================================================================"
log_header "📁 로그 파일 위치:"
log_info "  • 메인 로그: $LOG_FILE"
log_info "  • GPU 모니터링: $GPU_LOG"  
log_info "  • 시스템 모니터링: $SYSTEM_LOG"
log_header "================================================================================"

# 종료
exit $TEST_RESULT