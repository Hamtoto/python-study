#!/bin/bash
# Phase 3 수정된 멀티스트림 테스트 실행 스크립트
#
# NVENC 세션 제한 문제가 해결된 Phase 3 테스트를 실행합니다.
# 
# 주요 개선 사항:
# - NVENC 동시 세션 수 제한 (4개 → 2개)
# - 배치 처리 방식 (2+2)
# - 자동 소프트웨어 폴백
# - 안전한 리소스 정리
# - 향상된 에러 처리
#
# 사용법:
#     ./run_phase3_fixed_test.sh
#     
# 출력:
#     - 콘솔에 실시간 진행 상황 출력
#     - phase3_fixed_test_log.txt에 상세 로그 저장
#     - GPU 모니터링 결과
#
# Author: Dual-Face High-Speed Processing System
# Date: 2025.01
# Version: 1.0.0 (Fixed)

set -e  # 에러 발생 시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 파일 설정
LOG_FILE="logs/phase3_fixed_test_log.txt"
GPU_LOG="logs/gpu_monitor_fixed.log"
ERROR_LOG="logs/phase3_fixed_errors.log"

# 로그 함수들
log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}[${timestamp}]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[${timestamp}]${NC} ✅ $1" | tee -a "$LOG_FILE"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[${timestamp}]${NC} ⚠️ $1" | tee -a "$LOG_FILE"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[${timestamp}]${NC} ❌ $1" | tee -a "$LOG_FILE"
    echo -e "${RED}[${timestamp}]${NC} ❌ $1" >> "$ERROR_LOG"
}

# 환경 확인 함수
check_environment() {
    log_info "🔍 환경 확인 중..."
    
    # GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1)
        log_success "GPU 감지: $gpu_info"
    else
        log_error "nvidia-smi를 찾을 수 없음"
        return 1
    fi
    
    # Python 환경 확인
    if python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
        local torch_info=$(python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null)
        log_success "PyTorch: $torch_info"
    else
        log_error "PyTorch CUDA 환경 문제"
        return 1
    fi
    
    # PyAV 확인
    if python3 -c "import av; print(f'PyAV: {av.__version__}')" 2>/dev/null; then
        local av_info=$(python3 -c "import av; print(f'PyAV: {av.__version__}')" 2>/dev/null)
        log_success "PyAV: $av_info"
    else
        log_error "PyAV 가져오기 실패"
        return 1
    fi
    
    # OpenCV 확인
    if python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null; then
        local cv_info=$(python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null)
        log_success "OpenCV: $cv_info"
    else
        log_error "OpenCV 가져오기 실패" 
        return 1
    fi
    
    return 0
}

# GPU 모니터링 시작
start_gpu_monitoring() {
    log_info "📊 GPU 모니터링 시작..."
    
    # GPU 모니터링을 백그라운드에서 실행
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv >> "$GPU_LOG" 2>/dev/null || break
            sleep 2
        done
    ) &
    
    GPU_MONITOR_PID=$!
    log_success "GPU 모니터링 시작 (PID: $GPU_MONITOR_PID)"
}

# GPU 모니터링 정지
stop_gpu_monitoring() {
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
        wait $GPU_MONITOR_PID 2>/dev/null || true
        log_success "GPU 모니터링 정지"
    fi
}

# 테스트 결과 분석
analyze_test_results() {
    log_info "📊 테스트 결과 분석 중..."
    
    # 로그에서 성공/실패 패턴 찾기
    local total_tests=$(grep -c "테스트.*:" "$LOG_FILE" || echo "0")
    local passed_tests=$(grep -c "✅.*테스트.*성공" "$LOG_FILE" || echo "0")
    local failed_tests=$(grep -c "❌.*테스트.*실패" "$LOG_FILE" || echo "0")
    
    # NVENC vs 소프트웨어 인코딩 비율
    local nvenc_count=$(grep -c "method.*nvenc" "$LOG_FILE" || echo "0")
    local software_count=$(grep -c "method.*software" "$LOG_FILE" || echo "0")
    local fallback_count=$(grep -c "폴백" "$LOG_FILE" || echo "0")
    
    # 세션 관리 통계
    local session_acquired=$(grep -c "세션.*획득" "$LOG_FILE" || echo "0")
    local session_released=$(grep -c "세션.*해제" "$LOG_FILE" || echo "0")
    
    log_info "📈 테스트 통계:"
    log_info "  • 총 테스트: $total_tests"
    log_info "  • 성공: $passed_tests"
    log_info "  • 실패: $failed_tests"
    
    if [ "$total_tests" -gt 0 ]; then
        local pass_rate=$((passed_tests * 100 / total_tests))
        log_info "  • 성공률: ${pass_rate}%"
    fi
    
    log_info "📊 인코딩 통계:"
    log_info "  • NVENC 인코딩: $nvenc_count"
    log_info "  • 소프트웨어 인코딩: $software_count"
    log_info "  • 폴백 발생: $fallback_count"
    
    log_info "🔧 세션 관리:"
    log_info "  • 세션 획득: $session_acquired"
    log_info "  • 세션 해제: $session_released"
    
    # GPU 사용률 분석 (마지막 10줄)
    if [ -f "$GPU_LOG" ] && [ -s "$GPU_LOG" ]; then
        log_info "📊 GPU 사용률 (마지막 측정값들):"
        tail -5 "$GPU_LOG" | while read line; do
            if [[ $line != *"timestamp"* ]]; then
                log_info "  $line"
            fi
        done
    fi
}

# 정리 함수
cleanup() {
    log_info "🧹 정리 중..."
    
    # GPU 모니터링 정지
    stop_gpu_monitoring
    
    # Python 프로세스 정리
    pkill -f "test_phase3_fixed.py" 2>/dev/null || true
    
    # GPU 메모리 정리
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    log_success "정리 완료"
}

# 시그널 핸들러 설정
trap cleanup EXIT
trap 'log_warning "중단 신호 받음"; cleanup; exit 130' INT TERM

# 메인 실행
main() {
    # 헤더 출력
    echo "=================================================================================" | tee "$LOG_FILE"
    echo "🚀 Phase 3 Fixed 멀티스트림 테스트 (NVENC 세션 제한 해결 버전)" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
    
    local start_time=$(date +%s)
    local test_success=false
    
    # 기존 로그 파일들 초기화
    > "$LOG_FILE"
    > "$GPU_LOG"
    > "$ERROR_LOG"
    
    log_info "테스트 시작 시간: $(date)"
    log_info "주요 개선사항:"
    log_info "  • NVENC 동시 세션: 4개 → 2개 제한"
    log_info "  • 배치 처리: 2+2 방식"
    log_info "  • 자동 소프트웨어 폴백"
    log_info "  • 안전한 리소스 정리"
    
    # 환경 확인
    if ! check_environment; then
        log_error "환경 확인 실패"
        return 1
    fi
    
    # GPU 모니터링 시작
    start_gpu_monitoring
    
    # 메인 테스트 실행
    log_info "🎯 Phase 3 Fixed 테스트 실행 중..."
    
    if python3 test_phase3_fixed.py 2>&1 | tee -a "$LOG_FILE"; then
        local exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            test_success=true
            log_success "Phase 3 Fixed 테스트 성공!"
        else
            log_error "Python 테스트 실패 (exit code: $exit_code)"
        fi
    else
        log_error "Python 테스트 실행 실패"
    fi
    
    # 테스트 결과 분석
    analyze_test_results
    
    # 최종 결과
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo "=================================================================================" | tee -a "$LOG_FILE"
    echo "📊 Phase 3 Fixed 테스트 최종 결과" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
    
    if [ "$test_success" = true ]; then
        log_success "전체 테스트: 성공"
        log_success "NVENC 세션 제한 문제 해결 확인"
        log_success "소프트웨어 폴백 메커니즘 동작 확인"
    else
        log_error "전체 테스트: 실패"
        log_warning "에러 로그 확인: $ERROR_LOG"
    fi
    
    log_info "총 소요 시간: ${minutes}분 ${seconds}초"
    log_info "로그 파일:"
    log_info "  • 상세 로그: $LOG_FILE"
    log_info "  • GPU 모니터링: $GPU_LOG"
    log_info "  • 에러 로그: $ERROR_LOG"
    
    echo "=================================================================================" | tee -a "$LOG_FILE"
    
    if [ "$test_success" = true ]; then
        return 0
    else
        return 1
    fi
}

# 스크립트 실행
if main; then
    echo ""
    echo "🎉 Phase 3 Fixed 테스트가 성공적으로 완료되었습니다!"
    echo "   NVENC 세션 제한 문제가 해결되었으며, 멀티스트림 처리가 안정적으로 동작합니다."
    echo ""
    exit 0
else
    echo ""
    echo "❌ Phase 3 Fixed 테스트가 실패했습니다."
    echo "   로그 파일을 확인하여 문제를 파악하세요: $LOG_FILE"
    echo ""
    exit 1
fi