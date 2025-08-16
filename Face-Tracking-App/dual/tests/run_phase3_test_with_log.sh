#!/bin/bash
# Phase 3 멀티스트림 테스트 실행 스크립트 (로그 저장 버전)
#
# DevContainer 환경에서 4-Stream 병렬 처리 테스트를 실행하고
# 모든 출력을 로그 파일에도 저장합니다.
#
# 사용법:
#     ./run_phase3_test_with_log.sh
#     
# 출력 파일:
#     - phase3_test_log.txt (전체 실행 로그)
#     - gpu_monitor.log (GPU 모니터링)

set -e  # 오류 시 중단

# 로그 파일 설정
LOG_FILE="phase3_test_log.txt"
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# 로그 파일 초기화
echo "Phase 3 멀티스트림 테스트 실행 로그" > "$LOG_FILE"
echo "시작 시간: $(timestamp)" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# tee를 사용해서 콘솔과 파일에 동시 출력하는 함수
log_and_display() {
    while IFS= read -r line; do
        echo "$line"
        echo "[$(timestamp)] $line" >> "$LOG_FILE"
    done
}

# 실제 테스트 실행 부분을 함수로 분리
run_test() {
    echo "🚀 Phase 3 멀티스트림 테스트 시작"
    echo "=================================================="

    # 환경 확인
    echo "📋 환경 확인 중..."

    # Python 경로 확인
    PYTHON_PATH=$(which python3 || which python)
    echo "  • Python: $PYTHON_PATH"

    # GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        echo "  • GPU 상태:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1 | \
        awk -F', ' '{printf "    - %s: %d/%dMB VRAM\n", $1, $3, $2}'
    else
        echo "  • ⚠️ nvidia-smi 없음 - GPU 확인 불가"
    fi

    # CUDA 확인
    echo -n "  • CUDA: "
    python3 -c "import torch; print('✅ Available' if torch.cuda.is_available() else '❌ Not Available')" 2>/dev/null || echo "❌ PyTorch 없음"

    # PyAV 확인
    echo -n "  • PyAV: "
    python3 -c "import av; print('✅', av.version)" 2>/dev/null || echo "❌ PyAV 없음"

    # OpenCV 확인
    echo -n "  • OpenCV: "
    python3 -c "import cv2; print('✅', cv2.__version__)" 2>/dev/null || echo "❌ OpenCV 없음"

    echo ""

    # 디렉토리 준비
    echo "📁 테스트 디렉토리 준비..."
    mkdir -p test_videos test_output
    echo "  • test_videos/ 생성"
    echo "  • test_output/ 생성"

    # 권한 확인
    if [ ! -w "test_videos" ] || [ ! -w "test_output" ]; then
        echo "❌ 쓰기 권한 없음 - 디렉토리 권한 확인 필요"
        return 1
    fi

    echo "  ✅ 디렉토리 준비 완료"
    echo ""

    # Python 모듈 경로 설정
    export PYTHONPATH="${PYTHONPATH}:/workspace"

    # 테스트 실행
    echo "🧪 멀티스트림 테스트 실행..."
    echo "=================================================="

    # 메모리 모니터링 시작 (백그라운드)
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU 모니터링 시작 (nvidia-smi)..."
        nvidia-smi dmon -s pucvmet -d 10 > gpu_monitor.log 2>&1 &
        NVIDIA_PID=$!
        echo "  • PID: $NVIDIA_PID (로그: gpu_monitor.log)"
    else
        NVIDIA_PID=""
    fi

    # 실제 테스트 실행
    START_TIME=$(date +%s)

    echo ""
    echo "🔥 테스트 시작 - $(date)"
    echo "--------------------------------------------------"

    if python3 test_multi_stream.py; then
        TEST_RESULT="SUCCESS"
        echo "✅ 테스트 성공!"
    else
        TEST_RESULT="FAILED"
        echo "❌ 테스트 실패!"
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "--------------------------------------------------"
    echo "🏁 테스트 완료 - $(date)"
    echo "⏱️ 총 소요 시간: ${DURATION}초 ($((DURATION/60))분 $((DURATION%60))초)"
    echo ""

    # 모니터링 중지
    if [ ! -z "$NVIDIA_PID" ]; then
        echo "📊 GPU 모니터링 중지..."
        kill $NVIDIA_PID 2>/dev/null || true
        
        # 모니터링 요약
        if [ -f "gpu_monitor.log" ]; then
            echo "  • GPU 사용률 요약:"
            # 마지막 몇 줄에서 평균 계산 (간단한 방법)
            tail -10 gpu_monitor.log | grep -E "^\s*[0-9]" | tail -5 | \
            awk '{sum+=$3; count++} END {if(count>0) printf "    - 평균 GPU 활용률: %.1f%%\n", sum/count}' || true
        fi
    fi

    # 결과 요약
    echo "=================================================="
    echo "📋 Phase 3 테스트 결과 요약"
    echo "=================================================="
    echo "  • 결과: $TEST_RESULT"
    echo "  • 소요 시간: ${DURATION}초"
    echo "  • 목표 달성: $([ "$TEST_RESULT" = "SUCCESS" ] && echo "✅ 성공" || echo "❌ 실패")"

    # 생성된 파일들 확인
    echo ""
    echo "📁 생성된 파일들:"
    if [ -d "test_output" ]; then
        find test_output -name "*.mp4" -exec ls -lh {} \; | head -10 | \
        awk '{printf "  • %s (%s)\n", $9, $5}'
    else
        echo "  • 출력 파일 없음"
    fi

    # 정리
    echo ""
    echo "🧹 정리 작업..."
    echo "  • 임시 파일 정리 중..."

    # GPU 로그 파일 크기 확인 후 압축
    if [ -f "gpu_monitor.log" ] && [ $(wc -l < gpu_monitor.log) -gt 100 ]; then
        echo "  • GPU 모니터링 로그 압축 중..."
        gzip gpu_monitor.log 2>/dev/null || true
    fi

    echo "  ✅ 정리 완료"

    # 최종 메시지
    echo ""
    echo "=================================================="
    if [ "$TEST_RESULT" = "SUCCESS" ]; then
        echo "🎉 Phase 3 멀티스트림 테스트 성공!"
        echo "   4-Stream 병렬 처리 시스템이 정상 동작합니다."
        echo ""
        echo "📈 다음 단계:"
        echo "   • 실제 대용량 비디오로 성능 벤치마크 진행"
        echo "   • GPU 사용률 80% 목표 달성 확인"
        echo "   • 15분 내 처리 목표 검증"
    else
        echo "❌ Phase 3 멀티스트림 테스트 실패"
        echo "   문제점을 분석하고 수정이 필요합니다."
        echo ""
        echo "🔧 문제 해결 체크리스트:"
        echo "   • DevContainer 환경에서 실행했는지 확인"
        echo "   • GPU 메모리 충분한지 확인 (nvidia-smi)"
        echo "   • 의존성 모듈 정상 설치되었는지 확인"
        echo "   • 로그 파일에서 상세 오류 메시지 확인"
    fi

    echo "=================================================="

    return $([ "$TEST_RESULT" = "SUCCESS" ] && echo 0 || echo 1)
}

# 메인 실행 부분
echo "🚀 Phase 3 테스트 시작 (로그 저장 버전)"
echo "로그 파일: $LOG_FILE"
echo "=================================================="

# 테스트 실행하고 출력을 콘솔과 파일에 동시 저장
if run_test | log_and_display; then
    FINAL_RESULT=0
else
    FINAL_RESULT=1
fi

# 로그 파일에 완료 정보 추가
echo "" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "완료 시간: $(timestamp)" >> "$LOG_FILE"
echo "최종 결과: $([ $FINAL_RESULT -eq 0 ] && echo "SUCCESS" || echo "FAILED")" >> "$LOG_FILE"

# 사용자에게 로그 파일 위치 알림
echo ""
echo "📄 완전한 실행 로그가 저장되었습니다:"
echo "   파일: $(pwd)/$LOG_FILE"
echo "   크기: $(wc -l < "$LOG_FILE") 줄"
echo ""
echo "📋 로그 확인 명령어:"
echo "   전체 보기: cat $LOG_FILE"
echo "   실시간 보기: tail -f $LOG_FILE"
echo "   오류만 보기: grep -i error $LOG_FILE"

exit $FINAL_RESULT