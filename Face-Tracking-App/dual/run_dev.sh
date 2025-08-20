#!/bin/bash
# 🚀 Dual-Face Tracking 통합 실행 스크립트
# 사용법:
#   ./run_dev.sh                              # 기본 비디오 실행
#   ./run_dev.sh input/sample.mp4            # 특정 비디오
#   ./run_dev.sh input/sample.mp4 output/result.mp4  # 입출력 지정

set -e

# 기본값 설정
INPUT_VIDEO="${1:-input/2people_sample1.mp4}"
OUTPUT_VIDEO="${2:-output/$(basename ${INPUT_VIDEO%.*})_tracked.mp4}"

# 입력 파일 확인
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "❌ 입력 파일을 찾을 수 없습니다: $INPUT_VIDEO"
    echo ""
    echo "📁 사용 가능한 비디오:"
    ls -la input/ 2>/dev/null | grep "\.mp4" | awk '{print "   " $9}' || echo "   (input 폴더가 비어있습니다)"
    exit 1
fi

# 출력 디렉토리 생성
OUTPUT_DIR=$(dirname "$OUTPUT_VIDEO")
mkdir -p "$OUTPUT_DIR"

# 로그 파일 설정
mkdir -p logs
VIDEO_NAME=$(basename "${INPUT_VIDEO%.*}")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${VIDEO_NAME}_${TIMESTAMP}.log"

# 로그 헤더 작성
{
    echo "🚀 Dual-Face Tracking System"
    echo "========================================"
    echo "시작 시간: $(date)"
    echo "입력: $INPUT_VIDEO"
    echo "출력: $OUTPUT_VIDEO"
    echo "로그 파일: $LOG_FILE"
    echo "========================================"
    echo ""
} | tee "$LOG_FILE"

echo "========================================"
echo "▶ System Check"

# GPU 환경 확인
if command -v nvidia-smi >/dev/null 2>&1; then
    {
        echo "✅ GPU 확인:"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
        echo ""
    } | tee -a "$LOG_FILE"
fi
echo "========================================"

# Face Tracking 시스템 실행
echo "Face Tracking Start..." | tee -a "$LOG_FILE"

# Python 실행을 서브셸에서 실행하고 명시적으로 종료 상태 확인
(
    python3 -m dual_face_tracker.core.face_tracking_system \
        --input "$INPUT_VIDEO" \
        --output "$OUTPUT_VIDEO" \
        --auto-speaker \
        --debug
) 2>&1 | tee -a "$LOG_FILE"

# 실행 결과 저장
RESULT=${PIPESTATUS[0]}

# 완료 정보 로그에 기록
{
    echo ""
    echo "종료 시간: $(date)"
    if [ $RESULT -eq 0 ]; then
        echo "✅ 처리 완료!"
    else
        echo "⚠️ 처리 중 오류 발생 (코드: $RESULT)"
    fi
    echo "📂 결과: $OUTPUT_VIDEO"
    echo "📋 로그 저장: $LOG_FILE"
} | tee -a "$LOG_FILE"

exit $RESULT