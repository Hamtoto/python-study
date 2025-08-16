#!/bin/bash

# ==============================================================================
# Phase 5: Dual-Face Tracking System 실행 스크립트
# ==============================================================================
# 
# 기능: 2명의 얼굴을 검출하고 추적하여 각각 크롭한 스플릿 스크린 비디오 생성
# 입력: 2people_sample1.mp4 (또는 다른 비디오)
# 출력: 1920x1080 스플릿 스크린 (Person1 + Person2 얼굴 추적)
# 
# 사용법:
#   ./run_face_tracking.sh [input_video] [output_video]
#   ./run_face_tracking.sh                                    # 기본값 사용
#   ./run_face_tracking.sh custom_video.mp4                   # 커스텀 입력
#   ./run_face_tracking.sh input.mp4 output_tracked.mp4       # 커스텀 입출력
# 
# ==============================================================================

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로고 출력
echo -e "${BLUE}"
echo "🚀 Dual-Face Tracking System v5.0"
echo "=================================="
echo "  Phase 5: 실제 얼굴 트래킹 & 크롭"
echo "  2명 얼굴 검출 → 추적 → 스플릿 스크린"
echo -e "${NC}"

# 파라미터 설정
INPUT_VIDEO=${1:-"tests/videos/2people_sample1.mp4"}
OUTPUT_VIDEO=${2:-"output/2people_sample1_tracked.mp4"}

echo -e "${YELLOW}📋 실행 설정:${NC}"
echo "   📥 입력 비디오: $INPUT_VIDEO"
echo "   📤 출력 비디오: $OUTPUT_VIDEO"
echo "   🎯 처리 모드: dual_face (2명 얼굴 추적)"
echo "   🖥️  GPU 가속: 활성화"
echo

# DevContainer 확인
if [ -f /.dockerenv ]; then
    echo -e "${GREEN}✅ DevContainer 환경 감지${NC}"
    CURRENT_DIR="/workspace"
    cd "$CURRENT_DIR"
    
    # 환경 확인
    echo -e "${BLUE}🔍 환경 확인 중...${NC}"
    python3 -c "
import sys
print(f'   🐍 Python: {sys.version.split()[0]}')
try:
    import cv2
    print(f'   📹 OpenCV: {cv2.__version__}')
except ImportError:
    print('   ❌ OpenCV 없음')
try:
    import torch
    print(f'   🔥 PyTorch CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   🖥️  GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('   ❌ PyTorch 없음')
"
    echo
    
    # 직접 실행 (DevContainer 내부)
    echo -e "${GREEN}🚀 얼굴 트래킹 시스템 시작...${NC}"
    python3 face_tracking_system.py \
        --input "$INPUT_VIDEO" \
        --output "$OUTPUT_VIDEO" \
        --mode dual_face \
        --gpu 0
    
    exit_code=$?
    
else
    echo -e "${YELLOW}🐳 DevContainer 실행 중...${NC}"
    
    # DevContainer가 실행 중인지 확인
    if ! docker ps | grep -q "dual-face-dev-container"; then
        echo -e "${RED}❌ DevContainer가 실행되지 않음${NC}"
        echo -e "${YELLOW}💡 다음 명령으로 DevContainer를 시작하세요:${NC}"
        echo "   ./run_dev.sh"
        exit 1
    fi
    
    # Docker exec로 실행
    docker exec dual-face-dev-container bash -c "
        cd /workspace && 
        echo '🚀 얼굴 트래킹 시스템 시작...' &&
        python3 face_tracking_system.py \
            --input '$INPUT_VIDEO' \
            --output '$OUTPUT_VIDEO' \
            --mode dual_face \
            --gpu 0
    "
    
    exit_code=$?
fi

echo

# 결과 확인
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ 얼굴 트래킹 처리 완료!${NC}"
    echo
    echo -e "${BLUE}📊 결과 확인:${NC}"
    
    # 출력 파일 확인
    if [ -f "$OUTPUT_VIDEO" ] || docker exec dual-face-dev-container test -f "/workspace/$OUTPUT_VIDEO" 2>/dev/null; then
        # 파일 크기 확인
        if [ -f /.dockerenv ]; then
            # DevContainer 내부
            file_size=$(du -h "$OUTPUT_VIDEO" | cut -f1 2>/dev/null || echo "알 수 없음")
        else
            # 호스트에서 Docker 실행
            file_size=$(docker exec dual-face-dev-container du -h "/workspace/$OUTPUT_VIDEO" | cut -f1 2>/dev/null || echo "알 수 없음")
        fi
        
        echo "   📄 출력 파일: $OUTPUT_VIDEO"
        echo "   💾 파일 크기: $file_size"
        
        # 호스트 output 디렉토리로 복사
        if [ ! -f /.dockerenv ]; then
            echo "   📋 호스트로 복사 중..."
            mkdir -p "/home/hamtoto/work/python-study/Face-Tracking-App/output"
            docker cp "dual-face-dev-container:/workspace/$OUTPUT_VIDEO" "/home/hamtoto/work/python-study/Face-Tracking-App/output/"
            echo "   ✅ 호스트 복사 완료: /home/hamtoto/work/python-study/Face-Tracking-App/output/"
        fi
        
    else
        echo -e "${RED}   ❌ 출력 파일이 생성되지 않았습니다${NC}"
    fi
    
    echo
    echo -e "${GREEN}🎉 Phase 5 완료!${NC}"
    echo "   • 2명 얼굴 검출 ✅"
    echo "   • 실시간 얼굴 추적 ✅"
    echo "   • 얼굴 중심 크롭 ✅"
    echo "   • 1920x1080 스플릿 스크린 ✅"
    echo
    echo -e "${BLUE}💡 다음 단계:${NC}"
    echo "   • 출력 비디오 재생하여 결과 확인"
    echo "   • 다른 비디오로 테스트"
    echo "   • 트래킹 파라미터 조정 (필요시)"
    
else
    echo -e "${RED}❌ 얼굴 트래킹 처리 실패 (종료 코드: $exit_code)${NC}"
    echo
    echo -e "${YELLOW}🔍 트러블슈팅:${NC}"
    echo "   1. 입력 비디오 파일 경로 확인"
    echo "   2. DevContainer GPU 접근 확인: nvidia-smi"
    echo "   3. 의존성 확인: pip install -r requirements.txt"
    echo "   4. 로그 확인: 위의 에러 메시지 참조"
    
    exit $exit_code
fi

echo
echo "==============================================="
echo "🎯 Dual-Face Tracking System 실행 완료"
echo "==============================================="