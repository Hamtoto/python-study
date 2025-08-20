#!/bin/bash
# run_phase1_test.sh - Phase 1 완료 테스트 실행 스크립트

echo "🚀 Dual-Face Tracker Phase 1 테스트 실행"
echo "========================================"

# DevContainer 환경 확인
if [ ! -f /.dockerenv ]; then
    echo "⚠️  DevContainer 환경에서 실행하는 것을 권장합니다"
    echo "   ./run_dev.sh 명령으로 DevContainer를 시작하세요"
    echo ""
fi

# 현재 디렉토리로 이동
cd "$(dirname "$0")"

# Python 경로 확인
if ! command -v python &> /dev/null; then
    echo "❌ Python을 찾을 수 없습니다"
    exit 1
fi

# CUDA 확인
python -c "import torch; print(f'CUDA 사용가능: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch 또는 CUDA를 확인할 수 없습니다"
    echo "   DevContainer에서 실행하고 있는지 확인하세요"
    exit 1
fi

echo ""
echo "🔍 환경 정보:"
python -c "
import torch
print(f'  - PyTorch: {torch.__version__}')
print(f'  - CUDA: {torch.version.cuda}')
print(f'  - GPU 개수: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  - GPU 이름: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "📋 Phase 1 테스트 항목:"
echo "  1. 환경 검증 (CUDA, GPU 메모리)"
echo "  2. HybridConfigManager 동작 확인"
echo "  3. 테스트 비디오 생성 (1080p)"
echo "  4. NvDecoder 초기화"
echo "  5. 단일 프레임 디코딩"
echo "  6. 배치 프레임 디코딩"
echo "  7. 색공간 변환 (NV12 → RGB)"
echo "  8. GPU 메모리 모니터링"
echo "  9. 성능 측정"
echo ""

# 테스트 실행
echo "🧪 테스트 시작..."
python test_single_stream.py

# 결과 확인
TEST_RESULT=$?

echo ""
echo "========================================"

if [ $TEST_RESULT -eq 0 ]; then
    echo "🎉 Phase 1 테스트 완료!"
    echo ""
    echo "📊 결과: 모든 테스트 통과"
    echo "📈 진행률: 70% → 100%"
    echo ""
    echo "✅ Phase 1 성공 기준 달성:"
    echo "  - 환경 검증 완료"
    echo "  - GPU 하드웨어 가속 동작"
    echo "  - DevContainer 환경 검증"
    echo "  - 1080p NVDEC 디코딩 성공"
    echo ""
    echo "🚀 Phase 2 진행 준비 완료!"
    echo "   다음 단계: TensorRT 추론 엔진 구현"
else
    echo "❌ Phase 1 테스트 실패"
    echo ""
    echo "💡 문제 해결 방법:"
    echo "  1. dual_face_tracker.log 파일 확인"
    echo "  2. DevContainer 환경 재시작: docker system prune && ./run_dev.sh"
    echo "  3. GPU 상태 확인: nvidia-smi"
    echo "  4. PyAV 코덱 확인: python check_av_codecs.py"
fi

echo ""
echo "📄 로그 파일 위치: dual_face_tracker.log"
echo "🔧 DevContainer 재시작: ./run_dev.sh"

exit $TEST_RESULT