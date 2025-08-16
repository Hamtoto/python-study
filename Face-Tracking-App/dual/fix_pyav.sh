#!/bin/bash
# PyAV 설치 문제 해결 스크립트
# DevContainer 내부에서 실행

set -e

echo "🔧 PyAV 설치 문제 해결 시작"
echo "=================================================="

# 현재 환경 확인
echo "📋 현재 환경 상태:"
echo "  • Python: $(which python3)"
echo "  • pip: $(which pip)"
echo "  • Virtual env: $VIRTUAL_ENV"

# PyAV 설치 상태 확인
echo ""
echo "🔍 PyAV 현재 상태 확인..."
if python3 -c "import av; print(f'✅ PyAV: {av.__version__}')" 2>/dev/null; then
    echo "  ✅ PyAV 이미 정상 설치됨"
    exit 0
else
    echo "  ❌ PyAV 설치되지 않음 또는 인식 불가"
fi

# 필수 시스템 라이브러리 확인
echo ""
echo "🔍 시스템 FFmpeg 라이브러리 확인..."
MISSING_LIBS=()

for lib in libavcodec libavformat libavutil libswscale; do
    if ! pkg-config --exists $lib; then
        MISSING_LIBS+=($lib)
    fi
done

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    echo "  ❌ 누락된 라이브러리: ${MISSING_LIBS[*]}"
    echo "  📦 FFmpeg 개발 라이브러리 설치 중..."
    apt-get update -qq
    apt-get install -y \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
        libavdevice-dev
else
    echo "  ✅ FFmpeg 라이브러리 모두 설치됨"
fi

# venv 환경 확인
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "⚠️ Virtual environment 비활성화 상태"
    echo "🔧 venv 활성화 중..."
    source /opt/venv/bin/activate
    export PATH="/opt/venv/bin:$PATH"
fi

echo ""
echo "📋 최종 환경 상태:"
echo "  • Python: $(which python3)"
echo "  • pip: $(which pip)"
echo "  • Virtual env: $VIRTUAL_ENV"

# 기존 PyAV 제거 (있다면)
echo ""
echo "🧹 기존 PyAV 제거..."
pip uninstall -y av 2>/dev/null || echo "  (기존 PyAV 없음)"

# PyAV 재설치 (소스 빌드)
echo ""
echo "📦 PyAV 11.0.0 소스 빌드 설치..."
echo "  (NVENC/NVDEC 하드웨어 가속 지원)"
echo "  ⏱️ 약 2-3분 소요될 수 있습니다..."

# 소스 빌드를 위한 임시 패키지 설치
pip install --no-cache-dir \
    wheel \
    setuptools \
    cython

# PyAV 소스 빌드 (--no-binary로 강제)
pip install av==11.0.0 --no-binary av --no-cache-dir --verbose

# 설치 확인
echo ""
echo "🧪 PyAV 설치 확인..."
if python3 -c "import av; print(f'✅ PyAV 설치 성공: {av.__version__}')" 2>/dev/null; then
    echo "  ✅ PyAV 정상 설치 및 인식됨"
else
    echo "  ❌ PyAV 설치 실패"
    echo ""
    echo "🔧 대체 방법 시도 중..."
    
    # 대체 방법 1: --force-reinstall
    echo "  • 강제 재설치 시도..."
    pip install av==11.0.0 --no-binary av --force-reinstall --no-cache-dir
    
    if python3 -c "import av; print('✅ PyAV 강제 재설치 성공')" 2>/dev/null; then
        echo "  ✅ 강제 재설치 성공"
    else
        echo "  ❌ 강제 재설치도 실패"
        
        # 대체 방법 2: 최신 버전 시도
        echo "  • 최신 버전 설치 시도..."
        pip install av --no-binary av --no-cache-dir
        
        if python3 -c "import av; print(f'✅ PyAV 최신 버전 설치 성공: {av.__version__}')" 2>/dev/null; then
            echo "  ✅ 최신 버전 설치 성공"
        else
            echo "  ❌ 모든 시도 실패"
            echo ""
            echo "🚨 PyAV 설치 실패 - 수동 해결 필요:"
            echo "   1. DevContainer 재빌드: Ctrl+Shift+P → 'Rebuild Container'"
            echo "   2. 또는 시스템 FFmpeg 버전 확인: ffmpeg -version"
            echo "   3. 또는 Python 버전 호환성 확인"
            exit 1
        fi
    fi
fi

# 하드웨어 가속 코덱 확인
echo ""
echo "🎬 하드웨어 가속 코덱 확인..."
python3 -c "
import av
print(f'PyAV 버전: {av.__version__}')
print(f'사용 가능한 코덱 수: {len(av.codec.codecs_available)}')

# NVENC/NVDEC 코덱 확인
hw_codecs = [c for c in av.codec.codecs_available if 'nv' in c.lower() or 'cuda' in c.lower()]
if hw_codecs:
    print(f'하드웨어 가속 코덱: {len(hw_codecs)}개')
    for codec in sorted(hw_codecs)[:10]:  # 처음 10개만
        print(f'  - {codec}')
    if len(hw_codecs) > 10:
        print(f'  ... 및 {len(hw_codecs)-10}개 더')
else:
    print('⚠️ 하드웨어 가속 코덱 없음 (소프트웨어 처리만 가능)')
"

echo ""
echo "=================================================="
echo "✅ PyAV 설치 문제 해결 완료!"
echo "   이제 Phase 3 테스트를 실행할 수 있습니다:"
echo "   ./run_phase3_test.sh"
echo "=================================================="