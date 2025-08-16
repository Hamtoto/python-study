#!/bin/bash

# Phase 4 컴포넌트 테스트 실행 스크립트

echo "🚀 Phase 4 운영화 컴포넌트 테스트 시작"
echo "================================================================================"
echo "📋 테스트 범위:"
echo "   • HardwareMonitor (CSV 기반 모니터링)"
echo "   • PerformanceReporter (리포트 생성)"
echo "   • StreamRecoveryManager (자동 복구)"
echo "   • MemoryManager (VRAM 관리)"
echo "   • ErrorHandlerRegistry (에러 처리)"
echo "   • AutoTuner (자동 최적화)"
echo "   • ProductionTestSuite (프로덕션 테스트)"
echo "================================================================================"
echo

# 현재 작업 디렉토리 확인
echo "📁 현재 작업 디렉토리: $(pwd)"
echo "🐍 Python 버전: $(python3 --version)"
echo

# GPU 확인
echo "🖥️ GPU 상태 확인:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
else
    echo "   ⚠️ nvidia-smi를 찾을 수 없습니다"
fi
echo

# PyTorch CUDA 확인
echo "🔥 PyTorch CUDA 확인:"
python3 -c "
try:
    import torch
    print(f'   ✅ PyTorch 버전: {torch.__version__}')
    print(f'   🔥 CUDA 사용 가능: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   🖥️ GPU 디바이스 수: {torch.cuda.device_count()}')
        print(f'   📝 현재 디바이스: {torch.cuda.current_device()}')
except ImportError:
    print('   ❌ PyTorch가 설치되지 않았습니다')
"
echo

# 필수 라이브러리 확인
echo "📚 필수 라이브러리 확인:"
python3 -c "
import sys
libraries = ['psutil', 'pathlib', 'asyncio', 'dataclasses', 'collections', 'threading']
for lib in libraries:
    try:
        __import__(lib)
        print(f'   ✅ {lib}')
    except ImportError:
        print(f'   ❌ {lib} - 설치 필요')
"
echo

# 테스트 실행
echo "🧪 Phase 4 컴포넌트 테스트 실행..."
echo "⏱️ 예상 소요 시간: 1-2분"
echo

# Python 테스트 스크립트 실행
if python3 test_phase4_components.py; then
    echo
    echo "✅ Phase 4 컴포넌트 테스트 성공!"
    echo
    
    # 결과 파일 확인
    echo "📁 생성된 테스트 결과:"
    if [ -d "phase4_test_results" ]; then
        find phase4_test_results -type f -name "*.log" -o -name "*.txt" -o -name "*.json" -o -name "*.csv" | head -10 | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo "   📄 $file ($size)"
        done
        
        echo
        echo "💡 결과 확인 방법:"
        echo "   📊 모니터링 로그: phase4_test_results/monitoring/"
        echo "   📋 성능 리포트: phase4_test_results/performance/"
        echo "   🧪 테스트 결과: phase4_test_results/test_results/"
    fi
    
    echo
    echo "🎉 Phase 4 운영화 시스템 구현 완료!"
    echo "   • 실시간 모니터링 ✅"
    echo "   • 자동 복구 시스템 ✅"
    echo "   • 성능 최적화 ✅"
    echo "   • 프로덕션 테스트 ✅"
    
else
    echo
    echo "❌ Phase 4 컴포넌트 테스트 실패"
    echo "   🔍 상세 에러는 위의 로그를 확인해주세요"
    exit 1
fi

echo
echo "================================================================================"
echo "🎯 Phase 4 완료 - 프로덕션 준비 완료!"
echo "================================================================================"