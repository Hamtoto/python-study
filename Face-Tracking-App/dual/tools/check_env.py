#!/usr/bin/env python3
"""
환경 검증 스크립트
DevContainer 환경의 OpenCV, CUDA, cuDNN 자동 검증 및 수정
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_env():
    """Python 환경 확인"""
    print("🐍 Python 환경 확인:")
    print(f"   • Python 버전: {sys.version}")
    print(f"   • Python 경로: {sys.executable}")
    print(f"   • Virtual ENV: {os.environ.get('VIRTUAL_ENV', '없음')}")
    return True


def check_opencv():
    """OpenCV 확인 및 자동 수정"""
    print("\n🎨 OpenCV 확인:")
    
    try:
        import cv2
        print(f"   • ✅ OpenCV 버전: {cv2.__version__}")
        
        # CUDA 지원 확인
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"   • ✅ CUDA 지원: {cuda_devices}개 디바이스")
            
            # 간단한 CUDA 작업 테스트
            test_mat = cv2.cuda.GpuMat(100, 100, cv2.CV_8UC3)
            test_mat.setTo((255, 255, 255))
            print("   • ✅ CUDA 기능 테스트: 성공")
            
        except Exception as cuda_error:
            print(f"   • ⚠️ CUDA 기능 오류: {cuda_error}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"   • ❌ OpenCV import 실패: {e}")
        
        # 자동 수정 시도
        print("   • 🔧 자동 수정 시도 중...")
        opencv_paths = [
            "/usr/lib/python3.10/dist-packages/cv2.so",
            "/usr/local/lib/python3.10/site-packages/cv2.so"
        ]
        
        venv_site_packages = Path(sys.executable).parent.parent / "lib/python3.10/site-packages"
        
        for opencv_path in opencv_paths:
            if Path(opencv_path).exists():
                try:
                    subprocess.run([
                        "cp", "-r", 
                        str(Path(opencv_path).parent / "cv2*"),
                        str(venv_site_packages)
                    ], check=True)
                    print(f"   • ✅ OpenCV 복사 완료: {opencv_path} → venv")
                    return check_opencv()  # 재귀 호출로 재검증
                except subprocess.CalledProcessError:
                    continue
        
        print("   • ❌ 자동 수정 실패")
        return False


def check_pytorch():
    """PyTorch CUDA 확인"""
    print("\n🔥 PyTorch CUDA 확인:")
    
    try:
        import torch
        print(f"   • ✅ PyTorch 버전: {torch.__version__}")
        print(f"   • ✅ CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   • ✅ CUDA 디바이스 수: {torch.cuda.device_count()}")
            print(f"   • ✅ 현재 디바이스: {torch.cuda.get_device_name(0)}")
        
        return True
        
    except ImportError as e:
        print(f"   • ❌ PyTorch import 실패: {e}")
        return False


def check_cudnn():
    """cuDNN 라이브러리 확인"""
    print("\n🧠 cuDNN 라이브러리 확인:")
    
    cudnn_paths = [
        "/usr/local/cuda/lib64/libcudnn.so",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so",
        "/usr/local/cuda/lib64/libcudnn.so.9"
    ]
    
    found_cudnn = False
    for path in cudnn_paths:
        if Path(path).exists():
            print(f"   • ✅ cuDNN 라이브러리 발견: {path}")
            found_cudnn = True
            break
    
    if not found_cudnn:
        print("   • ⚠️ cuDNN 라이브러리를 찾을 수 없음")
        
    # LD_LIBRARY_PATH 확인
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if '/usr/local/cuda/lib64' in ld_library_path:
        print("   • ✅ LD_LIBRARY_PATH에 CUDA 경로 포함됨")
    else:
        print("   • ⚠️ LD_LIBRARY_PATH에 CUDA 경로 없음")
        print("   • 💡 다음 명령어 실행: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    
    return found_cudnn


def check_gpu_pipeline():
    """GPU 파이프라인 간단 테스트"""
    print("\n🚀 GPU 파이프라인 테스트:")
    
    try:
        import cv2
        import torch
        import numpy as np
        
        # 1. NumPy → OpenCV CUDA 테스트
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gpu_mat = cv2.cuda.GpuMat()
        gpu_mat.upload(test_image)
        
        # 2. GPU 리사이즈 테스트
        resized = cv2.cuda.resize(gpu_mat, (320, 240))
        
        # 3. GPU → CPU 다운로드 테스트
        result = resized.download()
        
        print(f"   • ✅ 파이프라인 테스트: {test_image.shape} → {result.shape}")
        return True
        
    except Exception as e:
        print(f"   • ❌ 파이프라인 테스트 실패: {e}")
        return False


def main():
    """메인 검증 함수"""
    print("🔍 DevContainer 환경 검증 시작")
    print("=" * 50)
    
    results = {
        "python": check_python_env(),
        "opencv": check_opencv(),
        "pytorch": check_pytorch(),
        "cudnn": check_cudnn(),
        "pipeline": check_gpu_pipeline()
    }
    
    print("\n📊 검증 결과 요약:")
    print("=" * 50)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for component, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"   • {component.upper()}: {status}")
    
    success_rate = (passed_checks / total_checks) * 100
    print(f"\n🎯 전체 성공률: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ 환경이 정상적으로 설정되었습니다!")
        return 0
    else:
        print("⚠️ 환경 설정에 문제가 있습니다.")
        print("💡 run_dev.sh를 다시 실행하거나 Dockerfile을 재빌드하세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())