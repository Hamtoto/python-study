#!/usr/bin/env python3
"""
CUDA/TensorRT 환경 진단 도구
D7.1 이슈 해결을 위한 종합 진단
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description="", capture_output=True):
    """명령 실행 및 결과 반환"""
    print(f"\n🔍 {description}")
    print(f"$ {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, 
                              text=True, timeout=30)
        
        if capture_output:
            if result.returncode == 0:
                print(f"✅ Success")
                if result.stdout:
                    print(result.stdout.strip())
                return True, result.stdout
            else:
                print(f"❌ Failed (return code: {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr.strip()}")
                return False, result.stderr
        else:
            return result.returncode == 0, ""
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout after 30 seconds")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, str(e)

def check_system_info():
    """시스템 기본 정보 확인"""
    print("=" * 60)
    print("🖥️  시스템 정보")
    print("=" * 60)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # GPU 확인
    success, output = run_command("lspci | grep -i nvidia", "NVIDIA GPU 확인")
    if success and output:
        print(f"GPU 발견: {len(output.splitlines())} 개")
    
    return True

def check_nvidia_driver():
    """NVIDIA 드라이버 확인"""
    print("\n=" * 60)
    print("🚗 NVIDIA 드라이버")
    print("=" * 60)
    
    # nvidia-smi 실행
    success, output = run_command("nvidia-smi", "nvidia-smi 실행")
    if not success:
        print("❌ nvidia-smi 실행 실패 - NVIDIA 드라이버 문제")
        return False
    
    # 드라이버 버전 추출
    if "Driver Version:" in output:
        lines = output.split('\n')
        for line in lines:
            if "Driver Version:" in line:
                print(f"드라이버 버전 확인됨: {line.strip()}")
                break
    
    # CUDA 버전 확인
    if "CUDA Version:" in output:
        lines = output.split('\n')
        for line in lines:
            if "CUDA Version:" in line:
                print(f"CUDA 버전 확인됨: {line.strip()}")
                break
    
    return True

def check_cuda_installation():
    """CUDA 설치 확인"""
    print("\n=" * 60)
    print("⚡ CUDA 설치")
    print("=" * 60)
    
    # nvcc 확인
    success, output = run_command("nvcc --version", "CUDA Compiler (nvcc) 확인")
    if success:
        print("CUDA 컴파일러 정상")
    else:
        print("⚠️ CUDA 컴파일러 없음 - 런타임만 설치됨")
    
    # CUDA 경로 확인
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda"
    ]
    
    print("\nCUDA 설치 경로 확인:")
    for path in cuda_paths:
        if Path(path).exists():
            print(f"✅ {path} 존재")
            lib_path = Path(path) / "lib64"
            if lib_path.exists():
                print(f"  └─ {lib_path} 존재")
        else:
            print(f"❌ {path} 없음")
    
    # 환경 변수 확인
    print(f"\nCUDA 환경 변수:")
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    
    print(f"CUDA_HOME: {cuda_home}")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"LD_LIBRARY_PATH: {ld_path}")
    
    return True

def check_python_packages():
    """Python 패키지 확인"""
    print("\n=" * 60)
    print("🐍 Python 패키지")
    print("=" * 60)
    
    packages_to_check = [
        "torch",
        "torchvision", 
        "tensorrt",
        "onnx",
        "onnxruntime-gpu",
        "ultralytics",
        "numpy"
    ]
    
    for package in packages_to_check:
        try:
            if package == "torch":
                import torch
                print(f"✅ torch {torch.__version__}")
                print(f"   CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   CUDA version: {torch.version.cuda}")
                    print(f"   GPU count: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                        
            elif package == "tensorrt":
                import tensorrt as trt
                print(f"✅ tensorrt {trt.__version__}")
                
            elif package == "onnxruntime-gpu":
                import onnxruntime as ort
                print(f"✅ onnxruntime {ort.__version__}")
                print(f"   Available providers: {ort.get_available_providers()}")
                
            else:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    print(f"✅ {package} {module.__version__}")
                else:
                    print(f"✅ {package} (no version info)")
                    
        except ImportError:
            print(f"❌ {package} not installed")
        except Exception as e:
            print(f"⚠️ {package} error: {e}")
    
    return True

def check_cuda_runtime():
    """CUDA 런타임 테스트"""
    print("\n=" * 60)
    print("🧪 CUDA 런타임 테스트")
    print("=" * 60)
    
    try:
        import torch
        
        # 기본 CUDA 테스트
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA 사용 가능")
            
            # 간단한 텐서 연산
            device = torch.device('cuda:0')
            print(f"사용 중인 디바이스: {device}")
            
            # 메모리 테스트
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print(f"✅ GPU 텐서 연산 성공 - 결과 shape: {z.shape}")
            
            # GPU 메모리 정보
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU 메모리 - 할당: {memory_allocated:.2f}GB, 예약: {memory_reserved:.2f}GB")
            
            return True
        else:
            print("❌ PyTorch CUDA 사용 불가")
            return False
            
    except Exception as e:
        print(f"❌ CUDA 런타임 테스트 실패: {e}")
        return False

def check_tensorrt():
    """TensorRT 상세 확인"""
    print("\n=" * 60)
    print("🚀 TensorRT 진단")
    print("=" * 60)
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT 버전: {trt.__version__}")
        
        # TensorRT Logger 생성 테스트
        logger = trt.Logger(trt.Logger.INFO)
        print("✅ TensorRT Logger 생성 성공")
        
        # Builder 생성 테스트
        try:
            builder = trt.Builder(logger)
            print("✅ TensorRT Builder 생성 성공")
            
            # Runtime 생성 테스트
            runtime = trt.Runtime(logger)
            print("✅ TensorRT Runtime 생성 성공")
            
            return True
            
        except Exception as e:
            print(f"❌ TensorRT Builder/Runtime 생성 실패: {e}")
            print("   이것은 CUDA Error 35의 원인일 수 있습니다")
            return False
            
    except ImportError:
        print("❌ TensorRT 패키지 없음")
        return False
    except Exception as e:
        print(f"❌ TensorRT 테스트 실패: {e}")
        return False

def suggest_solutions():
    """해결 방안 제시"""
    print("\n" + "=" * 60)
    print("💡 해결 방안")
    print("=" * 60)
    
    print("""
🔧 CUDA Error 35 해결 방안:

1. **NVIDIA 드라이버 재설치**:
   sudo apt purge nvidia-*
   sudo apt install nvidia-driver-525  # 또는 최신 버전
   sudo reboot

2. **CUDA 런타임 재설치**:
   # Docker 환경에서는 호스트의 CUDA와 버전 맞추기
   pip uninstall tensorrt
   pip install tensorrt==10.5.0  # CUDA 12.8 호환

3. **대안: ONNX Runtime 사용**:
   # TensorRT 대신 ONNX Runtime으로 Phase 2 진행
   # 성능은 약간 낮지만 안정적

4. **환경 변수 설정**:
   export CUDA_VISIBLE_DEVICES=0
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

5. **DevContainer 재시작**:
   # GPU 권한 문제일 수 있음
   docker system prune
   ./run_dev.sh
""")

def main():
    """메인 진단 함수"""
    print("🔬 CUDA/TensorRT 환경 종합 진단")
    print("D7.1 이슈 해결을 위한 시스템 분석")
    
    results = {}
    
    # 1. 시스템 정보
    results['system'] = check_system_info()
    
    # 2. NVIDIA 드라이버
    results['driver'] = check_nvidia_driver()
    
    # 3. CUDA 설치
    results['cuda'] = check_cuda_installation()
    
    # 4. Python 패키지
    results['packages'] = check_python_packages()
    
    # 5. CUDA 런타임
    results['runtime'] = check_cuda_runtime()
    
    # 6. TensorRT
    results['tensorrt'] = check_tensorrt()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 진단 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name.ljust(10)}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체: {passed}/{total} 통과")
    
    if not results.get('tensorrt', False):
        print("\n⚠️ TensorRT 실패 - ONNX Runtime으로 대안 진행 권장")
    
    # 해결 방안 제시
    suggest_solutions()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)