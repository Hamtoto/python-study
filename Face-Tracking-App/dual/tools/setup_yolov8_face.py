#!/usr/bin/env python3
"""
YOLOv8 Face Detection 모델 설정 스크립트
Ultralytics YOLOv8을 설치하고 face detection 모델을 다운로드/변환합니다.
"""

import os
import sys
import subprocess
from pathlib import Path
import requests

def run_command(cmd, description=""):
    """명령 실행 헬퍼"""
    if description:
        print(f"\n📌 {description}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def install_ultralytics():
    """Ultralytics 패키지 설치"""
    print("🚀 Installing Ultralytics YOLOv8...")
    
    # Ultralytics 설치
    if not run_command("pip install ultralytics", "Installing ultralytics package"):
        return False
    
    # 추가 의존성
    deps = [
        "onnx",
        "onnxruntime-gpu",  # GPU 지원
        "onnx-simplifier",
        "tensorrt"  # TensorRT는 별도 설치 필요할 수 있음
    ]
    
    for dep in deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    return True

def download_yolov8_face_models():
    """YOLOv8 face detection 모델 다운로드"""
    print("\n📥 Downloading YOLOv8 face detection models...")
    
    # 디렉토리 생성
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # GitHub에서 직접 다운로드 가능한 모델들
    # akanametov/yolo-face 리포지토리 기반
    models_to_download = {
        "yolov8n-face": {
            "url": "https://github.com/akanametov/yolo-face/releases/download/v0.1.0/yolov8n-face.pt",
            "description": "YOLOv8 nano face detection (fastest)"
        },
        "yolov8s-face": {
            "url": "https://github.com/akanametov/yolo-face/releases/download/v0.1.0/yolov8s-face.pt",
            "description": "YOLOv8 small face detection (balanced)"
        }
    }
    
    downloaded_models = []
    
    for model_name, info in models_to_download.items():
        pt_path = weights_dir / f"{model_name}.pt"
        
        # 이미 존재하면 스킵
        if pt_path.exists():
            print(f"✅ {model_name}.pt already exists")
            downloaded_models.append(pt_path)
            continue
        
        print(f"\nDownloading {model_name}...")
        print(f"  {info['description']}")
        
        try:
            # 다운로드
            response = requests.get(info["url"], stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(pt_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r  Progress: {percent:.1f}%", end="")
            
            print(f"\n✅ Downloaded: {pt_path}")
            downloaded_models.append(pt_path)
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            # 대안: ultralytics 기본 모델 사용
            print("  Trying alternative: using base YOLOv8 model...")
            alternative_cmd = f"yolo task=detect mode=download model=yolov8n.pt"
            if run_command(alternative_cmd):
                downloaded_models.append(Path("yolov8n.pt"))
    
    return downloaded_models

def export_to_onnx(model_paths):
    """PyTorch 모델을 ONNX로 변환"""
    print("\n🔄 Exporting models to ONNX...")
    
    models_dir = Path("models")
    exported_models = []
    
    for pt_path in model_paths:
        if not pt_path.exists():
            print(f"⚠️ Model not found: {pt_path}")
            continue
        
        onnx_path = models_dir / f"{pt_path.stem}.onnx"
        
        # 이미 존재하면 스킵
        if onnx_path.exists():
            print(f"✅ {onnx_path.name} already exists")
            exported_models.append(onnx_path)
            continue
        
        print(f"\nExporting {pt_path.name} to ONNX...")
        
        # Ultralytics CLI 사용
        export_cmd = f"yolo export model={pt_path} format=onnx opset=16 simplify=True imgsz=640"
        
        if run_command(export_cmd, "Using YOLO export command"):
            # 생성된 ONNX 파일 이동
            expected_onnx = pt_path.with_suffix('.onnx')
            if expected_onnx.exists():
                import shutil
                shutil.move(str(expected_onnx), str(onnx_path))
                print(f"✅ Exported: {onnx_path}")
                exported_models.append(onnx_path)
        else:
            print(f"❌ Failed to export {pt_path.name}")
            
            # 대안: Python API 사용
            print("  Trying Python API...")
            try:
                from ultralytics import YOLO
                model = YOLO(str(pt_path))
                model.export(format="onnx", opset=16, simplify=True, imgsz=640)
                
                expected_onnx = pt_path.with_suffix('.onnx')
                if expected_onnx.exists():
                    import shutil
                    shutil.move(str(expected_onnx), str(onnx_path))
                    print(f"✅ Exported via Python API: {onnx_path}")
                    exported_models.append(onnx_path)
            except Exception as e:
                print(f"❌ Python API also failed: {e}")
    
    return exported_models

def export_to_tensorrt(onnx_paths):
    """ONNX 모델을 TensorRT로 변환"""
    print("\n🚀 Exporting models to TensorRT...")
    
    engines_dir = Path("engines")
    engines_dir.mkdir(exist_ok=True)
    
    exported_engines = []
    
    for onnx_path in onnx_paths:
        if not onnx_path.exists():
            print(f"⚠️ ONNX model not found: {onnx_path}")
            continue
        
        engine_path = engines_dir / f"{onnx_path.stem}_fp16.engine"
        
        # 이미 존재하면 스킵
        if engine_path.exists():
            print(f"✅ {engine_path.name} already exists")
            exported_engines.append(engine_path)
            continue
        
        print(f"\nExporting {onnx_path.name} to TensorRT...")
        
        # TensorRT 변환 스크립트 사용
        trt_cmd = f"python tools/convert_to_tensorrt.py --onnx {onnx_path} --engine {engine_path} --precision fp16 --max-batch 8"
        
        if run_command(trt_cmd, "Using TensorRT conversion script"):
            print(f"✅ Exported: {engine_path}")
            exported_engines.append(engine_path)
        else:
            print(f"⚠️ TensorRT conversion failed for {onnx_path.name}")
            print("  This is optional - ONNX model can still be used")
    
    return exported_engines

def test_models():
    """모델 테스트"""
    print("\n🧪 Testing models...")
    
    # 테스트 이미지 생성 또는 다운로드
    test_image = Path("test_face.jpg")
    
    if not test_image.exists():
        print("Downloading test image...")
        test_url = "https://ultralytics.com/images/bus.jpg"
        try:
            response = requests.get(test_url)
            response.raise_for_status()
            with open(test_image, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded test image: {test_image}")
        except:
            print("⚠️ Could not download test image")
            return
    
    # ONNX 모델 테스트
    models_dir = Path("models")
    for onnx_file in models_dir.glob("*.onnx"):
        print(f"\nTesting {onnx_file.name}...")
        test_cmd = f"yolo predict model={onnx_file} source={test_image} save=True"
        if run_command(test_cmd):
            print(f"✅ {onnx_file.name} works!")
        else:
            print(f"⚠️ {onnx_file.name} test failed")

def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 YOLOv8 Face Detection Setup")
    print("=" * 60)
    
    # 1. Ultralytics 설치
    if not install_ultralytics():
        print("❌ Failed to install ultralytics")
        return 1
    
    # 2. Face detection 모델 다운로드
    pt_models = download_yolov8_face_models()
    
    if not pt_models:
        print("❌ No models downloaded")
        # 기본 YOLOv8 모델 사용
        print("Using default YOLOv8 model...")
        run_command("yolo task=detect mode=download model=yolov8n.pt")
        pt_models = [Path("yolov8n.pt")]
    
    # 3. ONNX 변환
    onnx_models = export_to_onnx(pt_models)
    
    if not onnx_models:
        print("⚠️ No ONNX models created")
    
    # 4. TensorRT 변환 (선택적)
    if onnx_models:
        trt_engines = export_to_tensorrt(onnx_models)
        if trt_engines:
            print(f"\n✅ Created {len(trt_engines)} TensorRT engines")
    
    # 5. 테스트
    if onnx_models:
        test_models()
    
    # 6. 요약
    print("\n" + "=" * 60)
    print("📊 Setup Summary")
    print("=" * 60)
    
    print(f"✅ PyTorch models: {len(pt_models)}")
    for p in pt_models:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"   - {p.name} ({size_mb:.1f} MB)")
    
    print(f"\n✅ ONNX models: {len(onnx_models)}")
    for p in onnx_models:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"   - {p.name} ({size_mb:.1f} MB)")
    
    engines_dir = Path("engines")
    if engines_dir.exists():
        engines = list(engines_dir.glob("*.engine"))
        if engines:
            print(f"\n✅ TensorRT engines: {len(engines)}")
            for p in engines:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"   - {p.name} ({size_mb:.1f} MB)")
    
    print("\n🎉 Setup complete! You can now use these models for face detection.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())