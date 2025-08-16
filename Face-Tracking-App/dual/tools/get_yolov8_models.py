#!/usr/bin/env python3
"""
YOLOv8 모델 다운로드 및 변환 (올바른 방법)
Context7에서 확인한 정확한 Ultralytics 명령 사용
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, desc=""):
    """명령 실행"""
    if desc:
        print(f"\n📌 {desc}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def main():
    print("🚀 Getting YOLOv8 models the correct way...")
    
    # 1. 기본 YOLOv8 모델 다운로드 (자동으로 다운로드됨)
    print("\n📥 Getting base YOLOv8 model...")
    from ultralytics import YOLO
    
    # 모델 로드 시 자동으로 다운로드됨
    models_to_get = ["yolov8n.pt", "yolov8s.pt"]
    
    for model_name in models_to_get:
        print(f"\nGetting {model_name}...")
        try:
            model = YOLO(model_name)  # 자동 다운로드
            print(f"✅ {model_name} downloaded successfully")
            
            # 크기 확인
            model_path = Path(model_name)
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"   Size: {size_mb:.1f} MB")
        except Exception as e:
            print(f"❌ Failed to get {model_name}: {e}")
    
    # 2. ONNX 변환
    print("\n🔄 Exporting to ONNX...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_get:
        if Path(model_name).exists():
            print(f"\nExporting {model_name} to ONNX...")
            export_cmd = f"yolo export model={model_name} format=onnx opset=16 simplify=True imgsz=640"
            if run_cmd(export_cmd):
                # 생성된 ONNX 파일을 models 디렉토리로 이동
                onnx_name = model_name.replace('.pt', '.onnx')
                if Path(onnx_name).exists():
                    import shutil
                    dest = models_dir / onnx_name
                    shutil.move(onnx_name, dest)
                    print(f"✅ Moved to {dest}")
    
    # 3. 테스트
    print("\n🧪 Testing models...")
    test_cmd = "yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' save=True"
    if run_cmd(test_cmd, "Testing YOLOv8n"):
        print("✅ Model works!")
    
    # 4. 요약
    print("\n📊 Summary:")
    
    # PT 파일들
    pt_files = list(Path(".").glob("*.pt"))
    print(f"PyTorch models: {len(pt_files)}")
    for f in pt_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    # ONNX 파일들
    if models_dir.exists():
        onnx_files = list(models_dir.glob("*.onnx"))
        print(f"\nONNX models: {len(onnx_files)}")
        for f in onnx_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    main()