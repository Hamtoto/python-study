#!/usr/bin/env python3
"""
Ultralytics API를 사용한 TensorRT 변환
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    print("🚀 Exporting YOLOv8 to TensorRT...")
    
    # 엔진 디렉토리 생성
    engines_dir = Path("engines")
    engines_dir.mkdir(exist_ok=True)
    
    models_to_export = ["yolov8n.pt", "yolov8s.pt"]
    
    for model_name in models_to_export:
        if not Path(model_name).exists():
            print(f"⚠️ {model_name} not found, skipping...")
            continue
            
        print(f"\n📦 Exporting {model_name}...")
        
        try:
            # 모델 로드
            model = YOLO(model_name)
            
            # TensorRT 엔진으로 변환
            model.export(
                format="engine",
                half=True,  # FP16 정밀도
                dynamic=True,  # 동적 배치
                batch=8,  # 최대 배치 크기
                workspace=4,  # 4GB workspace
                verbose=True
            )
            
            # 생성된 엔진을 engines 디렉토리로 이동
            engine_name = model_name.replace('.pt', '.engine')
            if Path(engine_name).exists():
                import shutil
                dest = engines_dir / engine_name
                shutil.move(engine_name, dest)
                
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"✅ Engine saved: {dest} ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"❌ Failed to export {model_name}: {e}")
            print("  Note: TensorRT export requires NVIDIA GPU and CUDA")
    
    # 결과 요약
    print("\n📊 Export Summary:")
    if engines_dir.exists():
        engines = list(engines_dir.glob("*.engine"))
        print(f"TensorRT engines: {len(engines)}")
        for engine in engines:
            size_mb = engine.stat().st_size / (1024 * 1024)
            print(f"  - {engine.name} ({size_mb:.1f} MB)")
    else:
        print("No engines created")
    
    print("\n💡 If TensorRT export failed, you can still use ONNX models for Phase 2")

if __name__ == "__main__":
    main()