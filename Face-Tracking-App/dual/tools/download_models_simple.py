#!/usr/bin/env python3
"""
간단한 모델 다운로드 스크립트 (최소 의존성)
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# 모델 다운로드 URL
MODELS = {
    "yolov8n-face": {
        "url": "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt",
        "description": "YOLOv8n face detection model (lightweight)"
    },
    "scrfd_500m": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/scrfd_500m_bnkps.onnx",
        "description": "SCRFD 500M parameters model"
    }
}

def download_file(url: str, dest_path: Path) -> bool:
    """URL에서 파일을 다운로드합니다."""
    try:
        if dest_path.exists():
            print(f"✅ {dest_path.name} already exists")
            return True
            
        print(f"📥 Downloading: {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {dest_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    """메인 함수"""
    print("🚀 Starting model download...")
    
    # 디렉토리 생성
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # 선택된 모델
    selected_models = ["yolov8n-face", "scrfd_500m"]
    
    print(f"Selected models: {', '.join(selected_models)}")
    
    success_count = 0
    
    for model_name in selected_models:
        if model_name not in MODELS:
            print(f"⚠️ Unknown model: {model_name}")
            continue
            
        model_info = MODELS[model_name]
        
        # 파일 확장자 결정
        if model_info["url"].endswith(".onnx"):
            file_ext = ".onnx"
        elif model_info["url"].endswith(".pt"):
            file_ext = ".pt"
        else:
            file_ext = ""
            
        dest_path = weights_dir / f"{model_name}{file_ext}"
        
        print(f"\n📦 {model_name}: {model_info['description']}")
        
        if download_file(model_info["url"], dest_path):
            success_count += 1
    
    # 결과
    print(f"\n📊 Results: {success_count}/{len(selected_models)} successful")
    
    if success_count == len(selected_models):
        print("✅ All models downloaded successfully!")
        
        # 다운로드된 파일 목록
        print("\n📁 Downloaded files:")
        for file_path in weights_dir.glob("*"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file_path.name} ({size_mb:.1f} MB)")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())