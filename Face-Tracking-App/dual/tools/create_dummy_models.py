#!/usr/bin/env python3
"""
테스트용 더미 모델 생성 스크립트
실제 모델 대신 개발/테스트용 더미 ONNX 모델을 생성합니다.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import onnx

class DummyFaceDetector(nn.Module):
    """더미 얼굴 검출 모델"""
    
    def __init__(self, num_classes=1):
        super().__init__()
        # 간단한 CNN 구조
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Detection head
        self.detection_head = nn.Conv2d(128, 5 + num_classes, 1)  # x,y,w,h,conf,class
        
    def forward(self, x):
        # 간단한 forward pass
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        
        x = torch.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        
        # Detection output
        out = self.detection_head(x)
        
        # Reshape to (batch, num_predictions, 6)
        batch_size = out.shape[0]
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(batch_size, -1, 6)
        
        return out

class DummyReIDModel(nn.Module):
    """더미 ReID 모델"""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        # 간단한 CNN + FC 구조
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Global average pooling 후 FC
        self.fc = nn.Linear(128, embedding_dim)
        
    def forward(self, x):
        # Feature extraction
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        
        x = torch.relu(self.conv3(x))
        
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Embedding
        embedding = self.fc(x)
        
        # L2 normalization
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

def create_face_detector_onnx(output_path: Path):
    """얼굴 검출 ONNX 모델 생성"""
    print(f"Creating face detector model: {output_path}")
    
    model = DummyFaceDetector()
    model.eval()
    
    # 더미 입력 (batch_size=1, channels=3, height=640, width=640)
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # ONNX 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 검증
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Created: {output_path}")
    return True

def create_reid_model_onnx(output_path: Path):
    """ReID ONNX 모델 생성"""
    print(f"Creating ReID model: {output_path}")
    
    model = DummyReIDModel(embedding_dim=128)
    model.eval()
    
    # 더미 입력 (batch_size=1, channels=3, height=256, width=128)
    dummy_input = torch.randn(1, 3, 256, 128)
    
    # ONNX 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    
    # 검증
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Created: {output_path}")
    return True

def main():
    """메인 함수"""
    print("🚀 Creating dummy models for development...")
    
    # 디렉토리 생성
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    weights_dir = Path("weights") 
    weights_dir.mkdir(exist_ok=True)
    
    # 모델 생성
    models_to_create = [
        ("yolov8n-face.onnx", create_face_detector_onnx),
        ("scrfd_500m.onnx", create_face_detector_onnx),
        ("reid_lightweight.onnx", create_reid_model_onnx),
    ]
    
    success_count = 0
    
    for model_name, create_func in models_to_create:
        model_path = models_dir / model_name
        try:
            if create_func(model_path):
                success_count += 1
                
                # weights 디렉토리에도 복사 (호환성을 위해)
                import shutil
                weight_path = weights_dir / model_name
                shutil.copy2(model_path, weight_path)
                
        except Exception as e:
            print(f"❌ Failed to create {model_name}: {e}")
    
    # 결과
    print(f"\n📊 Results: {success_count}/{len(models_to_create)} models created")
    
    if success_count == len(models_to_create):
        print("✅ All dummy models created successfully!")
        
        # 생성된 파일 목록
        print("\n📁 Created models:")
        for file_path in models_dir.glob("*.onnx"):
            size_kb = file_path.stat().st_size / 1024
            print(f"  - {file_path.name} ({size_kb:.1f} KB)")
        
        print("\n⚠️ Note: These are dummy models for development.")
        print("Replace with real models for production use.")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())