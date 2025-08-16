#!/usr/bin/env python3
"""
간단한 TensorRT 변환 스크립트 (의존성 최소화)
"""

import sys
import argparse
from pathlib import Path
import tensorrt as trt
import numpy as np

def convert_onnx_to_tensorrt(onnx_path, engine_path, precision="fp16", max_batch_size=8):
    """ONNX를 TensorRT로 변환"""
    print(f"🔧 Converting {onnx_path} to TensorRT...")
    
    # TensorRT Logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Builder 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # ONNX 파일 파싱
    print("  Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(f"  Error: {parser.get_error(error)}")
            return False
    
    # Config 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # 정밀도 설정
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 precision enabled")
        else:
            print("  Warning: FP16 not supported, using FP32")
    
    # 동적 배치 크기 설정
    profile = builder.create_optimization_profile()
    
    # 입력 텐서 정보
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        print(f"  Input: {input_name} - shape: {input_shape}")
        
        # 동적 shape 설정 (배치만 동적)
        min_shape = [1] + list(input_shape[1:])
        opt_shape = [max_batch_size // 2] + list(input_shape[1:])
        max_shape = [max_batch_size] + list(input_shape[1:])
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        print(f"    Dynamic shapes: {min_shape} / {opt_shape} / {max_shape}")
    
    config.add_optimization_profile(profile)
    
    # 엔진 빌드
    print("  Building TensorRT engine... (this may take a while)")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("❌ Failed to build engine")
        return False
    
    # 엔진 저장
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    
    # 파일 크기 출력
    size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"✅ Engine saved: {engine_path} ({size_mb:.1f} MB)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--engine", help="Output engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--max-batch", type=int, default=8)
    
    args = parser.parse_args()
    
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"❌ ONNX file not found: {onnx_path}")
        return 1
    
    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix('.engine')
    
    # 엔진 디렉토리 생성
    engine_path.parent.mkdir(exist_ok=True)
    
    if convert_onnx_to_tensorrt(onnx_path, engine_path, args.precision, args.max_batch):
        print("✅ Conversion successful!")
        return 0
    else:
        print("❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())