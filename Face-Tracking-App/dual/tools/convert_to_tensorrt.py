#!/usr/bin/env python3
"""
ONNX 모델을 TensorRT 엔진으로 변환하는 스크립트
FP16/FP32 정밀도 및 동적 배치 크기 지원
"""

import os
import sys
import argparse
from pathlib import Path
import tensorrt as trt
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_face_tracker.utils.logger import UnifiedLogger
from dual_face_tracker.utils.cuda_utils import get_gpu_memory_info

logger = UnifiedLogger()

# TensorRT 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class ONNXToTensorRT:
    """ONNX를 TensorRT 엔진으로 변환하는 클래스"""
    
    def __init__(
        self,
        onnx_path: str,
        engine_path: str,
        precision: str = "fp32",
        max_batch_size: int = 8,
        min_batch_size: int = 1,
        opt_batch_size: int = 4,
        max_workspace_gb: float = 4.0
    ):
        """
        Args:
            onnx_path: 입력 ONNX 파일 경로
            engine_path: 출력 TensorRT 엔진 경로
            precision: 정밀도 ("fp32", "fp16", "int8")
            max_batch_size: 최대 배치 크기
            min_batch_size: 최소 배치 크기
            opt_batch_size: 최적 배치 크기
            max_workspace_gb: 최대 작업 공간 (GB)
        """
        self.onnx_path = Path(onnx_path)
        self.engine_path = Path(engine_path)
        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.opt_batch_size = opt_batch_size
        self.max_workspace_gb = max_workspace_gb
        
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX 파일을 찾을 수 없음: {self.onnx_path}")
    
    def build_engine(self) -> bool:
        """TensorRT 엔진을 빌드합니다."""
        try:
            logger.stage(f"🔧 TensorRT 엔진 빌드 시작: {self.onnx_path.name}")
            logger.info(f"  정밀도: {self.precision.upper()}")
            logger.info(f"  배치 크기: {self.min_batch_size}-{self.opt_batch_size}-{self.max_batch_size}")
            
            # GPU 메모리 상태 확인
            gpu_info = get_gpu_memory_info()
            logger.info(f"  GPU 메모리: {gpu_info['used_gb']:.1f}/{gpu_info['total_gb']:.1f} GB 사용 중")
            
            # Builder 생성
            builder = trt.Builder(TRT_LOGGER)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # ONNX 파일 파싱
            logger.info("ONNX 파일 파싱 중...")
            with open(self.onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX 파싱 오류: {parser.get_error(error)}")
                    return False
            
            # Builder 설정
            config = builder.create_builder_config()
            
            # 작업 공간 메모리 설정
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 
                int(self.max_workspace_gb * (1 << 30))  # GB to bytes
            )
            
            # 정밀도 설정
            if self.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 정밀도 활성화")
                else:
                    logger.warning("⚠️ FP16이 지원되지 않음, FP32 사용")
            elif self.precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("INT8 정밀도 활성화 (캘리브레이션 필요)")
                    # INT8 캘리브레이션 구현 필요
                else:
                    logger.warning("⚠️ INT8이 지원되지 않음, FP32 사용")
            
            # 동적 배치 크기 설정
            profile = builder.create_optimization_profile()
            
            # 네트워크 입력 설정
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                input_name = input_tensor.name
                input_shape = input_tensor.shape
                
                logger.info(f"입력 텐서: {input_name} - {input_shape}")
                
                # 동적 shape 설정 (배치 차원만 동적)
                min_shape = [self.min_batch_size] + list(input_shape[1:])
                opt_shape = [self.opt_batch_size] + list(input_shape[1:])
                max_shape = [self.max_batch_size] + list(input_shape[1:])
                
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.debug(f"  동적 shape: {min_shape} / {opt_shape} / {max_shape}")
            
            config.add_optimization_profile(profile)
            
            # 추가 최적화 플래그
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            
            # 엔진 빌드
            logger.info("TensorRT 엔진 빌드 중... (시간이 걸릴 수 있습니다)")
            engine_bytes = builder.build_serialized_network(network, config)
            
            if engine_bytes is None:
                logger.error("엔진 빌드 실패")
                return False
            
            # 엔진 저장
            with open(self.engine_path, 'wb') as f:
                f.write(engine_bytes)
            
            # 엔진 정보
            size_mb = self.engine_path.stat().st_size / (1024 * 1024)
            logger.success(f"✅ TensorRT 엔진 생성 완료: {self.engine_path}")
            logger.info(f"  엔진 크기: {size_mb:.1f} MB")
            
            # 엔진 검증
            return self.verify_engine()
            
        except Exception as e:
            logger.error(f"엔진 빌드 실패: {e}")
            if self.engine_path.exists():
                self.engine_path.unlink()
            return False
    
    def verify_engine(self) -> bool:
        """생성된 TensorRT 엔진을 검증합니다."""
        try:
            logger.info("엔진 검증 중...")
            
            # 런타임 생성
            runtime = trt.Runtime(TRT_LOGGER)
            
            # 엔진 로드
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                logger.error("엔진 역직렬화 실패")
                return False
            
            # 실행 컨텍스트 생성
            context = engine.create_execution_context()
            if context is None:
                logger.error("실행 컨텍스트 생성 실패")
                return False
            
            # 입출력 정보 출력
            logger.info("엔진 입출력 정보:")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                dtype = engine.get_tensor_dtype(name)
                shape = engine.get_tensor_shape(name)
                mode = engine.get_tensor_mode(name)
                
                io_type = "입력" if mode == trt.TensorIOMode.INPUT else "출력"
                logger.info(f"  {io_type}: {name} - {shape} ({dtype})")
            
            # 메모리 요구사항
            logger.info(f"디바이스 메모리 요구량: {engine.device_memory_size / (1024**2):.1f} MB")
            
            logger.success("✅ 엔진 검증 완료")
            return True
            
        except Exception as e:
            logger.error(f"엔진 검증 실패: {e}")
            return False
    
    def benchmark_engine(self, num_iterations: int = 100) -> dict:
        """엔진 성능을 벤치마크합니다."""
        try:
            import torch
            import time
            
            logger.stage("⏱️ 엔진 벤치마크 시작")
            
            # 런타임 및 엔진 로드
            runtime = trt.Runtime(TRT_LOGGER)
            with open(self.engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            context = engine.create_execution_context()
            
            # 입력 shape 설정
            input_name = engine.get_tensor_name(0)
            input_shape = [self.opt_batch_size, 3, 640, 640]  # 기본 shape
            context.set_input_shape(input_name, input_shape)
            
            # GPU 메모리 할당
            bindings = []
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = context.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                size = np.prod(shape)
                
                # GPU 메모리 할당
                device_mem = torch.empty(size, dtype=torch.float32, device='cuda')
                bindings.append(device_mem.data_ptr())
            
            # 워밍업
            logger.info("워밍업 중...")
            for _ in range(10):
                context.execute_v2(bindings)
            
            # 벤치마크
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                context.execute_v2(bindings)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 결과 계산
            total_time = end_time - start_time
            avg_time = total_time / num_iterations * 1000  # ms
            fps = num_iterations * self.opt_batch_size / total_time
            
            results = {
                "total_time": total_time,
                "avg_latency_ms": avg_time,
                "throughput_fps": fps,
                "batch_size": self.opt_batch_size,
                "iterations": num_iterations
            }
            
            logger.success("✅ 벤치마크 완료")
            logger.info(f"  평균 지연시간: {avg_time:.2f} ms")
            logger.info(f"  처리량: {fps:.1f} FPS")
            logger.info(f"  배치 크기: {self.opt_batch_size}")
            
            return results
            
        except Exception as e:
            logger.error(f"벤치마크 실패: {e}")
            return {}

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="ONNX를 TensorRT로 변환")
    parser.add_argument("--onnx", type=str, help="입력 ONNX 파일 경로")
    parser.add_argument("--engine", type=str, help="출력 TensorRT 엔진 경로")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16", help="정밀도")
    parser.add_argument("--max-batch", type=int, default=8, help="최대 배치 크기")
    parser.add_argument("--min-batch", type=int, default=1, help="최소 배치 크기")
    parser.add_argument("--opt-batch", type=int, default=4, help="최적 배치 크기")
    parser.add_argument("--workspace", type=float, default=4.0, help="작업 공간 크기 (GB)")
    parser.add_argument("--benchmark", action="store_true", help="벤치마크 실행")
    parser.add_argument("--all", action="store_true", help="모든 ONNX 모델 변환")
    
    args = parser.parse_args()
    
    logger.stage("🚀 TensorRT 변환 시작")
    
    models_dir = Path("models")
    engines_dir = Path("engines")
    engines_dir.mkdir(exist_ok=True)
    
    if args.all:
        # 모든 ONNX 모델 변환
        success_count = 0
        total_count = 0
        
        for onnx_file in models_dir.glob("*.onnx"):
            total_count += 1
            engine_path = engines_dir / f"{onnx_file.stem}_{args.precision}.engine"
            
            converter = ONNXToTensorRT(
                onnx_file,
                engine_path,
                args.precision,
                args.max_batch,
                args.min_batch,
                args.opt_batch,
                args.workspace
            )
            
            if converter.build_engine():
                success_count += 1
                
                if args.benchmark:
                    converter.benchmark_engine()
        
        logger.stage(f"📊 변환 결과: {success_count}/{total_count} 성공")
        
        # 생성된 엔진 파일 목록
        logger.info("\n📁 TensorRT 엔진 파일:")
        for engine_file in engines_dir.glob("*.engine"):
            size_mb = engine_file.stat().st_size / (1024 * 1024)
            logger.info(f"  - {engine_file.name} ({size_mb:.1f} MB)")
            
    else:
        # 단일 모델 변환
        if not args.onnx:
            logger.error("--onnx 옵션이 필요합니다")
            return 1
        
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            logger.error(f"ONNX 파일을 찾을 수 없음: {onnx_path}")
            return 1
        
        engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix('.engine')
        
        converter = ONNXToTensorRT(
            onnx_path,
            engine_path,
            args.precision,
            args.max_batch,
            args.min_batch,
            args.opt_batch,
            args.workspace
        )
        
        if not converter.build_engine():
            return 1
        
        if args.benchmark:
            converter.benchmark_engine()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())