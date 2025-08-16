"""
ONNX Runtime inference engine for optimized GPU acceleration.

이 모듈은 ONNX Runtime을 사용한 고성능 GPU 추론 엔진을 제공합니다.
RTX 5090에서 CUDA Execution Provider를 통해 최적화된 추론을 수행합니다.
"""

import os
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from ..utils.logger import UnifiedLogger
from ..utils.exceptions import InferenceError, ModelLoadError


class ONNXRuntimeEngine:
    """
    ONNX Runtime GPU 추론 엔진.
    
    Context7 best practices를 적용한 고성능 GPU 추론을 제공합니다.
    CUDAExecutionProvider를 우선 사용하며, CPU fallback을 지원합니다.
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        provider_options: Optional[Dict] = None,
        enable_profiling: bool = False,
        enable_optimization: bool = True
    ):
        """
        ONNX Runtime 엔진 초기화.
        
        Args:
            model_path: ONNX 모델 파일 경로
            providers: 실행 제공자 리스트 (기본값: CUDA → CPU)
            provider_options: 제공자별 옵션 설정
            enable_profiling: 프로파일링 활성화 여부
            enable_optimization: 모델 최적화 활성화 여부
        """
        self.logger = UnifiedLogger("ONNXRuntimeEngine")
        
        if not ORT_AVAILABLE:
            raise ImportError("onnxruntime not available. Please install onnxruntime-gpu")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        
        # 기본 제공자 설정
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.providers = providers
        self.provider_options = provider_options or {}
        self.enable_profiling = enable_profiling
        self.enable_optimization = enable_optimization
        
        # 세션 변수
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        
        # 성능 메트릭
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.warmup_completed = False
        
        # 초기화
        self._initialize_session()
        self.logger.success(f"ONNXRuntimeEngine initialized: {self.model_path.name}")
    
    def _initialize_session(self):
        """ONNX Runtime 세션 초기화."""
        try:
            # Context7 best practices: 세션 옵션 최적화
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
            
            if self.enable_optimization:
                # 그래프 최적화 레벨 설정 (99: 모든 최적화)
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # 메모리 패턴 최적화
                sess_options.enable_mem_pattern = True
                sess_options.enable_mem_reuse = True
            
            if self.enable_profiling:
                sess_options.enable_profiling = True
                prof_file = f"onnx_profile_{int(time.time())}.json"
                sess_options.profile_file_prefix = prof_file
                self.logger.debug(f"Profiling enabled: {prof_file}")
            
            # 제공자 가용성 확인
            available_providers = ort.get_available_providers()
            self.logger.debug(f"Available providers: {available_providers}")
            
            # 요청된 제공자 중 사용 가능한 것만 필터링
            filtered_providers = []
            for provider in self.providers:
                if provider in available_providers:
                    filtered_providers.append(provider)
                    self.logger.debug(f"✅ Provider available: {provider}")
                else:
                    self.logger.warning(f"⚠️ Provider not available: {provider}")
            
            if not filtered_providers:
                raise InferenceError("No valid execution providers available")
            
            # 세션 생성
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=filtered_providers
            )
            
            # 입출력 메타데이터 수집
            self._collect_model_metadata()
            
            # 실제 사용된 제공자 로깅
            used_providers = self.session.get_providers()
            self.logger.stage(f"Active providers: {used_providers}")
            
            # CUDA 사용 여부 확인
            if 'CUDAExecutionProvider' in used_providers:
                self.logger.success("🚀 CUDA acceleration enabled")
            else:
                self.logger.warning("💻 Running on CPU only")
                
        except Exception as e:
            self.logger.error(f"Session initialization failed: {e}")
            raise InferenceError(f"Failed to initialize ONNX Runtime session: {e}")
    
    def _collect_model_metadata(self):
        """모델 입출력 메타데이터 수집."""
        # 입력 정보
        for input_meta in self.session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            self.input_names.append(name)
            self.input_shapes[name] = shape
            
        # 출력 정보  
        for output_meta in self.session.get_outputs():
            name = output_meta.name
            shape = output_meta.shape
            self.output_names.append(name)
            self.output_shapes[name] = shape
            
        self.logger.debug(f"Model inputs: {self.input_shapes}")
        self.logger.debug(f"Model outputs: {self.output_shapes}")
    
    def warmup(self, num_runs: int = 3) -> float:
        """
        모델 워밍업으로 초기 지연시간 제거.
        
        Args:
            num_runs: 워밍업 실행 횟수
            
        Returns:
            평균 워밍업 시간 (ms)
        """
        if self.warmup_completed:
            self.logger.debug("Warmup already completed")
            return 0.0
            
        self.logger.stage("🔥 Starting model warmup...")
        
        # 더미 입력 생성
        dummy_inputs = {}
        for name, shape in self.input_shapes.items():
            # 동적 배치 크기 처리 (첫 번째 차원이 -1이면 1로 설정)
            actual_shape = [1 if dim == -1 else dim for dim in shape]
            dummy_inputs[name] = np.random.randn(*actual_shape).astype(np.float32)
        
        warmup_times = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            try:
                _ = self.session.run(self.output_names, dummy_inputs)
                elapsed = (time.time() - start_time) * 1000  # ms
                warmup_times.append(elapsed)
                self.logger.debug(f"Warmup {i+1}/{num_runs}: {elapsed:.2f}ms")
                
            except Exception as e:
                self.logger.error(f"Warmup failed at run {i+1}: {e}")
                raise InferenceError(f"Model warmup failed: {e}")
        
        avg_warmup_time = np.mean(warmup_times)
        self.warmup_completed = True
        
        self.logger.success(f"✅ Warmup completed: {avg_warmup_time:.2f}ms average")
        return avg_warmup_time
    
    def run(
        self, 
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        모델 추론 실행.
        
        Args:
            inputs: 입력 데이터 딕셔너리 {input_name: numpy_array}
            output_names: 반환할 출력 이름들 (None이면 모든 출력)
            
        Returns:
            출력 데이터 딕셔너리 {output_name: numpy_array}
        """
        if self.session is None:
            raise InferenceError("Session not initialized")
        
        # 입력 검증
        self._validate_inputs(inputs)
        
        # 출력 이름 결정
        if output_names is None:
            output_names = self.output_names
        
        start_time = time.time()
        
        try:
            # 추론 실행
            raw_outputs = self.session.run(output_names, inputs)
            
            # 출력 딕셔너리 구성
            outputs = {}
            for i, name in enumerate(output_names):
                outputs[name] = raw_outputs[i]
            
            # 성능 메트릭 업데이트
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise InferenceError(f"Model inference failed: {e}")
    
    def _validate_inputs(self, inputs: Dict[str, np.ndarray]):
        """입력 데이터 검증."""
        for name in self.input_names:
            if name not in inputs:
                raise InferenceError(f"Missing required input: {name}")
            
            input_array = inputs[name]
            expected_shape = self.input_shapes[name]
            
            # 동적 배치 크기 고려한 shape 검증
            if len(input_array.shape) != len(expected_shape):
                raise InferenceError(
                    f"Input '{name}' shape mismatch: "
                    f"expected {len(expected_shape)}D, got {len(input_array.shape)}D"
                )
            
            # 각 차원 검증 (동적 차원 -1은 제외)
            for i, (actual, expected) in enumerate(zip(input_array.shape, expected_shape)):
                if expected != -1 and actual != expected:
                    raise InferenceError(
                        f"Input '{name}' dimension {i} mismatch: "
                        f"expected {expected}, got {actual}"
                    )
    
    @property
    def average_inference_time(self) -> float:
        """평균 추론 시간 (ms) 반환."""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count
    
    @property 
    def fps(self) -> float:
        """초당 프레임 수 계산."""
        if self.average_inference_time == 0:
            return 0.0
        return 1000.0 / self.average_inference_time
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """성능 통계 반환."""
        return {
            'inference_count': self.inference_count,
            'total_time_ms': self.total_inference_time,
            'average_time_ms': self.average_inference_time,
            'fps': self.fps,
            'warmup_completed': self.warmup_completed,
            'providers': self.session.get_providers() if self.session else []
        }
    
    def reset_stats(self):
        """성능 통계 초기화."""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.logger.debug("Performance statistics reset")
    
    def close(self):
        """리소스 정리."""
        if self.session is not None:
            self.session = None
        self.logger.debug("ONNXRuntimeEngine closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()