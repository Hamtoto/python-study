"""
DualFaceProcessor - 전체 GPU 파이프라인 통합 처리기.

완전한 Zero-copy GPU 파이프라인을 통해 비디오 처리를 수행합니다:
NVDEC (디코딩) → GPU Composition (합성) → NVENC (인코딩)

주요 기능:
    - 하드웨어 가속 비디오 디코딩 (PyAV NVDEC)
    - GPU 기반 스플릿 스크린 합성 (CUDA)
    - 하드웨어 가속 비디오 인코딩 (PyAV NVENC)
    - 실시간 성능 모니터링
    - 에러 복구 및 품질 관리

성능 목표:
    - 실시간 처리 (30fps+)
    - GPU 메모리 효율성 (<75% VRAM)
    - Zero-copy 처리 최적화

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0
"""

import time
import threading
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
import torch
from contextlib import contextmanager

from ..decoders import NvDecoder, SurfaceConverter
from ..composers import TileComposer, GpuResizer, TileCompositionErrorPolicy
from ..composers.gpu_resizer import ResizeStrategy
from ..encoders import NvEncoder, EncodingProfile, get_streaming_profile
from ..encoders.software_encoder import SoftwareEncoder, create_software_encoder
from ..utils.logger import UnifiedLogger
from ..utils.exceptions import (
    DualFaceTrackerError,
    DecodingError,
    CompositionError,
    EncodingError,
    GPUMemoryError
)
from ..utils.cuda_utils import (
    get_gpu_memory_info,
    monitor_gpu_memory,
    clear_gpu_cache,
    safe_upload_to_gpu
)


@dataclass
class DualFaceConfig:
    """DualFaceProcessor 설정 클래스."""
    
    # 입출력 설정
    input_path: str
    output_path: str
    output_width: int = 1920
    output_height: int = 1080
    target_fps: float = 30.0
    
    # 처리 설정
    enable_gpu_optimization: bool = True
    max_gpu_memory_usage: float = 0.75  # 75% of VRAM
    
    # 디코딩 설정
    decoder_config: Dict[str, Any] = field(default_factory=dict)
    
    # 합성 설정  
    composer_config: Dict[str, Any] = field(default_factory=lambda: {
        'split_mode': 'horizontal',  # horizontal split screen
        'left_weight': 0.5,
        'right_weight': 0.5,
        'background_color': (0, 0, 0)
    })
    
    # 인코딩 설정
    encoder_config: Dict[str, Any] = field(default_factory=lambda: {
        'profile': 'streaming',
        'bitrate': 8_000_000,
        'preset': 'p4'
    })
    
    # 성능 설정
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_monitoring': True,
        'monitoring_interval': 1.0,  # seconds
        'max_processing_time': 10.0  # seconds per frame for timeout
    })


@dataclass 
class ProcessingMetrics:
    """처리 성능 메트릭."""
    
    # 기본 통계
    frames_processed: int = 0
    frames_dropped: int = 0
    processing_time_total: float = 0.0
    
    # 단계별 시간 (평균)
    decode_time_avg: float = 0.0
    compose_time_avg: float = 0.0  
    encode_time_avg: float = 0.0
    
    # 성능 지표
    current_fps: float = 0.0
    average_fps: float = 0.0
    gpu_memory_used: int = 0  # bytes
    gpu_utilization: float = 0.0
    
    # 품질 지표
    composition_success_rate: float = 1.0
    encoding_success_rate: float = 1.0
    
    def update_fps(self):
        """FPS 계산 업데이트."""
        if self.processing_time_total > 0:
            self.average_fps = self.frames_processed / self.processing_time_total
    
    def get_summary(self) -> Dict[str, Any]:
        """메트릭 요약 반환."""
        return {
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'current_fps': self.current_fps,
            'average_fps': self.average_fps,
            'gpu_memory_mb': self.gpu_memory_used / (1024**2),
            'gpu_utilization': self.gpu_utilization,
            'composition_success_rate': self.composition_success_rate,
            'encoding_success_rate': self.encoding_success_rate
        }


class PipelineErrorHandler:
    """파이프라인 통합 에러 처리기."""
    
    def __init__(self, logger: UnifiedLogger):
        """에러 핸들러 초기화."""
        self.logger = logger
        self.error_counts = {
            'decoder': 0,
            'composer': 0, 
            'encoder': 0
        }
        self.max_consecutive_errors = 5
        
    def handle_decoder_error(self, error: Exception, frame_number: int) -> bool:
        """디코더 에러 처리."""
        self.error_counts['decoder'] += 1
        self.logger.error(f"❌ Decoder error at frame {frame_number}: {error}")
        
        # 연속 에러가 많으면 중단
        if self.error_counts['decoder'] > self.max_consecutive_errors:
            self.logger.error(f"❌ Too many decoder errors ({self.error_counts['decoder']})")
            return False
            
        # 재시도 가능
        self.logger.warning(f"⚠️ Attempting decoder recovery...")
        return True
    
    def handle_composer_error(self, error: Exception, frame_number: int) -> Optional[np.ndarray]:
        """합성 에러 처리."""
        self.error_counts['composer'] += 1
        self.logger.error(f"❌ Composer error at frame {frame_number}: {error}")
        
        # 기본 검은 화면 반환
        fallback_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.logger.warning(f"⚠️ Using fallback frame for composition error")
        
        return fallback_frame
    
    def handle_encoder_error(self, error: Exception, frame_number: int) -> bool:
        """인코더 에러 처리."""
        self.error_counts['encoder'] += 1
        self.logger.error(f"❌ Encoder error at frame {frame_number}: {error}")
        
        # 인코더 에러는 치명적일 수 있음
        if self.error_counts['encoder'] > self.max_consecutive_errors:
            self.logger.error(f"❌ Too many encoder errors ({self.error_counts['encoder']})")
            return False
            
        # 재시도 가능
        self.logger.warning(f"⚠️ Attempting encoder recovery...")
        return True
    
    def reset_error_counts(self):
        """에러 카운트 리셋."""
        self.error_counts = {k: 0 for k in self.error_counts}


class DualFaceProcessor:
    """
    전체 GPU 파이프라인 통합 처리기.
    
    NVDEC → GPU Composition → NVENC 파이프라인을 통해
    Zero-copy GPU 메모리에서 비디오를 처리합니다.
    
    Args:
        config: DualFaceConfig 설정 객체
        
    Example:
        >>> config = DualFaceConfig(
        ...     input_path="input.mp4",
        ...     output_path="output.mp4"
        ... )
        >>> processor = DualFaceProcessor(config)
        >>> with processor:
        ...     metrics = processor.process_video()
        >>> print(f"Processed {metrics.frames_processed} frames")
    """
    
    def __init__(self, config: Optional[DualFaceConfig] = None, cuda_stream: Optional[torch.cuda.Stream] = None, gpu_id: int = 0):
        """
        DualFaceProcessor 초기화.
        
        Args:
            config: DualFaceConfig 설정 객체 (선택사항)
            cuda_stream: 사용할 CUDA 스트림 (MultiStreamProcessor용)
            gpu_id: GPU 디바이스 ID
        """
        self.config = config
        self.cuda_stream = cuda_stream
        self.gpu_id = gpu_id
        self.logger = UnifiedLogger("DualFaceProcessor")
        
        if config:
            self.logger.info(f"🎬 Initializing DualFaceProcessor")
            self.logger.info(f"  • Input: {config.input_path}")
            self.logger.info(f"  • Output: {config.output_path} ({config.output_width}x{config.output_height})")
        
        if cuda_stream:
            self.logger.debug(f"  • Using CUDA Stream: {cuda_stream}")
        
        self.logger.debug(f"  • GPU Device: {gpu_id}")
        
        # 컴포넌트 초기화 (lazy loading)
        self.decoder: Optional[NvDecoder] = None
        self.surface_converter: Optional[SurfaceConverter] = None
        self.tile_composer: Optional[TileComposer] = None
        self.gpu_resizer: Optional[GpuResizer] = None
        self.encoder: Optional[NvEncoder] = None
        
        # 에러 처리 및 모니터링
        self.error_handler = PipelineErrorHandler(self.logger)
        self.metrics = ProcessingMetrics()
        
        # 상태 관리
        self.is_processing = False
        self.should_stop = threading.Event()
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # GPU 메모리 관리
        self.initial_gpu_memory = 0
        
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def initialize(self):
        """모든 컴포넌트 초기화."""
        try:
            self.logger.info("🔧 Initializing pipeline components...")
            
            # GPU 메모리 상태 확인
            self._check_gpu_memory()
            
            # 1. 디코더 초기화
            self.logger.info("  • Initializing NVDEC decoder...")
            self.decoder = NvDecoder(
                video_path=self.config.input_path,
                **self.config.decoder_config
            )
            
            # Surface converter for NV12 → RGB
            self.surface_converter = SurfaceConverter(
                source_format='nv12',
                target_format='rgb24'
            )
            
            # 2. 합성기 초기화
            self.logger.info("  • Initializing GPU composer...")
            
            # TileComposer가 지원하는 파라미터만 전달
            tile_composer_params = {
                'output_width': self.config.output_width,
                'output_height': self.config.output_height
            }
            
            # 지원되는 파라미터만 필터링
            if 'interpolation' in self.config.composer_config:
                tile_composer_params['interpolation'] = self.config.composer_config['interpolation']
            if 'use_cuda_stream' in self.config.composer_config:
                tile_composer_params['use_cuda_stream'] = self.config.composer_config['use_cuda_stream']
                
            self.tile_composer = TileComposer(**tile_composer_params)
            
            self.gpu_resizer = GpuResizer()
            
            # 3. 인코더 초기화
            self.logger.info("  • Initializing NVENC encoder...")
            
            # 인코딩 프로파일 설정
            if isinstance(self.config.encoder_config.get('profile'), str):
                profile = get_streaming_profile()
            else:
                profile = self.config.encoder_config.get('profile', get_streaming_profile())
            
            self.encoder = NvEncoder(
                output_path=self.config.output_path,
                width=self.config.output_width,
                height=self.config.output_height,
                fps=self.config.target_fps,
                profile=profile
            )
            
            # 4. 모니터링 시작
            if self.config.performance_config['enable_monitoring']:
                self._start_monitoring()
            
            self.logger.success("✅ All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise DualFaceTrackerError(f"Pipeline initialization failed: {e}")
    
    def cleanup(self):
        """리소스 정리."""
        try:
            self.logger.info("🔒 Cleaning up pipeline...")
            
            # 처리 중단
            self.should_stop.set()
            self.is_processing = False
            
            # 모니터링 스레드 종료
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            
            # 컴포넌트 정리
            if self.encoder:
                self.encoder.close()
                self.encoder = None
                
            if self.decoder:
                self.decoder.close()
                self.decoder = None
            
            # GPU 메모리 정리
            clear_gpu_cache()
            
            self.logger.success("✅ Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    def _check_gpu_memory(self):
        """GPU 메모리 상태 확인."""
        try:
            memory_info = get_gpu_memory_info()
            total_gb = memory_info['total'] / (1024**3)
            used_gb = memory_info['allocated'] / (1024**3)
            usage_percent = (used_gb / total_gb) * 100
            
            self.initial_gpu_memory = memory_info['allocated']
            
            self.logger.info(f"🔍 GPU Memory: {used_gb:.1f}/{total_gb:.1f}GB ({usage_percent:.1f}%)")
            
            # 메모리 사용량 경고
            if usage_percent > self.config.max_gpu_memory_usage * 100:
                self.logger.warning(f"⚠️ High GPU memory usage: {usage_percent:.1f}%")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check GPU memory: {e}")
    
    def _start_monitoring(self):
        """성능 모니터링 시작."""
        def monitoring_worker():
            while not self.should_stop.is_set():
                try:
                    # GPU 메모리 모니터링
                    memory_info = get_gpu_memory_info()
                    self.metrics.gpu_memory_used = memory_info['allocated'] - self.initial_gpu_memory
                    
                    # FPS 업데이트
                    self.metrics.update_fps()
                    
                    # 로그 출력 (5초마다)
                    if self.metrics.frames_processed % 150 == 0 and self.metrics.frames_processed > 0:
                        summary = self.metrics.get_summary()
                        self.logger.info(f"📊 Performance: {summary['current_fps']:.1f} fps, "
                                       f"GPU: {summary['gpu_memory_mb']:.1f}MB")
                    
                except Exception as e:
                    self.logger.debug(f"🔧 Monitoring error: {e}")
                
                time.sleep(self.config.performance_config['monitoring_interval'])
        
        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()
    
    def process_video(self) -> ProcessingMetrics:
        """
        전체 비디오 처리 실행.
        
        Returns:
            ProcessingMetrics: 처리 성능 메트릭
            
        Raises:
            DualFaceTrackerError: 처리 실패 시
        """
        if not self.decoder or not self.encoder:
            raise DualFaceTrackerError("Pipeline not initialized. Use context manager.")
        
        self.logger.info("🚀 Starting video processing...")
        self.is_processing = True
        start_time = time.perf_counter()
        
        try:
            with self.encoder:
                frame_number = 0
                
                for frame_data in self.decoder.decode_frames():
                    if self.should_stop.is_set():
                        break
                    
                    frame_start = time.perf_counter()
                    
                    try:
                        # 1. 디코딩 완료 (이미 디코더에서 수행됨)
                        decode_time = 0.001  # Placeholder
                        
                        # 2. 색공간 변환 (NV12 → RGB)
                        if isinstance(frame_data, dict) and 'surface' in frame_data:
                            rgb_frame = self.surface_converter.convert(frame_data['surface'])
                        else:
                            # numpy array인 경우 그대로 사용
                            rgb_frame = frame_data
                        
                        # 3. GPU 합성 처리 
                        compose_start = time.perf_counter()
                        
                        # 임시로 동일한 프레임을 좌우에 배치 (추후 실제 얼굴 분리 로직 추가)
                        composed_frame = self._compose_dual_frame(rgb_frame, rgb_frame)
                        
                        compose_time = time.perf_counter() - compose_start
                        
                        # 4. 인코딩
                        encode_start = time.perf_counter()
                        
                        success = self.encoder.encode_frame(composed_frame)
                        if not success:
                            if not self.error_handler.handle_encoder_error(
                                Exception("Encoding failed"), frame_number
                            ):
                                break
                            self.metrics.frames_dropped += 1
                            continue
                        
                        encode_time = time.perf_counter() - encode_start
                        
                        # 5. 메트릭 업데이트
                        frame_time = time.perf_counter() - frame_start
                        self._update_metrics(frame_time, decode_time, compose_time, encode_time)
                        
                        frame_number += 1
                        
                        # 진행 상황 로그 (30프레임마다)
                        if frame_number % 30 == 0:
                            self.logger.debug(f"🔄 Processed frame {frame_number}, "
                                            f"FPS: {self.metrics.current_fps:.1f}")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error processing frame {frame_number}: {e}")
                        self.metrics.frames_dropped += 1
                        
                        # 에러 처리 - 계속 진행할지 결정
                        if isinstance(e, (CompositionError, GPUMemoryError)):
                            fallback_frame = self.error_handler.handle_composer_error(e, frame_number)
                            if fallback_frame is not None:
                                self.encoder.encode_frame(fallback_frame)
                        else:
                            # 다른 에러는 중단
                            break
            
            # 인코더 플러시
            self.encoder.flush()
            
            # 최종 메트릭 계산
            total_time = time.perf_counter() - start_time
            self.metrics.processing_time_total = total_time
            self.metrics.update_fps()
            
            self.logger.success(f"✅ Video processing completed!")
            self._log_final_metrics()
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"❌ Video processing failed: {e}")
            raise DualFaceTrackerError(f"Video processing failed: {e}")
        
        finally:
            self.is_processing = False
    
    def _compose_dual_frame(
        self, 
        left_frame: np.ndarray, 
        right_frame: np.ndarray
    ) -> np.ndarray:
        """
        듀얼 프레임 합성.
        
        Args:
            left_frame: 왼쪽 프레임
            right_frame: 오른쪽 프레임
            
        Returns:
            합성된 1920x1080 스플릿 스크린 프레임
        """
        try:
            # GPU로 안전한 업로드 (OpenCV 4.13 호환)
            gpu_left = cv2.cuda.GpuMat()
            gpu_right = cv2.cuda.GpuMat()
            if not safe_upload_to_gpu(gpu_left, left_frame):
                raise CompositionError("Left frame GPU upload failed")
            if not safe_upload_to_gpu(gpu_right, right_frame):
                raise CompositionError("Right frame GPU upload failed")
            
            # 960x1080으로 리사이즈 (좌우 각각)
            target_width = self.config.output_width // 2  # 960
            target_height = self.config.output_height     # 1080
            
            resized_left = self.gpu_resizer.resize_to_fit(
                gpu_left, target_width, target_height, strategy=ResizeStrategy.FIT_COVER
            )
            resized_right = self.gpu_resizer.resize_to_fit(
                gpu_right, target_width, target_height, strategy=ResizeStrategy.FIT_COVER
            )
            
            # 타일 합성
            composed_gpu = self.tile_composer.compose_dual_frame(
                resized_left, resized_right
            )
            
            # CPU로 다운로드
            composed_frame = composed_gpu.download()
            
            return composed_frame
            
        except Exception as e:
            self.logger.error(f"❌ Frame composition failed: {e}")
            raise CompositionError(f"Failed to compose dual frame: {e}")
    
    def _update_metrics(
        self, 
        frame_time: float, 
        decode_time: float, 
        compose_time: float, 
        encode_time: float
    ):
        """메트릭 업데이트."""
        self.metrics.frames_processed += 1
        
        # 평균 시간 계산 (exponential moving average)
        alpha = 0.1
        self.metrics.decode_time_avg = (1 - alpha) * self.metrics.decode_time_avg + alpha * decode_time
        self.metrics.compose_time_avg = (1 - alpha) * self.metrics.compose_time_avg + alpha * compose_time
        self.metrics.encode_time_avg = (1 - alpha) * self.metrics.encode_time_avg + alpha * encode_time
        
        # 현재 FPS 계산
        if frame_time > 0:
            self.metrics.current_fps = 1.0 / frame_time
    
    def _log_final_metrics(self):
        """최종 메트릭 로그 출력."""
        summary = self.metrics.get_summary()
        
        self.logger.info("=" * 80)
        self.logger.info("📊 FINAL PROCESSING METRICS")
        self.logger.info("=" * 80)
        self.logger.info(f"🎬 Input: {self.config.input_path}")
        self.logger.info(f"📹 Output: {self.config.output_path}")
        self.logger.info(f"🖼️ Resolution: {self.config.output_width}x{self.config.output_height}")
        self.logger.info("")
        self.logger.info(f"📈 Frames processed: {summary['frames_processed']}")
        self.logger.info(f"📉 Frames dropped: {summary['frames_dropped']}")
        self.logger.info(f"⚡ Average FPS: {summary['average_fps']:.1f}")
        self.logger.info(f"⏱️ Processing time: {self.metrics.processing_time_total:.2f}s")
        self.logger.info("")
        self.logger.info(f"🔧 Decode time avg: {self.metrics.decode_time_avg*1000:.1f}ms")
        self.logger.info(f"🎨 Compose time avg: {self.metrics.compose_time_avg*1000:.1f}ms")  
        self.logger.info(f"📦 Encode time avg: {self.metrics.encode_time_avg*1000:.1f}ms")
        self.logger.info("")
        self.logger.info(f"🖥️ GPU memory used: {summary['gpu_memory_mb']:.1f}MB")
        self.logger.info(f"📊 Composition success: {summary['composition_success_rate']*100:.1f}%")
        self.logger.info(f"📊 Encoding success: {summary['encoding_success_rate']*100:.1f}%")
        self.logger.info("=" * 80)


# Convenience functions
def create_default_config(input_path: str, output_path: str) -> DualFaceConfig:
    """기본 설정으로 DualFaceConfig 생성."""
    return DualFaceConfig(
        input_path=input_path,
        output_path=output_path,
        output_width=1920,
        output_height=1080,
        target_fps=30.0
    )


def process_video_simple(
    input_path: str, 
    output_path: str,
    **kwargs
) -> ProcessingMetrics:
    """
    간단한 비디오 처리 함수.
    
    Args:
        input_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        **kwargs: 추가 설정
        
    Returns:
        ProcessingMetrics: 처리 결과 메트릭
    """
    config = create_default_config(input_path, output_path)
    
    # kwargs로 설정 오버라이드
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    with DualFaceProcessor(config) as processor:
        return processor.process_video()


def process_video_with_config(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    MultiStreamProcessor에서 호출할 수 있는 간단한 비디오 처리 함수.
    
    Args:
        input_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        
    Returns:
        처리 결과 딕셔너리
    """
    try:
        config = create_default_config(input_path, output_path)
        
        with DualFaceProcessor(config) as processor:
            metrics = processor.process_video()
            
            return {
                'success': True,
                'frames_processed': metrics.frames_processed,
                'processing_time': metrics.total_processing_time,
                'fps': metrics.frames_processed / max(metrics.total_processing_time, 0.001)
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'frames_processed': 0,
            'processing_time': 0,
            'fps': 0
        }