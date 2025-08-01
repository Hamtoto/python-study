"""
CUDA 스트림 활용 GPU 처리 시스템 - Phase 2 최적화
RTX 5090 32GB VRAM 최대 활용을 위한 비동기 스트림 처리
"""
import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from concurrent.futures import ThreadPoolExecutor
from moviepy import VideoFileClip
import warnings

# MoviePy 경고 억제
warnings.filterwarnings("ignore", message=".*bytes wanted but.*bytes read.*")

from ..core.models import ModelManager
from ..utils.logging import logger
from ..config import DEVICE


class CUDAStreamProcessor:
    """CUDA 스트림 기반 비동기 GPU 처리 시스템"""
    
    def __init__(self, device=DEVICE, num_streams: int = 4):
        """
        Args:
            device: CUDA 디바이스
            num_streams: CUDA 스트림 수 (RTX 5090 최적화 기준)
        """
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
            
        self.num_streams = num_streams
        self.streams = []
        self.model_manager = None
        
        # 스트림별 메모리 풀 사전 할당
        self.memory_pools = []
        
        # 스트림별 독립 MTCNN 인스턴스 (안정성 향상)
        self.stream_mtcnns = []
        
    def initialize(self):
        """CUDA 스트림 및 모델 초기화"""
        try:
            # 모델 초기화
            self.model_manager = ModelManager(self.device)
            
            # CUDA 스트림 생성
            self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_streams)]
            
            # 스트림별 메모리 풀 초기화
            self._initialize_memory_pools()
            
            # 스트림별 독립 MTCNN 인스턴스 초기화
            self._initialize_stream_mtcnns()
            
            # GPU 워밍업
            self._warmup_gpu()
            
            logger.info(f"CUDA 스트림 프로세서 초기화 완료 - {self.num_streams}개 스트림, 디바이스: {self.device}")
            
        except Exception as e:
            logger.error(f"CUDA 스트림 초기화 실패: {str(e)}")
            raise
    
    def _initialize_memory_pools(self):
        """스트림별 메모리 풀 사전 할당"""
        try:
            # RTX 5090 32GB 기준 스트림당 6GB 할당
            pool_size_mb = 6 * 1024  # 6GB
            
            for i in range(self.num_streams):
                with torch.cuda.stream(self.streams[i]):
                    # 스트림별 텐서 풀 생성
                    pool = {
                        'frame_buffer': torch.zeros((256, 3, 224, 224), dtype=torch.float32, device=self.device),
                        'face_buffer': torch.zeros((128, 3, 160, 160), dtype=torch.float32, device=self.device),
                        'embedding_buffer': torch.zeros((128, 512), dtype=torch.float32, device=self.device)
                    }
                    self.memory_pools.append(pool)
            
            # 메모리 동기화
            torch.cuda.synchronize()
            logger.info(f"스트림별 메모리 풀 초기화 완료 - {self.num_streams}개 풀")
            
        except Exception as e:
            logger.error(f"메모리 풀 초기화 실패: {str(e)}")
            raise
    
    def _initialize_stream_mtcnns(self):
        """스트림별 독립 MTCNN 인스턴스 초기화"""
        try:
            from facenet_pytorch import MTCNN
            
            # 각 스트림마다 독립적인 MTCNN 인스턴스 생성
            for i in range(self.num_streams):
                stream_mtcnn = MTCNN(
                    image_size=160,
                    margin=0,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=True,
                    device=self.device,
                    keep_all=True
                )
                self.stream_mtcnns.append(stream_mtcnn)
            
            logger.info(f"스트림별 독립 MTCNN 초기화 완료 - {self.num_streams}개 인스턴스")
            
        except Exception as e:
            logger.error(f"스트림별 MTCNN 초기화 실패: {str(e)}")
            # fallback: 공통 MTCNN 사용
            self.stream_mtcnns = [self.model_manager.mtcnn] * self.num_streams
            logger.warning("공통 MTCNN으로 fallback")
    
    def _warmup_gpu(self):
        """GPU 워밍업 - 초기 컴파일 시간 제거"""
        try:
            logger.info("GPU 워밍업 시작...")
            
            # 더미 데이터로 모델 워밍업
            dummy_frames = torch.randn(32, 3, 224, 224, device=self.device)
            dummy_faces = torch.randn(16, 3, 160, 160, device=self.device)
            
            # 각 스트림에서 워밍업 실행
            for i, stream in enumerate(self.streams):
                with torch.cuda.stream(stream):
                    # MTCNN 워밍업
                    _ = self.model_manager.mtcnn.detect(dummy_frames[:8])
                    
                    # ResNet 워밍업
                    _ = self.model_manager.resnet(dummy_faces[:8])
            
            # 모든 스트림 동기화
            torch.cuda.synchronize()
            logger.success("GPU 워밍업 완료")
            
        except Exception as e:
            logger.warning(f"GPU 워밍업 실패 (계속 진행): {str(e)}")
    
    def process_video_segment_async(self, seg_input: str, seg_cropped: str) -> bool:
        """
        CUDA 스트림을 활용한 비동기 세그먼트 처리
        
        Args:
            seg_input: 입력 세그먼트 경로
            seg_cropped: 크롭된 출력 경로
            
        Returns:
            처리 성공 여부
        """
        try:
            logger.info(f"CUDA 스트림 비동기 처리 시작: {os.path.basename(seg_input)}")
            start_time = time.time()
            
            # 1) 비디오 메타데이터 확인
            with VideoFileClip(seg_input) as clip:
                fps = clip.fps
                duration = clip.duration
                total_frames = int(fps * duration)
                
                if duration < 1.0:
                    logger.warning(f"세그먼트 너무 짧음: {duration:.1f}초")
                    return False
                
                logger.info(f"비디오 정보: {duration:.1f}초, {fps:.1f} FPS, {total_frames}프레임")
                
                # 2) 프레임 추출 및 스트림별 배치 분할
                frame_batches = self._extract_frames_to_batches(clip, fps, duration)
                
                if not frame_batches:
                    logger.warning("추출된 프레임 없음")
                    return False
                
                # 3) 다중 스트림 비동기 처리
                processed_frames = self._process_batches_async(frame_batches)
                
                # 4) 크롭된 비디오 생성
                success = self._create_cropped_video(processed_frames, seg_cropped, fps)
                
                elapsed = time.time() - start_time
                if success:
                    logger.success(f"CUDA 스트림 처리 완료: {os.path.basename(seg_input)} ({elapsed:.1f}초)")
                else:
                    logger.error(f"CUDA 스트림 처리 실패: {os.path.basename(seg_input)}")
                
                return success
                
        except Exception as e:
            logger.error(f"CUDA 스트림 처리 오류: {str(e)}")
            return False
    
    def _extract_frames_to_batches(self, clip: VideoFileClip, fps: float, duration: float) -> List[List[np.ndarray]]:
        """프레임 추출 및 스트림별 배치 분할"""
        try:
            frames = []
            
            # 프레임 추출
            for t in np.arange(0, duration, 1.0/fps):
                if t >= duration:
                    break
                try:
                    frame = clip.get_frame(t)
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except:
                    continue
            
            if not frames:
                return []
            
            # 스트림 수에 맞게 배치 분할
            batch_size = max(1, len(frames) // self.num_streams)
            batches = []
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                if batch:  # 빈 배치 제외
                    batches.append(batch)
            
            logger.info(f"프레임 배치 분할: {len(frames)}프레임 → {len(batches)}배치")
            return batches
            
        except Exception as e:
            logger.error(f"프레임 추출 실패: {str(e)}")
            return []
    
    def _process_batches_async(self, frame_batches: List[List[np.ndarray]]) -> List[Dict[str, Any]]:
        """다중 스트림으로 배치 비동기 처리"""
        try:
            logger.info(f"다중 스트림 비동기 처리 시작 - {len(frame_batches)}개 배치")
            start_time = time.time()
            
            # ThreadPoolExecutor로 스트림별 병렬 처리
            processed_results = []
            
            with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
                # 각 배치를 다른 스트림에 할당
                futures = []
                
                for i, batch in enumerate(frame_batches):
                    stream_idx = i % self.num_streams
                    future = executor.submit(self._process_single_batch_stream, batch, stream_idx)
                    futures.append(future)
                
                # 결과 수집
                for future in futures:
                    try:
                        result = future.result()
                        processed_results.extend(result)
                    except Exception as e:
                        logger.error(f"배치 처리 오류: {str(e)}")
            
            # 모든 스트림 동기화
            torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            logger.success(f"다중 스트림 처리 완료 - {len(processed_results)}프레임, {elapsed:.1f}초")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"비동기 배치 처리 실패: {str(e)}")
            return []
    
    def _process_single_batch_stream(self, frames: List[np.ndarray], stream_idx: int) -> List[Dict[str, Any]]:
        """단일 스트림에서 배치 처리"""
        try:
            stream = self.streams[stream_idx]
            memory_pool = self.memory_pools[stream_idx]
            
            results = []
            
            with torch.cuda.stream(stream):
                # 프레임을 텐서로 변환
                frame_tensors = []
                for i, frame in enumerate(frames):
                    try:
                        # 프레임 유효성 검사
                        if frame is None or frame.size == 0:
                            logger.warning(f"스트림 {stream_idx}: 프레임 {i} 무효")
                            continue
                            
                        # OpenCV BGR → RGB, normalize
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 프레임 크기 검증
                        if frame_rgb.shape[0] < 50 or frame_rgb.shape[1] < 50:
                            logger.warning(f"스트림 {stream_idx}: 프레임 {i} 너무 작음 {frame_rgb.shape}")
                            continue
                            
                        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                        frame_tensors.append(frame_tensor.to(self.device, non_blocking=True))
                        
                    except Exception as e:
                        logger.error(f"스트림 {stream_idx}: 프레임 {i} 텐서 변환 오류: {str(e)}")
                        continue
                
                if not frame_tensors:
                    logger.warning(f"스트림 {stream_idx}: 유효한 프레임 텐서 없음 ({len(frames)}개 입력)")
                    return results
                
                batch_tensor = torch.stack(frame_tensors)
                logger.info(f"스트림 {stream_idx}: 배치 텐서 생성 완료 - {batch_tensor.shape}")
                
                # MTCNN 단일 프레임 순차 처리 (안정성 우선)
                try:
                    all_faces = []
                    all_probs = []
                    successful_detections = 0
                    
                    logger.info(f"스트림 {stream_idx}: MTCNN 단일 프레임 처리 시작 - {batch_tensor.size(0)}개 프레임")
                    
                    # 각 프레임을 개별적으로 처리
                    for frame_idx in range(batch_tensor.size(0)):
                        try:
                            # 단일 프레임 추출 [1, C, H, W]
                            single_frame = batch_tensor[frame_idx:frame_idx+1]
                            
                            # 스트림별 독립 MTCNN 인스턴스 사용
                            stream_mtcnn = self.stream_mtcnns[stream_idx]
                            
                            # MTCNN 단일 프레임 처리 (스트림별 독립 인스턴스)
                            with torch.cuda.stream(stream):
                                frame_faces, frame_probs = stream_mtcnn.detect(single_frame)
                            
                            # 결과 저장
                            all_faces.append(frame_faces)
                            all_probs.append(frame_probs)
                            
                            # 성공적인 검출 카운트
                            if frame_faces is not None and len(frame_faces) > 0:
                                successful_detections += 1
                                
                        except Exception as frame_e:
                            logger.warning(f"스트림 {stream_idx}: 프레임 {frame_idx} MTCNN 처리 실패: {str(frame_e)}")
                            # 실패한 프레임은 None으로 처리
                            all_faces.append(None)
                            all_probs.append(None)
                    
                    faces = all_faces
                    probs = all_probs
                    
                    logger.info(f"스트림 {stream_idx}: MTCNN 처리 완료 - {successful_detections}/{len(faces)}개 프레임에서 얼굴 감지")
                    
                    if successful_detections == 0:
                        logger.warning(f"스트림 {stream_idx}: 모든 프레임에서 얼굴 감지 실패")
                        return results
                    
                    # 얼굴이 감지된 프레임 처리
                    for i, (frame_faces, frame_probs) in enumerate(zip(faces, probs)):
                        try:
                            if frame_faces is not None and len(frame_faces) > 0 and frame_probs is not None:
                                # 가장 확실한 얼굴 선택
                                best_idx = torch.argmax(frame_probs)
                                face_tensor = frame_faces[best_idx].unsqueeze(0)
                                
                                # ResNet으로 임베딩 추출 (비동기)
                                embedding = self.model_manager.resnet(face_tensor)
                                
                                # 원본 프레임에서 얼굴 영역 크롭
                                original_frame = frames[i] if i < len(frames) else None
                                if original_frame is not None:
                                    # 바운딩 박스를 사용해 얼굴 크롭
                                    bbox = frame_faces[best_idx].cpu().numpy()
                                    # 실제 얼굴 크롭 로직은 나중에 구현
                                    face_crop = original_frame  # 임시로 원본 프레임 사용
                                    
                                    results.append({
                                        'frame_idx': i,
                                        'face': face_crop,
                                        'embedding': embedding.cpu().numpy(),
                                        'bbox': bbox,
                                        'confidence': frame_probs[best_idx].item()
                                    })
                        except Exception as e:
                            logger.error(f"스트림 {stream_idx}: 프레임 {i} 얼굴 처리 오류: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.error(f"스트림 {stream_idx}: MTCNN 전체 처리 오류: {str(e)}")
                    # 심각한 오류 시 빈 결과 반환 (상위에서 fallback 처리)
                    return results
            
            logger.info(f"스트림 {stream_idx}: {len(frames)}프레임 → {len(results)}개 얼굴 처리")
            return results
            
        except Exception as e:
            logger.error(f"스트림 {stream_idx} 배치 처리 오류: {str(e)}")
            return []
    
    def _create_cropped_video(self, processed_frames: List[Dict[str, Any]], output_path: str, fps: float) -> bool:
        """처리된 프레임으로 크롭된 비디오 생성"""
        try:
            if not processed_frames:
                logger.warning("처리된 프레임 없음")
                return False
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 첫 번째 프레임으로 비디오 크기 결정
            first_face = processed_frames[0]['face']
            if len(first_face.shape) == 3:
                h, w = first_face.shape[:2]
            else:
                logger.error("잘못된 프레임 형태")
                return False
            
            # VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            if not out.isOpened():
                logger.error("VideoWriter 열기 실패")
                return False
            
            # 처리된 프레임들을 비디오로 저장
            for frame_data in processed_frames:
                face_frame = frame_data['face']
                
                # 크기 조정 (필요한 경우)
                if face_frame.shape[:2] != (h, w):
                    face_frame = cv2.resize(face_frame, (w, h))
                
                # BGR 형식으로 변환 (필요한 경우)
                if len(face_frame.shape) == 3 and face_frame.shape[2] == 3:
                    out.write(face_frame.astype(np.uint8))
            
            out.release()
            
            # 파일 생성 확인
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.success(f"크롭된 비디오 생성 완료: {output_path}")
                return True
            else:
                logger.error("비디오 파일 생성 실패 또는 빈 파일")
                return False
            
        except Exception as e:
            logger.error(f"크롭된 비디오 생성 실패: {str(e)}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 모든 스트림 동기화 및 정리
            if self.streams:
                torch.cuda.synchronize()
                self.streams.clear()
            
            # 메모리 풀 정리
            self.memory_pools.clear()
            
            # 스트림별 MTCNN 정리
            self.stream_mtcnns.clear()
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            logger.info("CUDA 스트림 프로세서 정리 완료")
            
        except Exception as e:
            logger.error(f"CUDA 스트림 정리 오류: {str(e)}")


def create_cuda_stream_processor(device=DEVICE, num_streams: int = 4) -> CUDAStreamProcessor:
    """
    CUDA 스트림 프로세서 생성
    
    Args:
        device: CUDA 디바이스
        num_streams: 스트림 수 (RTX 5090 기준 4개 최적)
        
    Returns:
        CUDAStreamProcessor 인스턴스
    """
    processor = CUDAStreamProcessor(device=device, num_streams=num_streams)
    processor.initialize()
    return processor