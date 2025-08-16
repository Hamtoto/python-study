"""
소프트웨어 비디오 인코더 (NVENC 폴백용)

NVENC 하드웨어 인코더가 사용 불가능할 때 사용하는 소프트웨어 인코더입니다.
OpenCV VideoWriter 또는 FFmpeg libx264를 사용합니다.

Author: Dual-Face High-Speed Processing System
Date: 2025.01  
Version: 1.0.0 (NVENC Fallback)
"""

import cv2
import numpy as np
import subprocess
from typing import Optional, Dict, Any, Union
from pathlib import Path
import time
import threading
from queue import Queue, Empty

from ..utils.logger import UnifiedLogger
from ..utils.exceptions import EncodingError


class SoftwareEncoder:
    """
    소프트웨어 비디오 인코더 (NVENC 대체용)
    
    NVENC이 사용 불가능할 때 CPU 기반 인코딩을 제공합니다.
    OpenCV VideoWriter 또는 FFmpeg subprocess를 사용합니다.
    """
    
    def __init__(self, 
                 output_path: str,
                 width: int = 1920,
                 height: int = 1080, 
                 fps: float = 30.0,
                 method: str = "opencv"):
        """
        소프트웨어 인코더 초기화
        
        Args:
            output_path: 출력 비디오 경로
            width: 비디오 너비
            height: 비디오 높이  
            fps: 프레임레이트
            method: 인코딩 방법 ("opencv" 또는 "ffmpeg")
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.method = method
        
        self.logger = UnifiedLogger("SoftwareEncoder")
        
        # 상태 관리
        self.is_initialized = False
        self.frame_count = 0
        self.total_encode_time = 0.0
        
        # OpenCV VideoWriter
        self.writer: Optional[cv2.VideoWriter] = None
        
        # FFmpeg subprocess
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        
        self.logger.debug(f"SoftwareEncoder 생성: {method} 방식, {width}x{height}@{fps}fps")
    
    def initialize(self) -> None:
        """인코더 초기화"""
        if self.is_initialized:
            return
            
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.method == "opencv":
                self._init_opencv_writer()
            elif self.method == "ffmpeg":
                self._init_ffmpeg_process()
            else:
                raise EncodingError(f"지원하지 않는 인코딩 방법: {self.method}")
            
            self.is_initialized = True
            self.logger.success(f"SoftwareEncoder 초기화 완료: {self.method}")
            
        except Exception as e:
            self.logger.error(f"SoftwareEncoder 초기화 실패: {e}")
            raise EncodingError(f"초기화 실패: {e}")
    
    def _init_opencv_writer(self) -> None:
        """OpenCV VideoWriter 초기화"""
        # 다양한 코덱 시도 (호환성 순서)
        codecs_to_try = [
            ('mp4v', 'MPEG-4'),
            ('XVID', 'Xvid'),
            ('X264', 'H.264'),
        ]
        
        for fourcc_str, name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                self.writer = cv2.VideoWriter(
                    str(self.output_path), 
                    fourcc, 
                    self.fps, 
                    (self.width, self.height)
                )
                
                if self.writer.isOpened():
                    self.logger.debug(f"OpenCV VideoWriter 초기화 성공: {name} ({fourcc_str})")
                    return
                else:
                    self.writer.release()
                    self.writer = None
                    
            except Exception as e:
                self.logger.warning(f"{name} 코덱 실패: {e}")
                continue
        
        raise EncodingError("사용 가능한 OpenCV 코덱이 없습니다")
    
    def _init_ffmpeg_process(self) -> None:
        """FFmpeg subprocess 초기화"""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV는 BGR 순서
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(self.output_path)
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            self.logger.debug("FFmpeg subprocess 초기화 성공")
            
        except FileNotFoundError:
            raise EncodingError("FFmpeg가 시스템에 설치되지 않음")
        except Exception as e:
            raise EncodingError(f"FFmpeg 프로세스 생성 실패: {e}")
    
    def encode_frame(self, frame: np.ndarray) -> bool:
        """
        프레임 인코딩
        
        Args:
            frame: BGR 형식 프레임 (OpenCV 형식)
            
        Returns:
            인코딩 성공 여부
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # 프레임 크기 검증
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            if self.method == "opencv":
                success = self._encode_frame_opencv(frame)
            elif self.method == "ffmpeg":
                success = self._encode_frame_ffmpeg(frame)
            else:
                raise EncodingError(f"지원하지 않는 방법: {self.method}")
            
            if success:
                self.frame_count += 1
                encode_time = time.perf_counter() - start_time
                self.total_encode_time += encode_time
                
                if self.frame_count % 30 == 0:  # 30프레임마다 로그
                    avg_fps = self.frame_count / self.total_encode_time
                    self.logger.debug(
                        f"소프트웨어 인코딩: {self.frame_count}프레임, {avg_fps:.1f}fps"
                    )
            
            return success
            
        except Exception as e:
            self.logger.error(f"프레임 인코딩 실패: {e}")
            return False
    
    def _encode_frame_opencv(self, frame: np.ndarray) -> bool:
        """OpenCV로 프레임 인코딩"""
        if not self.writer or not self.writer.isOpened():
            return False
        
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            self.logger.error(f"OpenCV 쓰기 실패: {e}")
            return False
    
    def _encode_frame_ffmpeg(self, frame: np.ndarray) -> bool:
        """FFmpeg로 프레임 인코딩"""
        if not self.ffmpeg_process or self.ffmpeg_process.stdin.closed:
            return False
        
        try:
            self.ffmpeg_process.stdin.write(frame.tobytes())
            self.ffmpeg_process.stdin.flush()
            return True
        except BrokenPipeError:
            self.logger.error("FFmpeg 프로세스가 종료됨")
            return False
        except Exception as e:
            self.logger.error(f"FFmpeg 쓰기 실패: {e}")
            return False
    
    def finalize(self) -> Dict[str, Any]:
        """인코딩 완료 및 정리"""
        result = {
            'success': False,
            'frame_count': self.frame_count,
            'total_time': self.total_encode_time,
            'avg_fps': 0.0,
            'output_path': str(self.output_path),
            'file_size': 0
        }
        
        try:
            if self.method == "opencv" and self.writer:
                self.writer.release()
                self.writer = None
                
            elif self.method == "ffmpeg" and self.ffmpeg_process:
                self.ffmpeg_process.stdin.close()
                return_code = self.ffmpeg_process.wait(timeout=30)
                
                if return_code != 0:
                    stderr_output = self.ffmpeg_process.stderr.read().decode()
                    self.logger.error(f"FFmpeg 종료 오류: {stderr_output}")
                    return result
            
            # 결과 파일 확인
            if self.output_path.exists():
                result['success'] = True
                result['file_size'] = self.output_path.stat().st_size
                result['avg_fps'] = self.frame_count / max(self.total_encode_time, 0.001)
                
                self.logger.success(
                    f"소프트웨어 인코딩 완료: {self.frame_count}프레임, "
                    f"{result['avg_fps']:.1f}fps, {result['file_size']}bytes"
                )
            else:
                self.logger.error("출력 파일이 생성되지 않음")
        
        except Exception as e:
            self.logger.error(f"인코딩 완료 처리 실패: {e}")
        
        finally:
            self.is_initialized = False
        
        return result
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finalize()


def create_software_encoder(
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0
) -> SoftwareEncoder:
    """
    소프트웨어 인코더 생성 (자동 방법 선택)
    
    Args:
        output_path: 출력 경로
        width: 비디오 너비
        height: 비디오 높이
        fps: 프레임레이트
        
    Returns:
        초기화된 소프트웨어 인코더
    """
    # FFmpeg 사용 가능 여부 확인
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            method = "ffmpeg"
        else:
            method = "opencv"
    except:
        method = "opencv"
    
    return SoftwareEncoder(output_path, width, height, fps, method)