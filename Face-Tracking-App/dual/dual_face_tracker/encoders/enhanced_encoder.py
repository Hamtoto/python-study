"""
향상된 인코더 (NVENC 세션 관리 + 소프트웨어 폴백)

NVENC 세션 제한 문제를 해결하기 위한 향상된 인코더입니다:
- NVENC 세션 관리자 통합
- 자동 소프트웨어 폴백
- 에러 복구 메커니즘
- 성능 최적화

Author: Dual-Face High-Speed Processing System  
Date: 2025.01
Version: 1.0.0 (Session Managed)
"""

import asyncio
import time
import threading
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager

from .nvencoder import NvEncoder
from .software_encoder import SoftwareEncoder
from .session_manager import get_global_session_manager, NvencSessionManager
from .encoding_config import EncodingProfile, get_default_profile
from ..utils.logger import UnifiedLogger
from ..utils.exceptions import (
    EncodingError,
    HardwareError,
    ResourceError
)


class EnhancedEncoder:
    """
    향상된 인코더 (NVENC 세션 관리 + 소프트웨어 폴백)
    
    특징:
    - NVENC 세션 제한 관리
    - 자동 소프트웨어 폴백
    - 에러 복구
    - 성능 모니터링
    """
    
    def __init__(self, 
                 output_path: str,
                 width: int = 1920,
                 height: int = 1080,
                 fps: float = 30.0,
                 profile: Optional[EncodingProfile] = None,
                 max_nvenc_sessions: int = 2,
                 enable_fallback: bool = True,
                 session_timeout: float = 30.0):
        """
        향상된 인코더 초기화
        
        Args:
            output_path: 출력 파일 경로
            width: 비디오 너비
            height: 비디오 높이
            fps: 프레임레이트
            profile: 인코딩 프로파일
            max_nvenc_sessions: 최대 NVENC 세션 수
            enable_fallback: 소프트웨어 폴백 활성화
            session_timeout: 세션 대기 타임아웃 (초)
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.profile = profile or get_default_profile("streaming")
        self.enable_fallback = enable_fallback
        self.session_timeout = session_timeout
        
        self.logger = UnifiedLogger("EnhancedEncoder")
        
        # 세션 관리자
        self.session_manager = get_global_session_manager(max_nvenc_sessions)
        
        # 인코더 인스턴스
        self.nvenc_encoder: Optional[NvEncoder] = None
        self.software_encoder: Optional[SoftwareEncoder] = None
        self.current_encoder = None
        self.encoder_type = None  # "nvenc" 또는 "software"
        
        # 세션 관리
        self.session_id: Optional[int] = None
        self.session_acquired = False
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'nvenc_frames': 0,
            'software_frames': 0,
            'encoding_errors': 0,
            'fallback_count': 0,
            'start_time': None,
            'encoding_time': 0.0
        }
        
        self.logger.info(f"🚀 향상된 인코더 초기화: {self.output_path}")
        self.logger.info(f"📋 설정: NVENC세션={max_nvenc_sessions}, 폴백={enable_fallback}")
    
    async def initialize(self) -> bool:
        """
        인코더 초기화 (NVENC 우선, 실패 시 소프트웨어)
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.stats['start_time'] = time.perf_counter()
            
            # NVENC 시도
            nvenc_success = await self._try_initialize_nvenc()
            if nvenc_success:
                return True
            
            # 소프트웨어 폴백
            if self.enable_fallback:
                software_success = await self._try_initialize_software()
                if software_success:
                    self.stats['fallback_count'] += 1
                    return True
            
            raise EncodingError("모든 인코더 초기화 실패")
            
        except Exception as e:
            self.logger.error(f"❌ 인코더 초기화 실패: {e}")
            return False
    
    async def _try_initialize_nvenc(self) -> bool:
        """NVENC 인코더 초기화 시도"""
        try:
            self.logger.info("🎯 NVENC 인코더 초기화 시도...")
            
            # 세션 획득 시도
            try:
                self.session_id = await self.session_manager.get_session(
                    timeout=self.session_timeout
                )
                self.session_acquired = True
                self.logger.debug(f"✅ NVENC 세션 획득: {self.session_id}")
                
            except ResourceError as e:
                self.logger.warning(f"⚠️ NVENC 세션 획득 실패: {e}")
                return False
            
            # NVENC 인코더 생성
            try:
                self.nvenc_encoder = NvEncoder(
                    output_path=str(self.output_path),
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    profile=self.profile
                )
                
                self.nvenc_encoder.open()
                self.current_encoder = self.nvenc_encoder
                self.encoder_type = "nvenc"
                
                self.logger.success("✅ NVENC 인코더 초기화 성공")
                return True
                
            except Exception as nvenc_error:
                # NVENC 초기화 실패 시 세션 해제
                if self.session_acquired:
                    await self.session_manager.release_session(self.session_id)
                    self.session_acquired = False
                    
                # 세션 관리자에 에러 기록
                if self.session_id is not None:
                    self.session_manager.mark_session_error(self.session_id, nvenc_error)
                
                self.logger.warning(f"⚠️ NVENC 초기화 실패: {nvenc_error}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ NVENC 초기화 중 예상치 못한 에러: {e}")
            return False
    
    async def _try_initialize_software(self) -> bool:
        """소프트웨어 인코더 초기화 시도"""
        try:
            self.logger.info("🔧 소프트웨어 인코더로 폴백...")
            
            self.software_encoder = SoftwareEncoder(
                output_path=str(self.output_path),
                width=self.width,
                height=self.height,
                fps=self.fps,
                method="opencv"
            )
            
            success = self.software_encoder.initialize()
            if success:
                self.current_encoder = self.software_encoder
                self.encoder_type = "software"
                self.logger.success("✅ 소프트웨어 인코더 초기화 성공")
                return True
            else:
                self.logger.error("❌ 소프트웨어 인코더 초기화 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 소프트웨어 인코더 초기화 에러: {e}")
            return False
    
    async def encode_frame(self, frame: Union[np.ndarray, torch.Tensor, cv2.cuda.GpuMat]) -> bool:
        """
        프레임 인코딩
        
        Args:
            frame: 입력 프레임
            
        Returns:
            bool: 인코딩 성공 여부
        """
        if not self.current_encoder:
            raise EncodingError("인코더가 초기화되지 않음")
        
        try:
            start_time = time.perf_counter()
            
            # 현재 인코더로 시도
            if self.encoder_type == "nvenc":
                success = self.nvenc_encoder.encode_frame(frame)
                if success:
                    self.stats['nvenc_frames'] += 1
                else:
                    # NVENC 실패 시 소프트웨어로 폴백 시도
                    success = await self._fallback_to_software_for_frame(frame)
                    
            else:  # software encoder
                success = self.software_encoder.encode_frame(frame)
                if success:
                    self.stats['software_frames'] += 1
            
            if success:
                self.stats['total_frames'] += 1
                encoding_time = time.perf_counter() - start_time
                self.stats['encoding_time'] += encoding_time
            else:
                self.stats['encoding_errors'] += 1
                
            return success
            
        except Exception as e:
            self.stats['encoding_errors'] += 1
            self.logger.error(f"❌ 프레임 인코딩 에러: {e}")
            
            # 에러 발생 시 폴백 시도
            if self.encoder_type == "nvenc" and self.enable_fallback:
                return await self._fallback_to_software_for_frame(frame)
            
            return False
    
    async def _fallback_to_software_for_frame(self, frame) -> bool:
        """단일 프레임에 대해 소프트웨어 폴백"""
        if not self.enable_fallback:
            return False
            
        try:
            # 소프트웨어 인코더가 없다면 초기화
            if not self.software_encoder:
                fallback_path = self.output_path.with_name(
                    f"{self.output_path.stem}_fallback{self.output_path.suffix}"
                )
                
                self.software_encoder = SoftwareEncoder(
                    output_path=str(fallback_path),
                    width=self.width,
                    height=self.height,
                    fps=self.fps
                )
                
                success = self.software_encoder.initialize()
                if not success:
                    return False
            
            # 소프트웨어 인코더로 프레임 처리
            success = self.software_encoder.encode_frame(frame)
            if success:
                self.stats['software_frames'] += 1
                self.stats['fallback_count'] += 1
                self.logger.debug("🔧 소프트웨어 폴백으로 프레임 처리")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 소프트웨어 폴백 에러: {e}")
            return False
    
    async def close(self):
        """인코더 종료 및 리소스 정리"""
        try:
            self.logger.info("🔒 향상된 인코더 종료 중...")
            
            # 현재 인코더 종료
            if self.nvenc_encoder:
                try:
                    self.nvenc_encoder.close()
                except Exception as e:
                    self.logger.warning(f"⚠️ NVENC 인코더 종료 에러: {e}")
            
            if self.software_encoder:
                try:
                    self.software_encoder.close()
                except Exception as e:
                    self.logger.warning(f"⚠️ 소프트웨어 인코더 종료 에러: {e}")
            
            # NVENC 세션 해제
            if self.session_acquired and self.session_id is not None:
                await self.session_manager.release_session(self.session_id)
                self.session_acquired = False
            
            # 통계 로그
            self._log_final_stats()
            
            self.logger.success("✅ 향상된 인코더 종료 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 인코더 종료 에러: {e}")
    
    def _log_final_stats(self):
        """최종 통계 출력"""
        if self.stats['start_time']:
            total_time = time.perf_counter() - self.stats['start_time']
        else:
            total_time = 0.0
        
        total_frames = self.stats['total_frames']
        nvenc_frames = self.stats['nvenc_frames']
        software_frames = self.stats['software_frames']
        errors = self.stats['encoding_errors']
        fallbacks = self.stats['fallback_count']
        
        nvenc_ratio = (nvenc_frames / max(1, total_frames)) * 100
        software_ratio = (software_frames / max(1, total_frames)) * 100
        error_rate = (errors / max(1, total_frames + errors)) * 100
        
        avg_fps = total_frames / max(0.001, total_time)
        
        self.logger.info("📊 최종 인코딩 통계:")
        self.logger.info(f"  • 총 프레임: {total_frames}")
        self.logger.info(f"  • NVENC: {nvenc_frames} ({nvenc_ratio:.1f}%)")
        self.logger.info(f"  • 소프트웨어: {software_frames} ({software_ratio:.1f}%)")
        self.logger.info(f"  • 폴백 횟수: {fallbacks}")
        self.logger.info(f"  • 에러 비율: {error_rate:.1f}%")
        self.logger.info(f"  • 평균 FPS: {avg_fps:.1f}")
        self.logger.info(f"  • 총 처리 시간: {total_time:.2f}초")
    
    def get_stats(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            stats['elapsed_time'] = time.perf_counter() - stats['start_time']
            stats['current_fps'] = stats['total_frames'] / max(0.001, stats['elapsed_time'])
        else:
            stats['elapsed_time'] = 0.0
            stats['current_fps'] = 0.0
        
        stats['encoder_type'] = self.encoder_type
        stats['session_id'] = self.session_id
        stats['session_acquired'] = self.session_acquired
        
        return stats
    
    @property 
    def is_nvenc_active(self) -> bool:
        """NVENC 인코더 활성 상태"""
        return self.encoder_type == "nvenc" and self.nvenc_encoder is not None
    
    @property
    def is_software_active(self) -> bool:
        """소프트웨어 인코더 활성 상태"""
        return self.encoder_type == "software" and self.software_encoder is not None


@asynccontextmanager
async def create_enhanced_encoder(
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    **kwargs
):
    """
    향상된 인코더 컨텍스트 매니저
    
    사용 예:
        async with create_enhanced_encoder("output.mp4") as encoder:
            await encoder.encode_frame(frame)
    """
    encoder = EnhancedEncoder(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        **kwargs
    )
    
    try:
        success = await encoder.initialize()
        if not success:
            raise EncodingError("인코더 초기화 실패")
        
        yield encoder
        
    finally:
        await encoder.close()