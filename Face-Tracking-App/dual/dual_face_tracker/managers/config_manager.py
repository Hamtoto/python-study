"""
HybridConfigManager - 하이브리드 설정 관리 시스템

사용자 수동 설정 → 자동 프로빙 → 안전한 기본값 3단계 우선순위 시스템
"""

import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from ..utils.exceptions import DualFaceTrackerError
from .hardware_prober import HardwareProber


class HybridConfigManager:
    """
    하이브리드 설정 관리자
    
    설정 우선순위:
    1. manual_config.yaml - 사용자 수동 설정 (최우선)
    2. auto_detected.yaml - 자동 프로빙 결과
    3. fallback_config.yaml - 안전한 기본값
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            config_dir: 설정 파일 디렉토리 경로. None이면 현재 디렉토리 사용
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.config_priority = [
            'manual_config.yaml',      # 1순위: 사용자 수동 설정
            'auto_detected.yaml',      # 2순위: 자동 프로빙 결과
            'fallback_config.yaml'     # 3순위: 안전한 기본값
        ]
        self.hardware_prober = HardwareProber()
        self.current_config: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)
        
    def load_optimal_config(self) -> Dict[str, Any]:
        """
        최적 설정 로드 (우선순위 기반)
        
        Returns:
            Dict[str, Any]: 로드된 설정 딕셔너리
            
        Raises:
            DualFaceTrackerError: 모든 설정 로드 실패시
        """
        self.logger.info("🔧 하이브리드 설정 관리 시작...")
        
        # 1단계: 수동 설정 파일 확인
        manual_config_path = self.config_dir / 'manual_config.yaml'
        if self._exists_and_valid(manual_config_path):
            self.logger.info("✅ 사용자 수동 설정 발견 - 최우선 적용")
            self.current_config = self._load_yaml(manual_config_path)
            return self.current_config
            
        # 2단계: 자동 프로빙 실행
        self.logger.info("🔍 하드웨어 자동 프로빙 실행 중...")
        try:
            auto_config = self.hardware_prober.generate_optimal_config()
            auto_config_path = self.config_dir / 'auto_detected.yaml'
            self._save_yaml(auto_config_path, auto_config)
            self.logger.info("✅ 자동 프로빙 성공 - 감지된 설정 적용")
            self.current_config = auto_config
            return self.current_config
        except Exception as e:
            self.logger.warning(f"⚠️ 자동 프로빙 실패: {e}")
            
        # 3단계: 안전한 기본값 사용
        self.logger.info("🛡️ 기본 안전 설정 적용")
        fallback_config_path = self.config_dir / 'fallback_config.yaml'
        if self._exists_and_valid(fallback_config_path):
            self.current_config = self._load_yaml(fallback_config_path)
            return self.current_config
        else:
            # fallback_config.yaml이 없으면 하드코딩된 기본값 사용
            self.current_config = self._get_hardcoded_defaults()
            return self.current_config
            
    def allow_user_override(self, section: str, key: str, value: Any) -> None:
        """
        사용자 설정 재정의 허용
        
        Args:
            section: 설정 섹션명
            key: 설정 키
            value: 설정값
        """
        override_config = {
            section: {key: value},
            'override_timestamp': datetime.now().isoformat(),
            'override_reason': f'User manual override for {section}.{key}'
        }
        
        manual_config_path = self.config_dir / 'manual_config.yaml'
        
        # 기존 manual_config.yaml이 있으면 병합
        if manual_config_path.exists():
            existing_config = self._load_yaml(manual_config_path)
            if section in existing_config:
                existing_config[section].update(override_config[section])
            else:
                existing_config[section] = override_config[section]
            existing_config.update({
                'override_timestamp': override_config['override_timestamp'],
                'override_reason': override_config['override_reason']
            })
        else:
            existing_config = override_config
            
        self._save_yaml(manual_config_path, existing_config)
        self.logger.info(f"✅ 사용자 재정의 저장: {section}.{key} = {value}")
        
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """
        설정값 조회 (우선순위 적용)
        
        Args:
            section: 설정 섹션명
            key: 설정 키
            default: 기본값
            
        Returns:
            Any: 설정값
        """
        if self.current_config is None:
            self.load_optimal_config()
            
        return self.current_config.get(section, {}).get(key, default)
        
    def _exists_and_valid(self, config_path: Path) -> bool:
        """
        설정 파일 존재 및 유효성 확인
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            bool: 유효한 설정 파일 여부
        """
        if not config_path.exists():
            return False
            
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                yaml.safe_load(file)
            return True
        except yaml.YAMLError as e:
            self.logger.warning(f"⚠️ YAML 파싱 오류: {config_path} - {e}")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ 파일 읽기 오류: {config_path} - {e}")
            return False
            
    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """
        YAML 파일 로드
        
        Args:
            config_path: YAML 파일 경로
            
        Returns:
            Dict[str, Any]: 로드된 설정
            
        Raises:
            DualFaceTrackerError: 로드 실패시
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            raise DualFaceTrackerError(f"설정 파일 로드 실패: {config_path} - {e}")
            
    def _save_yaml(self, config_path: Path, config: Dict[str, Any]) -> None:
        """
        YAML 파일 저장
        
        Args:
            config_path: 저장할 파일 경로
            config: 저장할 설정 딕셔너리
            
        Raises:
            DualFaceTrackerError: 저장 실패시
        """
        try:
            # 디렉토리 생성
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            raise DualFaceTrackerError(f"설정 파일 저장 실패: {config_path} - {e}")
            
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """
        하드코딩된 안전한 기본값 반환
        
        Returns:
            Dict[str, Any]: 기본 설정
        """
        return {
            'hardware': {
                'gpu_name': 'Unknown GPU',
                'nvdec_max_sessions': 2,  # 안전한 기본값
                'nvenc_max_sessions': 2,
                'vram_gb': 8,
                'driver_version': 'Unknown'
            },
            'performance': {
                'max_concurrent_streams': 2,  # 보수적 기본값
                'batch_size_analyze': 32,
                'vram_safety_margin': 0.25,  # 25% 안전 마진
                'target_gpu_utilization': 0.7  # 70% 목표
            },
            'nvdec_settings': {
                'max_sessions': 2,
                'preferred_format': 'nv12'
            },
            'nvenc_settings': {
                'max_sessions': 2,
                'preset': 'medium',
                'rc_mode': 'cbr'
            },
            'fallback_timestamp': datetime.now().isoformat(),
            'fallback_reason': 'Hardcoded safe defaults used'
        }