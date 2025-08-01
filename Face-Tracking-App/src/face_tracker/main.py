"""
Face-Tracking-App 메인 진입점
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.face_tracker.processing.processor import process_all_videos
from src.face_tracker.utils.logging import logger


def main(mode="single"):
    """
    Face-Tracking-App 메인 실행 함수
    
    Args:
        mode: 실행 모드 ("single" 또는 "dual")
    """
    # 공통 config 로드
    import src.face_tracker.config as config_module
    
    # 모드별 config 로드 및 설정 적용
    if mode == "single":
        import src.face_tracker.single_config as mode_config
        logger.info("🎯 SINGLE 모드로 실행 (1인 화자)")
    elif mode == "dual":
        import src.face_tracker.dual_config as mode_config
        logger.info("🎯 DUAL 모드로 실행 (2인 화자)")
    else:
        raise ValueError(f"지원하지 않는 모드: {mode}")
    
    # 모드별 설정을 config 모듈에 동적 적용
    config_module.TRACKING_MODE = mode_config.TRACKING_MODE
    config_module.OUTPUT_ROOT = mode_config.OUTPUT_ROOT
    
    logger.info(f"현재 설정 모드: {mode_config.TRACKING_MODE}")
    logger.info(f"출력 경로: {mode_config.OUTPUT_ROOT}")
    
    # 통합 로거로 세션 관리
    with logger.session_context():
        process_all_videos()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face-Tracking-App')
    parser.add_argument('--mode', choices=['single', 'dual'], 
                       help='실행 모드 (single: 1인 화자, dual: 2인 화자)')
    
    args = parser.parse_args()
    main(args.mode)