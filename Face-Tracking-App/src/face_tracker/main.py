"""
Face-Tracking-App 메인 진입점
"""
import sys
import os
from datetime import datetime

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
    elif mode == "dual_split":
        import src.face_tracker.dual_split_config as mode_config
        logger.info("🎯 DUAL_SPLIT 모드로 실행 (화면분할 2인 추적)")
        logger.debug("🎯 DEBUG: DUAL_SPLIT 모드 선택됨")
        logger.debug(f"🎯 DEBUG: dual_split_config 모듈 로드됨: {mode_config.__file__}")
        print("🎯 DEBUG: DUAL_SPLIT 모드 선택됨")  # 콘솔 출력용
    else:
        raise ValueError(f"지원하지 않는 모드: {mode}")
    
    # 모드별 설정을 config 모듈에 동적 적용
    config_module.TRACKING_MODE = mode_config.TRACKING_MODE
    config_module.OUTPUT_ROOT = mode_config.OUTPUT_ROOT
    
    logger.info(f"현재 설정 모드: {mode_config.TRACKING_MODE}")
    logger.debug(f"🔍 DEBUG: 현재 TRACKING_MODE: {config_module.TRACKING_MODE}")                    
    logger.info(f"출력 경로: {mode_config.OUTPUT_ROOT}")
    
    # DUAL_SPLIT 모드의 경우 추가 디버그 정보
    if mode == "dual_split":
        print(f"🎯 CONSOLE: DUAL_SPLIT 모드 분기 진입! ({datetime.now().strftime('%H:%M:%S')})")
        logger.debug("🎯 DEBUG: DUAL_SPLIT 모드 분기 진입!")
        logger.debug(f"🔍 DEBUG: TRACKING_MODE 확인: {config_module.TRACKING_MODE}")
        logger.debug(f"🔍 DEBUG: OUTPUT_ROOT 확인: {config_module.OUTPUT_ROOT}")
        print(f"🔍 CONSOLE: 설정 완료 - TRACKING_MODE: {config_module.TRACKING_MODE}")
    
    # 통합 로거로 세션 관리
    print(f"🔍 CONSOLE: process_all_videos() 호출 시작... ({datetime.now().strftime('%H:%M:%S')})")
    with logger.session_context():
        logger.debug("🔍 DEBUG: session_context 진입, process_all_videos() 호출...")
        process_all_videos()
        logger.debug("🔍 DEBUG: process_all_videos() 완료")
    print(f"🔍 CONSOLE: process_all_videos() 완료 ({datetime.now().strftime('%H:%M:%S')})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face-Tracking-App')
    parser.add_argument('--mode', choices=['single', 'dual', 'dual_split'], 
                       help='실행 모드 (single: 1인 화자, dual: 2인 화자, dual_split: 화면분할 2인 추적)')
    
    args = parser.parse_args()
    main(args.mode)