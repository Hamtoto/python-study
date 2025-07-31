"""
Face-Tracking-App 메인 진입점
"""
from src.face_tracker.processing.processor import process_all_videos
from src.face_tracker.utils.logging import logger


if __name__ == "__main__":
    # 통합 로거로 세션 관리
    with logger.session_context():
        process_all_videos()