"""
Face-Tracking-App 메인 진입점
"""
from processors.video_processor import process_all_videos
from utils.console_logger import ConsoleLogger


if __name__ == "__main__":
    # 모든 콘솔 출력을 log.log에 저장
    with ConsoleLogger("log.log"):
        process_all_videos()