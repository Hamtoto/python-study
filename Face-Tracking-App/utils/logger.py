"""
에러 로그 수집 유틸리티
"""
import os
import logging
from datetime import datetime
from config import OUTPUT_ROOT


class ErrorLogger:
    """에러 로그만 수집하는 로거"""
    
    def __init__(self, log_file="error_log.txt"):
        self.log_file = os.path.join(OUTPUT_ROOT, log_file)
        self._setup_logger()
    
    def _setup_logger(self):
        """로거 설정"""
        # 로거 생성
        self.logger = logging.getLogger('error_logger')
        self.logger.setLevel(logging.ERROR)
        
        # 기존 핸들러 제거 (중복 방지)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.ERROR)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('[%(asctime)s][Error] %(message)s', 
                                    datefmt='%m.%dT%H:%M')
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
    
    def log_error(self, message: str):
        """에러 로그 기록"""
        self.logger.error(message)
    
    def log_segment_error(self, segment_name: str, error_msg: str):
        """세그먼트 처리 에러 로그"""
        self.log_error(f"{segment_name} - {error_msg}")
    
    def log_video_error(self, video_name: str, error_msg: str):
        """비디오 처리 에러 로그"""
        self.log_error(f"{video_name} - {error_msg}")
    
    def clear_log(self):
        """로그 파일 초기화"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)


# 전역 에러 로거 인스턴스
error_logger = ErrorLogger()