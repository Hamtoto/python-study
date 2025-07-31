"""
통합 로깅 시스템 - 단일 로그 파일로 모든 로그 관리
"""
import os
import sys
import logging
from datetime import datetime
from contextlib import contextmanager


class UnifiedLogger:
    """모든 로그를 하나의 파일에 통합 관리하는 로거"""
    
    def __init__(self, log_file="face_tracker.log"):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._setup_logger()
    
    def _setup_logger(self):
        """통합 로거 설정"""
        # 로거 생성
        self.logger = logging.getLogger('face_tracker')
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 (중복 방지)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 로그 포맷 설정 (간결하게)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', 
                                    datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """정보 로그"""
        self.logger.info(message)
    
    def error(self, message: str):
        """에러 로그"""
        self.logger.error(f"❌ {message}")
    
    def success(self, message: str):
        """성공 로그"""
        self.logger.info(f"✅ {message}")
    
    def stage(self, message: str):
        """단계별 로그"""
        self.logger.info(f"🔄 {message}")
    
    def warning(self, message: str):
        """경고 로그"""
        self.logger.warning(f"⚠️ {message}")
    
    def clear_log(self):
        """로그 파일 초기화"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    
    @contextmanager
    def session_context(self):
        """처리 세션 컨텍스트"""
        # 세션 시작
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Face-Tracking-App 시작: {timestamp}")
        self.logger.info(f"{'='*50}")
        
        try:
            yield self
        except Exception as e:
            self.error(f"처리 중 오류 발생: {str(e)}")
            raise
        finally:
            # 세션 종료
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"Face-Tracking-App 완료: {end_time}")
            self.logger.info(f"{'='*50}")


class TeeOutput:
    """콘솔과 파일에 동시 출력 (기존 코드와의 호환성 유지)"""
    def __init__(self, console, file_handle):
        self.console = console
        self.file_handle = file_handle
        
    def write(self, text):
        # 콘솔에 출력
        self.console.write(text)
        # 파일에도 저장
        self.file_handle.write(text)
        self.file_handle.flush()
        
    def flush(self):
        self.console.flush()
        self.file_handle.flush()
        
    def __getattr__(self, name):
        return getattr(self.console, name)


# 전역 통합 로거 인스턴스
logger = UnifiedLogger()

# 하위 호환성을 위한 에러 로거 인터페이스
class ErrorLoggerCompat:
    """기존 error_logger 코드와의 호환성 유지"""
    def log_error(self, message: str):
        logger.error(message)
    
    def log_segment_error(self, segment_name: str, error_msg: str):
        logger.error(f"{segment_name} - {error_msg}")
    
    def log_video_error(self, video_name: str, error_msg: str):
        logger.error(f"{video_name} - {error_msg}")

error_logger = ErrorLoggerCompat()

# 하위 호환성을 위한 ConsoleLogger
class ConsoleLogger:
    """기존 ConsoleLogger와의 호환성 유지"""
    def __init__(self, log_file=None):
        pass  # 이제 단일 로거를 사용하므로 별도 설정 불필요
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass