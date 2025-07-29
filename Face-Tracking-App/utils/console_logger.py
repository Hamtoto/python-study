"""
콘솔 출력을 log.log 파일에 저장하는 로거
"""
import sys
import os
from datetime import datetime


class ConsoleLogger:
    def __init__(self, log_file="log.log"):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        """컨텍스트 매니저 시작 - 로깅 활성화"""
        self.log_handle = open(self.log_file, 'a', encoding='utf-8')
        
        # 세션 시작 마크
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_handle.write(f"\n{'='*60}\n")
        self.log_handle.write(f"Face-Tracking-App 실행 시작: {timestamp}\n")
        self.log_handle.write(f"{'='*60}\n")
        self.log_handle.flush()
        
        # stdout, stderr을 파일과 콘솔 동시 출력으로 변경
        sys.stdout = TeeOutput(self.original_stdout, self.log_handle)
        sys.stderr = TeeOutput(self.original_stderr, self.log_handle)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 - 원래 출력으로 복원"""
        # 세션 종료 마크
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_handle.write(f"\nFace-Tracking-App 실행 완료: {timestamp}\n")
        self.log_handle.write(f"{'='*60}\n\n")
        
        # 원래 출력으로 복원
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        self.log_handle.close()


class TeeOutput:
    """콘솔과 파일에 동시 출력하는 클래스"""
    def __init__(self, console, file_handle):
        self.console = console
        self.file_handle = file_handle
        
    def write(self, text):
        # 콘솔에 출력
        self.console.write(text)
        # 파일에도 저장
        self.file_handle.write(text)
        self.file_handle.flush()  # 즉시 파일에 쓰기
        
    def flush(self):
        self.console.flush()
        self.file_handle.flush()
        
    def __getattr__(self, name):
        # 기타 속성들은 콘솔 객체에서 가져오기
        return getattr(self.console, name)