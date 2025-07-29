
# -*- coding: utf-8 -*-
"""
프로젝트 전용 커스텀 예외 클래스 정의 모듈
"""

class VideoProcessingError(Exception):
    """비디오 처리 파이프라인의 특정 단계에서 발생하는 일반적인 오류"""
    def __init__(self, message="An error occurred during video processing.", details=None):
        super().__init__(message)
        self.details = details

    def __str__(self):
        if self.details:
            return f"{super().__str__()} Details: {self.details}"
        return super().__str__()

class GPUMemoryError(VideoProcessingError):
    """GPU 메모리 관련 오류 (예: OOM)"""
    def __init__(self, message="GPU out of memory.", details=None):
        super().__init__(message, details)

class InputValidationError(VideoProcessingError):
    """입력 파일(비디오)의 유효성 검사 실패 시 발생하는 오류"""
    def __init__(self, message="Input video validation failed.", details=None):
        super().__init__(message, details)

class FFmpegError(VideoProcessingError):
    """FFmpeg 명령어 실행 실패 시 발생하는 오류"""
    def __init__(self, message="FFmpeg command failed.", command=None, stderr=None):
        super().__init__(message)
        self.command = command
        self.stderr = stderr

    def __str__(self):
        error_str = super().__str__()
        if self.command:
            error_str += f"\n  Command: {' '.join(self.command)}"
        if self.stderr:
            error_str += f"\n  Stderr: {self.stderr}"
        return error_str

class ModelLoadError(VideoProcessingError):
    """딥러닝 모델 로딩 실패 시 발생하는 오류"""
    def __init__(self, message="Failed to load a model.", model_name=None):
        super().__init__(message)
        self.model_name = model_name

    def __str__(self):
        if self.model_name:
            return f"{super().__str__()} Model: {self.model_name}"
        return super().__str__()

