# -*- coding: utf-8 -*-
"""
입력 파일 유효성 검증 모듈
"""
import os
import subprocess
from src.face_tracker.utils.exceptions import InputValidationError
from src.face_tracker.utils.logging import logger

def validate_video_file(file_path: str):
    """
    비디오 파일의 유효성을 검증합니다.

    1. 파일 존재 여부 확인
    2. ffprobe를 사용하여 비디오 스트림이 있는지 확인

    Args:
        file_path: 검증할 비디오 파일의 경로

    Raises:
        InputValidationError: 파일이 유효하지 않을 경우 발생
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise InputValidationError(f"File does not exist: {file_path}")

    # 2. ffprobe로 비디오 스트림 확인
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',  # 비디오 스트림 선택
            '-show_entries', 'stream=codec_name',  # 코덱 이름 요청
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        
        if not result.stdout.strip():
            raise InputValidationError(
                message="No video stream found in the file.",
                details=f"File: {file_path}"
            )
        
        logger.success(f"입력 파일 검증 성공: {os.path.basename(file_path)}")
        return True

    except subprocess.CalledProcessError as e:
        raise InputValidationError(
            message="ffprobe failed to analyze the file. It might be corrupted or not a valid video file.",
            details=f"File: {file_path}\nStderr: {e.stderr}"
        )
    except subprocess.TimeoutExpired:
        raise InputValidationError(
            message="ffprobe timed out while analyzing the file.",
            details=f"File: {file_path}"
        )
    except FileNotFoundError:
        raise InputValidationError(
            message="ffprobe command not found. Please ensure FFmpeg is installed and in your system's PATH."
        )