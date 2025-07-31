#!/usr/bin/env python3
"""
로깅 시스템 테스트
"""
import sys
import os
sys.path.append('/home/hamtoto/work/python-study/Face-Tracking-App')

from src.face_tracker.utils.logging import logger

def test_logging():
    """로깅 시스템 테스트"""
    print("로깅 테스트 시작...")
    
    # 로그 파일 경로 확인
    print(f"로그 파일 경로: {logger.log_file}")
    
    # 테스트 로그 생성
    logger.info("테스트 정보 로그")
    logger.error("테스트 에러 로그")
    logger.success("테스트 성공 로그")
    logger.stage("테스트 단계 로그")
    logger.warning("테스트 경고 로그")
    
    # 로그 파일 존재 확인
    if os.path.exists(logger.log_file):
        print(f"로그 파일 생성됨: {logger.log_file}")
        with open(logger.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print("로그 파일 내용:")
            print(content)
    else:
        print(f"로그 파일 생성 안됨: {logger.log_file}")
    
    print("로깅 테스트 완료")

if __name__ == "__main__":
    test_logging()