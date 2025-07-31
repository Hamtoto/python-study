#!/usr/bin/env python3
"""
세그먼트 분할 문제 디버깅 스크립트
"""
import os
import sys
sys.path.append('/home/hamtoto/work/python-study/Face-Tracking-App')

from src.face_tracker.processing.trimmer import slice_video_parallel_ffmpeg
from src.face_tracker.utils.logging import logger

def test_segmentation():
    """세그먼트 분할 테스트"""
    
    # 테스트 파일 경로
    input_video = "/home/hamtoto/work/python-study/Face-Tracking-App/videos/input/sample.mp4"
    output_folder = "/tmp/debug_segments"
    
    print(f"입력 파일: {input_video}")
    print(f"출력 폴더: {output_folder}")
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_video):
        print("ERROR: 입력 파일이 존재하지 않습니다")
        return False
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    print("세그먼트 분할 시작...")
    
    try:
        # 세그먼트 분할 실행
        result = slice_video_parallel_ffmpeg(input_video, output_folder, segment_length=10)
        
        print(f"분할 결과: {result}")
        
        # 생성된 파일 확인
        if os.path.exists(output_folder):
            files = [f for f in os.listdir(output_folder) if f.lower().endswith('.mp4')]
            print(f"생성된 파일 수: {len(files)}개")
            
            for f in files:
                file_path = os.path.join(output_folder, f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size} bytes)")
        else:
            print("ERROR: 출력 폴더가 존재하지 않습니다")
            
    except Exception as e:
        print(f"ERROR: 세그먼트 분할 실패 - {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_segmentation()