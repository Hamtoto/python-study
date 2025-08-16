#!/usr/bin/env python3
"""
NVENC 문제 시 소프트웨어 인코더 폴백 테스트
"""

import cv2
import numpy as np
from pathlib import Path
import time

def test_software_encoding():
    """소프트웨어 H.264 인코딩 테스트"""
    print("🧪 소프트웨어 H.264 인코딩 테스트")
    
    try:
        output_path = Path("test_output/software_test.mp4")
        output_path.parent.mkdir(exist_ok=True)
        
        # OpenCV VideoWriter로 소프트웨어 인코딩
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 코덱
        writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (640, 480))
        
        if not writer.isOpened():
            print("❌ VideoWriter 열기 실패")
            return False
        
        print("✅ VideoWriter 열기 성공")
        
        # 30프레임 테스트 생성
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
        
        writer.release()
        
        # 결과 확인
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ 소프트웨어 인코딩 성공: {output_path} ({output_path.stat().st_size} bytes)")
            return True
        else:
            print("❌ 출력 파일 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ 소프트웨어 인코딩 실패: {e}")
        return False

def test_cpu_h264_encoding():
    """CPU H.264 인코딩 테스트 (libx264)"""
    print("\n🧪 CPU H.264 (libx264) 인코딩 테스트")
    
    try:
        import subprocess
        
        output_path = Path("test_output/cpu_h264_test.mp4")
        
        # FFmpeg으로 직접 CPU 인코딩
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', '640x480',
            '-r', '30',
            '-i', 'pipe:0',
            '-c:v', 'libx264',  # 소프트웨어 H.264
            '-preset', 'ultrafast',
            '-crf', '23',
            str(output_path)
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 30프레임 생성하여 FFmpeg에 전송
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            process.stdin.write(frame.tobytes())
        
        process.stdin.close()
        _, stderr = process.communicate(timeout=30)
        
        if process.returncode == 0 and output_path.exists():
            print(f"✅ CPU H.264 인코딩 성공: {output_path} ({output_path.stat().st_size} bytes)")
            return True
        else:
            print(f"❌ CPU H.264 인코딩 실패: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ CPU H.264 인코딩 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔧 NVENC 대신 소프트웨어 인코더 테스트")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    Path("test_output").mkdir(exist_ok=True)
    
    success1 = test_software_encoding()
    success2 = test_cpu_h264_encoding()
    
    if success1 or success2:
        print("\n🎉 소프트웨어 인코딩 사용 가능!")
        print("NVENC 없이도 멀티스트림 테스트를 진행할 수 있습니다.")
    else:
        print("\n❌ 모든 인코딩 방법 실패")
        print("시스템 설정을 확인해야 합니다.")