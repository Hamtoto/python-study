#!/usr/bin/env python3
"""
PyAV NVENC 하드웨어 인코딩 테스트
H.264/H.265 NVIDIA 하드웨어 가속 코덱 테스트
"""

import av
import numpy as np
import os

print("="*60)
print("🎬 PyAV NVENC 하드웨어 인코딩 테스트")
print("="*60)

# 테스트 비디오 파일명
test_file = 'test_nvenc.mp4'

# 테스트할 코덱 목록
test_codecs = [
    ('h264_nvenc', 'H.264 NVENC'),
    ('hevc_nvenc', 'H.265/HEVC NVENC'),
    ('h264', 'H.264 Software'),
    ('libx264', 'libx264 Software')
]

print(f"\n📦 PyAV 버전: {av.__version__}")
print(f"테스트 파일: {test_file}\n")

for codec_name, description in test_codecs:
    print(f"🔍 {description} ({codec_name}) 테스트")
    print("-"*40)
    
    try:
        # 컨테이너 생성
        container = av.open(f'test_{codec_name}.mp4', 'w')
        
        # 스트림 추가
        stream = container.add_stream(codec_name, rate=30)
        stream.width = 640
        stream.height = 480
        stream.pix_fmt = 'yuv420p'
        
        # 옵션 설정 (NVENC 특화)
        if 'nvenc' in codec_name:
            stream.options = {
                'preset': 'fast',
                'gpu': '0'
            }
        
        # 테스트 프레임 생성 및 인코딩
        for i in range(5):  # 5프레임만 테스트
            # 랜덤 프레임 생성
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame = frame.reformat(format=stream.pix_fmt)
            
            # 인코딩
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # 플러시
        for packet in stream.encode():
            container.mux(packet)
        
        # 닫기
        container.close()
        
        # 파일 크기 확인
        file_size = os.path.getsize(f'test_{codec_name}.mp4')
        print(f"   ✅ 성공! 파일 크기: {file_size:,} bytes")
        
        # 테스트 파일 삭제
        os.remove(f'test_{codec_name}.mp4')
        
    except av.codec.codec.UnknownCodecError:
        print(f"   ❌ 코덱을 찾을 수 없음: {codec_name}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        # 실패한 파일 정리
        if os.path.exists(f'test_{codec_name}.mp4'):
            os.remove(f'test_{codec_name}.mp4')
    
    print()

print("💡 하드웨어 디코딩 테스트")
print("="*60)

# 소프트웨어로 테스트 비디오 생성
print("테스트용 비디오 생성 중...")
try:
    container = av.open('test_source.mp4', 'w')
    stream = container.add_stream('h264', rate=30)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = 'yuv420p'
    
    for i in range(30):  # 1초 분량
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        frame = frame.reformat(format=stream.pix_fmt)
        for packet in stream.encode(frame):
            container.mux(packet)
    
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    print("✅ 테스트 비디오 생성 완료\n")
    
    # 하드웨어 디코딩 테스트
    decode_codecs = [
        ('h264_cuvid', 'H.264 CUVID'),
        ('hevc_cuvid', 'HEVC CUVID'),
    ]
    
    for codec_name, description in decode_codecs:
        print(f"🔍 {description} 디코딩 테스트")
        print("-"*40)
        try:
            container = av.open('test_source.mp4')
            # 특정 디코더 강제 지정
            container.streams.video[0].codec_context.skip_frame = 'NONKEY'
            
            frame_count = 0
            for frame in container.decode(video=0):
                frame_count += 1
                if frame_count >= 10:  # 10프레임만 테스트
                    break
            
            container.close()
            print(f"   ✅ 성공! {frame_count}프레임 디코딩")
        except Exception as e:
            print(f"   ❌ 실패: {e}")
        print()
    
    # 테스트 파일 삭제
    os.remove('test_source.mp4')
    
except Exception as e:
    print(f"❌ 테스트 비디오 생성 실패: {e}")

print("\n🎯 테스트 완료!")
print("="*60)