#!/usr/bin/env python3
"""
PyAV 하드웨어 가속 코덱 확인 스크립트
NVDEC, NVENC, CUDA 지원 코덱 검색
"""

import av

print("="*60)
print("🎬 PyAV 하드웨어 가속 코덱 확인")
print("="*60)

print(f"\n📦 PyAV 버전: {av.__version__}")
print(f"전체 코덱 수: {len(av.codec.codecs_available)}")

print("\n🔍 하드웨어 가속 코덱 검색 중...")
print("-"*40)

hardware_keywords = ['nvdec', 'cuda', 'nvenc', 'cuvid', 'npp']
found_codecs = []

for codec in sorted(av.codec.codecs_available):
    codec_lower = codec.lower()
    for keyword in hardware_keywords:
        if keyword in codec_lower:
            found_codecs.append(codec)
            break

if found_codecs:
    print(f"✅ 하드웨어 가속 코덱 {len(found_codecs)}개 발견:")
    for codec in found_codecs:
        print(f"   - {codec}")
else:
    print("⚠️ 하드웨어 가속 코덱을 찾을 수 없습니다")

print("\n🎥 H.264/H.265 관련 코덱:")
print("-"*40)
h264_codecs = [c for c in av.codec.codecs_available if 'h264' in c.lower() or 'h265' in c.lower() or 'hevc' in c.lower()]
for codec in sorted(h264_codecs):
    print(f"   - {codec}")

print("\n💡 FFmpeg 하드웨어 지원 확인:")
print("-"*40)
print("다음 명령어로 시스템 FFmpeg 확인:")
print("  ffmpeg -codecs | grep -E 'nvdec|nvenc|cuda'")
print("  ffmpeg -hwaccels")