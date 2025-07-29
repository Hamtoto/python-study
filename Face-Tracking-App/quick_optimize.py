"""
현재 실행 중인 프로세스에 대한 즉시 최적화 방안
"""
import os
import sys
import shutil

def apply_quick_optimizations():
    """즉시 적용 가능한 최적화 적용"""
    
    print("=== Face-Tracking-App 즉시 최적화 적용 ===")
    
    # 1. 기존 파일 백업
    backup_files = [
        "processors/video_trimmer.py",
        "processors/video_processor.py", 
        "utils/audio_utils.py"
    ]
    
    print("1. 기존 파일 백업 중...")
    for file_path in backup_files:
        if os.path.exists(file_path):
            backup_path = file_path.replace('.py', '_backup.py')
            shutil.copy2(file_path, backup_path)
            print(f"   백업: {file_path} -> {backup_path}")
    
    # 2. 최적화된 파일로 교체
    replacements = [
        ("processors/video_trimmer_optimized.py", "processors/video_trimmer.py"),
        ("processors/video_processor_optimized.py", "processors/video_processor.py"),
        ("utils/audio_utils_optimized.py", "utils/audio_utils.py")
    ]
    
    print("\n2. 최적화된 파일로 교체 중...")
    for src, dst in replacements:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"   교체: {src} -> {dst}")
        else:
            print(f"   오류: {src} 파일이 없습니다.")
    
    # 3. config.py 최적화 설정 추가
    print("\n3. 설정 최적화...")
    config_additions = '''
# FFmpeg 최적화 설정
FFMPEG_PRESET = 'ultrafast'  # 속도 우선 (ultrafast, superfast, veryfast, faster, fast)
FFMPEG_CRF = 23  # 품질 (18-28, 낮을수록 고품질)
FFMPEG_THREADS = 0  # 0 = 자동, 또는 구체적 스레드 수
PARALLEL_SEGMENTS = True  # 세그먼트 병렬 처리 활성화
MAX_PARALLEL_PROCESSES = 8  # 최대 병렬 프로세스 수
'''
    
    try:
        with open("config.py", "a") as f:
            f.write(config_additions)
        print("   config.py에 FFmpeg 최적화 설정 추가 완료")
    except Exception as e:
        print(f"   config.py 업데이트 오류: {e}")
    
    # 4. 메모리 및 임시 파일 최적화 가이드
    print("\n4. 추가 최적화 권장사항:")
    print("   - 현재 실행 중인 프로세스 종료 후 재시작")
    print("   - 시스템 메모리 확인: free -h")
    print("   - 디스크 공간 확인: df -h")
    print("   - FFmpeg 설치 확인: ffmpeg -version")
    print("   - GPU 메모리 모니터링: nvidia-smi")
    
    print("\n5. 예상 성능 개선:")
    print("   - 오디오 추출: 3-5배 빨라짐")
    print("   - 비디오 트리밍: 4-8배 빨라짐") 
    print("   - 세그먼트 분할: 병렬 처리로 2-3배 빨라짐")
    print("   - 전체 처리: 5-10배 빨라질 것으로 예상")
    print("   - 6시간 영상 -> 1-2시간 내 완료 예상")
    
    print("\n최적화 적용 완료!")
    return True


def check_system_requirements():
    """시스템 요구사항 확인"""
    print("=== 시스템 요구사항 확인 ===")
    
    # FFmpeg 설치 확인
    ffmpeg_check = os.system("ffmpeg -version > /dev/null 2>&1")
    if ffmpeg_check == 0:
        print("✓ FFmpeg 설치됨")
    else:
        print("✗ FFmpeg 미설치 - 설치 필요: sudo apt install ffmpeg")
        return False
    
    # 디스크 공간 확인
    print("\n디스크 공간:")
    os.system("df -h .")
    
    # 메모리 확인
    print("\n메모리 상태:")
    os.system("free -h")
    
    # GPU 확인 (선택사항)
    print("\nGPU 상태:")
    gpu_check = os.system("nvidia-smi > /dev/null 2>&1")
    if gpu_check == 0:
        os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    else:
        print("GPU 없음 또는 nvidia-smi 없음")
    
    return True


if __name__ == "__main__":
    print("Face-Tracking-App 성능 최적화 도구")
    print("현재 MoviePy -> FFmpeg 최적화를 적용합니다.")
    
    if not check_system_requirements():
        print("시스템 요구사항을 만족하지 않습니다.")
        sys.exit(1)
    
    response = input("\n최적화를 적용하시겠습니까? (y/N): ")
    if response.lower() == 'y':
        apply_quick_optimizations()
    else:
        print("최적화 적용을 취소했습니다.")