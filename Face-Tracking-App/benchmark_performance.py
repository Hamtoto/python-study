"""
MoviePy vs FFmpeg 성능 비교 테스트
"""
import time
import os
import sys
from processors.video_trimmer import create_condensed_video as moviepy_condensed
from processors.video_trimmer_optimized import create_condensed_video_ffmpeg
from utils.audio_utils import get_voice_segments as moviepy_audio
from utils.audio_utils_optimized import get_voice_segments_ffmpeg


def benchmark_condensed_video(video_path, timeline, fps, output_dir):
    """요약본 생성 성능 비교"""
    print("=== 요약본 생성 성능 비교 ===")
    
    # MoviePy 버전
    moviepy_output = os.path.join(output_dir, "condensed_moviepy.mp4")
    print("\n1. MoviePy 버전 테스트...")
    start_time = time.time()
    try:
        success_moviepy = moviepy_condensed(video_path, moviepy_output, timeline, fps)
        moviepy_time = time.time() - start_time
        print(f"MoviePy 완료: {moviepy_time:.2f}초, 성공: {success_moviepy}")
    except Exception as e:
        moviepy_time = time.time() - start_time
        print(f"MoviePy 오류: {e}, 소요시간: {moviepy_time:.2f}초")
        success_moviepy = False
    
    # FFmpeg 버전
    ffmpeg_output = os.path.join(output_dir, "condensed_ffmpeg.mp4")
    print("\n2. FFmpeg 버전 테스트...")
    start_time = time.time()
    try:
        success_ffmpeg = create_condensed_video_ffmpeg(video_path, ffmpeg_output, timeline, fps)
        ffmpeg_time = time.time() - start_time
        print(f"FFmpeg 완료: {ffmpeg_time:.2f}초, 성공: {success_ffmpeg}")
    except Exception as e:
        ffmpeg_time = time.time() - start_time
        print(f"FFmpeg 오류: {e}, 소요시간: {ffmpeg_time:.2f}초")
        success_ffmpeg = False
    
    # 결과 비교
    if success_moviepy and success_ffmpeg:
        speedup = moviepy_time / ffmpeg_time
        print(f"\n성능 개선: {speedup:.2f}배 빨라짐")
        print(f"시간 절약: {moviepy_time - ffmpeg_time:.2f}초")
    
    return moviepy_time, ffmpeg_time


def benchmark_audio_extraction(video_path):
    """오디오 추출 성능 비교"""
    print("\n=== 오디오 추출 성능 비교 ===")
    
    # MoviePy 버전
    print("\n1. MoviePy 버전 테스트...")
    start_time = time.time()
    try:
        moviepy_segments = moviepy_audio(video_path)
        moviepy_time = time.time() - start_time
        print(f"MoviePy 완료: {moviepy_time:.2f}초, 세그먼트 수: {len(moviepy_segments)}")
    except Exception as e:
        moviepy_time = time.time() - start_time
        print(f"MoviePy 오류: {e}, 소요시간: {moviepy_time:.2f}초")
        moviepy_segments = []
    
    # FFmpeg 버전
    print("\n2. FFmpeg 버전 테스트...")
    start_time = time.time()
    try:
        ffmpeg_segments = get_voice_segments_ffmpeg(video_path)
        ffmpeg_time = time.time() - start_time
        print(f"FFmpeg 완료: {ffmpeg_time:.2f}초, 세그먼트 수: {len(ffmpeg_segments)}")
    except Exception as e:
        ffmpeg_time = time.time() - start_time
        print(f"FFmpeg 오류: {e}, 소요시간: {ffmpeg_time:.2f}초")
        ffmpeg_segments = []
    
    # 결과 비교
    if moviepy_segments and ffmpeg_segments:
        speedup = moviepy_time / ffmpeg_time
        print(f"\n성능 개선: {speedup:.2f}배 빨라짐")
        print(f"시간 절약: {moviepy_time - ffmpeg_time:.2f}초")
    
    return moviepy_time, ffmpeg_time


def estimate_total_processing_time(video_duration_hours, current_fps=10.83, optimized_speedup=5.0):
    """전체 처리 시간 예측"""
    print(f"\n=== 전체 처리 시간 예측 ===")
    print(f"비디오 길이: {video_duration_hours}시간")
    print(f"현재 처리 속도: {current_fps} it/s")
    
    # 현재 예상 시간
    total_frames = video_duration_hours * 3600 * 30  # 30fps 가정
    current_total_time = total_frames / current_fps / 3600  # 시간 단위
    
    # 최적화 후 예상 시간
    optimized_total_time = current_total_time / optimized_speedup
    
    print(f"현재 예상 완료 시간: {current_total_time:.1f}시간")
    print(f"최적화 후 예상 완료 시간: {optimized_total_time:.1f}시간")
    print(f"시간 절약: {current_total_time - optimized_total_time:.1f}시간")


if __name__ == "__main__":
    # 테스트 설정
    video_path = "./videos/input/70.mp4"
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"테스트 비디오 파일이 없습니다: {video_path}")
        sys.exit(1)
    
    # 샘플 타임라인 생성 (실제 분석 없이 테스트용)
    sample_timeline = [True] * 1000 + [False] * 500 + [True] * 800  # 예시 타임라인
    fps = 30.0
    
    print("FFmpeg vs MoviePy 성능 벤치마크 시작...")
    
    # 1. 오디오 추출 테스트
    audio_moviepy_time, audio_ffmpeg_time = benchmark_audio_extraction(video_path)
    
    # 2. 요약본 생성 테스트 (작은 샘플로 테스트)
    # condensed_moviepy_time, condensed_ffmpeg_time = benchmark_condensed_video(video_path, sample_timeline, fps, output_dir)
    
    # 3. 전체 처리 시간 예측
    estimate_total_processing_time(6.0)  # 6시간 비디오
    
    print("\n벤치마크 완료!")