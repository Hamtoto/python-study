"""
최적화된 오디오 처리 유틸리티 - FFmpeg 및 멀티프로세싱 활용
"""
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.exceptions import FFmpegError
from config import AUDIO_SAMPLE_RATE, AUDIO_FRAME_DURATION


def get_voice_segments_ffmpeg(audio_path: str, threshold_db: int = -30, min_silence_duration: float = 0.5):
    """
    FFmpeg의 silencedetect 필터를 사용한 고속 음성 구간 검출
    
    Args:
        audio_path: 오디오 파일 경로
        threshold_db: 침묵으로 간주할 데시벨 임계값
        min_silence_duration: 최소 침묵 지속 시간 (초)
    
    Returns:
        list[tuple[float, float]]: 음성 구간 (시작, 끝) 리스트
    """
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-af', f'silencedetect=n={threshold_db}dB:d={min_silence_duration}',
        '-f', 'null', '-'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stderr_output = result.stderr
        
        silence_starts = [float(line.split(' ')[-1]) for line in stderr_output.splitlines() if 'silence_start' in line]
        silence_ends = [float(line.split(' ')[-1]) for line in stderr_output.splitlines() if 'silence_end' in line]
        
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(duration_result.stdout.strip())
        
        voice_segments = []
        last_silence_end = 0.0
        
        for start, end in zip(silence_starts, silence_ends):
            if start > last_silence_end:
                voice_segments.append((last_silence_end, start))
            last_silence_end = end
            
        if last_silence_end < duration:
            voice_segments.append((last_silence_end, duration))
            
        return voice_segments
        
    except (subprocess.CalledProcessError, ValueError) as e:
        raise FFmpegError(command=cmd, stderr=getattr(e, 'stderr', ''))

def extract_audio_segment(args):
    """멀티프로세싱을 위한 오디오 세그먼트 추출 함수"""
    video_path, segment, temp_dir = args
    start, end = segment
    
    temp_audio_path = os.path.join(temp_dir, f"temp_audio_{start}_{end}.wav")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(start),
        '-to', str(end),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(AUDIO_SAMPLE_RATE),
        '-ac', '1',
        temp_audio_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        print(f"오디오 세그먼트 추출 실패: {os.path.basename(video_path)} ({start}-{end}), Stderr: {e.stderr}")
        return None

def get_voice_segments_parallel(video_path: str, num_workers: int = 4):
    """
    멀티프로세싱을 활용한 병렬 음성 구간 검출
    """
    print("## 1단계: 병렬 음성 구간(VAD) 분석 시작")
    
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise FFmpegError(command=cmd, stderr=getattr(e, 'stderr', ''))

    chunk_size = duration / num_workers
    segments = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tasks = [(video_path, seg, temp_dir) for seg in segments]
        
        all_voice_segments = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_segment = {executor.submit(extract_audio_segment, task): task for task in tasks}
            
            for future in as_completed(future_to_segment):
                audio_chunk_path = future.result()
                if audio_chunk_path:
                    chunk_voice_segments = get_voice_segments_ffmpeg(audio_chunk_path)
                    original_start_time = future_to_segment[future][1][0]
                    for start, end in chunk_voice_segments:
                        all_voice_segments.append((start + original_start_time, end + original_start_time))
                    
                    os.remove(audio_chunk_path)
        
        all_voice_segments.sort()
        return all_voice_segments

def get_voice_segments(video_path: str):
    num_workers = max(1, os.cpu_count() // 2)
    return get_voice_segments_parallel(video_path, num_workers=num_workers)
