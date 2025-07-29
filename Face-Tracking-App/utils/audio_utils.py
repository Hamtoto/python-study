"""
오디오 처리 유틸리티 함수들 - FFmpeg 최대 성능 최적화
"""
import wave
import subprocess
import webrtcvad
import multiprocessing
import re
from moviepy import AudioFileClip
from config import AUDIO_SAMPLE_RATE, AUDIO_FRAME_DURATION


def get_optimal_cpu_threads():
    """동적 CPU 스레드 수 계산 (전체 코어 - 1)"""
    total_cores = multiprocessing.cpu_count()
    optimal_threads = max(1, total_cores - 1)  # 최소 1개, 최대 전체-1
    print(f"🖥️ CPU 최적화: {total_cores}코어 중 {optimal_threads}개 사용")
    return optimal_threads


def get_voice_segments(video_path: str, sample_rate=AUDIO_SAMPLE_RATE, frame_duration=AUDIO_FRAME_DURATION):
    """
    비디오에서 음성 구간을 추출하는 함수
    
    Args:
        video_path: 비디오 파일 경로
        sample_rate: 오디오 샘플 레이트
        frame_duration: 프레임 지속시간 (ms)
    
    Returns:
        list: (시작시간, 종료시간) 튜플들의 리스트
    """
    wav_path = video_path.replace('.mp4', '_audio.wav')
    
    # 동적 CPU 스레드 수 적용
    threads = get_optimal_cpu_threads()
    
    # FFmpeg 최대 성능 명령어 (실시간 출력)
    ffmpeg_cmd = [
        'ffmpeg', '-y', 
        '-threads', str(threads),  # 동적 CPU 스레드
        '-i', video_path, 
        '-ar', str(sample_rate), 
        '-ac', '1',  # 모노
        '-sample_fmt', 's16',
        '-progress', 'pipe:1',  # 실시간 진행률
        wav_path
    ]
    
    try:
        print(f"🚀 FFmpeg 최대 성능 오디오 추출 시작 ({threads}스레드)")
        
        # 실시간 출력으로 무한 대기 방지
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 실시간 진행률 파싱
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if 'time=' in line:
                # 진행률 표시 (선택적)
                time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
                if time_match:
                    print(f"⏳ 처리 중: {time_match.group(1)}", end='\r')
        
        # 프로세스 완료 대기
        return_code = process.wait()
        if return_code == 0:
            print("\n🎯 FFmpeg 최대 성능 오디오 추출 완료!")
        else:
            raise subprocess.CalledProcessError(return_code, ffmpeg_cmd)
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        # Fallback to MoviePy
        print(f"\n⚠️ FFmpeg 실패 ({e}), MoviePy fallback으로 오디오 추출")
        AudioFileClip(video_path).write_audiofile(wav_path, fps=sample_rate, nbytes=2, codec='pcm_s16le')
    vad = webrtcvad.Vad(3)
    wf = wave.open(wav_path, 'rb')
    frames = wf.readframes(wf.getnframes())
    segment_length = int(sample_rate * frame_duration / 1000) * 2
    voice_times = []
    is_speech = False
    t = 0.0
    for offset in range(0, len(frames), segment_length):
        chunk = frames[offset:offset+segment_length]
        
        # WebRTC VAD는 정확한 청크 길이를 요구함 (10ms, 20ms, 30ms)
        if len(chunk) != segment_length:
            # 마지막 청크가 불완전한 경우 패딩 또는 스킵
            if len(chunk) < segment_length // 2:  # 절반 미만이면 스킵
                break
            # 절반 이상이면 0으로 패딩
            chunk = chunk + b'\x00' * (segment_length - len(chunk))
        
        try:
            speech = vad.is_speech(chunk, sample_rate)
        except Exception as e:
            print(f"⚠️ VAD 프레임 스킵: {len(chunk)} bytes, 오류: {e}")
            speech = False  # 오류 시 음성 아님으로 처리
        if speech and not is_speech:
            start = t; is_speech = True
        if not speech and is_speech:
            voice_times.append((start, t)); is_speech = False
        t += frame_duration / 1000
    if is_speech:
        voice_times.append((start, t))
    wf.close()
    return voice_times