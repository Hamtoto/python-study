"""
최적화된 오디오 처리 유틸리티 - FFmpeg 직접 호출 버전
"""
import os
import wave
import webrtcvad
import subprocess
import tempfile
from config import AUDIO_SAMPLE_RATE, AUDIO_FRAME_DURATION


def get_voice_segments_ffmpeg(video_path: str, sample_rate=AUDIO_SAMPLE_RATE, frame_duration=AUDIO_FRAME_DURATION):
    """
    FFmpeg를 사용한 고성능 음성 구간 추출
    
    Args:
        video_path: 비디오 파일 경로
        sample_rate: 오디오 샘플 레이트
        frame_duration: 프레임 지속시간 (ms)
    
    Returns:
        list: (시작시간, 종료시간) 튜플들의 리스트
    """
    # 임시 WAV 파일 생성
    with tempfile.NamedTemporaryFile(suffix='_audio.wav', delete=False) as temp_wav:
        wav_path = temp_wav.name
    
    try:
        # FFmpeg를 사용한 고속 오디오 추출 (MoviePy보다 훨씬 빠름)
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # 비디오 스트림 제거
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',  # 모노로 변환
            '-f', 'wav',
            wav_path
        ]
        
        # FFmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # VAD 처리 (기존 로직 유지)
        vad = webrtcvad.Vad(3)
        wf = wave.open(wav_path, 'rb')
        frames = wf.readframes(wf.getnframes())
        segment_length = int(sample_rate * frame_duration / 1000) * 2
        voice_times = []
        is_speech = False
        t = 0.0
        
        for offset in range(0, len(frames), segment_length):
            chunk = frames[offset:offset+segment_length]
            if len(chunk) < segment_length:
                break
                
            try:
                speech = vad.is_speech(chunk, sample_rate)
                if speech and not is_speech:
                    start = t
                    is_speech = True
                if not speech and is_speech:
                    voice_times.append((start, t))
                    is_speech = False
                t += frame_duration / 1000
            except webrtcvad.Error:
                # VAD 오류 발생 시 해당 프레임 건너뛰기
                t += frame_duration / 1000
                continue
        
        if is_speech:
            voice_times.append((start, t))
            
        wf.close()
        return voice_times
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 오디오 추출 오류: {e}")
        print(f"stderr: {e.stderr}")
        return []
    finally:
        # 임시 파일 정리
        if os.path.exists(wav_path):
            os.remove(wav_path)


def extract_audio_segment_ffmpeg(video_path: str, output_path: str, start_time: float, duration: float):
    """
    FFmpeg를 사용한 특정 구간 오디오 추출
    
    Args:
        video_path: 입력 비디오 경로
        output_path: 출력 오디오 경로
        start_time: 시작 시간 (초)
        duration: 지속 시간 (초)
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-vn',  # 비디오 스트림 제거
        '-c:a', 'copy',  # 오디오 재인코딩 없이 복사
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def merge_audio_video_ffmpeg(video_path: str, audio_path: str, output_path: str):
    """
    FFmpeg를 사용한 오디오/비디오 병합
    
    Args:
        video_path: 비디오 파일 경로
        audio_path: 오디오 파일 경로  
        output_path: 출력 파일 경로
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # 비디오 재인코딩 없이 복사
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',  # 더 짧은 스트림에 맞춤
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


# 하위 호환성을 위한 함수명 유지
def get_voice_segments(video_path: str, sample_rate=AUDIO_SAMPLE_RATE, frame_duration=AUDIO_FRAME_DURATION):
    """MoviePy 대신 FFmpeg 버전 호출"""
    return get_voice_segments_ffmpeg(video_path, sample_rate, frame_duration)