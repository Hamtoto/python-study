"""
최적화된 오디오 처리 유틸리티 - FFmpeg 직접 호출 버전
"""
import os
import wave
import webrtcvad
import subprocess
import tempfile
from src.face_tracker.config import AUDIO_SAMPLE_RATE, AUDIO_FRAME_DURATION
from src.face_tracker.utils.logging import logger


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
        logger.error(f"오디오 추출 오류: {e.stderr}")
        return []
    finally:
        # 임시 파일 정리
        if os.path.exists(wav_path):
            os.remove(wav_path)


