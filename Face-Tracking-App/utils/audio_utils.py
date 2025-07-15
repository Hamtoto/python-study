"""
오디오 처리 유틸리티 함수들
"""
import wave
import webrtcvad
from moviepy import AudioFileClip
from config import AUDIO_SAMPLE_RATE, AUDIO_FRAME_DURATION


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
        speech = vad.is_speech(chunk, sample_rate)
        if speech and not is_speech:
            start = t; is_speech = True
        if not speech and is_speech:
            voice_times.append((start, t)); is_speech = False
        t += frame_duration / 1000
    if is_speech:
        voice_times.append((start, t))
    wf.close()
    return voice_times