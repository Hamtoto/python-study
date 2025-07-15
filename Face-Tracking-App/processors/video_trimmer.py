"""
비디오 트리밍 모듈
"""
import os
from moviepy import VideoFileClip, concatenate_videoclips
from config import CUT_THRESHOLD_SECONDS, FACE_DETECTION_THRESHOLD_FRAMES, VIDEO_CODEC, AUDIO_CODEC, SEGMENT_LENGTH_SECONDS


def create_condensed_video(video_path: str, output_path: str, timeline: list[bool], fps: float, cut_threshold: float = CUT_THRESHOLD_SECONDS) -> bool:
    """
    불필요한 구간을 제거하여 요약본 생성
    
    Args:
        video_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        timeline: 얼굴 탐지 타임라인
        fps: 비디오 fps
        cut_threshold: 잘라낼 구간 임계값 (초)
    
    Returns:
        bool: 성공 여부
    """
    print("## 2단계: 불필요 구간 제거 시작")
    if not timeline:
        print("오류: 타임라인 데이터 없음")
        return False
    
    clips = []
    seg_start = 0
    face_seg = timeline[0]
    
    for i in range(1, len(timeline)):
        if timeline[i] != face_seg:
            end = i
            duration = (end - seg_start) / fps
            if face_seg or duration <= cut_threshold:
                clips.append((seg_start / fps, end / fps))
            seg_start = i
            face_seg = timeline[i]
    
    # 마지막 구간 처리
    end = len(timeline)
    duration = (end - seg_start) / fps
    if face_seg or duration <= cut_threshold:
        clips.append((seg_start / fps, end / fps))
    
    if not clips:
        print("오류: 유지할 클립 없음")
        return False
    
    print("MoviePy 요약본 생성 중…")
    original = VideoFileClip(video_path)
    subclips = [original.subclipped(s, e) for s, e in clips]
    summary = concatenate_videoclips(subclips)
    summary.write_videofile(output_path, codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC, 
                           temp_audiofile='temp-audio.m4a', remove_temp=True)
    original.close()
    print("요약본 완료")
    return True


def trim_by_face_timeline(input_path: str, id_timeline: list, fps: float, 
                         threshold_frames: int = FACE_DETECTION_THRESHOLD_FRAMES, 
                         output_path: str = None):
    """
    얼굴이 threshold_frames 이상 미탐지된 구간을 제거하여 트리밍
    
    Args:
        input_path: 입력 비디오 경로
        id_timeline: ID 타임라인 리스트
        fps: 비디오 fps
        threshold_frames: 트리밍 임계값 (프레임 수)
        output_path: 출력 비디오 경로
    
    Returns:
        bool: 성공 여부
    """
    # Build keep mask
    n = len(id_timeline)
    keep = [True] * n
    i = 0
    
    while i < n:
        if id_timeline[i] is None:
            j = i
            while j < n and id_timeline[j] is None:
                j += 1
            if j - i >= threshold_frames:
                for k in range(i, j):
                    keep[k] = False
            i = j
        else:
            i += 1

    # Build segments to keep
    segments = []
    i = 0
    while i < n:
        if keep[i]:
            start = i
            while i < n and keep[i]:
                i += 1
            segments.append((start / fps, i / fps))
        else:
            i += 1

    if not segments:
        print("## 경고: 유지할 구간이 없습니다.")
        return False

    # Perform trimming
    clip = VideoFileClip(input_path)
    duration = clip.duration
    subclips = []
    for s, e in segments:
        end_clamped = min(e, duration)
        subclips.append(clip.subclipped(s, end_clamped))
    final = concatenate_videoclips(subclips)
    final.write_videofile(output_path, codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC, 
                         temp_audiofile='temp-audio.m4a', remove_temp=True)
    clip.close()
    final.close()
    return True


def slice_video(input_path: str, output_folder: str, segment_length: int = SEGMENT_LENGTH_SECONDS):
    """
    비디오를 지정된 길이의 세그먼트로 분할
    
    Args:
        input_path: 입력 비디오 경로
        output_folder: 출력 폴더 경로
        segment_length: 세그먼트 길이 (초)
    """
    print(f"Slicing {input_path} into {segment_length}-second segments in {output_folder}")
    clip = VideoFileClip(input_path)
    duration = clip.duration
    
    for start in range(0, int(duration), segment_length):
        end = min(start + segment_length, duration)
        if (end - start) < segment_length:
            continue
        segment = clip.subclipped(start, end)
        segment_filename = f"segment_{start}_{int(end)}.mp4"
        output_path = os.path.join(output_folder, segment_filename)
        segment.write_videofile(output_path, codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC)
    
    clip.close()