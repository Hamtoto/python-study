"""
비디오 트리밍 모듈 - FFmpeg 최대 성능 최적화
"""
import os
import subprocess
import tempfile
import multiprocessing
import re
from moviepy import VideoFileClip, concatenate_videoclips
from config import CUT_THRESHOLD_SECONDS, FACE_DETECTION_THRESHOLD_FRAMES, VIDEO_CODEC, AUDIO_CODEC, SEGMENT_LENGTH_SECONDS


def get_optimal_cpu_threads():
    """동적 CPU 스레드 수 계산 (전체 코어 - 1)"""
    total_cores = multiprocessing.cpu_count()
    optimal_threads = max(1, total_cores - 1)
    return optimal_threads


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
    
    # 동적 CPU 최적화
    threads = get_optimal_cpu_threads()
    print(f"🚀 FFmpeg 최대 성능 요약본 생성 ({threads}스레드)...")
    
    # FFmpeg filter_complex 생성 (MoviePy 대신 직접 호출)
    filter_parts = []
    for i, (start, end) in enumerate(clips):
        filter_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
    
    # concat 필터
    video_inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    audio_inputs = "".join(f"[a{i}]" for i in range(len(clips)))
    concat_filter = f"{video_inputs}{audio_inputs}concat=n={len(clips)}:v=1:a=1[outv][outa]"
    
    filter_complex = ";".join(filter_parts + [concat_filter])
    
    # filter_complex가 너무 긴 경우 임시 파일 사용
    import tempfile
    filter_file = None
    
    if len(filter_complex) > 8000:  # 명령어 길이 제한 대비
        # 클립 수가 너무 많으면 MoviePy로 직접 처리
        print(f"⚠️ 클립 수 너무 많음 ({len(clips)}개), MoviePy 직접 사용")
        original = VideoFileClip(video_path)
        video_duration = original.duration
        
        # 안전한 클립 생성
        safe_clips = []
        for s, e in clips:
            if e >= video_duration:
                e = video_duration - 0.001
            safe_clips.append((s, e))
        
        subclips = [original.subclipped(s, e) for s, e in safe_clips]
        summary = concatenate_videoclips(subclips)
        summary.write_videofile(output_path, codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC, 
                               temp_audiofile='temp-audio.m4a', remove_temp=True)
        original.close()
        for clip in subclips:
            clip.close()
        summary.close()
        print("🎯 MoviePy 대용량 처리 완료")
        return True
    else:
        # 기존 방식 (짧은 경우)
        ffmpeg_cmd = [
            'ffmpeg', '-y', 
            '-threads', str(threads),
            '-i', video_path,
            '-filter_complex', filter_complex,
            '-map', '[outv]', '-map', '[outa]',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'fast',
            '-progress', 'pipe:1',
            output_path
        ]
    
    try:
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
                time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
                if time_match:
                    print(f"⏳ 요약본 생성: {time_match.group(1)}", end='\r')
        
        # 프로세스 완료 대기
        return_code = process.wait()
        if return_code == 0:
            print("\n🎯 FFmpeg 최대 성능 요약본 완료!")
        else:
            raise subprocess.CalledProcessError(return_code, ffmpeg_cmd)
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"FFmpeg 오류: {e}")
        # Fallback to MoviePy
        print("⚠️ MoviePy fallback 사용")
        original = VideoFileClip(video_path)
        video_duration = original.duration
        
        # end_time이 video duration과 같거나 큰 경우 조정
        safe_clips = []
        for s, e in clips:
            if e >= video_duration:
                e = video_duration - 0.001  # 1ms 여유
                print(f"⚠️ 클립 끝시간 조정: {e:.3f} -> {video_duration - 0.001:.3f}")
            safe_clips.append((s, e))
        
        subclips = [original.subclipped(s, e) for s, e in safe_clips]
        summary = concatenate_videoclips(subclips)
        summary.write_videofile(output_path, codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC, 
                               temp_audiofile='temp-audio.m4a', remove_temp=True)
        original.close()
        print("MoviePy fallback 완료")
    finally:
        # 임시 파일 정리
        if filter_file:
            try:
                os.unlink(filter_file.name)
            except:
                pass
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