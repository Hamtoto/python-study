"""
최적화된 비디오 트리밍 모듈 - FFmpeg 직접 호출 버전
"""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from config import CUT_THRESHOLD_SECONDS, FACE_DETECTION_THRESHOLD_FRAMES, VIDEO_CODEC, AUDIO_CODEC, SEGMENT_LENGTH_SECONDS


def create_condensed_video_ffmpeg(video_path: str, output_path: str, timeline: list[bool], fps: float, cut_threshold: float = CUT_THRESHOLD_SECONDS) -> bool:
    """
    FFmpeg를 사용한 고성능 요약본 생성
    
    Args:
        video_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        timeline: 얼굴 탐지 타임라인
        fps: 비디오 fps
        cut_threshold: 잘라낼 구간 임계값 (초)
    
    Returns:
        bool: 성공 여부
    """
    print("## 2단계: FFmpeg 고성능 불필요 구간 제거 시작")
    if not timeline:
        print("오류: 타임라인 데이터 없음")
        return False
    
    # 유지할 구간 계산
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
    
    # FFmpeg filter_complex 명령어 생성
    filter_parts = []
    input_parts = []
    
    for i, (start, end) in enumerate(clips):
        # 각 구간을 선택하는 필터
        filter_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        input_parts.append(f"[v{i}][a{i}]")
    
    # 모든 클립을 연결하는 필터
    video_inputs = "".join([f"[v{i}]" for i in range(len(clips))])
    audio_inputs = "".join([f"[a{i}]" for i in range(len(clips))])
    
    filter_complex = ";".join(filter_parts) + f";{video_inputs}concat=n={len(clips)}:v=1:a=0[vout];{audio_inputs}concat=n={len(clips)}:v=0:a=1[aout]"
    
    # FFmpeg 명령어 실행
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-map', '[aout]',
        '-c:v', VIDEO_CODEC,
        '-c:a', AUDIO_CODEC,
        '-preset', 'ultrafast',  # 속도 우선
        '-crf', '23',  # 품질 설정
        output_path
    ]
    
    try:
        print("FFmpeg 요약본 생성 중...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("FFmpeg 요약본 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 오류: {e}")
        print(f"stderr: {e.stderr}")
        return False


def trim_by_face_timeline_ffmpeg(input_path: str, id_timeline: list, fps: float, 
                                threshold_frames: int = FACE_DETECTION_THRESHOLD_FRAMES, 
                                output_path: str = None):
    """
    FFmpeg를 사용한 고성능 얼굴 타임라인 기반 트리밍
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

    # FFmpeg를 사용한 트리밍 (create_condensed_video_ffmpeg와 동일한 로직)
    return _ffmpeg_trim_segments(input_path, output_path, segments)


def slice_video_parallel_ffmpeg(input_path: str, output_folder: str, segment_length: int = SEGMENT_LENGTH_SECONDS):
    """
    FFmpeg를 사용한 병렬 비디오 분할
    """
    print(f"FFmpeg 병렬 슬라이싱: {input_path} -> {segment_length}초 세그먼트들")
    
    # 비디오 길이 확인
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    # 병렬 처리를 위한 작업 리스트 생성
    tasks = []
    for start in range(0, int(duration), segment_length):
        end = min(start + segment_length, duration)
        if (end - start) < segment_length:
            continue
        
        segment_filename = f"segment_{start}_{int(end)}.mp4"
        output_path = os.path.join(output_folder, segment_filename)
        tasks.append((input_path, output_path, start, end))
    
    # 병렬 실행
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_slice_single_segment, tasks))
    
    success_count = sum(results)
    print(f"병렬 슬라이싱 완료: {success_count}/{len(tasks)} 세그먼트")
    return success_count == len(tasks)


def _slice_single_segment(task):
    """단일 세그먼트 슬라이싱"""
    input_path, output_path, start, end = task
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', input_path,
        '-t', str(end - start),
        '-c', 'copy',  # 재인코딩 없이 복사만
        '-avoid_negative_ts', 'make_zero',
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _ffmpeg_trim_segments(input_path, output_path, segments):
    """FFmpeg를 사용한 세그먼트 트리밍"""
    if len(segments) == 1:
        # 단일 세그먼트의 경우 간단한 trim 사용
        start, end = segments[0]
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(end - start),
            '-c:v', VIDEO_CODEC,
            '-c:a', AUDIO_CODEC,
            '-preset', 'ultrafast',
            output_path
        ]
    else:
        # 다중 세그먼트의 경우 filter_complex 사용
        filter_parts = []
        for i, (start, end) in enumerate(segments):
            filter_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
            filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        
        video_inputs = "".join([f"[v{i}]" for i in range(len(segments))])
        audio_inputs = "".join([f"[a{i}]" for i in range(len(segments))])
        
        filter_complex = ";".join(filter_parts) + f";{video_inputs}concat=n={len(segments)}:v=1:a=0[vout];{audio_inputs}concat=n={len(segments)}:v=0:a=1[aout]"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-filter_complex', filter_complex,
            '-map', '[vout]',
            '-map', '[aout]',
            '-c:v', VIDEO_CODEC,
            '-c:a', AUDIO_CODEC,
            '-preset', 'ultrafast',
            output_path
        ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 세그먼트 트리밍 오류: {e}")
        return False


# 하위 호환성을 위한 함수명 유지
def create_condensed_video(video_path: str, output_path: str, timeline: list[bool], fps: float, cut_threshold: float = CUT_THRESHOLD_SECONDS) -> bool:
    """MoviePy 대신 FFmpeg 버전 호출"""
    return create_condensed_video_ffmpeg(video_path, output_path, timeline, fps, cut_threshold)


def trim_by_face_timeline(input_path: str, id_timeline: list, fps: float, 
                         threshold_frames: int = FACE_DETECTION_THRESHOLD_FRAMES, 
                         output_path: str = None):
    """MoviePy 대신 FFmpeg 버전 호출"""
    return trim_by_face_timeline_ffmpeg(input_path, id_timeline, fps, threshold_frames, output_path)


def slice_video(input_path: str, output_folder: str, segment_length: int = SEGMENT_LENGTH_SECONDS):
    """MoviePy 대신 FFmpeg 병렬 버전 호출"""
    return slice_video_parallel_ffmpeg(input_path, output_folder, segment_length)