"""
최적화된 비디오 트리밍 모듈 - FFmpeg 직접 호출 버전
"""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from src.face_tracker.config import CUT_THRESHOLD_SECONDS, FACE_DETECTION_THRESHOLD_FRAMES, VIDEO_CODEC, AUDIO_CODEC, SEGMENT_LENGTH_SECONDS
from src.face_tracker.utils.logging import logger
from src.face_tracker.utils.exceptions import FFmpegError


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
    if not timeline:
        logger.error("타임라인 데이터 없음")
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
        logger.error("유지할 클립 없음")
        return False
    
    # 공통 FFmpeg 로직 사용
    cmd = _build_ffmpeg_segments_command(video_path, output_path, clips)
    
    # CRF 품질 옵션 추가 (요약본에만 적용)
    if '-crf' not in cmd:
        cmd.insert(-1, '-crf')
        cmd.insert(-1, '23')
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise FFmpegError(command=cmd, stderr=e.stderr)


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
        logger.warning("유지할 구간이 없음")
        return False

    # FFmpeg를 사용한 트리밍 (create_condensed_video_ffmpeg와 동일한 로직)
    return _ffmpeg_trim_segments(input_path, output_path, segments)


def slice_video_parallel_ffmpeg(input_path: str, output_folder: str, segment_length: int = SEGMENT_LENGTH_SECONDS):
    """
    FFmpeg를 사용한 병렬 비디오 분할
    """
    
    logger.info(f"비디오 분할 시작 - 입력: {input_path}")
    logger.info(f"출력 폴더: {output_folder}")
    logger.info(f"세그먼트 길이: {segment_length}초")
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_path):
        logger.error(f"입력 파일 없음: {input_path}")
        return False
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 비디오 길이 확인
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', input_path]
    logger.info(f"비디오 길이 확인 명령: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"비디오 총 길이: {duration:.1f}초")
    except Exception as e:
        logger.error(f"비디오 길이 확인 실패: {str(e)}")
        return False
    
    # 병렬 처리를 위한 작업 리스트 생성
    tasks = []
    for start in range(0, int(duration), segment_length):
        end = min(start + segment_length, duration)
        if (end - start) < segment_length:
            logger.info(f"세그먼트 스킵 - 너무 짧음: {start}초~{end:.1f}초 (길이: {end-start:.1f}초)")
            continue
        
        segment_filename = f"segment_{start}_{int(end)}.mp4"
        output_path = os.path.join(output_folder, segment_filename)
        tasks.append((input_path, output_path, start, end))
        logger.info(f"세그먼트 생성 예정: {segment_filename} ({start}초~{end:.1f}초)")
    
    logger.info(f"총 {len(tasks)}개 세그먼트 생성 작업 준비 완료")
    
    if len(tasks) == 0:
        logger.warning("생성할 세그먼트가 없습니다")
        return False
    
    # 병렬 실행
    logger.info(f"ThreadPoolExecutor로 {len(tasks)}개 세그먼트 병렬 처리 시작")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_slice_single_segment, tasks))
    
    success_count = sum(results)
    logger.info(f"세그먼트 분할 완료: {success_count}/{len(tasks)}개 성공")
    
    # 생성된 파일 확인
    created_files = [f for f in os.listdir(output_folder) if f.lower().endswith('.mp4')]
    logger.info(f"실제 생성된 파일 수: {len(created_files)}개")
    for f in created_files:
        fpath = os.path.join(output_folder, f)
        fsize = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        logger.info(f"  생성됨: {f} ({fsize} bytes)")
    
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 파일 생성 확인
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"세그먼트 생성 성공: {os.path.basename(output_path)} ({file_size} bytes)")
            return True
        else:
            logger.error(f"세그먼트 파일 생성되지 않음: {os.path.basename(output_path)}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 세그먼트 생성 실패: {os.path.basename(output_path)}")
        logger.error(f"  명령: {' '.join(cmd)}")
        logger.error(f"  stderr: {e.stderr}")
        return False


def _build_ffmpeg_segments_command(input_path, output_path, segments):
    """FFmpeg 세그먼트 명령어 생성 (공통 로직)"""
    if len(segments) == 1:
        # 단일 세그먼트: 간단한 trim 사용
        start, end = segments[0]
        return [
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
        # 다중 세그먼트: filter_complex 사용
        filter_parts = []
        for i, (start, end) in enumerate(segments):
            filter_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
            filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        
        video_inputs = "".join([f"[v{i}]" for i in range(len(segments))])
        audio_inputs = "".join([f"[a{i}]" for i in range(len(segments))])
        
        filter_complex = f"{';'.join(filter_parts)};{video_inputs}concat=n={len(segments)}:v=1:a=0[vout];{audio_inputs}concat=n={len(segments)}:v=0:a=1[aout]"
        
        return [
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


def _ffmpeg_trim_segments(input_path, output_path, segments):
    """FFmpeg를 사용한 세그먼트 트리밍 (공통 로직 사용)"""
    cmd = _build_ffmpeg_segments_command(input_path, output_path, segments)
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise FFmpegError(command=cmd, stderr=e.stderr)


# 하위 호환성을 위한 함수명 별칭 (Alias)
create_condensed_video = create_condensed_video_ffmpeg
trim_by_face_timeline = trim_by_face_timeline_ffmpeg  
slice_video = slice_video_parallel_ffmpeg