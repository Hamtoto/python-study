"""
최적화된 비디오 처리 워크플로우 - FFmpeg 기반
"""
import os
import time
import shutil
import subprocess
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
from concurrent.futures import ThreadPoolExecutor

# pkg_resources 경고 억제
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# CUDA 메모리 최적화 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from src.face_tracker.utils.audio import get_voice_segments_ffmpeg
from src.face_tracker.processing.analyzer import analyze_video_faces
from src.face_tracker.processing.timeline import generate_id_timeline
from src.face_tracker.processing.trimmer import (
    create_condensed_video_ffmpeg, 
    trim_by_face_timeline_ffmpeg, 
    slice_video_parallel_ffmpeg
)
from src.face_tracker.processing.tracker import track_and_crop_video
from src.face_tracker.processing.selector import TargetSelector
from src.face_tracker.utils.logging import logger
from src.face_tracker.utils.performance_reporter import start_video_report, get_current_reporter
from src.face_tracker.config import (
    DEVICE, INPUT_DIR, OUTPUT_ROOT, TEMP_ROOT, 
    SUPPORTED_VIDEO_EXTENSIONS, BATCH_SIZE_ANALYZE, 
    BATCH_SIZE_ID_TIMELINE, VIDEO_CODEC, AUDIO_CODEC, TRACKING_MODE
)


def process_single_segment_ffmpeg(task_data):
    """
    FFmpeg를 사용한 단일 세그먼트 처리 (멀티프로세싱용)
    
    Args:
        task_data: 세그먼트 처리에 필요한 데이터 딕셔너리
    """
    try:
        seg_fname = task_data['seg_fname']
        seg_input = task_data['seg_input']
        seg_cropped = task_data['seg_cropped']
        output_path = task_data['output_path']
        
        logger.info(f"세그먼트 처리: {seg_fname}")
        
        # 1) 모델 초기화 (각 프로세스에서 독립적으로)
        from src.face_tracker.core.models import ModelManager
        model_manager = ModelManager(DEVICE)
        mtcnn = model_manager.get_mtcnn()
        resnet = model_manager.get_resnet()
        
        # 2) 얼굴 크롭
        track_and_crop_video(seg_input, seg_cropped, mtcnn, resnet, DEVICE)
        
        # 3) FFmpeg를 사용한 오디오 동기화 (MoviePy 대신)
        cmd = [
            'ffmpeg', '-y',
            '-i', seg_cropped,  # 크롭된 비디오
            '-i', seg_input,    # 원본 오디오가 있는 비디오
            '-c:v', 'copy',     # 비디오 재인코딩 없이 복사
            '-c:a', AUDIO_CODEC,
            '-map', '0:v:0',    # 첫 번째 입력의 비디오
            '-map', '1:a:0',    # 두 번째 입력의 오디오
            '-shortest',        # 더 짧은 스트림에 맞춤
            output_path
        ]
        
        # 4) FFmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 5) 임시 파일 정리
        if os.path.exists(seg_cropped):
            os.remove(seg_cropped)
            
        logger.success(f"세그먼트 완료: {seg_fname}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{seg_fname} FFmpeg 오류: {e.stderr}")
    except Exception as e:
        logger.error(f"{seg_fname} 처리 오류: {str(e)}")


def process_single_video_optimized(fname: str):
    """
    최적화된 단일 비디오 파일 처리
    
    Args:
        fname: 처리할 비디오 파일명
    """
    basename = os.path.splitext(fname)[0]
    temp_dir = os.path.join(TEMP_ROOT, basename)
    os.makedirs(temp_dir, exist_ok=True)

    input_path = os.path.join(INPUT_DIR, fname)
    condensed = os.path.join(temp_dir, f"condensed_{fname}")
    trimmed = os.path.join(temp_dir, f"trimmed_{fname}")

    logger.stage(f"비디오 처리 시작: {fname}")
    start_time = time.time()
    
    # 성능 리포트 시작
    reporter = start_video_report(fname)
    
    try:
        # 0) FFmpeg를 사용한 고속 오디오 VAD
        reporter.start_stage("오디오 VAD")
        logger.stage("오디오 VAD 처리...")
        voice_timeline = get_voice_segments_ffmpeg(input_path)
        logger.info(f"음성 구간 {len(voice_timeline)}개 감지")
        reporter.end_stage("오디오 VAD", segments=len(voice_timeline))
        
        # 1) 얼굴 감지 타임라인
        reporter.start_stage("얼굴 감지")
        logger.stage("얼굴 감지 분석...")
        timeline, fps = analyze_video_faces(input_path, batch_size=BATCH_SIZE_ANALYZE, device=DEVICE)
        reporter.end_stage("얼굴 감지", frames=len(timeline), batch_size=BATCH_SIZE_ANALYZE)
        
        # 2) FFmpeg를 사용한 고속 요약본 생성
        reporter.start_stage("요약본 생성")
        logger.stage("요약본 생성...")
        if create_condensed_video_ffmpeg(input_path, condensed, timeline, fps):
            logger.success("요약본 생성 완료")
            reporter.end_stage("요약본 생성", frames=len(timeline))
            
            # 2a) ID 타임라인 생성 및 타겟 인물 자동 선택
            reporter.start_stage("얼굴 인식")
            logger.stage("얼굴 인식 및 타겟 선택...")
            id_timeline, fps2 = generate_id_timeline(condensed, DEVICE, batch_size=BATCH_SIZE_ID_TIMELINE)
            reporter.end_stage("얼굴 인식", frames=len(id_timeline), batch_size=BATCH_SIZE_ID_TIMELINE)
            
            # 모드별 타겟 ID 선택 (임베딩 정보 포함)
            from src.face_tracker.core.embeddings import SmartEmbeddingManager
            emb_manager = SmartEmbeddingManager()
            embeddings = emb_manager.get_all_embeddings()
            
            target_id = TargetSelector.select_target(id_timeline, TRACKING_MODE, embeddings)
            if target_id is not None:
                stats = TargetSelector.get_target_stats(id_timeline, target_id)
                logger.success(f"타겟 선택: ID={target_id} ({stats['coverage_ratio']:.1%} 커버리지)")
            else:
                logger.warning(f"타겟 찾기 실패 (모드={TRACKING_MODE})")
            
            # 타겟 아닌 프레임은 None으로 표시
            if target_id is not None:
                id_timeline_bool = [tid if tid == target_id else None for tid in id_timeline]
            else:
                id_timeline_bool = id_timeline
            
            # FFmpeg를 사용한 고속 트리밍
            reporter.start_stage("비디오 트리밍")
            logger.stage("비디오 트리밍...")
            if trim_by_face_timeline_ffmpeg(condensed, id_timeline_bool, fps2, threshold_frames=30, output_path=trimmed):
                source_for_crop = trimmed
            else:
                source_for_crop = condensed
            reporter.end_stage("비디오 트리밍")
            
            # 3) FFmpeg를 사용한 병렬 세그먼트 분할
            reporter.start_stage("세그먼트 분할")
            logger.stage("세그먼트 분할...")
            segment_temp_folder = os.path.join(temp_dir, "segments")
            os.makedirs(segment_temp_folder, exist_ok=True)
            slice_video_parallel_ffmpeg(source_for_crop, segment_temp_folder, segment_length=10)
            reporter.end_stage("세그먼트 분할")

            # 4) 각 세그먼트별 얼굴 크롭 및 오디오 동기 병합 (병렬 처리)
            reporter.start_stage("얼굴 크롭")
            logger.stage("얼굴 크롭 및 오디오 동기화...")
            final_segment_folder = os.path.join(OUTPUT_ROOT, basename)
            os.makedirs(final_segment_folder, exist_ok=True)
            
            segment_files = [f for f in os.listdir(segment_temp_folder) 
                           if f.lower().endswith(".mp4")]
            segment_files.sort()  # 순서 보장
            
            logger.info(f"세그먼트 {len(segment_files)}개 병렬 처리 시작")
            
            # 멀티프로세싱을 위한 작업 데이터 준비
            segment_tasks = []
            for seg_fname in segment_files:
                seg_input = os.path.join(segment_temp_folder, seg_fname)
                seg_cropped = os.path.join(temp_dir, f"crop_{seg_fname}")
                output_seg_path = os.path.join(final_segment_folder, seg_fname)
                
                segment_tasks.append({
                    'seg_fname': seg_fname,
                    'seg_input': seg_input,
                    'seg_cropped': seg_cropped,
                    'output_path': output_seg_path
                })
            
            # CPU 코어 기반 최적 프로세스 수 계산
            num_processes = max(1, min(8, int(cpu_count() * 0.8)))
            
            # 성능 리포트에 정보 설정
            reporter.set_processing_info(len(timeline), len(segment_files), num_processes)
            
            with Pool(processes=num_processes) as pool:
                pool.map(process_single_segment_ffmpeg, segment_tasks)
                    
            reporter.end_stage("얼굴 크롭", segments=len(segment_files))
            logger.success(f"세그먼트 병렬 처리 완료 ({len(segment_files)}개)")
        else:
            logger.error(f"요약본 생성 실패: {fname}")

    except Exception as e:
        logger.error(f"{fname} 비디오 처리 오류: {str(e)}")
    finally:
        elapsed = time.time() - start_time
        logger.success(f"{fname} 처리 완료 ({int(elapsed)}초)")
        
        # 성능 리포트 생성
        if reporter:
            reporter.generate_report()
        
        # 임시 디렉토리 정리
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_all_videos_optimized():
    """
    최적화된 모든 비디오 파일 처리
    """
    # CUDA 멀티프로세싱을 위한 spawn 방식 설정
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        # 이미 설정된 경우 무시
        pass
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)

    # 로그 초기화
    logger.clear_log()
    logger.info("비디오 처리 시스템 시작")

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            continue
        
        process_single_video_optimized(fname)

    logger.success("모든 비디오 처리 완료")
    
    # 전체 임시 폴더 정리
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT)
        logger.info("임시 폴더 정리 완료")


# 하위 호환성을 위한 함수명 유지
def process_single_video(fname: str):
    """최적화된 버전 호출"""
    return process_single_video_optimized(fname)


def process_all_videos():
    """최적화된 버전 호출"""
    return process_all_videos_optimized()