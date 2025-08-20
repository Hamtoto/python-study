"""
최적화된 비디오 처리 워크플로우 - FFmpeg 기반
"""
import os
import time
import shutil
import subprocess
import warnings
import cv2
import numpy as np
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
        
        # 2) 얼굴 크롭 (파일 존재 확인)
        if not os.path.exists(seg_input):
            logger.error(f"입력 파일 없음: {seg_input}")
            return
        
        # 출력 디렉토리 생성 확인
        seg_cropped_dir = os.path.dirname(seg_cropped)
        os.makedirs(seg_cropped_dir, exist_ok=True)
        logger.info(f"얼굴 크롭 시작: {seg_input} -> {seg_cropped}")
            
        track_and_crop_video(seg_input, seg_cropped, mtcnn, resnet, DEVICE)
        
        # 크롭된 파일 생성 확인
        if not os.path.exists(seg_cropped):
            logger.error(f"얼굴 크롭 실패: {seg_fname} - 크롭된 파일 생성되지 않음")
            return
        
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
            
            # 모드별 처리 분기
            from src.face_tracker.core.embeddings import SmartEmbeddingManager
            emb_manager = SmartEmbeddingManager()
            embeddings = emb_manager.get_all_embeddings()
            
            # 동적으로 업데이트된 TRACKING_MODE 값 가져오기
            from src.face_tracker.config import TRACKING_MODE as current_mode
            
            print(f"🔍 DEBUG: 현재 TRACKING_MODE: {current_mode}")
            logger.info(f"🔍 현재 TRACKING_MODE: {current_mode}")
            
            if current_mode == "dual":
                print("🔍 DEBUG: DUAL 모드 분기 진입")
                logger.info("🔍 DEBUG: DUAL 모드 분기 진입")
                # DUAL 모드 전용 처리 함수 호출
                process_dual_mode_segments(
                    id_timeline, fps2, condensed, 
                    temp_dir, basename, reporter, len(timeline)
                )
                
            elif current_mode == "dual_split":
                print("🎯 DEBUG: DUAL_SPLIT 모드 분기 진입!")
                logger.info("🎯 DUAL_SPLIT 모드 분기 진입!")
                # DUAL_SPLIT 모드 전용 처리 함수 호출
                try:
                    process_dual_split_mode_segments(
                        id_timeline, fps2, condensed, 
                        temp_dir, basename, reporter, len(timeline)
                    )
                    print("🎯 DEBUG: DUAL_SPLIT 모드 처리 완료!")
                    logger.info("🎯 DUAL_SPLIT 모드 처리 완료!")
                except Exception as e:
                    print(f"❌ DEBUG: DUAL_SPLIT 처리 오류: {e}")
                    logger.error(f"DUAL_SPLIT 처리 오류: {e}")
                    import traceback
                    traceback.print_exc()
                
            else:
                print(f"🔍 DEBUG: SINGLE 모드 또는 기타 모드 분기 진입: {current_mode}")
                logger.info(f"🔍 DEBUG: SINGLE 모드 또는 기타 모드 분기 진입: {current_mode}")
                # SINGLE 모드: 기존 로직
                target_id = TargetSelector.select_target(id_timeline, current_mode, embeddings)
                if target_id is not None:
                    stats = TargetSelector.get_target_stats(id_timeline, target_id)
                    logger.success(f"타겟 선택: ID={target_id} ({stats['coverage_ratio']:.1%} 커버리지)")
                else:
                    logger.warning(f"타겟 찾기 실패 (모드={current_mode})")
                
                # 타겟 아닌 프레임은 None으로 표시
                if target_id is not None:
                    id_timeline_bool = [tid if tid == target_id else None for tid in id_timeline]
                else:
                    id_timeline_bool = id_timeline
            
                # SINGLE 모드: 기존 트리밍 및 세그먼트 처리
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

def process_dual_mode_segments(id_timeline, fps, source_video, temp_dir, basename, reporter, total_frames):
    """
    DUAL 모드 전용: 상위 2명의 화자에 대해 각각 10초 단위 세그먼트 생성
    
    Args:
        id_timeline: 얼굴 ID 타임라인
        fps: 비디오 FPS
        source_video: 소스 비디오 경로 (요약본)
        temp_dir: 임시 디렉토리
        basename: 베이스 파일명
        reporter: 성능 리포터
    """
    from src.face_tracker.dual_config import DUAL_PERSON_FOLDER_PREFIX
    from collections import Counter

    reporter.start_stage("DUAL 모드 처리")
    logger.stage("DUAL 모드: 상위 2명 화자별 10초 세그먼트 생성...")

    # 1. L2 정규화를 사용한 ID 병합으로 상위 2명 face_id 추출
    from src.face_tracker.core.embeddings import SmartEmbeddingManager
    from src.face_tracker.processing.selector import TargetSelector
    from src.face_tracker.config import L2_NORMALIZATION_ENABLED
    
    emb_manager = SmartEmbeddingManager()
    embeddings = emb_manager.get_all_embeddings()
    
    # 동적 임계값 계산 및 L2 정규화 적용된 ID 병합
    if embeddings:
        from src.face_tracker.utils.adaptive_threshold import (
            calculate_optimal_threshold, 
            log_threshold_optimization,
            should_apply_adaptive_threshold
        )
        from src.face_tracker.config import (
            DUAL_MODE_SIMILARITY_THRESHOLD,
            ENABLE_ADAPTIVE_THRESHOLD,
            ADAPTIVE_THRESHOLD_MIN_CONFIDENCE,
            ADAPTIVE_THRESHOLD_MIN_IMPROVEMENT,
            ADAPTIVE_THRESHOLD_SAFETY_RANGE,
            ADAPTIVE_THRESHOLD_MIN_SAMPLES
        )
        
        # 기본 임계값 설정
        default_threshold = DUAL_MODE_SIMILARITY_THRESHOLD
        final_threshold = default_threshold
        optimal_threshold = None
        confidence = "disabled"
        statistics = {}
        
        # 동적 임계값 계산 (설정에 따라)
        if ENABLE_ADAPTIVE_THRESHOLD:
            optimal_threshold, confidence, statistics = calculate_optimal_threshold(
                embeddings, 
                use_l2_norm=L2_NORMALIZATION_ENABLED,
                min_same_samples=ADAPTIVE_THRESHOLD_MIN_SAMPLES,
                min_different_samples=ADAPTIVE_THRESHOLD_MIN_SAMPLES,
                safety_range=ADAPTIVE_THRESHOLD_SAFETY_RANGE
            )
            
            # 적용할 임계값 결정
            if optimal_threshold is not None and should_apply_adaptive_threshold(
                confidence, optimal_threshold, default_threshold,
                min_confidence_level=ADAPTIVE_THRESHOLD_MIN_CONFIDENCE,
                min_improvement_percent=ADAPTIVE_THRESHOLD_MIN_IMPROVEMENT
            ):
                final_threshold = optimal_threshold
                logger.success(f"동적 임계값 적용: {final_threshold:.3f} (신뢰도: {confidence})")
            else:
                logger.info(f"기본 임계값 사용: {final_threshold:.3f} (동적: {confidence})")
        else:
            logger.info(f"동적 임계값 비활성화, 기본값 사용: {final_threshold:.3f}")
        
        # 최적화 결과 로깅
        log_threshold_optimization(
            optimal_threshold, confidence, statistics, 
            default_threshold, L2_NORMALIZATION_ENABLED
        )
        
        # ID 병합 수행
        merged_timeline = TargetSelector._merge_similar_ids(
            id_timeline, embeddings, 
            similarity_threshold=final_threshold,
            use_l2_norm=L2_NORMALIZATION_ENABLED
        )
        logger.info(f"DUAL 모드 ID 병합 완료 (임계값: {final_threshold:.3f}, L2: {L2_NORMALIZATION_ENABLED})")
        
        # 성능 리포터에 임계값 최적화 정보 전달
        if optimal_threshold is not None:
            reporter.set_threshold_optimization_info(
                final_threshold, default_threshold, confidence, statistics
            )
    else:
        merged_timeline = id_timeline
    
    valid_ids = [fid for fid in merged_timeline if fid is not None]
    if not valid_ids:
        logger.warning("DUAL 모드: 인식된 얼굴이 없습니다.")
        reporter.end_stage("DUAL 모드 처리", segments=0)
        return

    face_id_counts = Counter(valid_ids)
    top_2_faces = face_id_counts.most_common(2)

    logger.info(f"DUAL 모드: 상위 {len(top_2_faces)}명 화자 선택됨")
    total_segments_processed = 0

    # 2. 각 화자에 대해 SINGLE 모드와 동일한 로직 수행
    for i, (target_id, count) in enumerate(top_2_faces, 1):
        person_folder_name = f"{DUAL_PERSON_FOLDER_PREFIX}{i}"
        logger.info(f"{person_folder_name} (face_id={target_id}, 등장 프레임={count}) 처리 시작")

        # 2a. 해당 화자의 타임라인 생성 (병합된 타임라인 사용)
        target_timeline_bool = [tid if tid == target_id else None for tid in merged_timeline]
        
        # 2b. 해당 화자 영상만 트리밍
        person_trimmed_path = os.path.join(temp_dir, f"trimmed_{person_folder_name}.mp4")
        if not trim_by_face_timeline_ffmpeg(source_video, target_timeline_bool, fps, threshold_frames=30, output_path=person_trimmed_path):
            logger.warning(f"{person_folder_name}: 트리밍할 영상이 없어 건너뜁니다.")
            continue

        # 2c. 10초 단위로 세그먼트 분할
        person_segment_temp_folder = os.path.join(temp_dir, f"segments_{person_folder_name}")
        os.makedirs(person_segment_temp_folder, exist_ok=True)
        slice_video_parallel_ffmpeg(person_trimmed_path, person_segment_temp_folder, segment_length=10)

        # 2d. 각 세그먼트 크롭 및 저장
        final_person_output_dir = os.path.join(OUTPUT_ROOT, basename, person_folder_name)
        os.makedirs(final_person_output_dir, exist_ok=True)

        segment_files = [f for f in os.listdir(person_segment_temp_folder) if f.lower().endswith(".mp4")]
        segment_files.sort()

        if not segment_files:
            logger.warning(f"{person_folder_name}: 생성된 세그먼트가 없습니다.")
            continue

        logger.info(f"{person_folder_name}: {len(segment_files)}개 세그먼트 처리 시작")

        segment_tasks = []
        for seg_fname in segment_files:
            # 출력 파일명에 person_folder_name을 포함하여 구분
            output_seg_fname = f"{person_folder_name}_{seg_fname}"
            seg_input = os.path.join(person_segment_temp_folder, seg_fname)
            seg_cropped = os.path.join(temp_dir, f"crop_{output_seg_fname}")
            output_seg_path = os.path.join(final_person_output_dir, output_seg_fname)

            segment_tasks.append({
                'seg_fname': output_seg_fname,
                'seg_input': seg_input,
                'seg_cropped': seg_cropped,
                'output_path': output_seg_path
            })
        
        # 순차 처리
        for task in segment_tasks:
            process_single_segment_ffmpeg(task)
        
        logger.success(f"{person_folder_name}: {len(segment_tasks)}개 세그먼트 처리 완료")
        total_segments_processed += len(segment_tasks)

        # 임시 폴더 정리
        if os.path.exists(person_segment_temp_folder):
            shutil.rmtree(person_segment_temp_folder)
        if os.path.exists(person_trimmed_path):
            os.remove(person_trimmed_path)

    # 최종 리포트 정보 업데이트
    num_processes = max(1, min(8, int(cpu_count() * 0.8)))
    reporter.set_processing_info(total_frames, total_segments_processed, num_processes)

    reporter.end_stage("DUAL 모드 처리", segments=total_segments_processed)
    logger.success(f"DUAL 모드 전체 처리 완료: 총 {total_segments_processed}개 세그먼트, {len(top_2_faces)}명 인식")


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


def process_dual_split_mode_segments(id_timeline, fps, source_video, temp_dir, basename, reporter, total_frames):
    """
    DUAL_SPLIT 모드 전용: 화면을 좌우로 분할하여 2명을 동시에 추적
    
    Args:
        id_timeline: 얼굴 ID 타임라인
        fps: 비디오 FPS
        source_video: 소스 비디오 경로 (요약본)
        temp_dir: 임시 디렉토리
        basename: 베이스 파일명
        reporter: 성능 리포터
        total_frames: 총 프레임 수
    """
    from collections import Counter
    import cv2
    from moviepy import VideoFileClip  # MoviePy v2
    import numpy as np
    from src.face_tracker.utils.logging import logger
    from datetime import datetime

    print(f"🔄 CONSOLE: DUAL_SPLIT 모드 처리 시작 ({datetime.now().strftime('%H:%M:%S')})")
    reporter.start_stage("DUAL_SPLIT 모드 처리")
    logger.stage("DUAL_SPLIT 모드: 화면 분할 2인 동시 추적...")
    logger.debug("🔍 DUAL_SPLIT: process_dual_split_mode_segments() 함수 시작")
    print(f"🔄 CONSOLE: process_dual_split_mode_segments() 진입 완료 - 파라미터: fps={fps}, total_frames={total_frames}")

    try:
        # 1. 하이브리드 Person 할당 전략 수행
        print(f"🔄 CONSOLE: DUAL_SPLIT 의존성 로드 시작 ({datetime.now().strftime('%H:%M:%S')})")
        logger.debug("🔍 DUAL_SPLIT: import 구문 로드 중...")
        try:
            from src.face_tracker.core.embeddings import SmartEmbeddingManager
            from src.face_tracker.processing.selector import TargetSelector
            from src.face_tracker.config import L2_NORMALIZATION_ENABLED
            logger.debug("✅ DUAL_SPLIT: import 구문 로드 완료")
            print(f"✅ CONSOLE: DUAL_SPLIT 의존성 로드 완료 ({datetime.now().strftime('%H:%M:%S')})")
        except Exception as e:
            logger.error(f"❌ DUAL_SPLIT: import 구문 로드 실패 - {e}")
            print(f"❌ CONSOLE: DUAL_SPLIT 의존성 로드 실패: {e}")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        logger.debug("🔍 DUAL_SPLIT: SmartEmbeddingManager 초기화 중...")
        try:
            emb_manager = SmartEmbeddingManager()
            logger.debug("✅ DUAL_SPLIT: SmartEmbeddingManager 초기화 완료")
        except Exception as e:
            logger.error(f"❌ DUAL_SPLIT: SmartEmbeddingManager 초기화 실패 - {e}")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        logger.debug("🔍 DUAL_SPLIT: embeddings 로드 중...")
        try:
            embeddings = emb_manager.get_all_embeddings()
            logger.debug(f"✅ DUAL_SPLIT: embeddings 로드 완료 - {len(embeddings) if embeddings else 0}개")
        except Exception as e:
            logger.error(f"❌ DUAL_SPLIT: embeddings 로드 실패 - {e}")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        # L2 정규화 적용된 ID 병합 (간단한 임계값 사용)
        if embeddings:
            logger.debug("🔍 DUAL_SPLIT: ID 병합 시작...")
            try:
                from src.face_tracker.config import DUAL_MODE_SIMILARITY_THRESHOLD
                
                logger.debug(f"🔍 DUAL_SPLIT: ID 병합 파라미터 - 임계값: {DUAL_MODE_SIMILARITY_THRESHOLD}, L2: {L2_NORMALIZATION_ENABLED}")
                merged_timeline = TargetSelector._merge_similar_ids(
                    id_timeline, embeddings, 
                    similarity_threshold=DUAL_MODE_SIMILARITY_THRESHOLD,
                    use_l2_norm=L2_NORMALIZATION_ENABLED
                )
                logger.info(f"DUAL_SPLIT 모드 ID 병합 완료 (임계값: {DUAL_MODE_SIMILARITY_THRESHOLD:.3f}, L2: {L2_NORMALIZATION_ENABLED})")
                logger.debug("✅ DUAL_SPLIT: ID 병합 성공적으로 완료")
            except Exception as e:
                logger.error(f"❌ DUAL_SPLIT: ID 병합 실패 - {e}")
                logger.warning("🔄 DUAL_SPLIT: ID 병합 실패로 원본 타임라인 사용")
                merged_timeline = id_timeline
        else:
            logger.debug("🔍 DUAL_SPLIT: embeddings가 없어 ID 병합 스킵")
            merged_timeline = id_timeline
        
        # 2. 하이브리드 Person 할당
        print(f"🔄 CONSOLE: Person 할당 시작 ({datetime.now().strftime('%H:%M:%S')})")
        logger.debug("🔍 DUAL_SPLIT: assign_persons_hybrid() 호출 시작...")
        try:
            person1_id, person2_id = assign_persons_hybrid(merged_timeline, source_video, fps)
            logger.debug(f"✅ DUAL_SPLIT: assign_persons_hybrid() 완료 - Person1: {person1_id}, Person2: {person2_id}")
            print(f"✅ CONSOLE: Person 할당 완료 - Person1: {person1_id}, Person2: {person2_id} ({datetime.now().strftime('%H:%M:%S')})")
        except Exception as e:
            logger.error(f"❌ DUAL_SPLIT: assign_persons_hybrid() 실패 - {e}")
            print(f"❌ CONSOLE: Person 할당 실패: {e}")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        if person1_id is None or person2_id is None:
            logger.warning("DUAL_SPLIT 모드: 2명의 주요 인물을 찾을 수 없습니다.")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        logger.info(f"개선된 DUAL_SPLIT 모드: Person1(좌측)={person1_id}, Person2(우측)={person2_id}")
        
        # 3. 분할 영상 생성
        logger.debug("🔍 DUAL_SPLIT: 출력 폴더 생성 중...")
        try:
            from src.face_tracker.config import OUTPUT_ROOT
            output_folder = os.path.join(OUTPUT_ROOT, basename)
            os.makedirs(output_folder, exist_ok=True)
            
            split_video_path = os.path.join(output_folder, f"{basename}_dual_split.mp4")
            logger.debug(f"✅ DUAL_SPLIT: 출력 파일 경로 생성 완료: {split_video_path}")
        except Exception as e:
            logger.error(f"❌ DUAL_SPLIT: 출력 폴더 생성 실패 - {e}")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        # 4. 화면 분할 및 중앙 정렬 처리
        print(f"🔄 CONSOLE: 화면 분할 영상 생성 시작 ({datetime.now().strftime('%H:%M:%S')})")
        logger.debug("🔍 DUAL_SPLIT: create_split_screen_video() 호출 시작...")
        try:
            split_result = create_split_screen_video(
                source_video, merged_timeline, fps, 
                person1_id, person2_id, split_video_path
            )
            logger.debug("✅ DUAL_SPLIT: create_split_screen_video() 호출 완료")
            print(f"✅ CONSOLE: 화면 분할 영상 생성 완료 ({datetime.now().strftime('%H:%M:%S')})")
            
            # 처리 결과 정보 로그
            if split_result:
                logger.info(f"🎬 DUAL_SPLIT 처리 결과:")
                logger.info(f"  • 처리된 프레임: {split_result['processed_frames']:,}개")
                logger.info(f"  • 총 프레임: {split_result['total_frames']:,}개")
                logger.info(f"  • Person1 검출률: {split_result['person1_ratio']:.1f}%")
                logger.info(f"  • Person2 검출률: {split_result['person2_ratio']:.1f}%")
                logger.info(f"  • 전체 품질: {split_result['quality_status']}")
                
                # 성능 리포터에 정확한 정보 전달
                reporter.set_dual_split_stats(
                    processed_frames=split_result['processed_frames'],
                    total_frames=split_result['total_frames'],
                    segments=1  # DUAL_SPLIT 모드는 단일 영상 생성
                )
            else:
                logger.warning("🔍 DUAL_SPLIT: create_split_screen_video()에서 결과를 반환하지 않음")
                
        except Exception as e:
            logger.error(f"❌ DUAL_SPLIT: create_split_screen_video() 실패 - {e}")
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
            return
        
        logger.success(f"DUAL_SPLIT 모드 처리 완료: {split_video_path}")
        
        # 최종 결과를 리포터에 전달
        if split_result:
            reporter.end_stage("DUAL_SPLIT 모드 처리", 
                            segments=1, 
                            frames=split_result['processed_frames'])
        else:
            reporter.end_stage("DUAL_SPLIT 모드 처리", segments=1)
        
    except Exception as e:
        logger.error(f"❌ DUAL_SPLIT: process_dual_split_mode_segments() 전체 실행 실패 - {e}")
        reporter.end_stage("DUAL_SPLIT 모드 처리", segments=0)
        return


def assign_persons_hybrid(id_timeline, source_video, fps):
    """
    개선된 Person 할당 전략
    가장 많이 나온 2명을 일관되게 Person1/Person2로 할당
    Person1 = 더 많이 나온 사람 (좌측 출력 영역)
    Person2 = 두 번째로 많이 나온 사람 (우측 출력 영역)
    
    Returns:
        tuple: (person1_id, person2_id)
    """
    from collections import Counter
    from src.face_tracker.utils.logging import logger
    from datetime import datetime
    
    print(f"🔄 CONSOLE: assign_persons_hybrid() 시작 ({datetime.now().strftime('%H:%M:%S')})")
    logger.info("개선된 Person 할당 시작...")
    logger.debug("🔍 ASSIGN: assign_persons_hybrid() 함수 시작")
    
    # 1. 빈도 분석으로 가장 많이 나온 2명 찾기
    print(f"🔄 CONSOLE: ID 타임라인 분석 시작 - 전체 길이: {len(id_timeline)} ({datetime.now().strftime('%H:%M:%S')})")
    logger.debug("🔍 ASSIGN: ID 타임라인 분석 시작...")
    valid_ids = [fid for fid in id_timeline if fid is not None]
    logger.debug(f"🔍 ASSIGN: 유효한 ID 개수: {len(valid_ids)}")
    print(f"🔄 CONSOLE: 유효한 ID 개수: {len(valid_ids)}개")
    
    if not valid_ids:
        logger.warning("유효한 얼굴 ID가 없습니다.")
        logger.debug("🔍 ASSIGN: 유효한 ID가 없어서 종료")
        return None, None
    
    logger.debug("🔍 ASSIGN: Counter를 이용한 빈도 분석 중...")
    face_counts = Counter(valid_ids)
    logger.debug(f"🔍 ASSIGN: 전체 얼굴 ID별 카운트: {dict(face_counts.most_common())}")
    
    top_2_faces = face_counts.most_common(2)
    logger.debug(f"🔍 ASSIGN: 상위 2명 결과: {top_2_faces}")
    
    if len(top_2_faces) < 2:
        logger.warning("2명 미만의 얼굴만 감지됨")
        logger.debug("🔍 ASSIGN: 2명 미만으로 인한 조기 종료")
        return top_2_faces[0][0] if top_2_faces else None, None
    
    # 2. 일관된 Person 할당
    logger.debug("🔍 ASSIGN: Person1/Person2 할당 중...")
    # Person1 = 가장 많이 나온 사람 (좌측 출력 영역 담당)
    # Person2 = 두 번째로 많이 나온 사람 (우측 출력 영역 담당)
    person1_id = top_2_faces[0][0]  # 최다 출현자
    person2_id = top_2_faces[1][0]  # 차순 출현자
    logger.debug(f"🔍 ASSIGN: 할당 결과 - Person1: {person1_id}, Person2: {person2_id}")
    
    person1_count = top_2_faces[0][1]
    person2_count = top_2_faces[1][1]
    
    logger.info(f"Person 할당 완료:")
    logger.info(f"  Person1 (좌측): ID={person1_id} ({person1_count}회 출현)")
    logger.info(f"  Person2 (우측): ID={person2_id} ({person2_count}회 출현)")
    print(f"✅ CONSOLE: Person 할당 결과 - Person1: {person1_id}({person1_count}회), Person2: {person2_id}({person2_count}회)")
    
    # 3. 할당 비율 검증
    total_appearances = person1_count + person2_count
    person1_ratio = person1_count / total_appearances
    person2_ratio = person2_count / total_appearances
    
    logger.info(f"출현 비율: Person1={person1_ratio:.1%}, Person2={person2_ratio:.1%}")
    print(f"🔄 CONSOLE: 출현 비율 - Person1: {person1_ratio:.1%}, Person2: {person2_ratio:.1%} ({datetime.now().strftime('%H:%M:%S')})")
    
    # 극단적인 불균형 경고 (한 사람이 90% 이상)
    if person1_ratio > 0.9:
        logger.warning(f"Person1이 {person1_ratio:.1%} 비율로 압도적 출현 - 단일 인물 영상일 가능성")
    
    print(f"✅ CONSOLE: assign_persons_hybrid() 완료 ({datetime.now().strftime('%H:%M:%S')})")
    return person1_id, person2_id

class DualPersonTracker:
    """
    dual_split 전용 Person 트래커
    벡터 기반 Person1/Person2 식별 + 위치 기반 연속성 보장
    """
    
    def __init__(self, person1_id, person2_id, embeddings_manager):
        """
        Args:
            person1_id: 좌측 출력 영역 담당 Person ID
            person2_id: 우측 출력 영역 담당 Person ID
            embeddings_manager: SmartEmbeddingManager 인스턴스
        """
        from src.face_tracker.utils.logging import logger
        import numpy as np
        
        self.person1_id = person1_id  # 좌측 영역 담당
        self.person2_id = person2_id  # 우측 영역 담당
        
        # 기준 임베딩 벡터 추출 및 메모리 보관
        embeddings = embeddings_manager.get_all_embeddings()
        self.person1_embedding = embeddings.get(person1_id)
        self.person2_embedding = embeddings.get(person2_id)
        
        # L2 정규화 적용
        if self.person1_embedding is not None:
            self.person1_embedding = self._l2_normalize(self.person1_embedding)
        if self.person2_embedding is not None:
            self.person2_embedding = self._l2_normalize(self.person2_embedding)
        
        # 위치 트래킹 상태
        self.person1_last_pos = None  # (x, y, w, h)
        self.person2_last_pos = None
        self.person1_lost_frames = 0  # 트래킹 실패 카운터
        self.person2_lost_frames = 0
        
        # 메모리 효율성을 위한 캐시 관리
        self._embedding_cache = {}  # 임베딩 계산 결과 캐시
        self._cache_size_limit = 100  # 캐시 최대 크기
        
        logger.info(f"DualPersonTracker 초기화 완료:")
        logger.info(f"  Person1 (좌측): ID={person1_id}, 임베딩={'✅' if self.person1_embedding is not None else '❌'}")
        logger.info(f"  Person2 (우측): ID={person2_id}, 임베딩={'✅' if self.person2_embedding is not None else '❌'}")
    
    def _l2_normalize(self, embedding):
        """L2 정규화 적용"""
        import numpy as np
        try:
            # 임베딩을 1차원 배열로 변환
            emb_flat = np.array(embedding).flatten()
            
            # L2 norm 계산
            norm = np.linalg.norm(emb_flat)
            
            # 정규화 (norm이 0인 경우 원본 반환)
            if norm > 1e-8:  # 매우 작은 값으로 0 체크
                return emb_flat / norm
            else:
                print(f"⚠️ CONSOLE: L2 norm이 0에 가까움 - norm: {norm}")
                return emb_flat
        except Exception as e:
            print(f"❌ CONSOLE: L2 정규화 에러 - {e}")
            return np.array(embedding).flatten()  # 에러 시 평탄화만 수행
    
    def _calculate_cosine_similarity(self, emb1, emb2):
        """코사인 유사도 계산"""
        import numpy as np
        try:
            # 임베딩을 1차원 배열로 변환 (안전성 확보)
            emb1_flat = np.array(emb1).flatten()
            emb2_flat = np.array(emb2).flatten()
            
            # 내적 계산 후 스칼라 값으로 변환
            dot_product = np.dot(emb1_flat, emb2_flat)
            
            # 배열인 경우 스칼라로 변환
            if isinstance(dot_product, np.ndarray):
                dot_product = float(dot_product.item())
            
            return dot_product
        except Exception as e:
            print(f"❌ CONSOLE: 코사인 유사도 계산 에러 - {e}")
            return 0.0  # 에러 시 0 반환  # L2 정규화된 벡터는 내적이 코사인 유사도
    
    def _calculate_position_distance(self, pos1, pos2):
        """위치 간 거리 계산"""
        import numpy as np
        
        if pos1 is None or pos2 is None:
            return float('inf')
        
        try:
            # 튜플/배열을 안전하게 스칼라 값으로 변환
            pos1 = np.array(pos1).flatten()
            pos2 = np.array(pos2).flatten()
            
            if len(pos1) < 4 or len(pos2) < 4:
                return float('inf')
            
            # 중심점 기준 거리 계산 (스칼라 값 보장)
            center1_x = float(pos1[0] + pos1[2] // 2)
            center1_y = float(pos1[1] + pos1[3] // 2)
            center2_x = float(pos2[0] + pos2[2] // 2)
            center2_y = float(pos2[1] + pos2[3] // 2)
            
            distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            
            # 스칼라 값으로 변환
            if isinstance(distance, np.ndarray):
                distance = float(distance.item())
            
            return distance
        except Exception as e:
            print(f"❌ CONSOLE: 위치 거리 계산 에러 - {e}")
            return float('inf')
    
    def match_faces_to_persons(self, faces, face_embeddings):
        """
        여러 얼굴 중에서 Person1/Person2에 해당하는 얼굴 찾기 (완전 안전 버전)
        
        Args:
            faces: MTCNN 검출된 얼굴 bbox 리스트 [(x1,y1,x2,y2), ...]
            face_embeddings: 각 얼굴의 임베딩 벡터 리스트

        Returns:
            tuple: (person1_face_box, person2_face_box)
        """
        from src.face_tracker.dual_split_config import (
            TRACKING_POSITION_THRESHOLD, 
            TRACKING_SIMILARITY_THRESHOLD,
            TRACKING_MAX_LOST_FRAMES
        )
        
        # 안전한 입력 검증
        try:
            if faces is None:
                self.person1_lost_frames += 1
                self.person2_lost_frames += 1
                return None, None
            
            # faces를 안전한 리스트로 변환
            import numpy as np
            if isinstance(faces, np.ndarray):
                if faces.size == 0:
                    self.person1_lost_frames += 1
                    self.person2_lost_frames += 1
                    return None, None
                faces_list = faces.tolist()
            elif isinstance(faces, list):
                if len(faces) == 0:
                    self.person1_lost_frames += 1
                    self.person2_lost_frames += 1
                    return None, None
                faces_list = faces
            else:
                self.person1_lost_frames += 1
                self.person2_lost_frames += 1
                return None, None
            
            face_count = len(faces_list)
            if face_count == 0:
                self.person1_lost_frames += 1
                self.person2_lost_frames += 1
                return None, None
                
        except Exception as e:
            print(f"❌ CONSOLE: 입력 검증 에러 - {e}")
            self.person1_lost_frames += 1
            self.person2_lost_frames += 1
            return None, None
        
        # 1단계: 위치 기반 매칭 (연속성 우선)
        person1_candidates = []
        person2_candidates = []
        
        for i in range(face_count):
            try:
                face = faces_list[i]
                # 안전한 좌표 추출
                x1, y1, x2, y2 = float(face[0]), float(face[1]), float(face[2]), float(face[3])
                face_pos = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
                
                # Person1 위치 기반 매칭
                if self.person1_last_pos is not None:
                    try:
                        distance = self._calculate_position_distance(face_pos, self.person1_last_pos)
                        distance_val = float(distance)
                        threshold_val = float(TRACKING_POSITION_THRESHOLD)
                        
                        if distance_val < threshold_val:
                            person1_candidates.append((i, face, distance_val))
                    except Exception as e:
                        print(f"❌ CONSOLE: Person1 위치 매칭 에러 - {e}")
                        continue
                
                # Person2 위치 기반 매칭  
                if self.person2_last_pos is not None:
                    try:
                        distance = self._calculate_position_distance(face_pos, self.person2_last_pos)
                        distance_val = float(distance)
                        threshold_val = float(TRACKING_POSITION_THRESHOLD)
                        
                        if distance_val < threshold_val:
                            person2_candidates.append((i, face, distance_val))
                    except Exception as e:
                        print(f"❌ CONSOLE: Person2 위치 매칭 에러 - {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ CONSOLE: 얼굴 {i} 위치 처리 에러 - {e}")
                continue
        
        # 2단계: 벡터 유사도 검증
        person1_face = None
        person2_face = None
        
        # Person1 매칭
        if len(person1_candidates) > 0 and self.person1_embedding is not None:
            best_candidate = None
            best_similarity_val = -1.0
            
            for idx, face, pos_distance in person1_candidates:
                try:
                    if idx < len(face_embeddings) and face_embeddings[idx] is not None:
                        similarity = self._calculate_cosine_similarity(
                            self.person1_embedding, 
                            self._l2_normalize(face_embeddings[idx])
                        )
                        
                        # 안전한 스칼라 변환
                        similarity_val = float(similarity)
                        threshold_val = float(TRACKING_SIMILARITY_THRESHOLD)
                        
                        # 안전한 조건 비교
                        if similarity_val > best_similarity_val and similarity_val > threshold_val:
                            best_similarity_val = similarity_val
                            best_candidate = (idx, face)
                            
                except Exception as e:
                    print(f"❌ CONSOLE: Person1 유사도 계산 에러 - {e}")
                    continue
            
            if best_candidate:
                person1_face = best_candidate[1]
                # 안전한 위치 업데이트
                try:
                    x1, y1, x2, y2 = float(person1_face[0]), float(person1_face[1]), float(person1_face[2]), float(person1_face[3])
                    self.person1_last_pos = (x1, y1, x2 - x1, y2 - y1)
                    self.person1_lost_frames = 0
                except Exception as e:
                    print(f"❌ CONSOLE: Person1 위치 업데이트 에러 - {e}")
        
        # Person2 매칭
        if len(person2_candidates) > 0 and self.person2_embedding is not None:
            best_candidate = None
            best_similarity_val = -1.0
            
            for idx, face, pos_distance in person2_candidates:
                try:
                    if idx < len(face_embeddings) and face_embeddings[idx] is not None:
                        similarity = self._calculate_cosine_similarity(
                            self.person2_embedding, 
                            self._l2_normalize(face_embeddings[idx])
                        )
                        
                        # 안전한 스칼라 변환
                        similarity_val = float(similarity)
                        threshold_val = float(TRACKING_SIMILARITY_THRESHOLD)
                        
                        # 안전한 조건 비교
                        if similarity_val > best_similarity_val and similarity_val > threshold_val:
                            best_similarity_val = similarity_val
                            best_candidate = (idx, face)
                            
                except Exception as e:
                    print(f"❌ CONSOLE: Person2 유사도 계산 에러 - {e}")
                    continue
            
            if best_candidate:
                person2_face = best_candidate[1]
                # 안전한 위치 업데이트
                try:
                    x1, y1, x2, y2 = float(person2_face[0]), float(person2_face[1]), float(person2_face[2]), float(person2_face[3])
                    self.person2_last_pos = (x1, y1, x2 - x1, y2 - y1)
                    self.person2_lost_frames = 0
                except Exception as e:
                    print(f"❌ CONSOLE: Person2 위치 업데이트 에러 - {e}")
        
        # 3단계: Fallback - 초기 프레임이거나 위치 정보가 없는 경우
        if person1_face is None and self.person1_last_pos is None and face_count > 0:
            try:
                person1_face = faces_list[0]
                x1, y1, x2, y2 = float(person1_face[0]), float(person1_face[1]), float(person1_face[2]), float(person1_face[3])
                self.person1_last_pos = (x1, y1, x2 - x1, y2 - y1)
            except Exception as e:
                print(f"❌ CONSOLE: Person1 Fallback 에러 - {e}")
        
        if person2_face is None and self.person2_last_pos is None and face_count > 1:
            try:
                person2_face = faces_list[1]
                x1, y1, x2, y2 = float(person2_face[0]), float(person2_face[1]), float(person2_face[2]), float(person2_face[3])
                self.person2_last_pos = (x1, y1, x2 - x1, y2 - y1)
            except Exception as e:
                print(f"❌ CONSOLE: Person2 Fallback 에러 - {e}")
        
        return person1_face, person2_face

    def _get_cached_embedding(self, face_hash):
        """캐시된 임베딩 조회"""
        return self._embedding_cache.get(face_hash)
    
    def _cache_embedding(self, face_hash, embedding):
        """임베딩 캐시 저장"""
        # 캐시 크기 제한
        if len(self._embedding_cache) >= self._cache_size_limit:
            # 가장 오래된 항목 제거 (단순 구현)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[face_hash] = embedding
    
    def get_tracking_stats(self):
        """트래킹 통계 반환"""
        return {
            'person1_lost_frames': self.person1_lost_frames,
            'person2_lost_frames': self.person2_lost_frames,
            'cache_size': len(self._embedding_cache),
            'person1_has_embedding': self.person1_embedding is not None,
            'person2_has_embedding': self.person2_embedding is not None
        }


# 추가 import 구문들
from PIL import Image

def check_gpu_compatibility():
    """GPU/CPU 호환성 및 메모리 상태 확인"""
    import torch
    from src.face_tracker.config import DEVICE
    
    print(f"🔧 CONSOLE: GPU 호환성 검사 시작...")
    
    # CUDA 가용성 확인
    cuda_available = torch.cuda.is_available()
    print(f"🔧 CONSOLE: CUDA 사용 가능: {cuda_available}")
    
    # 설정된 디바이스 확인
    device_is_cuda = 'cuda' in str(DEVICE)
    print(f"🔧 CONSOLE: 설정된 디바이스: {DEVICE} ({'CUDA' if device_is_cuda else 'CPU'})")
    
    if device_is_cuda and not cuda_available:
        print(f"⚠️ CONSOLE: CUDA가 설정되었으나 사용할 수 없습니다. CPU로 대체 권장")
        return False, "CUDA 불가"
    
    # GPU 메모리 상태 확인 (CUDA 사용시)
    if cuda_available and device_is_cuda:
        try:
            # 메모리 정리
            torch.cuda.empty_cache()
            
            # 메모리 상태 확인
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            
            print(f"🔧 CONSOLE: GPU 메모리 - 총: {total_memory/1024**3:.1f}GB, 사용: {allocated_memory/1024**3:.1f}GB, 여유: {free_memory/1024**3:.1f}GB")
            
            # 최소 1GB 여유 메모리 필요
            if free_memory < 1024**3:
                print(f"⚠️ CONSOLE: GPU 메모리 부족 ({free_memory/1024**3:.1f}GB < 1.0GB)")
                return False, "GPU 메모리 부족"
            
            return True, "GPU 호환 가능"
            
        except Exception as e:
            print(f"❌ CONSOLE: GPU 메모리 확인 실패 - {e}")
            return False, f"GPU 확인 실패: {e}"
    
    # CPU 모드
    print(f"🔧 CONSOLE: CPU 모드로 동작")
    return True, "CPU 모드"

def create_split_screen_video(source_video, id_timeline, fps, person1_id, person2_id, output_path):
    """
    개선된 분할 화면 영상 생성
    - 트래커 기반 정확한 Person1/Person2 매칭
    - 프레임 스킵 없이 모든 프레임 처리 (연속성 보장)
    - 벡터 유사도 + 위치 기반 하이브리드 매칭
    
    Args:
        source_video: 소스 비디오 경로 (요약본)
        id_timeline: 얼굴 ID 타임라인
        fps: 비디오 FPS
        person1_id: 좌측 출력 영역 담당 Person ID
        person2_id: 우측 출력 영역 담당 Person ID
        output_path: 출력 경로
    """
    # 필수 import들을 가장 먼저 실행
    try:
        from datetime import datetime
        import cv2
        import numpy as np
        from src.face_tracker.core.models import ModelManager
        from src.face_tracker.core.embeddings import SmartEmbeddingManager
        from src.face_tracker.config import DEVICE
        from src.face_tracker.utils.logging import logger
        from src.face_tracker.dual_split_config import SKIP_NO_FACE_FRAMES
        import torch
        
        # 이제 logger를 안전하게 사용 가능
        print(f"🚀 CONSOLE: create_split_screen_video() 시작 ({datetime.now().strftime('%H:%M:%S')})")
        logger.debug("🚀 CREATE_SPLIT: create_split_screen_video() 함수 진입")
        logger.debug(f"🚀 CREATE_SPLIT: 입력 파라미터 - source_video: {source_video}")
        logger.debug(f"🚀 CREATE_SPLIT: 입력 파라미터 - person1_id: {person1_id}, person2_id: {person2_id}")
        logger.debug(f"🚀 CREATE_SPLIT: 입력 파라미터 - output_path: {output_path}")
        logger.debug(f"🚀 CREATE_SPLIT: 입력 파라미터 - fps: {fps}, timeline 길이: {len(id_timeline) if id_timeline else 'None'}")
        print(f"🔄 CONSOLE: 기본 파라미터 설정 - Person1: {person1_id}, Person2: {person2_id}, FPS: {fps}")
        logger.debug("🔧 CREATE_SPLIT: 모든 import 완료")
        
    except ImportError as e:
        print(f"❌ CONSOLE: import 오류 - {e}")
        print("❌ CONSOLE: 기본 import만 사용하여 진행")
        # 기본 import만 사용
        from datetime import datetime
        import cv2
        import numpy as np
        print(f"🚀 CONSOLE: create_split_screen_video() 시작 ({datetime.now().strftime('%H:%M:%S')}) - 제한된 모드")
        # logger 없이 진행
        logger = None
    
    except Exception as e:
        print(f"❌ CONSOLE: 예상치 못한 오류 - {e}")
        from datetime import datetime
        logger = None
        print(f"🚀 CONSOLE: create_split_screen_video() 시작 ({datetime.now().strftime('%H:%M:%S')}) - 비상 모드")
        
        # logger 없이는 기본 처리 결과만 반환
        return {
            'processed_frames': 0,
            'total_frames': 0,
            'person1_detected_count': 0,
            'person2_detected_count': 0,
            'person1_ratio': 0.0,
            'person2_ratio': 0.0,
            'overall_detection_rate': 0.0,
            'person_balance': 0.0,
            'quality_status': '🔴 시스템 오류',
            'tracking_stats': {},
            'output_path': output_path
        }
    
    # logger 사용 시 안전 처리
    if logger:
        logger.info(f"개선된 분할 화면 영상 생성 시작: {output_path}")
        logger.info(f"Person1(좌측)={person1_id}, Person2(우측)={person2_id}")
    else:
        print(f"🎬 CONSOLE: 분할 화면 생성 시작: {output_path}")
        print(f"🎬 CONSOLE: Person1(좌측)={person1_id}, Person2(우측)={person2_id}")
    
    logger.debug("🏗️ CREATE_SPLIT: 모델 및 트래커 초기화 시작...")
    
    # GPU/CPU 호환성 검사
    gpu_compatible, gpu_status = check_gpu_compatibility()
    if not gpu_compatible:
        print(f"⚠️ CONSOLE: GPU 호환성 문제 감지 - {gpu_status}")
        if logger:
            logger.warning(f"GPU 호환성 문제: {gpu_status}")
    else:
        print(f"✅ CONSOLE: GPU 호환성 확인 - {gpu_status}")
        if logger:
            logger.info(f"GPU 호환성: {gpu_status}")
    
    # 모델 및 트래커 초기화
    logger.debug("🏗️ CREATE_SPLIT: ModelManager 초기화...")
    model_manager = ModelManager(DEVICE)
    logger.debug("🏗️ CREATE_SPLIT: ModelManager 초기화 완료")
    
    logger.debug("🏗️ CREATE_SPLIT: MTCNN 모델 로드...")
    mtcnn = model_manager.get_mtcnn()
    logger.debug("🏗️ CREATE_SPLIT: MTCNN 모델 로드 완료")
    
    logger.debug("🏗️ CREATE_SPLIT: FaceNet 모델 로드...")
    facenet = model_manager.get_resnet()
    logger.debug("🏗️ CREATE_SPLIT: FaceNet 모델 로드 완료")
    
    logger.debug("🏗️ CREATE_SPLIT: SmartEmbeddingManager 초기화...")
    emb_manager = SmartEmbeddingManager()
    logger.debug("🏗️ CREATE_SPLIT: SmartEmbeddingManager 초기화 완료")
    
    # DualPersonTracker 초기화
    logger.debug("🏗️ CREATE_SPLIT: DualPersonTracker 초기화...")
    tracker = DualPersonTracker(person1_id, person2_id, emb_manager)
    logger.debug("🏗️ CREATE_SPLIT: DualPersonTracker 초기화 완료")
    
    logger.debug("🎬 CREATE_SPLIT: 비디오 파일 열기 시작...")
    # 비디오 설정
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        logger.error(f"🎬 CREATE_SPLIT: 비디오 파일 열기 실패 - {source_video}")
        return
    logger.debug("🎬 CREATE_SPLIT: 비디오 파일 열기 완료")
    
    logger.debug("🎬 CREATE_SPLIT: 비디오 속성 읽기...")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"🎬 CREATE_SPLIT: 비디오 속성 - 크기: {frame_width}x{frame_height}, 총 프레임: {total_frames}")
    
    logger.debug("🎬 CREATE_SPLIT: VideoWriter 초기화...")
    # 출력 비디오 설정 (1920x1080)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    if not out.isOpened():
        logger.error(f"🎬 CREATE_SPLIT: VideoWriter 초기화 실패 - {output_path}")
        cap.release()
        return
    logger.debug("🎬 CREATE_SPLIT: VideoWriter 초기화 완료")
    
    frame_idx = 0
    processed_frames = 0
    person1_detected_count = 0
    person2_detected_count = 0
    
    logger.info(f"총 {total_frames}프레임 처리 시작 (스킵 없음: {not SKIP_NO_FACE_FRAMES})")
    logger.debug("🔄 CREATE_SPLIT: 프레임별 처리 루프 시작...")
    
    while True:
        logger.debug(f"🔄 CREATE_SPLIT: Frame {frame_idx} 읽기...")
        ret, frame = cap.read()
        if not ret or frame_idx >= len(id_timeline):
            logger.debug(f"🔄 CREATE_SPLIT: 프레임 읽기 종료 - ret: {ret}, frame_idx: {frame_idx}, timeline 길이: {len(id_timeline)}")
            break
        
        current_id = id_timeline[frame_idx] if frame_idx < len(id_timeline) else None
        logger.debug(f"🔄 CREATE_SPLIT: Frame {frame_idx} - current_id: {current_id}")
        
        # 🚀 모든 프레임 처리 (스킵 없음)
        split_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        try:
            logger.debug(f"🔍 CREATE_SPLIT: Frame {frame_idx} MTCNN 얼굴 검출...")
            # 얼굴 검출
            faces, _ = mtcnn.detect(frame)
            face_embeddings = []
            
            # 얼굴이 검출된 경우 임베딩 추출 (안전한 배열 검사)
            has_faces = False
            if faces is not None:
                # NumPy 배열인 경우 안전하게 처리
                import numpy as np
                if isinstance(faces, np.ndarray):
                    has_faces = faces.size > 0 and len(faces.shape) >= 2
                elif isinstance(faces, list):
                    has_faces = len(faces) > 0
                else:
                    # 기타 타입의 경우 길이 확인
                    try:
                        has_faces = len(faces) > 0
                    except:
                        has_faces = False
            
            if has_faces:
                logger.debug(f"🔍 CREATE_SPLIT: Frame {frame_idx} - {len(faces)}개 얼굴 검출됨, 임베딩 추출 시작...")
                for face_idx, face in enumerate(faces):
                    try:
                        # 얼굴 크롭 후 임베딩 추출
                        face_crop = extract_face_for_embedding(frame, face)
                        if face_crop is not None:
                            with torch.no_grad():
                                # GPU/CPU 디바이스 호환성 확인
                                if hasattr(face_crop, 'device') and face_crop.device != facenet.device:
                                    face_crop = face_crop.to(facenet.device)
                                
                                embedding = facenet(face_crop.unsqueeze(0)).detach().cpu().numpy().flatten()
                                face_embeddings.append(embedding)
                                logger.debug(f"🔍 CREATE_SPLIT: Frame {frame_idx} - 얼굴 {face_idx} 임베딩 추출 성공")
                        else:
                            face_embeddings.append(None)
                            logger.debug(f"🔍 CREATE_SPLIT: Frame {frame_idx} - 얼굴 {face_idx} 크롭 실패")
                    except Exception as e:
                        logger.warning(f"Frame {frame_idx} 임베딩 추출 실패: {e}")
                        face_embeddings.append(None)
                
                logger.debug(f"🎯 CREATE_SPLIT: Frame {frame_idx} - 트래커로 Person 매칭 시작...")
                # 🎯 트래커로 정확한 Person1/Person2 매칭 (안전한 호출)
                try:
                    person1_face, person2_face = tracker.match_faces_to_persons(faces, face_embeddings)
                except Exception as tracker_error:
                    print(f"❌ CONSOLE: 트래커 매칭 에러 - {tracker_error}")
                    person1_face, person2_face = None, None
                logger.debug(f"🎯 CREATE_SPLIT: Frame {frame_idx} - 매칭 결과: Person1={person1_face is not None}, Person2={person2_face is not None}")
            else:
                logger.debug(f"🔍 CREATE_SPLIT: Frame {frame_idx} - 얼굴 검출되지 않음")
                person1_face, person2_face = None, None
            
            # Person1 처리 (좌측 960x1080)
            if person1_face is not None:
                logger.debug(f"✂️ CREATE_SPLIT: Frame {frame_idx} - Person1 얼굴 크롭 시작...")
                person1_crop = create_centered_face_crop(frame, person1_face, 960, 1080)
                split_frame[:, :960] = person1_crop
                person1_detected_count += 1
                logger.debug(f"✂️ CREATE_SPLIT: Frame {frame_idx} - Person1 크롭 완료")
            # Person1이 없어도 빈 영역 유지 (검은색)
            
            # Person2 처리 (우측 960x1080)
            if person2_face is not None:
                logger.debug(f"✂️ CREATE_SPLIT: Frame {frame_idx} - Person2 얼굴 크롭 시작...")
                person2_crop = create_centered_face_crop(frame, person2_face, 960, 1080)
                split_frame[:, 960:] = person2_crop
                person2_detected_count += 1
                logger.debug(f"✂️ CREATE_SPLIT: Frame {frame_idx} - Person2 크롭 완료")
            # Person2가 없어도 빈 영역 유지 (검은색)
            
        except Exception as e:
            logger.warning(f"Frame {frame_idx} 처리 중 오류: {e}")
            # 오류 발생해도 빈 프레임 출력 (연속성 보장)
        
        # ✅ 모든 프레임 출력 (1920x1080 보장)
        logger.debug(f"💾 CREATE_SPLIT: Frame {frame_idx} - 비디오 출력...")
        out.write(split_frame)
        processed_frames += 1
        frame_idx += 1
        
        # 진행률 로그 (1000프레임마다)
        if frame_idx % 1000 == 0:
            progress = (frame_idx / total_frames) * 100
            logger.info(f"처리 진행률: {progress:.1f}% ({frame_idx}/{total_frames})")
    
    logger.debug("🔚 CREATE_SPLIT: 프레임 처리 루프 종료, 리소스 정리...")
    cap.release()
    out.release()
    logger.debug("🔚 CREATE_SPLIT: 비디오 리소스 정리 완료")
    
    # 결과 통계
    person1_ratio = (person1_detected_count / processed_frames) * 100
    person2_ratio = (person2_detected_count / processed_frames) * 100
    
    # 트래킹 통계 수집
    tracking_stats = tracker.get_tracking_stats()
    
    # 결과 통계 계산
    person1_ratio = (person1_detected_count / processed_frames) * 100 if processed_frames > 0 else 0
    person2_ratio = (person2_detected_count / processed_frames) * 100 if processed_frames > 0 else 0
    
    # 품질 지표 계산
    overall_detection_rate = ((person1_detected_count + person2_detected_count) / (processed_frames * 2)) * 100
    person_balance = abs(person1_ratio - person2_ratio)  # 균형도 (낮을수록 좋음)
    
    # 종합 품질 평가
    if overall_detection_rate >= 80 and person_balance <= 30:
        quality_status = "🟢 우수"
    elif overall_detection_rate >= 60 and person_balance <= 50:
        quality_status = "🟡 양호"
    else:
        quality_status = "🔴 개선 필요"
    
    logger.success(f"분할 화면 영상 생성 완료 - 품질: {quality_status}")
    logger.info(f"📊 처리 통계:")
    logger.info(f"  • 처리된 프레임: {processed_frames:,}")
    logger.info(f"  • 전체 검출률: {overall_detection_rate:.1f}%")
    logger.info(f"  • Person1(좌측) 검출률: {person1_ratio:.1f}% ({person1_detected_count:,}/{processed_frames:,})")
    logger.info(f"  • Person2(우측) 검출률: {person2_ratio:.1f}% ({person2_detected_count:,}/{processed_frames:,})")
    logger.info(f"  • Person 균형도: {person_balance:.1f}% (낮을수록 좋음)")
    
    logger.info(f"🎯 트래킹 성능:")
    logger.info(f"  • Person1 트래킹 실패: {tracking_stats['person1_lost_frames']:,}프레임")
    logger.info(f"  • Person2 트래킹 실패: {tracking_stats['person2_lost_frames']:,}프레임") 
    logger.info(f"  • 임베딩 캐시 크기: {tracking_stats['cache_size']}")
    logger.info(f"  • Person1 기준 임베딩: {'✅' if tracking_stats['person1_has_embedding'] else '❌'}")
    logger.info(f"  • Person2 기준 임베딩: {'✅' if tracking_stats['person2_has_embedding'] else '❌'}")
    
    logger.info(f"📁 출력 파일: {output_path}")
    
    # 품질 개선 제안
    if person_balance > 50:
        logger.warning("⚠️ Person 불균형이 큽니다. assign_persons_hybrid() 로직 검토 필요")
    if overall_detection_rate < 60:
        logger.warning("⚠️ 전체 검출률이 낮습니다. 임베딩 임계값 조정 고려")
    if tracking_stats['person1_lost_frames'] > processed_frames * 0.1 or tracking_stats['person2_lost_frames'] > processed_frames * 0.1:
        logger.warning("⚠️ 트래킹 실패율이 높습니다. 위치 임계값 조정 고려")
    
    logger.debug("🎉 CREATE_SPLIT: create_split_screen_video() 함수 완료")
    
    # 처리 결과 반환
    return {
        'processed_frames': processed_frames,
        'total_frames': total_frames,
        'person1_detected_count': person1_detected_count,
        'person2_detected_count': person2_detected_count,
        'person1_ratio': person1_ratio,
        'person2_ratio': person2_ratio,
        'overall_detection_rate': overall_detection_rate,
        'person_balance': person_balance,
        'quality_status': quality_status,
        'tracking_stats': tracking_stats,
        'output_path': output_path
    }


def extract_face_for_embedding(frame, face_box):
    """
    얼굴 임베딩 추출용 크롭 생성
    
    Args:
        frame: 원본 프레임
        face_box: 얼굴 바운딩 박스 [x1, y1, x2, y2]

    Returns:
        torch.Tensor: 160x160 크기의 정규화된 얼굴 이미지 텐서 (GPU에 위치)
    """
    import cv2
    import torch
    from torchvision import transforms
    from src.face_tracker.config import DEVICE
    
    try:
        x1, y1, x2, y2 = face_box.astype(int)
        
        # 얼굴 영역 크롭
        face_crop = frame[y1:y2, x1:x2]
        
        # 160x160으로 리사이즈 (FaceNet 입력 크기)
        face_resized = cv2.resize(face_crop, (160, 160))
        
        # RGB 변환 및 정규화
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환 후 텐서 변환
        from PIL import Image
        face_pil = Image.fromarray(face_rgb)
        
        # 텐서 변환 및 정규화 (-1 ~ 1 범위)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        face_tensor = transform(face_pil)
        
        # GPU로 이동 (CUDA 가용성 확인) - 안전한 타입 검사
        if torch.cuda.is_available():
            device_str = str(DEVICE)
            if 'cuda' in device_str:
                face_tensor = face_tensor.to(DEVICE)
        
        return face_tensor
    
    except Exception as e:
        print(f"❌ CONSOLE: extract_face_for_embedding 에러 - {e}")
        return None


def create_centered_face_crop(frame, face_box, target_width, target_height):
    """
    얼굴을 중앙에 위치시킨 크롭 생성
    
    Args:
        frame: 원본 프레임
        face_box: 얼굴 바운딩 박스 [x1, y1, x2, y2]
        target_width: 목표 너비
        target_height: 목표 높이
    
    Returns:
        numpy.ndarray: 중앙 정렬된 크롭 이미지
    """
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_box.astype(int)
    
    # 얼굴 중심점
    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2
    
    # 얼굴 크기 기반 크롭 영역 계산
    face_width = x2 - x1
    face_height = y2 - y1
    face_size = max(face_width, face_height)
    
    # 여유 공간을 고려한 크롭 크기
    crop_size = int(face_size * 2.5)  # 얼굴의 2.5배 크기로 크롭
    
    # 크롭 영역 계산
    crop_x1 = max(0, face_center_x - crop_size // 2)
    crop_y1 = max(0, face_center_y - crop_size // 2)
    crop_x2 = min(w, face_center_x + crop_size // 2)
    crop_y2 = min(h, face_center_y + crop_size // 2)
    
    # 크롭 추출
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 목표 크기로 리사이즈
    resized = cv2.resize(cropped, (target_width, target_height))
    
    return resized