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
from src.face_tracker.processing.gpu_batch_processor import create_gpu_batch_processor, split_tasks_into_batches
from src.face_tracker.processing.gpu_process_pool import create_gpu_process_pool
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


def process_segments_with_gpu_pool(segment_tasks, reporter):
    """
    GPU 프로세스 풀로 세그먼트 극한 병렬 처리
    """
    try:
        logger.info(f"🚀 GPU 프로세스 풀 처리 시작 - {len(segment_tasks)}개 세그먼트")
        
        # Import 검증
        try:
            from src.face_tracker.processing.gpu_process_pool import create_gpu_process_pool
            logger.info("GPU 프로세스 풀 모듈 import 성공")
            print("DEBUG: GPU 프로세스 풀 import 성공")
        except Exception as import_error:
            logger.error(f"GPU 프로세스 풀 import 실패: {str(import_error)}")
            print(f"DEBUG: GPU 프로세스 풀 import 실패: {str(import_error)}")
            return False
        
        # GPU 프로세스 풀 생성
        try:
            gpu_pool = create_gpu_process_pool()
            logger.info("GPU 프로세스 풀 객체 생성 성공")
            print("DEBUG: GPU 프로세스 풀 객체 생성 성공")
        except Exception as create_error:
            logger.error(f"GPU 프로세스 풀 생성 실패: {str(create_error)}")
            print(f"DEBUG: GPU 프로세스 풀 생성 실패: {str(create_error)}")
            import traceback
            print(f"DEBUG: 스택 트레이스: {traceback.format_exc()}")
            return False
        
        # 세그먼트 수에 맞는 프로세스 풀 시작
        try:
            num_processes = min(len(segment_tasks), gpu_pool.max_processes)
            logger.info(f"GPU 프로세스 풀 시작 시도 - {num_processes}개 프로세스")
            print(f"DEBUG: GPU 프로세스 풀 시작 시도 - {num_processes}개 프로세스")
            
            if not gpu_pool.start_pool(num_processes):
                logger.error("GPU 프로세스 풀 시작 실패")
                print("DEBUG: GPU 프로세스 풀 시작 실패")
                return False
            
            logger.success(f"GPU 프로세스 풀 시작 성공 - {num_processes}개 프로세스")
            print(f"DEBUG: GPU 프로세스 풀 시작 성공 - {num_processes}개 프로세스")
        except Exception as start_error:
            logger.error(f"GPU 프로세스 풀 시작 오류: {str(start_error)}")
            print(f"DEBUG: GPU 프로세스 풀 시작 오류: {str(start_error)}")
            import traceback
            print(f"DEBUG: 스택 트레이스: {traceback.format_exc()}")
            return False
        
        # 풀 상태 로깅
        stats = gpu_pool.get_pool_stats()
        logger.info(f"GPU 풀 상태: {stats['active_processes']}개 프로세스, VRAM {stats['vram_usage']}")
        
        # 세그먼트 병렬 처리
        results = gpu_pool.process_segments(segment_tasks)
        
        # 결과 처리 및 FFmpeg 후처리
        processed_count = 0
        failed_count = 0
        
        for result in results:
            if result['success']:
                # FFmpeg 오디오 동기화
                task_data = result['task_data']
                seg_cropped = task_data['seg_cropped']
                output_path = task_data['output_path']
                seg_input = task_data['seg_input']
                
                try:
                    # FFmpeg 오디오 동기화
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
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    processed_count += 1
                    logger.success(f"세그먼트 완료: {result['task_data']['seg_fname']}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg 오디오 동기화 실패: {result['task_data']['seg_fname']} - {str(e)}")
                    failed_count += 1
            else:
                logger.error(f"세그먼트 GPU 처리 실패: {result.get('error', '알 수 없는 오류')}")
                failed_count += 1
        
        # GPU 프로세스 풀 종료
        gpu_pool.shutdown_pool()
        
        logger.success(f"GPU 프로세스 풀 처리 완료 - 성공: {processed_count}, 실패: {failed_count}")
        
        return processed_count > 0
        
    except Exception as e:
        logger.error(f"❌ GPU 프로세스 풀 처리 전체 오류: {str(e)}")
        logger.warning("🔄 기존 GPU 배치 처리로 Fallback 시도")
        
        # Fallback: 기존 GPU 배치 처리 사용
        try:
            return process_segments_with_gpu_batch(segment_tasks, reporter)
        except Exception as fallback_error:
            logger.error(f"❌ Fallback 처리도 실패: {str(fallback_error)}")
            return False


def process_segments_with_gpu_batch(segment_tasks, reporter):
    """
    GPU 배치 처리로 세그먼트 처리
    """
    try:
        logger.info(f"GPU 배치 처리 시작 - {len(segment_tasks)}개 세그먼트")
        
        # GPU 배치 프로세서 생성
        gpu_process, task_queue, result_queue = create_gpu_batch_processor()
        
        # GPU 프로세스 시작
        gpu_process.start()
        
        # 태스크를 배치로 분할 (4개씩)
        task_batches = split_tasks_into_batches(segment_tasks, batch_size=4)
        
        # 배치들을 GPU 프로세스에 전송
        for i, batch in enumerate(task_batches):
            logger.info(f"배치 {i+1}/{len(task_batches)} 전송 중...")
            task_queue.put(batch)
        
        # 종료 신호 전송
        task_queue.put("STOP")
        
        # 결과 수집
        processed_count = 0
        failed_count = 0
        
        for i in range(len(task_batches)):
            try:
                batch_results = result_queue.get(timeout=300)  # 5분 타임아웃
                
                if isinstance(batch_results, dict) and "error" in batch_results:
                    logger.error(f"배치 {i+1} 처리 오류: {batch_results['error']}")
                    failed_count += 4  # 배치 크기 가정
                    continue
                
                # 개별 결과를 task_id로 매칭하여 처리
                batch_start_idx = i * 4  # 배치 시작 인덱스
                for j, result in enumerate(batch_results):
                    global_task_idx = batch_start_idx + j
                    if global_task_idx < len(segment_tasks):
                        if result['success']:
                            # FFmpeg 후처리 - 오디오 동기화
                            task_data = segment_tasks[global_task_idx]
                            seg_cropped = task_data['seg_cropped']
                            output_path = task_data['output_path']
                            seg_input = task_data['seg_input']
                            
                            try:
                                # FFmpeg 오디오 동기화
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
                                
                                subprocess.run(cmd, check=True, capture_output=True)
                                processed_count += 1
                                logger.success(f"세그먼트 완료: {result['seg_fname']}")
                            except subprocess.CalledProcessError as e:
                                logger.error(f"FFmpeg 오디오 동기화 실패: {result['seg_fname']} - {str(e)}")
                                failed_count += 1
                        else:
                            logger.error(f"세그먼트 GPU 처리 실패: {result['seg_fname']} - {result['error']}")
                            failed_count += 1
                
                logger.info(f"배치 {i+1}/{len(task_batches)} 완료")
                
            except Exception as e:
                logger.error(f"배치 {i+1} 결과 수신 오류: {str(e)}")
                failed_count += 4
                
        # GPU 프로세스 정리
        gpu_process.join(timeout=30)
        if gpu_process.is_alive():
            gpu_process.terminate()
            gpu_process.join()
        
        logger.success(f"GPU 배치 처리 완료 - 성공: {processed_count}, 실패: {failed_count}")
        
        return processed_count > 0
        
    except Exception as e:
        logger.error(f"GPU 배치 처리 전체 오류: {str(e)}")
        return False


def process_single_segment_ffmpeg(task_data):
    """
    FFmpeg를 사용한 단일 세그먼트 처리 (하위 호환성 유지)
    
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
            
            logger.info(f"타겟 선택 시작 - 모드: {TRACKING_MODE}, 전체 프레임: {len(id_timeline)}")
            logger.info(f"얼굴 인식된 프레임 수: {len([x for x in id_timeline if x is not None])}")
            
            target_id = TargetSelector.select_target(id_timeline, TRACKING_MODE, embeddings)
            if target_id is not None:
                stats = TargetSelector.get_target_stats(id_timeline, target_id)
                logger.success(f"타겟 선택 성공 - ID: {target_id}, 커버리지: {stats['coverage_ratio']:.1%}")
            else:
                logger.warning(f"타겟 찾기 실패 - 모드: {TRACKING_MODE}")
                unique_ids = list(set([x for x in id_timeline if x is not None]))
                logger.info(f"인식된 고유 ID 목록: {unique_ids}")
                if len(unique_ids) > 0:
                    logger.info("타겟 선택 실패했지만 인식된 ID가 있음 - 첫 번째 ID 사용")
                    target_id = unique_ids[0]
                else:
                    logger.error("인식된 ID가 전혀 없음 - 처리 중단")
                    return
            
            # 타겟 아닌 프레임은 None으로 표시
            if target_id is not None:
                id_timeline_bool = [tid if tid == target_id else None for tid in id_timeline]
            else:
                id_timeline_bool = id_timeline
            
            # FFmpeg를 사용한 고속 트리밍
            reporter.start_stage("비디오 트리밍")
            logger.stage("비디오 트리밍...")
            
            logger.info(f"트리밍 시작 - 입력: {condensed}")
            logger.info(f"트리밍 출력 경로: {trimmed}")
            logger.info(f"타임라인 길이: {len(id_timeline_bool)}, 유효 프레임: {len([x for x in id_timeline_bool if x is not None])}")
            
            trim_success = trim_by_face_timeline_ffmpeg(condensed, id_timeline_bool, fps2, threshold_frames=30, output_path=trimmed)
            
            if trim_success:
                source_for_crop = trimmed
                logger.success(f"트리밍 성공 - 출력: {trimmed}")
            else:
                source_for_crop = condensed
                logger.warning(f"트리밍 실패 - 원본 사용: {condensed}")
            
            logger.info(f"크롭 소스 파일: {source_for_crop}")
            if os.path.exists(source_for_crop):
                logger.info(f"크롭 소스 파일 존재 확인됨: {source_for_crop}")
            else:
                logger.error(f"크롭 소스 파일 없음: {source_for_crop}")
                
            reporter.end_stage("비디오 트리밍")
            
            # 3) FFmpeg를 사용한 병렬 세그먼트 분할
            reporter.start_stage("세그먼트 분할")
            logger.stage("세그먼트 분할...")
            segment_temp_folder = os.path.join(temp_dir, "segments")
            os.makedirs(segment_temp_folder, exist_ok=True)
            
            logger.info(f"세그먼트 분할 시작")
            logger.info(f"입력 파일: {source_for_crop}")
            logger.info(f"출력 폴더: {segment_temp_folder}")
            logger.info(f"세그먼트 길이: 10초")
            
            print(f"DEBUG: 세그먼트 분할 시작 - 입력: {source_for_crop}")
            print(f"DEBUG: 세그먼트 분할 - 출력 폴더: {segment_temp_folder}")
            
            slice_video_parallel_ffmpeg(source_for_crop, segment_temp_folder, segment_length=10)
            
            # 분할 결과 확인
            created_segments = [f for f in os.listdir(segment_temp_folder) if f.lower().endswith(".mp4")]
            logger.info(f"세그먼트 분할 완료 - 생성된 파일 수: {len(created_segments)}")
            for seg_file in created_segments:
                seg_path = os.path.join(segment_temp_folder, seg_file)
                file_size = os.path.getsize(seg_path) if os.path.exists(seg_path) else 0
                logger.info(f"  - {seg_file} ({file_size} bytes)")
            
            # 세그먼트 파일이 없으면 상세 디버깅
            if len(created_segments) == 0:
                logger.error(f"세그먼트 분할 실패 - 폴더 내용 확인")
                try:
                    all_files = os.listdir(segment_temp_folder)
                    logger.info(f"폴더 전체 파일 목록: {all_files}")
                    for f in all_files:
                        fpath = os.path.join(segment_temp_folder, f)
                        fsize = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                        logger.info(f"  전체파일: {f} ({fsize} bytes)")
                except Exception as e:
                    logger.error(f"폴더 읽기 오류: {str(e)}")
            
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
            
            # 세그먼트 존재 여부 확인
            if len(segment_files) == 0:
                logger.warning("세그먼트 파일이 없습니다 - GPU 프로세스 풀 처리 건너뜀")
                reporter.end_stage("얼굴 크롭", segments=0)
                # return 제거 - 처리 계속 진행하되 GPU 처리는 건너뜀
                logger.info("세그먼트가 없으므로 처리 완료")
                return
            
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
            
            # 성능 리포트에 정보 설정 (GPU 프로세스 풀) - 실제 생성된 세그먼트 수 기준
            num_processes = min(len(segment_files), 5)  # 세그먼트별 독립 프로세스
            reporter.set_processing_info(len(timeline), len(segment_files), num_processes)
            logger.warning(f"성능 리포트에 설정된 세그먼트 수: {len(segment_files)}개 (실제 파일 확인 필요)")
            
            # GPU 프로세스 풀로 극한 병렬 처리
            print(f"DEBUG: GPU 프로세스 풀 시작 - {len(segment_tasks)}개 작업")
            success = process_segments_with_gpu_pool(segment_tasks, reporter)
            print(f"DEBUG: GPU 프로세스 풀 완료 - 성공: {success}")
            
            if success:
                reporter.end_stage("얼굴 크롭", segments=len(segment_files))
                logger.success(f"GPU 프로세스 풀 처리 완료 ({len(segment_files)}개)")
                print(f"DEBUG: GPU 프로세스 풀 성공 완료")
            else:
                logger.error("GPU 프로세스 풀 처리 실패")
                print(f"DEBUG: GPU 프로세스 풀 실패")
        else:
            logger.error(f"요약본 생성 실패: {fname}")

    except Exception as e:
        logger.error(f"❌ {fname} 비디오 처리 오류: {str(e)}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
    finally:
        elapsed = time.time() - start_time
        logger.success(f"{fname} 처리 완료 ({int(elapsed)}초)")
        
        # 성능 리포트 생성 - 실제 출력 파일 확인
        if reporter:
            # 최종 출력 폴더에서 실제 생성된 파일 수 확인
            final_output_folder = os.path.join(OUTPUT_ROOT, basename)
            if os.path.exists(final_output_folder):
                actual_output_files = [f for f in os.listdir(final_output_folder) if f.lower().endswith('.mp4')]
                logger.info(f"최종 출력 폴더 파일 수: {len(actual_output_files)}개")
                # 실제 출력 파일 수로 업데이트
                reporter.segments_count = len(actual_output_files)
            
            reporter.generate_report()
        
        # 임시 디렉토리 정리 (디버깅을 위해 일시 비활성화)
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        logger.info(f"디버깅을 위해 임시 디렉토리 보존: {temp_dir}")


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
    print("DEBUG: 로그 시스템 초기화됨")

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