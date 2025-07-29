"""
비디오 처리 워크플로우 메인 모듈
"""
import os
import time
import shutil
import warnings
import subprocess
import torch
from multiprocessing import Pool, cpu_count, set_start_method, Process, Queue

# pkg_resources 경고 억제
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# CUDA 메모리 최적화 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from utils.audio_utils import get_voice_segments
from processors.face_analyzer import analyze_video_faces
from processors.id_timeline_generator import generate_id_timeline
from processors.video_trimmer import create_condensed_video, trim_by_face_timeline, slice_video
from processors.gpu_worker import gpu_crop_worker
from processors.target_selector import TargetSelector
from utils.logger import error_logger
from utils.input_validator import validate_video_file
from utils.exceptions import FFmpegError, GPUMemoryError, VideoProcessingError, InputValidationError
from config import (
    DEVICE, INPUT_DIR, OUTPUT_ROOT, TEMP_ROOT,
    SUPPORTED_VIDEO_EXTENSIONS, BATCH_SIZE_ANALYZE,
    BATCH_SIZE_ID_TIMELINE, VIDEO_CODEC, AUDIO_CODEC, TRACKING_MODE
)

def process_cpu_task(task_data):
    """
    CPU 집약적 작업(FFmpeg 오디오 병합) 처리 함수 (멀티프로세싱용)
    """
    try:
        seg_fname = task_data['seg_fname']
        seg_input = task_data['seg_input']
        seg_cropped = task_data['seg_cropped']
        output_path = task_data['output_path']

        print(f"  [CPU Worker] FFmpeg 병합 시작: {seg_fname}")

        threads = max(1, cpu_count() // 4)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-threads', str(threads),
            '-i', seg_cropped,
            '-i', seg_input,
            '-c:v', VIDEO_CODEC, '-c:a', AUDIO_CODEC,
            '-preset', 'fast',
            '-map', '0:v:0', '-map', '1:a:0',
            output_path
        ]

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise FFmpegError(command=ffmpeg_cmd, stderr=stderr)

        if os.path.exists(seg_cropped):
            os.remove(seg_cropped)

        print(f"  [CPU Worker] 완료: {seg_fname}")

    except FFmpegError as e:
        print(f"  [CPU Worker] ❌ {e}")
        error_logger.log_segment_error(seg_fname, str(e))
    except Exception as e:
        error_msg = str(e)
        print(f"  [CPU Worker] 오류: {seg_fname} - {error_msg}")
        error_logger.log_segment_error(seg_fname, error_msg)

def process_single_video(fname: str):
    """
    단일 비디오 파일 처리 (GPU Worker + CPU Pool 아키텍처)
    """
    basename = os.path.splitext(fname)[0]
    temp_dir = os.path.join(TEMP_ROOT, basename)
    os.makedirs(temp_dir, exist_ok=True)

    input_path = os.path.join(INPUT_DIR, fname)
    condensed = os.path.join(temp_dir, f"condensed_{fname}")
    trimmed = os.path.join(temp_dir, f"trimmed_{fname}")

    print(f"## 처리 시작: {fname}")
    start_time = time.time()

    try:
        validate_video_file(input_path)

        voice_timeline = get_voice_segments(input_path)
        timeline, fps = analyze_video_faces(input_path, batch_size=BATCH_SIZE_ANALYZE, device=DEVICE)

        if create_condensed_video(input_path, condensed, timeline, fps):
            print("요약본 완료")
            id_timeline, fps2 = generate_id_timeline(condensed, DEVICE, batch_size=BATCH_SIZE_ID_TIMELINE)

            from core.embedding_manager import SmartEmbeddingManager
            emb_manager = SmartEmbeddingManager()
            embeddings = emb_manager.get_all_embeddings()
            target_id = TargetSelector.select_target(id_timeline, TRACKING_MODE, embeddings)

            if target_id is not None:
                id_timeline_bool = [tid if tid == target_id else None for tid in id_timeline]
            else:
                id_timeline_bool = id_timeline

            if trim_by_face_timeline(condensed, id_timeline_bool, fps2, threshold_frames=30, output_path=trimmed):
                source_for_crop = trimmed
            else:
                source_for_crop = condensed

            segment_temp_folder = os.path.join(temp_dir, "segments")
            os.makedirs(segment_temp_folder, exist_ok=True)
            slice_video(source_for_crop, segment_temp_folder, segment_length=10)

            final_segment_folder = os.path.join(OUTPUT_ROOT, basename)
            os.makedirs(final_segment_folder, exist_ok=True)
            segment_files = sorted([f for f in os.listdir(segment_temp_folder) if f.lower().endswith(".mp4")])

            if not segment_files:
                print("## 처리할 세그먼트가 없습니다.")
                return

            print(f"## GPU/CPU 파이프라인 시작: {len(segment_files)}개 세그먼트")

            tasks_queue = Queue()
            results_queue = Queue()
            gpu_worker_process = Process(target=gpu_crop_worker, args=(tasks_queue, results_queue, 0))
            gpu_worker_process.start()

            for seg_fname in segment_files:
                task = {
                    'seg_fname': seg_fname,
                    'seg_input': os.path.join(segment_temp_folder, seg_fname),
                    'seg_cropped': os.path.join(temp_dir, f"crop_{seg_fname}"),
                    'output_path': os.path.join(final_segment_folder, seg_fname)
                }
                tasks_queue.put(task)

            tasks_queue.put(None)

            cpu_tasks = []
            for _ in range(len(segment_files)):
                result = results_queue.get()
                if result and result['status'] == 'success':
                    cpu_tasks.append(result['task'])
                else:
                    error_logger.log_segment_error(result['task']['seg_fname'], result.get('error', 'Unknown GPU error'))

            gpu_worker_process.join()

            if cpu_tasks:
                num_processes = max(1, cpu_count() - 1)
                print(f"## CPU 병렬 처리 시작: {len(cpu_tasks)}개 FFmpeg 작업 ({num_processes}코어 활용)")
                with Pool(processes=num_processes) as pool:
                    pool.map(process_cpu_task, cpu_tasks)
                print("## 모든 CPU 작업 완료")
            else:
                print("## CPU에서 처리할 작업이 없습니다.")
        else:
            print(f"요약본 생성 실패: {fname}")

    except InputValidationError as e:
        print(f"입력 파일 오류: {e}")
        error_logger.log_video_error(fname, str(e))
    except torch.cuda.OutOfMemoryError as e:
        error = GPUMemoryError(details=str(e))
        print(f"오류 발생: {error}")
        error_logger.log_video_error(fname, str(error))
    except FFmpegError as e:
        print(f"FFmpeg 오류 발생: {e}")
        error_logger.log_video_error(fname, str(e))
    except VideoProcessingError as e:
        print(f"비디오 처리 오류 발생: {e}")
        error_logger.log_video_error(fname, str(e))
    except Exception as e:
        error_msg = str(e)
        print(f"예상치 못한 오류 발생: {fname} - {error_msg}")
        import traceback
        traceback.print_exc()
        error_logger.log_video_error(fname, error_msg)
    finally:
        elapsed = time.time() - start_time
        print(f"{fname} 처리시간 : {int(elapsed)}초")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"## 완료: {fname}")

def process_all_videos():
    """
    입력 디렉토리의 모든 비디오 파일 처리
    """
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)
    error_logger.clear_log()
    print(f"## 에러 로그 파일: {error_logger.log_file}")

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            continue
        process_single_video(fname)

    print("모든 비디오 처리 완료")
