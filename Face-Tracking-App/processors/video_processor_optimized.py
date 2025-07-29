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

from utils.audio_utils_optimized import get_voice_segments_ffmpeg
from processors.face_analyzer import analyze_video_faces
from processors.id_timeline_generator import generate_id_timeline
from processors.video_trimmer_optimized import (
    create_condensed_video_ffmpeg, 
    trim_by_face_timeline_ffmpeg, 
    slice_video_parallel_ffmpeg
)
from processors.video_tracker import track_and_crop_video
from processors.target_selector import TargetSelector
from utils.logger import error_logger
from config import (
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
        
        print(f"  처리 중: {seg_fname}")
        
        # 1) 얼굴 크롭
        track_and_crop_video(seg_input, seg_cropped)
        
        # 2) FFmpeg를 사용한 오디오 동기화 (MoviePy 대신)
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
        
        # 3) FFmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 4) 임시 파일 정리
        if os.path.exists(seg_cropped):
            os.remove(seg_cropped)
            
        print(f"  완료: {seg_fname}")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg 오류: {e.stderr}"
        print(f"  오류: {seg_fname} - {error_msg}")
        error_logger.log_segment_error(seg_fname, error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"  오류: {seg_fname} - {error_msg}")
        error_logger.log_segment_error(seg_fname, error_msg)


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

    print(f"## 처리 시작: {fname}")
    start_time = time.time()
    
    try:
        # 0) FFmpeg를 사용한 고속 오디오 VAD
        print("## 0단계: FFmpeg 오디오 VAD 처리...")
        voice_timeline = get_voice_segments_ffmpeg(input_path)
        print(f"## 음성 구간 {len(voice_timeline)}개 감지")
        
        # 1) 얼굴 감지 타임라인
        print("## 1단계: 얼굴 감지 분석...")
        timeline, fps = analyze_video_faces(input_path, batch_size=BATCH_SIZE_ANALYZE, device=DEVICE)
        
        # 2) FFmpeg를 사용한 고속 요약본 생성
        print("## 2단계: FFmpeg 요약본 생성...")
        if create_condensed_video_ffmpeg(input_path, condensed, timeline, fps):
            print("FFmpeg 요약본 완료")
            print(f"## 디버그: condensed 파일 경로 = {condensed}")
            print(f"## 디버그: 얼굴 트리밍 전 timeline 길이 = {len(timeline)}, fps = {fps}")
            print("## 2단계 완료, 이제 트리밍 및 세그먼트 분할을 시작합니다.")
            
            # 2a) ID 타임라인 생성 및 타겟 인물 자동 선택
            print("## 3단계: ID 타임라인 생성 및 타겟 선택...")
            id_timeline, fps2 = generate_id_timeline(condensed, DEVICE, batch_size=BATCH_SIZE_ID_TIMELINE)
            print(f"## 디버그: trimming 전 condensed 영상 프레임 수 예측 = {len(id_timeline)}")
            
            # 모드별 타겟 ID 선택 (임베딩 정보 포함)
            from core.embedding_manager import SmartEmbeddingManager
            emb_manager = SmartEmbeddingManager()
            embeddings = emb_manager.get_all_embeddings()
            
            target_id = TargetSelector.select_target(id_timeline, TRACKING_MODE, embeddings)
            if target_id is not None:
                stats = TargetSelector.get_target_stats(id_timeline, target_id)
                print(f"## 타겟 선택 완료: ID={target_id}, 모드={TRACKING_MODE}")
                print(f"## 타겟 통계: {stats['target_frames']}/{stats['total_frames']} 프레임 ({stats['coverage_ratio']:.2%})")
            else:
                print(f"## 경고: 타겟을 찾을 수 없습니다 (모드={TRACKING_MODE})")
            
            # 타겟 아닌 프레임은 None으로 표시
            if target_id is not None:
                id_timeline_bool = [tid if tid == target_id else None for tid in id_timeline]
            else:
                id_timeline_bool = id_timeline
            
            # FFmpeg를 사용한 고속 트리밍
            print("## 4단계: FFmpeg 트리밍...")
            if trim_by_face_timeline_ffmpeg(condensed, id_timeline_bool, fps2, threshold_frames=30, output_path=trimmed):
                source_for_crop = trimmed
            else:
                source_for_crop = condensed
            
            # 3) FFmpeg를 사용한 병렬 세그먼트 분할
            print("## 5단계: FFmpeg 병렬 세그먼트 분할...")
            segment_temp_folder = os.path.join(temp_dir, "segments")
            os.makedirs(segment_temp_folder, exist_ok=True)
            slice_video_parallel_ffmpeg(source_for_crop, segment_temp_folder, segment_length=10)

            # 4) 각 세그먼트별 얼굴 크롭 및 오디오 동기 병합 (병렬 처리)
            print("## 6단계: 세그먼트별 얼굴 크롭 및 오디오 동기화...")
            final_segment_folder = os.path.join(OUTPUT_ROOT, basename)
            os.makedirs(final_segment_folder, exist_ok=True)
            
            segment_files = [f for f in os.listdir(segment_temp_folder) 
                           if f.lower().endswith(".mp4")]
            segment_files.sort()  # 순서 보장
            
            print(f"## 세그먼트 병렬 처리 시작: {len(segment_files)}개 파일")
            
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
            print(f"## 병렬 처리 프로세스 수: {num_processes}")
            
            with Pool(processes=num_processes) as pool:
                pool.map(process_single_segment_ffmpeg, segment_tasks)
                    
            print(f"## 세그먼트 병렬 처리 완료")
        else:
            print(f"요약본 생성 실패: {fname}")

    except Exception as e:
        error_msg = str(e)
        print(f"비디오 처리 중 오류 발생: {fname} - {error_msg}")
        error_logger.log_video_error(fname, error_msg)
    finally:
        elapsed = time.time() - start_time
        print(f"{fname} 처리시간 : {int(elapsed)}초")
        
        # 임시 디렉토리 정리
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"## 완료: {fname}")


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

    # 에러 로그 초기화
    error_logger.clear_log()
    print(f"## 에러 로그 파일: {error_logger.log_file}")

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            continue
        
        process_single_video_optimized(fname)

    print("모든 비디오 처리 및 세그먼트별 얼굴 크롭/동기화 완료")


# 하위 호환성을 위한 함수명 유지
def process_single_video(fname: str):
    """최적화된 버전 호출"""
    return process_single_video_optimized(fname)


def process_all_videos():
    """최적화된 버전 호출"""
    return process_all_videos_optimized()