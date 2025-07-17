"""
비디오 처리 워크플로우 메인 모듈
"""
import os
import time
import shutil
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
from moviepy import VideoFileClip, AudioFileClip

# pkg_resources 경고 억제
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# CUDA 메모리 최적화 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from utils.audio_utils import get_voice_segments
from processors.face_analyzer import analyze_video_faces
from processors.id_timeline_generator import generate_id_timeline
from processors.video_trimmer import create_condensed_video, trim_by_face_timeline, slice_video
from processors.video_tracker import track_and_crop_video
from processors.target_selector import TargetSelector
from utils.logger import error_logger
from config import (
    DEVICE, INPUT_DIR, OUTPUT_ROOT, TEMP_ROOT, 
    SUPPORTED_VIDEO_EXTENSIONS, BATCH_SIZE_ANALYZE, 
    BATCH_SIZE_ID_TIMELINE, VIDEO_CODEC, AUDIO_CODEC, TRACKING_MODE
)


def process_single_segment(task_data):
    """
    단일 세그먼트 처리 함수 (멀티프로세싱용)
    
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
        
        # 2) 오디오 동기화
        vc = VideoFileClip(seg_cropped)
        ac = AudioFileClip(seg_input)
        final_seg = vc.with_audio(ac)
        
        # 3) 최종 파일 생성
        final_seg.write_videofile(output_path, codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC)
        
        # 4) 정리
        vc.close()
        ac.close()
        final_seg.close()
        if os.path.exists(seg_cropped):
            os.remove(seg_cropped)
            
        print(f"  완료: {seg_fname}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"  오류: {seg_fname} - {error_msg}")
        error_logger.log_segment_error(seg_fname, error_msg)


def process_single_video(fname: str):
    """
    단일 비디오 파일 처리
    
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
        # 0) 오디오 VAD로 말하는 구간 추출
        voice_timeline = get_voice_segments(input_path)
        
        # 1) 얼굴 감지 타임라인
        timeline, fps = analyze_video_faces(input_path, batch_size=BATCH_SIZE_ANALYZE, device=DEVICE)
        
        # 2) 요약본 생성
        if create_condensed_video(input_path, condensed, timeline, fps):
            print("요약본 완료")
            print(f"## 디버그: condensed 파일 경로 = {condensed}")
            print(f"## 디버그: 얼굴 트리밍 전 timeline 길이 = {len(timeline)}, fps = {fps}")
            print("## 2단계 완료, 이제 트리밍 및 세그먼트 분할을 시작합니다.")
            
            # 2a) ID 타임라인 생성 및 타겟 인물 자동 선택 후 30프레임 이상 미검출 구간 트리밍
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
            
            if trim_by_face_timeline(condensed, id_timeline_bool, fps2, threshold_frames=30, output_path=trimmed):
                source_for_crop = trimmed
            else:
                source_for_crop = condensed
            
            # 3) 연속된 트리밍된 영상 10초 세그먼트로 분할
            segment_temp_folder = os.path.join(temp_dir, "segments")
            os.makedirs(segment_temp_folder, exist_ok=True)
            slice_video(source_for_crop, segment_temp_folder, segment_length=10)

            # 4) 각 세그먼트별 얼굴 크롭 및 오디오 동기 병합 (순차 처리)
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
            
            # GPU 메모리 고려하여 프로세스 수 조정 (GPU 메모리 부족 방지)
            num_processes = max(1, min(6, int(cpu_count() * 0.75)))
            print(f"## 병렬 처리 프로세스 수: {num_processes}")
            
            with Pool(processes=num_processes) as pool:
                pool.map(process_single_segment, segment_tasks)
                    
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


def process_all_videos():
    """
    입력 디렉토리의 모든 비디오 파일 처리
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
        
        process_single_video(fname)

    print("모든 비디오 처리 및 세그먼트별 얼굴 크롭/동기화 완료")