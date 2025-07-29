"""
GPU Worker Process Module
"""
import os
import torch
from core.model_manager import ModelManager
from processors.video_tracker import track_and_crop_video

def gpu_crop_worker(tasks_queue, results_queue, device_id):
    """
    GPU를 사용하여 비디오 크롭 작업을 수행하는 워커 프로세스.
    단일 프로세스로 실행되어 GPU 메모리 충돌을 방지합니다.
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    resnet = model_manager.get_resnet()
    
    print(f"✅ GPU Worker (PID: {os.getpid()}) on {device} started.")

    while True:
        task_data = tasks_queue.get()
        if task_data is None:  # Sentinel for stopping
            break

        seg_input = task_data['seg_input']
        seg_cropped = task_data['seg_cropped']
        
        try:
            print(f"  [GPU Worker] 크롭 시작: {os.path.basename(seg_input)}")
            # GPU 모델을 인자로 전달하여 실제 크롭 수행
            track_and_crop_video(
                video_path=seg_input,
                output_path=seg_cropped,
                mtcnn=mtcnn,
                resnet=resnet,
                device=device
            )
            results_queue.put({'status': 'success', 'task': task_data})
            print(f"  [GPU Worker] 크롭 완료: {os.path.basename(seg_cropped)}")
        except Exception as e:
            error_msg = f"GPU Worker error during cropping {os.path.basename(seg_input)}: {e}"
            print(f"  [GPU Worker] ❌ {error_msg}")
            results_queue.put({'status': 'error', 'task': task_data, 'error': error_msg})
        finally:
            # GPU 메모리 정리 (선택적)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("✅ GPU Worker finished.")
