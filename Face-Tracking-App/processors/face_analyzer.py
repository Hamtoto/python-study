"""
얼굴 탐지 및 분석 모듈
"""
import cv2
from PIL import Image
from tqdm import tqdm
from core.model_manager import ModelManager
from config import DEVICE, BATCH_SIZE_ANALYZE


def analyze_video_faces(video_path: str, batch_size: int = BATCH_SIZE_ANALYZE, device=DEVICE) -> tuple[list[bool], float]:
    """
    비디오의 각 프레임에서 얼굴 탐지 여부를 분석
    
    Args:
        video_path: 비디오 파일 경로
        batch_size: 배치 처리 크기
        device: 사용할 디바이스 (CPU/GPU)
    
    Returns:
        tuple: (얼굴 탐지 타임라인, fps)
    """
    print("## 1단계: 얼굴 탐지 분석 시작")
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_detected_timeline = []
    buffer = []
    
    with tqdm(total=frame_count, desc="[GPU 분석]") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV 프레임을 직접 버퍼에 저장 (PIL 변환 건너뛰기)
            buffer.append(frame)
            pbar.update(1)
            if len(buffer) == batch_size or pbar.n == frame_count:
                # 배치 BGR→RGB 변환 최적화
                rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
                pil_images = [Image.fromarray(rgb_frame) for rgb_frame in rgb_frames]
                boxes_list, _ = mtcnn.detect(pil_images)
                face_detected_timeline.extend(b is not None for b in boxes_list)
                buffer.clear()
    
    if buffer:
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
        pil_images = [Image.fromarray(rgb_frame) for rgb_frame in rgb_frames]
        boxes_list, _ = mtcnn.detect(pil_images)
        face_detected_timeline.extend(b is not None for b in boxes_list)
    
    cap.release()
    print("분석 완료")
    return face_detected_timeline, fps