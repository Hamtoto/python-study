# 현재 폴더에서 아래 코드 실행 
# source ..venv/bin/activate

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
import subprocess
from tqdm import tqdm

# --------------------------------------------------
# RTX 5080 CUDA 전용 Face Crop & Mux
# --------------------------------------------------

# TensorFlow GPU 메모리 growth 활성화
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] TensorFlow GPU memory growth enabled on {len(gpus)} device(s)")
else:
    print("[WARN] No GPU devices found for TensorFlow")


def crop_and_mux_cuda(input_path, output_path, backend='retinaface'):
    """
    입력 비디오에서 얼굴 영역(250x250)을 추출하여 임시 비디오로 저장 후,
    CUDA 가속을 활용한 FFmpeg NVENC로 오디오와 합성 출력.
    """
    # GPU 기반 크롭 헬퍼
    def gpu_crop(frame, x, y, w, h):
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(frame)
            roi = gpu_mat.rowRange(y, y + h).colRange(x, x + w)
            return roi.download()
        except Exception:
            # CUDA 지원 실패 시 CPU fallback
            return frame[y:y + h, x:x + w]

    # 비디오 캡처 (FFmpeg + CUDA 가속)
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: cannot open '{input_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_video = "_tmp_video.mp4"
    out = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total, desc='Processing (CUDA)', unit='frame')

    frame_idx = 0
    current_box = None
    crop_w, crop_h = 250, 250
    initial_emb = None
    tracker = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        # 얼굴 검출
        try:
            dets = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=backend,
                enforce_detection=False,
                align=False
            )
        except Exception:
            dets = []

        face = None
        if dets:
            # 첫 프레임: 가장 큰 얼굴 선택 및 초기 임베딩
            if initial_emb is None:
                sizes = [(d['facial_area']['w'] * d['facial_area']['h'], d) for d in dets]
                _, face = max(sizes, key=lambda x: x[0])
                emb = DeepFace.represent(
                    img_path=face['face'], model_name='Facenet', enforce_detection=False
                )[0]['embedding']
                initial_emb = np.array(emb)
            else:
                # 이후: 초기 임베딩과 가장 유사한 얼굴 선택
                best = (float('inf'), None)
                for d in dets:
                    emb = DeepFace.represent(
                        img_path=d['face'], model_name='Facenet', enforce_detection=False
                    )[0]['embedding']
                    cos = np.dot(initial_emb, emb) / (np.linalg.norm(initial_emb) * np.linalg.norm(emb))
                    dist = 1 - cos
                    if dist < best[0]:
                        best = (dist, d)
                face = best[1]

            # 트래킹 박스 계산 및 초기화
            fa = face['facial_area']
            cx, cy = fa['x'] + fa['w']//2, fa['y'] + fa['h']//2
            x1 = max(0, cx - crop_w//2)
            y1 = max(0, cy - crop_h//2)
            x1 = min(x1, width - crop_w)
            y1 = min(y1, height - crop_h)
            current_box = (x1, y1, crop_w, crop_h)
            tracker = (cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create')
                       else cv2.legacy.TrackerCSRT_create())
            tracker.init(frame, tuple(current_box))
        elif tracker:
            ok, box = tracker.update(frame)
            if ok:
                current_box = tuple(map(int, box))
        else:
            continue

        # GPU 크롭
        x1, y1, w, h = current_box
        crop = gpu_crop(frame, x1, y1, w, h)

        # 메모리 연속화
        if crop.dtype != np.uint8:
            crop = cv2.convertScaleAbs(crop)
        if not crop.flags['C_CONTIGUOUS']:
            crop = np.ascontiguousarray(crop)

        # 비디오 라이터 초기화(nvenc)
        if out is None:
            h_c, w_c = crop.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(tmp_video, cv2.CAP_FFMPEG, fourcc, fps, (w_c, h_c))

        out.write(crop)

    cap.release()
    if out:
        out.release()
    pbar.close()

    # 오디오 추출 & NVENC Mux
    tmp_audio = "_tmp_audio.aac"
    subprocess.run([
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path,
        '-vn', '-acodec', 'copy', tmp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', tmp_video,
        '-i', tmp_audio,
        '-c:v', 'h264_nvenc', '-c:a', 'copy', output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(tmp_video)
    os.remove(tmp_audio)
    print(f"✅ Done (CUDA). Processed {frame_idx} frames, output saved to '{output_path}'")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <input_video> [<output_video>]" )
        sys.exit(1)

    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) >= 3 else None
    if not outp:
        base = os.path.basename(inp)
        stem, _ = os.path.splitext(base)
        os.makedirs('videos/output', exist_ok=True)
        outp = os.path.join('videos/output', f'{stem}_cuda.mp4')

    crop_and_mux_cuda(inp, outp)
