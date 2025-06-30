#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from deepface import DeepFace
import subprocess
from tqdm import tqdm

# CSRT tracker factory for OpenCV versions
try:
    TrackerCSRT_create = cv2.TrackerCSRT_create
except AttributeError:
    TrackerCSRT_create = cv2.legacy.TrackerCSRT_create

def crop_and_mux(input_path, output_path, backend='retinaface'):
    # 1) 준비
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{input_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_video = "_tmp_video.mp4"
    out = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total_frames, desc='Processing', unit='frame')

    # 2) 프레임 순회
    frame_idx = 0
    current_box = None  # (x1, y1, w, h) for tracking
    track_id = 1
    track_dims = None  # (h, w) of initial face box
    tracker = None  # OpenCV tracker instance
    crop_w, crop_h = 250, 250
    initial_emb = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        # Detect faces, but always handle frames
        try:
            dets = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=backend,
                enforce_detection=False,
                align=False
            )
        except ValueError:
            dets = []

        face_obj = None
        if dets:
            # If first detection, choose largest face and set initial embedding
            if initial_emb is None:
                # pick largest face by area
                areas = [(d['facial_area']['w']*d['facial_area']['h'], d) for d in dets]
                _, face_obj = max(areas, key=lambda x: x[0])
                # get embedding for this face
                emb_data = DeepFace.represent(
                    img_path=face_obj['face'], 
                    model_name='Facenet', enforce_detection=False
                )
                initial_emb = emb_data[0]['embedding']
            else:
                # choose face closest to initial embedding
                min_dist = float('inf')
                for d in dets:
                    emb = DeepFace.represent(
                        img_path=d['face'], model_name='Facenet', enforce_detection=False
                    )[0]['embedding']
                    # cosine distance
                    cos = np.dot(initial_emb, emb) / (np.linalg.norm(initial_emb)*np.linalg.norm(emb))
                    dist = 1 - cos
                    if dist < min_dist:
                        min_dist = dist
                        face_obj = d
            # compute new tracking box centered on face, size 200x200
            fa = face_obj['facial_area']
            cx = fa['x'] + fa['w']//2
            cy = fa['y'] + fa['h']//2
            x1 = max(0, cx - crop_w//2)
            y1 = max(0, cy - crop_h//2)
            # clamp to frame
            x1 = min(x1, width - crop_w)
            y1 = min(y1, height - crop_h)
            current_box = (x1, y1, crop_w, crop_h)
            tqdm.write(f"Tracking ID {track_id}: box updated to {current_box}")
            # (re)initialize tracker
            tracker = TrackerCSRT_create()
            tracker.init(frame, tuple(current_box))
        elif tracker is not None:
            # update tracker
            ok, box = tracker.update(frame)
            if ok:
                current_box = tuple(map(int, box))
                tqdm.write(f"Tracking ID {track_id}: tracker updated to {current_box}")
        else:
            # no box yet: skip
            continue

        # use current_box for cropping
        x1, y1, w, h = current_box
        crop = frame[y1:y1+h, x1:x1+w]

        # 메모리 연속화
        if crop.dtype != np.uint8:
            crop = cv2.convertScaleAbs(crop)
        if not crop.flags['C_CONTIGUOUS']:
            crop = np.ascontiguousarray(crop)

        # initialize writer dynamically on first face crop
        if out is None:
            h_crop, w_crop = crop.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(tmp_video, fourcc, fps, (w_crop, h_crop))

        # write frame and log
        out.write(crop)
        tqdm.write(f"Frame {frame_idx}: face written, crop size={crop.shape[1]}x{crop.shape[0]}")

    cap.release()
    if out is not None:
        out.release()
    pbar.close()

    # 3) 오디오 추출 & Mux
    tmp_audio = "_tmp_audio.aac"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "copy", tmp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_video,
        "-i", tmp_audio,
        "-c", "copy", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4) 임시 파일 정리
    os.remove(tmp_video)
    os.remove(tmp_audio)

    print(f"✅ Done. Processed {frame_idx} frames, saved to '{output_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tmp.py <input_video> [<output_video>]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base = os.path.basename(input_path)
        stem, _ = os.path.splitext(base)
        os.makedirs("videos/output", exist_ok=True)
        output_path = os.path.join("videos/output", f"{stem}_face.mp4")

    crop_and_mux(input_path, output_path)