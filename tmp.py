#!/usr/bin/env python3
"""
Video Face Cropper using DeepFace
- Reads an input video, detects faces in each frame
- Crops face regions and writes each face stream to its own video file
- Supports multiple people: creates separate output video per face index
- Saves all output videos under a specified folder
"""
import os
import sys
import cv2
from deepface.detectors import FaceDetector


def crop_faces_from_video(input_path: str, output_dir: str, detector_backend: str = 'retinaface'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture and face detector
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = FaceDetector.build_model(detector_backend)

    # VideoWriters for each face index
    writers = {}
    face_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # Detect faces, returns list of dicts with facial_area
        detections = FaceDetector.detect_faces(detector, detector_backend, frame, align=False)
        for idx, face in enumerate(detections):
            fac = face['facial_area']  # dict with x, y, w, h
            x, y, w, h = fac['x'], fac['y'], fac['w'], fac['h']
            # Ensure valid crop coordinates
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            crop = frame[y1:y2, x1:x2]

            # If writer for this face idx not created, initialize
            if idx not in writers:
                out_path = os.path.join(output_dir, f"face_{idx+1}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h_crop, w_crop = crop.shape[:2]
                writers[idx] = cv2.VideoWriter(out_path, fourcc, fps, (w_crop, h_crop))
                face_count += 1
                print(f"Created writer for face {idx+1}, output: {out_path}")

            # Write cropped frame
            writers[idx].write(crop)

    # Release resources
    cap.release()
    for w in writers.values():
        w.release()

    print(f"Done! Processed {frame_idx} frames, extracted {face_count} face streams.")


def main():
    if len(sys.argv) < 3:
        print("Usage: crop_faces.py <input_video> <output_folder> [detector_backend]")
        sys.exit(1)
    input_video = sys.argv[1]
    output_folder = sys.argv[2]
    backend = sys.argv[3] if len(sys.argv) > 3 else 'retinaface'
    crop_faces_from_video(input_video, output_folder, backend)

if __name__ == '__main__':
    main()
