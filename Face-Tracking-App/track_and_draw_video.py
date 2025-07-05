import cv2
import torch
import os
import argparse
import time
import shutil
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from facenet_pytorch import MTCNN
from tqdm import tqdm
import numpy as np

# --- 1단계: 얼굴 탐지 및 타임라인 생성 ---
def analyze_video_faces(video_path: str, batch_size: int, device: torch.device) -> tuple[list[bool], float]:
    print("## 1단계: 얼굴 탐지 분석 시작")
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
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
            buffer.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            pbar.update(1)
            if len(buffer) == batch_size or pbar.n == frame_count:
                boxes_list, _ = mtcnn.detect(buffer)
                face_detected_timeline.extend(b is not None for b in boxes_list)
                buffer.clear()
    if buffer:
        boxes_list, _ = mtcnn.detect(buffer)
        face_detected_timeline.extend(b is not None for b in boxes_list)
    cap.release()
    print("분석 완료")
    return face_detected_timeline, fps

# --- 2단계: 불필요 구간 제거 및 요약본 생성 ---
def create_condensed_video(video_path: str, output_path: str, timeline: list[bool], fps: float, cut_threshold: float = 2.0) -> bool:
    print("## 2단계: 불필요 구간 제거 시작")
    if not timeline:
        print("오류: 타임라인 데이터 없음")
        return False
    clips = []
    seg_start = 0
    face_seg = timeline[0]
    for i in range(1, len(timeline)):
        if timeline[i] != face_seg:
            end = i
            duration = (end - seg_start) / fps
            if face_seg or duration <= cut_threshold:
                clips.append((seg_start / fps, end / fps))
            seg_start = i
            face_seg = timeline[i]
    # 마지막
    end = len(timeline)
    duration = (end - seg_start) / fps
    if face_seg or duration <= cut_threshold:
        clips.append((seg_start / fps, end / fps))
    if not clips:
        print("오류: 유지할 클립 없음")
        return False
    print("MoviePy 요약본 생성 중…")
    original = VideoFileClip(video_path)
    subclips = [original.subclipped(s, e) for s, e in clips]
    summary = concatenate_videoclips(subclips)
    summary.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    original.close()
    print("요약본 완료")
    return True

# --- 3단계: 얼굴 중심 기준 크롭 (스무딩, 이상치, 하이브리드 트래킹, 클램핑) ---
def track_and_crop_video(video_path: str, output_path: str, batch_size: int = 16, crop_size: int = 250,
                         jump_thresh: float = 50.0, ema_alpha: float = 0.2, reinit_interval: int = 30):
    print("## 3단계: 크롭 시작 (스무딩·이상치·하이브리드·클램핑)")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))

    tracker = None
    prev_center = None
    smooth_center = None

    pbar = tqdm(total=total_frames, desc="[크롭 처리]")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        # 하이브리드: reinit 또는 tracker update
        if frame_idx % reinit_interval == 0 or tracker is None:
            # 탐지
            boxes_list, _ = mtcnn.detect([Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))])
            faces = boxes_list[0]
            if faces is not None and len(faces) > 0:
                x1, y1, x2, y2 = faces[0]
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                raw_cx = int((x1 + x2) / 2)
                raw_cy = int((y1 + y2) / 2)
            else:
                raw_cx, raw_cy = w // 2, h // 2
        else:
            ok, bb = tracker.update(frame)
            if ok:
                x, y, tw, th = bb
                raw_cx = int(x + tw / 2)
                raw_cy = int(y + th / 2)
            else:
                boxes_list, _ = mtcnn.detect([Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))])
                faces = boxes_list[0]
                if faces is not None and len(faces) > 0:
                    x1, y1, x2, y2 = faces[0]
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    raw_cx = int((x1 + x2) / 2)
                    raw_cy = int((y1 + y2) / 2)
                else:
                    raw_cx, raw_cy = w // 2, h // 2
        # 스무딩
        if smooth_center is None:
            smooth_center = (raw_cx, raw_cy)
        else:
            scx, scy = smooth_center
            smooth_center = (int(ema_alpha * raw_cx + (1 - ema_alpha) * scx),
                             int(ema_alpha * raw_cy + (1 - ema_alpha) * scy))
        # 이상치 제거
        if prev_center is not None:
            dx = smooth_center[0] - prev_center[0]
            dy = smooth_center[1] - prev_center[1]
            if np.hypot(dx, dy) > jump_thresh:
                smooth_center = prev_center
        prev_center = smooth_center
        cx, cy = smooth_center
        # 클램핑
        half = crop_size // 2
        x0 = max(min(cx - half, w - crop_size), 0)
        y0 = max(min(cy - half, h - crop_size), 0)
        crop = frame[y0:y0+crop_size, x0:x0+crop_size]
        # 패딩
        ch, cw = crop.shape[:2]
        if ch != crop_size or cw != crop_size:
            crop = cv2.copyMakeBorder(crop, 0, crop_size-ch, 0, crop_size-cw, cv2.BORDER_CONSTANT, value=[0,0,0])
        writer.write(crop)
        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    print(f"크롭된 영상 저장 완료: {output_path}")

# --- 메인 파이프라인 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="얼굴 요약·크롭 비디오 생성")
    parser.add_argument("video_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cut_threshold", type=float, default=2.0)
    parser.add_argument("--crop_size", type=int, default=250)
    parser.add_argument("--jump_thresh", type=float, default=50.0)
    parser.add_argument("--ema_alpha", type=float, default=0.2)
    parser.add_argument("--reinit_interval", type=int, default=30)
    args = parser.parse_args()

    os.makedirs("temp_proc", exist_ok=True)
    condensed = os.path.join("temp_proc", f"condensed_{os.path.basename(args.video_path)}")
    cropped_silent = os.path.join("temp_proc", f"cropped_silent_{os.path.basename(args.video_path)}")

    start = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tl, fps = analyze_video_faces(args.video_path, args.batch_size, device)
    if create_condensed_video(args.video_path, condensed, tl, fps, args.cut_threshold):
        track_and_crop_video(condensed, cropped_silent, args.batch_size, args.crop_size,
                             args.jump_thresh, args.ema_alpha, args.reinit_interval)
        print("## 4단계: 오디오 합치기")
        vc = VideoFileClip(cropped_silent)
        ac = AudioFileClip(condensed)
        final = vc.with_audio(ac)
        final.write_videofile(args.output_path, codec='libx264', audio_codec='aac')
        vc.close(); ac.close()
    shutil.rmtree("temp_proc")
    print(f"전체 완료 출력:{os.path.abspath(args.output_path)} 소요:{time.time()-start:.2f}s")
