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
import wave
import webrtcvad
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
from torchvision import transforms

def get_voice_segments(video_path: str, sample_rate=16000, frame_duration=30):
    wav_path = video_path.replace('.mp4', '_audio.wav')
    AudioFileClip(video_path).write_audiofile(wav_path, fps=sample_rate, nbytes=2, codec='pcm_s16le')
    vad = webrtcvad.Vad(3)
    wf = wave.open(wav_path, 'rb')
    frames = wf.readframes(wf.getnframes())
    segment_length = int(sample_rate * frame_duration / 1000) * 2
    voice_times = []
    is_speech = False
    t = 0.0
    for offset in range(0, len(frames), segment_length):
        chunk = frames[offset:offset+segment_length]
        speech = vad.is_speech(chunk, sample_rate)
        if speech and not is_speech:
            start = t; is_speech = True
        if not speech and is_speech:
            voice_times.append((start, t)); is_speech = False
        t += frame_duration / 1000
    if is_speech:
        voice_times.append((start, t))
    wf.close()
    return voice_times

def generate_id_timeline(video_path: str, device: torch.device):
    """
    Generate a list of track_ids (or None) for each frame of the input video.
    """
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    prev_embs = {}
    next_id = 1

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="[ID 타임라인 생성]")
    fps = cap.get(cv2.CAP_PROP_FPS)
    id_timeline = []

    while True:
        ret, frame = cap.read()
        pbar.update(1)
        if not ret:
            break
        face_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect([face_img])
        track_id = None
        if boxes is not None and boxes[0] is not None and len(boxes[0]) > 0:
            x1, y1, x2, y2 = boxes[0][0]
            face_crop_pil = face_img.crop((x1, y1, x2, y2)).resize((160, 160))
            emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
            sims = {tid: F.cosine_similarity(emb, e, dim=1).item() for tid, e in prev_embs.items()}
            if sims and max(sims.values()) > 0.8:
                track_id = max(sims, key=sims.get)
            else:
                track_id = next_id
                next_id += 1
            prev_embs[track_id] = emb
        id_timeline.append(track_id)
    pbar.close()
    cap.release()
    return id_timeline, fps


def trim_by_face_timeline(input_path: str, id_timeline: list, fps: float, threshold_frames: int, output_path: str):
    """
    Trim out segments where face is not detected for threshold_frames or more,
    keeping audio and video in sync, and write to output_path.
    """
    # Build keep mask
    n = len(id_timeline)
    keep = [True] * n
    i = 0
    while i < n:
        if id_timeline[i] is None:
            j = i
            while j < n and id_timeline[j] is None:
                j += 1
            if j - i >= threshold_frames:
                for k in range(i, j):
                    keep[k] = False
            i = j
        else:
            i += 1

    # Build segments to keep
    segments = []
    i = 0
    while i < n:
        if keep[i]:
            start = i
            while i < n and keep[i]:
                i += 1
            segments.append((start / fps, i / fps))
        else:
            i += 1

    if not segments:
        print("## 경고: 유지할 구간이 없습니다.")
        return False

    # Perform trimming
    clip = VideoFileClip(input_path)
    duration = clip.duration
    subclips = []
    for s, e in segments:
        end_clamped = min(e, duration)
        subclips.append(clip.subclipped(s, end_clamped))
    final = concatenate_videoclips(subclips)
    final.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    clip.close()
    final.close()
    return True

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
def track_and_crop_video(
    video_path : str,
    output_path : str,
    batch_size : int = 16,
    crop_size : int = 250,
    jump_thresh : float = 25.0,
    ema_alpha : float = 0.2,
    reinit_interval : int = 40
    ) :
    print("## 3단계: 크롭 시작 (스무딩·이상치·하이브리드·클램핑)")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    # initialize embedding-based ID tracker
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    prev_embs = {}
    next_id = 1

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
    track_id = None

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
                # ── embedding and ID assignment on reinit frames
                face_crop_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\
                                  .crop((x1, y1, x2, y2)).resize((160, 160))
                emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
                sims = {tid: F.cosine_similarity(emb, e, dim=1).item()
                        for tid, e in prev_embs.items()}
                if sims and max(sims.values()) > 0.8:
                    track_id = max(sims, key=sims.get)
                else:
                    track_id = next_id; next_id += 1
                prev_embs[track_id] = emb
            else:
                raw_cx, raw_cy = w // 2, h // 2
                track_id = None
        else:
            ok, bb = tracker.update(frame)
            if ok:
                x, y, tw, th = bb
                raw_cx = int(x + tw / 2)
                raw_cy = int(y + th / 2)
                # maintain last track_id
            else:
                boxes_list, _ = mtcnn.detect([Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))])
                faces = boxes_list[0]
                if faces is not None and len(faces) > 0:
                    x1, y1, x2, y2 = faces[0]
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    raw_cx = int((x1 + x2) / 2)
                    raw_cy = int((y1 + y2) / 2)
                    # after fallback detection, reassign ID as above
                    face_crop_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\
                                      .crop((x1, y1, x2, y2)).resize((160, 160))
                    emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
                    sims = {tid: F.cosine_similarity(emb, e, dim=1).item()
                            for tid, e in prev_embs.items()}
                    if sims and max(sims.values()) > 0.8:
                        track_id = max(sims, key=sims.get)
                    else:
                        track_id = next_id; next_id += 1
                    prev_embs[track_id] = emb
                else:
                    raw_cx, raw_cy = w // 2, h // 2
                    track_id = None
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

        # write every frame
        writer.write(crop)

        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    print(f"크롭된 영상 저장 완료: {output_path}")

# --- 비디오 분할 함수 ---
def slice_video(input_path: str, output_folder: str, segment_length: int = 10):
    print(f"Slicing {input_path} into {segment_length}-second segments in {output_folder}")
    clip = VideoFileClip(input_path)
    duration = clip.duration
    for start in range(0, int(duration), segment_length):
        end = min(start + segment_length, duration)
        if (end - start) < segment_length:
            continue
        segment = clip.subclipped(start, end)
        segment_filename = f"segment_{start}_{int(end)}.mp4"
        output_path = os.path.join(output_folder, segment_filename)
        segment.write_videofile(output_path, codec='libx264', audio_codec='aac')
    clip.close()


if __name__ == "__main__":
    input_dir = "./videos/input"
    output_root = "./videos/output"
    temp_root = "temp_proc"
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.mp4', '.mov', '.avi')):
            continue
        basename = os.path.splitext(fname)[0]
        temp_dir = os.path.join(temp_root, basename)
        os.makedirs(temp_dir, exist_ok=True)

        input_path = os.path.join(input_dir, fname)
        condensed = os.path.join(temp_dir, f"condensed_{fname}")
        cropped_silent = os.path.join(temp_dir, f"cropped_silent_{fname}")
        final_output = os.path.join(output_root, f"{basename}.mp4")

        print(f"## 처리 시작: {fname}")
        start_time = time.time()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 0) 오디오 VAD로 말하는 구간 추출
        voice_timeline = get_voice_segments(input_path)
        # 1) 얼굴 감지 타임라인
        timeline, fps = analyze_video_faces(input_path, batch_size=16, device=device)
        # 2) 요약본 생성
        if create_condensed_video(input_path, condensed, timeline, fps):
            print("요약본 완료")
            print(f"## 디버그: condensed 파일 경로 = {condensed}")
            print(f"## 디버그: 얼굴 트리밍 전 timeline 길이 = {len(timeline)}, fps = {fps}")
            print("## 2단계 완료, 이제 트리밍 및 세그먼트 분할을 시작합니다.")
            # 2a) ID 타임라인 생성 및 타겟 인물 자동 선택 후 30프레임 이상 미검출 구간 트리밍
            id_timeline, fps2 = generate_id_timeline(condensed, device)
            print(f"## 디버그: trimming 전 condensed 영상 프레임 수 예측 = {len(id_timeline)}")
            # 자동 타겟 ID: 첫 등장 인물
            target_id = next((tid for tid in id_timeline if tid is not None), None)
            # 타겟 아닌 프레임은 None으로 표시
            if target_id is not None:
                id_timeline_bool = [tid if tid == target_id else None for tid in id_timeline]
            else:
                id_timeline_bool = id_timeline
            trimmed = os.path.join(temp_dir, f"trimmed_{fname}")
            if trim_by_face_timeline(condensed, id_timeline_bool, fps2, threshold_frames=30, output_path=trimmed):
                source_for_crop = trimmed
            else:
                source_for_crop = condensed
            # 3) 연속된 트리밍된 영상 10초 세그먼트로 분할
            segment_temp_folder = os.path.join(temp_dir, "segments")
            os.makedirs(segment_temp_folder, exist_ok=True)
            slice_video(source_for_crop, segment_temp_folder, segment_length=10)

            # 4) 각 세그먼트별 얼굴 크롭 및 오디오 동기 병합
            final_segment_folder = os.path.join(output_root, basename)
            os.makedirs(final_segment_folder, exist_ok=True)
            for seg_fname in os.listdir(segment_temp_folder):
                if not seg_fname.lower().endswith(".mp4"):
                    continue
                seg_input = os.path.join(segment_temp_folder, seg_fname)
                seg_cropped = os.path.join(temp_dir, f"crop_{seg_fname}")
                track_and_crop_video(seg_input, seg_cropped)
                vc = VideoFileClip(seg_cropped)
                ac = AudioFileClip(seg_input)
                final_seg = vc.with_audio(ac)
                output_seg_path = os.path.join(final_segment_folder, seg_fname)
                final_seg.write_videofile(output_seg_path, codec='libx264', audio_codec='aac')
                vc.close(); ac.close()
        else:
            print(f"요약본 생성 실패: {fname}")

        elapsed = time.time() - start_time
        print(f"{fname} 처리시간 : {int(elapsed)}초")
        shutil.rmtree(temp_dir)
        print(f"## 완료: {fname}")

    print("모든 비디오 처리 및 세그먼트별 얼굴 크롭/동기화 완료")

# --------------------------------------------------
# Face-Tracking-App: 비디오에서 얼굴을 추적하고 크롭하여 요약본을 생성하는 스크립트
# python track_and_draw_video.py [input_file_path] [output_file_path]
# --------------------------------------------------