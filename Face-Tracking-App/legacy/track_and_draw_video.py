import cv2
import torch
import os
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
from collections import OrderedDict

class TrackerMonitor:
    """트래커 성능 모니터링 클래스"""
    def __init__(self, window_size=20):
        self.success_history = []
        self.window_size = window_size
        
    def update(self, success):
        self.success_history.append(success)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
            
    def get_success_rate(self):
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

class SmartEmbeddingManager:
    """LRU 캐시 + 시간 기반 정리를 조합한 임베딩 관리자"""
    def __init__(self, max_size=15, ttl_seconds=30):
        self.embeddings = OrderedDict()
        self.last_used = {}
        self.access_count = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
    def cleanup_old_embeddings(self):
        """TTL 기반 오래된 임베딩 정리"""
        current_time = time.time()
        expired_ids = []
        
        for track_id, last_time in self.last_used.items():
            if current_time - last_time > self.ttl_seconds:
                expired_ids.append(track_id)
        
        for track_id in expired_ids:
            del self.embeddings[track_id]
            del self.last_used[track_id]
            del self.access_count[track_id]
            print(f"  만료된 ID 정리: {track_id}")
    
    def get_embedding(self, track_id):
        """임베딩 조회 (LRU 업데이트)"""
        if track_id in self.embeddings:
            # 최근 사용으로 이동
            self.embeddings.move_to_end(track_id)
            self.last_used[track_id] = time.time()
            self.access_count[track_id] = self.access_count.get(track_id, 0) + 1
            return self.embeddings[track_id]
        return None
    
    def add_embedding(self, track_id, emb):
        """임베딩 추가 (크기 제한 + TTL 정리)"""
        # 주기적 정리
        self.cleanup_old_embeddings()
        
        if track_id in self.embeddings:
            # 기존 ID 업데이트
            self.embeddings.move_to_end(track_id)
        else:
            # 새 ID 추가
            if len(self.embeddings) >= self.max_size:
                # 가장 오래된 것 제거 (LRU)
                oldest_id = next(iter(self.embeddings))
                del self.embeddings[oldest_id]
                del self.last_used[oldest_id]
                del self.access_count[oldest_id]
                print(f"  LRU 정리: {oldest_id}")
        
        # 새 임베딩 추가
        self.embeddings[track_id] = emb
        self.last_used[track_id] = time.time()
        self.access_count[track_id] = self.access_count.get(track_id, 0) + 1
    
    def get_all_embeddings(self):
        """모든 임베딩 반환 (유사도 계산용)"""
        return dict(self.embeddings)
    
    def get_stats(self):
        """통계 정보 반환"""
        return {
            'count': len(self.embeddings),
            'ids': list(self.embeddings.keys()),
            'access_counts': dict(self.access_count)
        }

def get_adaptive_reinit_interval(success_rate, base_interval=40):
    """성공률에 따른 적응적 재초기화 간격 계산"""
    if success_rate > 0.95:      # 매우 좋음
        return base_interval * 2     # 80프레임
    elif success_rate > 0.8:     # 좋음  
        return base_interval         # 40프레임
    elif success_rate > 0.6:     # 보통
        return base_interval // 2    # 20프레임
    else:                        # 나쁨
        return base_interval // 4    # 10프레임

def evaluate_bbox_quality(bbox, frame_shape):
    """바운딩 박스 품질 평가"""
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]
    
    # 1. 크기 체크
    if w < 30 or h < 30:
        return False
        
    # 2. 경계 체크  
    if x < 0 or y < 0 or x+w > frame_w or y+h > frame_h:
        return False
        
    # 3. 종횡비 체크
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
        
    return True

class ModelManager:
    """모델 싱글톤 매니저 - 모델 재생성 방지로 성능 향상"""
    _instance = None
    _initialized = False
    
    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device=None):
        if not self._initialized:
            if device is None:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.device = device
            print(f"ModelManager 초기화: {device}")
            self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            
            # GPU 메모리 풀 초기화
            self._init_memory_pool()
            self._initialized = True
    
    def _init_memory_pool(self):
        """GPU 메모리 풀 사전 할당"""
        if self.device.type == 'cuda':
            # 배치 텐서 풀 (배치 크기 256 기준)
            self.tensor_pool_256 = torch.zeros(256, 3, 224, 224, device=self.device)
            # 단일 얼굴 임베딩용 텐서 풀
            self.face_tensor_pool = torch.zeros(1, 3, 160, 160, device=self.device)
            # 변환 객체 사전 생성
            self.to_tensor = transforms.ToTensor()
            print(f"GPU 메모리 풀 초기화 완료: {self.device}")
    
    def get_mtcnn(self):
        return self.mtcnn
    
    def get_resnet(self):
        return self.resnet
    
    def get_tensor_pool(self, batch_size=256):
        """사전 할당된 텐서 풀 반환"""
        if self.device.type == 'cuda':
            if batch_size <= 256:
                return self.tensor_pool_256[:batch_size]
            else:
                # 더 큰 배치 사이즈면 동적 할당
                return torch.zeros(batch_size, 3, 224, 224, device=self.device)
        return None
    
    def get_face_tensor_pool(self):
        """얼굴 임베딩용 텐서 풀 반환"""
        if self.device.type == 'cuda':
            return self.face_tensor_pool
        return None
    
    def get_transform(self):
        """사전 생성된 변환 객체 반환"""
        return self.to_tensor
    
    def opencv_to_tensor_batch(self, frames_list):
        """OpenCV 프레임들을 배치 텐서로 직접 변환 (PIL 건너뛰기)"""
        if not frames_list:
            return None
            
        batch_size = len(frames_list)
        if batch_size == 0:
            return None
            
        # OpenCV BGR → RGB 변환과 동시에 numpy 배열로 변환
        rgb_frames = []
        for frame in frames_list:
            if isinstance(frame, Image.Image):
                # PIL 이미지면 numpy로 변환
                rgb_frame = np.array(frame)
            else:
                # OpenCV 프레임이면 BGR → RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)
        
        # numpy 배열 스택
        batch_array = np.stack(rgb_frames, axis=0)
        
        # numpy → tensor 직접 변환 (HWC → CHW, 0-255 → 0-1)
        batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2).float() / 255.0
        
        # GPU로 전송
        if self.device.type == 'cuda':
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
            
        return batch_tensor

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

def generate_id_timeline(video_path: str, device: torch.device, batch_size: int = 128):
    """
    Generate a list of track_ids (or None) for each frame of the input video.
    배치 처리로 최적화된 버전
    """
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()
    resnet = model_manager.get_resnet()
    emb_manager = SmartEmbeddingManager(max_size=15, ttl_seconds=30)
    next_id = 1

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="[ID 타임라인 생성 - 배치 처리]")
    fps = cap.get(cv2.CAP_PROP_FPS)
    id_timeline = []

    frames_buffer = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_buffer.append(frame)
        frame_count += 1
        
        # 배치가 찼거나 마지막 프레임이면 처리
        if len(frames_buffer) == batch_size or frame_count == total_frames:
            # 배치 BGR→RGB 변환 (중복 제거)
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer]
            pil_images = [Image.fromarray(rgb_frame) for rgb_frame in rgb_frames]
            
            # 배치 얼굴 탐지
            boxes_list, _ = mtcnn.detect(pil_images)
            
            # 각 프레임별 ID 할당
            for i, (frame, boxes) in enumerate(zip(frames_buffer, boxes_list)):
                track_id = None
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    # 이미 변환된 PIL 객체 재사용
                    face_img = pil_images[i]
                    face_crop_pil = face_img.crop((x1, y1, x2, y2)).resize((160, 160))
                    
                    # 최적화된 텐서 변환
                    face_tensor = model_manager.get_face_tensor_pool()
                    if face_tensor is not None:
                        face_array = np.array(face_crop_pil)
                        face_tensor[0] = torch.from_numpy(face_array).permute(2, 0, 1).float() / 255.0
                        emb = resnet(face_tensor)
                    else:
                        emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
                    
                    # ID 할당 (최적화된 유사도 계산)
                    all_embs = emb_manager.get_all_embeddings()
                    sims = {tid: F.cosine_similarity(emb, e, dim=1).item() for tid, e in all_embs.items()}
                    if sims and max(sims.values()) > 0.8:
                        track_id = max(sims, key=sims.get)
                    else:
                        track_id = next_id
                        next_id += 1
                    emb_manager.add_embedding(track_id, emb)
                
                id_timeline.append(track_id)
                pbar.update(1)
            
            # 버퍼 초기화
            frames_buffer = []
    
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
    crop_size : int = 250,
    jump_thresh : float = 25.0,
    ema_alpha : float = 0.2,
    reinit_interval : int = 40
    ) :
    print("## 3단계: 크롭 시작 (스무딩·이상치·하이브리드·클램핑)")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_manager = ModelManager(device)
    mtcnn = model_manager.get_mtcnn()

    # initialize embedding-based ID tracker
    resnet = model_manager.get_resnet()
    emb_manager = SmartEmbeddingManager(max_size=15, ttl_seconds=30)
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

    # 동적 재초기화를 위한 변수들
    tracker_monitor = TrackerMonitor(window_size=20)
    frames_since_reinit = 0
    current_reinit_interval = reinit_interval

    pbar = tqdm(total=total_frames, desc="[크롭 처리 - 동적 최적화]")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        frames_since_reinit += 1
        
        # 트래커 업데이트 시도
        tracker_success = False
        if tracker is not None:
            ok, bb = tracker.update(frame)
            if ok:
                bbox_quality = evaluate_bbox_quality(bb, frame.shape)
                tracker_success = bbox_quality
                if tracker_success:
                    x, y, tw, th = bb
                    raw_cx = int(x + tw / 2)
                    raw_cy = int(y + th / 2)
        
        # 성능 모니터링 업데이트
        if tracker is not None:
            tracker_monitor.update(tracker_success)
        
        # 적응적 재초기화 간격 계산
        success_rate = tracker_monitor.get_success_rate()
        current_reinit_interval = get_adaptive_reinit_interval(success_rate, reinit_interval)
        
        # 재초기화 결정
        need_reinit = (
            frames_since_reinit >= current_reinit_interval or
            not tracker_success or
            tracker is None
        )
        
        if need_reinit:
            # BGR→RGB 변환 1번만 실행 (중복 제거)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # MTCNN 얼굴 탐지 실행 (변환된 PIL 객체 재사용)
            boxes_list, _ = mtcnn.detect([pil_image])
            faces = boxes_list[0]
            if faces is not None and len(faces) > 0:
                x1, y1, x2, y2 = faces[0]
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                raw_cx = int((x1 + x2) / 2)
                raw_cy = int((y1 + y2) / 2)
                
                # ResNet 임베딩 계산 (같은 PIL 객체 재사용)
                face_crop_pil = pil_image.crop((x1, y1, x2, y2)).resize((160, 160))
                face_tensor = model_manager.get_face_tensor_pool()
                if face_tensor is not None:
                    face_array = np.array(face_crop_pil)
                    face_tensor[0] = torch.from_numpy(face_array).permute(2, 0, 1).float() / 255.0
                    emb = resnet(face_tensor)
                else:
                    emb = resnet(transforms.ToTensor()(face_crop_pil).unsqueeze(0).to(device))
                
                # ID 할당 (최적화된 유사도 계산)
                all_embs = emb_manager.get_all_embeddings()
                sims = {tid: F.cosine_similarity(emb, e, dim=1).item()
                        for tid, e in all_embs.items()}
                if sims and max(sims.values()) > 0.8:
                    track_id = max(sims, key=sims.get)
                else:
                    track_id = next_id; next_id += 1
                emb_manager.add_embedding(track_id, emb)
                
                # 재초기화 완료
                frames_since_reinit = 0
                stats = emb_manager.get_stats()
                print(f"  재초기화 완료 (성공률: {success_rate:.2f}, 간격: {current_reinit_interval}, ID수: {stats['count']})")
            else:
                raw_cx, raw_cy = w // 2, h // 2
                track_id = None
                frames_since_reinit = 0
        # 트래커 업데이트는 이미 위에서 처리됨
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
        timeline, fps = analyze_video_faces(input_path, batch_size=256, device=device)
        # 2) 요약본 생성
        if create_condensed_video(input_path, condensed, timeline, fps):
            print("요약본 완료")
            print(f"## 디버그: condensed 파일 경로 = {condensed}")
            print(f"## 디버그: 얼굴 트리밍 전 timeline 길이 = {len(timeline)}, fps = {fps}")
            print("## 2단계 완료, 이제 트리밍 및 세그먼트 분할을 시작합니다.")
            # 2a) ID 타임라인 생성 및 타겟 인물 자동 선택 후 30프레임 이상 미검출 구간 트리밍
            id_timeline, fps2 = generate_id_timeline(condensed, device, batch_size=128)
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

            # 4) 각 세그먼트별 얼굴 크롭 및 오디오 동기 병합 (순차 처리로 복원)
            final_segment_folder = os.path.join(output_root, basename)
            os.makedirs(final_segment_folder, exist_ok=True)
            
            segment_files = [f for f in os.listdir(segment_temp_folder) 
                           if f.lower().endswith(".mp4")]
            segment_files.sort()  # 순서 보장
            
            print(f"## 세그먼트 순차 처리 시작: {len(segment_files)}개 파일")
            
            for seg_fname in segment_files:
                seg_input = os.path.join(segment_temp_folder, seg_fname)
                seg_cropped = os.path.join(temp_dir, f"crop_{seg_fname}")
                
                print(f"  처리 중: {seg_fname}")
                track_and_crop_video(seg_input, seg_cropped)
                
                vc = VideoFileClip(seg_cropped)
                ac = AudioFileClip(seg_input)
                final_seg = vc.with_audio(ac)
                
                output_seg_path = os.path.join(final_segment_folder, seg_fname)
                final_seg.write_videofile(output_seg_path, codec='libx264', audio_codec='aac')
                
                # 즉시 정리
                vc.close()
                ac.close()
                final_seg.close()
                if os.path.exists(seg_cropped):
                    os.remove(seg_cropped)
                    
            print(f"## 세그먼트 순차 처리 완료")
        else:
            print(f"요약본 생성 실패: {fname}")

        elapsed = time.time() - start_time
        print(f"{fname} 처리시간 : {int(elapsed)}초")
        shutil.rmtree(temp_dir)
        print(f"## 완료: {fname}")

    print("모든 비디오 처리 및 세그먼트별 얼굴 크롭/동기화 완료")
