"""
Face-Tracking-App 설정 파일
"""
import torch

# 디바이스 설정
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 배치 처리 설정
BATCH_SIZE_ANALYZE = 256  # GPU 메모리 안전을 위해 축소
BATCH_SIZE_ID_TIMELINE = 128  # GPU 메모리 안전을 위해 축소

# 임베딩 관리자 설정
EMBEDDING_MAX_SIZE = 15  # 원래 설정으로 복구
EMBEDDING_TTL_SECONDS = 30  # 원래 설정으로 복구

# 트래커 모니터 설정
TRACKER_MONITOR_WINDOW_SIZE = 20

# 유사도 계산 설정
SIMILARITY_THRESHOLD = 0.6  # 원래 설정으로 복구

# L2 정규화 관련 설정
L2_NORMALIZATION_ENABLED = True
DUAL_MODE_SIMILARITY_THRESHOLD = 0.70  # DUAL 모드 전용 (더 엄격)
SINGLE_MODE_SIMILARITY_THRESHOLD = 0.60  # SINGLE 모드 기존값 유지

# L2 정규화 디버깅 설정
L2_NORM_DEBUG_MODE = False  # 상세 로그 출력 여부
SIMILARITY_COMPARISON_LOG = False  # 기존/신규 방식 비교 로그

# 비디오 처리 설정
CROP_SIZE = 250
JUMP_THRESHOLD = 25.0
EMA_ALPHA = 0.2
BASE_REINIT_INTERVAL = 40

# 바운딩 박스 품질 평가 설정
MIN_BBOX_SIZE = 30
MAX_ASPECT_RATIO = 2.0
MIN_ASPECT_RATIO = 0.5

# 트리밍 설정
FACE_DETECTION_THRESHOLD_FRAMES = 90
CUT_THRESHOLD_SECONDS = 5.0

# 세그먼트 설정
SEGMENT_LENGTH_SECONDS = 10

# 트래킹 모드 설정 (모드별 config에서 오버라이드됨)
TRACKING_MODE = "most_frequent"  # 기본값, single_config.py 또는 dual_config.py에서 설정

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16000
AUDIO_FRAME_DURATION = 30

# 경로 설정
INPUT_DIR = "./videos/input"
OUTPUT_ROOT = "./videos/output"
TEMP_ROOT = "temp_proc"

# 지원 비디오 확장자
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')

# 비디오 인코딩 설정
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'
FOURCC = 'mp4v'