"""
Face-Tracking-App 설정 파일
"""
import torch

# 임베딩 차원 설정
FACE_EMBEDDING_DIM = 1024  # 512 → 1024로 확장 (더 정확한 인식)
USE_HIGH_DIM_EMBEDDING = True  # 고차원 임베딩 사용 여부

# 디바이스 설정
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 배치 처리 설정 (고차원 임베딩 고려)
BATCH_SIZE_ANALYZE = 128 if USE_HIGH_DIM_EMBEDDING else 256  # 1024차원: 배치 절반
BATCH_SIZE_ID_TIMELINE = 64 if USE_HIGH_DIM_EMBEDDING else 128  # 메모리 절약

# 임베딩 관리자 설정
EMBEDDING_MAX_SIZE = 15  # 원래 설정으로 복구
EMBEDDING_TTL_SECONDS = 30  # 원래 설정으로 복구

# 트래커 모니터 설정
TRACKER_MONITOR_WINDOW_SIZE = 20

# 유사도 계산 설정 (실제 비디오 데이터 기준 최적화)
SIMILARITY_THRESHOLD = 0.75  # 실제 데이터 분석 결과 기반

# L2 정규화 관련 설정
L2_NORMALIZATION_ENABLED = True
DUAL_MODE_SIMILARITY_THRESHOLD = 0.80  # DUAL 모드 전용 (더 엄격한 구분)
SINGLE_MODE_SIMILARITY_THRESHOLD = 0.70  # SINGLE 모드 (약간 관대)

# L2 정규화 디버깅 설정
L2_NORM_DEBUG_MODE = False  # 상세 로그 출력 여부
SIMILARITY_COMPARISON_LOG = False  # 기존/신규 방식 비교 로그

# 동적 임계값 최적화 설정
ENABLE_ADAPTIVE_THRESHOLD = True  # 동적 임계값 계산 활성화
ADAPTIVE_THRESHOLD_MIN_CONFIDENCE = "medium"  # 최소 신뢰도 레벨 (high/medium/low)
ADAPTIVE_THRESHOLD_MIN_IMPROVEMENT = 2.0  # 최소 개선율 (%)
ADAPTIVE_THRESHOLD_SAFETY_RANGE = (0.6, 0.95)  # 안전 임계값 범위
ADAPTIVE_THRESHOLD_MIN_SAMPLES = 5  # 최소 샘플 수 (같은 사람/다른 사람 각각)

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