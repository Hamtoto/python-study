"""
DUAL_SPLIT 모드 전용 설정
화면 분할 2인 추적 모드에서 사용되는 설정들
"""

# 트래킹 모드 설정
TRACKING_MODE = "dual_split"  # 화면 분할 2인 추적 모드

# 출력 구조 설정
OUTPUT_ROOT = "./videos/output"  # 분할 영상 저장

# DUAL_SPLIT 모드 전용 설정
DUAL_SPLIT_MODE_ENABLED = True

# 화면 분할 설정
SPLIT_WIDTH = 960           # 각 person 영역 너비 (1920/2)
SPLIT_HEIGHT = 1080         # 각 person 영역 높이
FULL_WIDTH = 1920          # 전체 화면 너비
FULL_HEIGHT = 1080         # 전체 화면 높이

# Person 할당 설정
PERSON_ASSIGNMENT_STRATEGY = "hybrid"  # hybrid, frequency, spatial
SPATIAL_SPLIT_THRESHOLD = 960          # 좌/우 구분 기준 (x 좌표)

# 처리 조건 설정
MIN_FACES_PER_FRAME = 1     # 최소 얼굴 수 (1명 이상 있으면 처리)
SKIP_NO_FACE_FRAMES = False  # dual_split에서는 모든 프레임 처리로 연속성 보장 = True  # 얼굴이 없는 프레임 스킵

# 얼굴 중앙 정렬 설정
FACE_CENTER_ALIGNMENT = True        # 얼굴 중앙 정렬 활성화
FACE_CROP_SIZE = 400               # 얼굴 크롭 기본 크기
FACE_MARGIN_RATIO = 0.3            # 얼굴 여백 비율

# 트래킹 안정화 설정 (기존 시스템과 동일)
TRACKING_SMOOTHING = True           # 트래킹 스무딩 활성화
TRACKING_CLAMP_ENABLED = True       # 트래킹 클램핑 활성화
TRACKING_SKIP_JUMPS = True          # 급격한 움직임 스킵

# 트래킹 임계값 설정
TRACKING_POSITION_THRESHOLD = 100   # 위치 변화 임계값 (픽셀)
TRACKING_SIMILARITY_THRESHOLD = 0.7 # 임베딩 유사도 임계값
TRACKING_MAX_LOST_FRAMES = 10       # 최대 트래킹 실패 허용 프레임 수

# 영상 출력 설정
SPLIT_VIDEO_CODEC = 'libx264'       # 분할 영상 코덱
SPLIT_AUDIO_CODEC = 'aac'           # 오디오 코덱
SPLIT_SEGMENT_LENGTH = 10           # 세그먼트 길이 (초)

# 디버깅 설정
DEBUG_SPLIT_MODE = False            # 분할 모드 디버깅
DEBUG_PERSON_ASSIGNMENT = False     # Person 할당 디버깅