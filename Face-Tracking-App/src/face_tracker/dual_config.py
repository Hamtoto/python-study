"""
DUAL 모드 전용 설정
2인 화자 모드에서 사용되는 설정들
"""

# 트래킹 모드 설정
TRACKING_MODE = "dual"  # 2인 화자 분리 모드

# 출력 구조 설정
OUTPUT_ROOT = "./videos/output"  # person_1/person_2 구조로 저장

# DUAL 모드 전용 설정
DUAL_MODE_ENABLED = True
DUAL_MIN_SEGMENTS_PER_SPEAKER = 1  # 화자별 최소 세그먼트 수
DUAL_PERSON_FOLDER_PREFIX = "person_"  # person_1, person_2 형태로 저장