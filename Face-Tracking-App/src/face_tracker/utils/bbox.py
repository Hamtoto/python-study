"""
바운딩 박스 처리 유틸리티 함수들
"""
from src.face_tracker.config import MIN_BBOX_SIZE, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO


def evaluate_bbox_quality(bbox, frame_shape):
    """
    바운딩 박스 품질 평가
    
    Args:
        bbox: (x, y, w, h) 형태의 바운딩 박스
        frame_shape: 프레임 크기 (height, width, channels)
    
    Returns:
        bool: 품질이 좋으면 True, 나쁘면 False
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]
    
    # 1. 크기 체크
    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        return False
        
    # 2. 경계 체크  
    if x < 0 or y < 0 or x+w > frame_w or y+h > frame_h:
        return False
        
    # 3. 종횡비 체크
    aspect_ratio = w / h
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
        
    return True