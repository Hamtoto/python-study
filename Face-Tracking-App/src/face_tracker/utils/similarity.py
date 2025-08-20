"""
유사도 계산 유틸리티 함수들
"""
import torch
import torch.nn.functional as F
from src.face_tracker.config import SIMILARITY_THRESHOLD



def find_matching_id_with_best_fallback(emb, all_embs, threshold=SIMILARITY_THRESHOLD):
    """
    개선된 유사도 매칭: 임계값 이상 우선, 없으면 최고 유사도 선택
    
    Args:
        emb: 현재 얼굴 임베딩
        all_embs: 기존 임베딩들 딕셔너리 {track_id: embedding}
        threshold: 유사도 임계값
    
    Returns:
        int or None: 매칭된 track_id 또는 None
    """
    if not all_embs:
        return None
    
    best_id = None
    best_sim = 0.0
    
    for tid, existing_emb in all_embs.items():
        sim = F.cosine_similarity(emb, existing_emb, dim=1).item()
        
        # 임계값 이상이면 즉시 반환 (조기 종료)
        if sim > threshold:
            return tid
            
        # 최고 유사도 추적
        if sim > best_sim:
            best_sim = sim
            best_id = tid
    
    # 원래 로직으로 복구: 임계값 이상만 같은 사람으로 인식
    return best_id if best_sim > threshold else None


def cosine_similarity_l2_normalized(emb1, emb2):
    """
    L2 정규화된 두 임베딩 간의 코사인 유사도 계산
    
    Args:
        emb1: 첫 번째 임베딩 텐서
        emb2: 두 번째 임베딩 텐서
        
    Returns:
        float: 정규화된 코사인 유사도 (0~1)
    """
    # L2 정규화 적용 - 단위 벡터로 변환
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    
    # 정규화된 벡터간 코사인 유사도 계산
    return F.cosine_similarity(emb1_norm, emb2_norm, dim=1).item()


def find_matching_id_with_l2_normalization(emb, all_embs, threshold=SIMILARITY_THRESHOLD):
    """
    L2 정규화 적용된 개선된 유사도 매칭
    
    Args:
        emb: 현재 얼굴 임베딩 (torch.Tensor)
        all_embs: 기존 임베딩들 딕셔너리 {track_id: embedding}
        threshold: 유사도 임계값 (기본값: config에서 가져옴)
    
    Returns:
        int or None: 매칭된 track_id 또는 None
    """
    if not all_embs:
        return None
    
    # L2 정규화 적용 - 단위 벡터로 변환
    emb_normalized = F.normalize(emb, p=2, dim=1)
    
    best_id = None
    best_sim = 0.0
    threshold_matches = []
    
    for tid, existing_emb in all_embs.items():
        # 기존 임베딩도 L2 정규화
        existing_emb_normalized = F.normalize(existing_emb, p=2, dim=1)
        
        # 정규화된 벡터간 코사인 유사도 (= 내적)
        sim = F.cosine_similarity(emb_normalized, existing_emb_normalized, dim=1).item()
        
        # 임계값 이상인 모든 후보 수집
        if sim > threshold:
            threshold_matches.append((tid, sim))
            
        # 최고 유사도 추적
        if sim > best_sim:
            best_sim = sim
            best_id = tid
    
    # 임계값 이상 후보가 있으면 최고 유사도 반환
    if threshold_matches:
        threshold_matches.sort(key=lambda x: x[1], reverse=True)
        return threshold_matches[0][0]
    
    # 없으면 None 반환 (엄격한 매칭)
    return None


def find_matching_id_with_best_fallback_enhanced(emb, all_embs, threshold=SIMILARITY_THRESHOLD, use_l2_norm=True):
    """
    기존 함수에 L2 정규화 옵션 추가 (하위 호환성 유지)
    
    Args:
        emb: 현재 얼굴 임베딩
        all_embs: 기존 임베딩들 딕셔너리
        threshold: 유사도 임계값
        use_l2_norm: L2 정규화 사용 여부
        
    Returns:
        int or None: 매칭된 track_id 또는 None
    """
    if use_l2_norm:
        return find_matching_id_with_l2_normalization(emb, all_embs, threshold)
    else:
        # 기존 로직 사용 (하위 호환성)
        return find_matching_id_with_best_fallback(emb, all_embs, threshold)


def calculate_face_similarity(emb1, emb2, use_l2_norm=True):
    """
    얼굴 임베딩 간 유사도 계산 (통합 인터페이스)
    
    Args:
        emb1, emb2: 비교할 임베딩들
        use_l2_norm: L2 정규화 사용 여부
        
    Returns:
        float: 유사도 점수 (0~1)
    """
    if use_l2_norm:
        return cosine_similarity_l2_normalized(emb1, emb2)
    else:
        return F.cosine_similarity(emb1, emb2, dim=1).item()