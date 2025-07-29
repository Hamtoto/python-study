"""
유사도 계산 유틸리티 함수들
"""
import torch.nn.functional as F
from config import SIMILARITY_THRESHOLD


def find_matching_id_early_exit(emb, all_embs, threshold=SIMILARITY_THRESHOLD):
    """
    조기 종료를 사용한 최적화된 ID 매칭
    임계값 이상의 첫 번째 매칭을 찾으면 즉시 반환
    
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
            
        # 최고 유사도 추적 (임계값 미만인 경우를 위해)
        if sim > best_sim:
            best_sim = sim
            best_id = tid
    
    # 모든 유사도가 임계값 미만이면 None 반환
    return None


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