"""
타겟 선택 모듈 - 다양한 모드를 지원하는 확장 가능한 시스템
"""
import torch
import torch.nn.functional as F
from collections import Counter
from typing import List, Optional, Dict, Any
from src.face_tracker.config import TRACKING_MODE
from src.face_tracker.utils.logging import logger


class TargetSelector:
    """트래킹 타겟 선택을 위한 확장 가능한 클래스"""
    
    @staticmethod
    def select_target(id_timeline: List[Optional[int]], mode: str = TRACKING_MODE, embeddings: Dict = None) -> Optional[int]:
        """
        지정된 모드에 따라 타겟 ID를 선택
        
        Args:
            id_timeline: 프레임별 ID 리스트
            mode: 트래킹 모드 ("first_person" | "most_frequent")
            embeddings: ID별 임베딩 딕셔너리
            
        Returns:
            선택된 타겟 ID (None이면 타겟 없음)
        """
        if mode == "first_person":
            return TargetSelector._select_first_person(id_timeline)
        elif mode == "most_frequent":
            # ID 병합 후 most_frequent 계산
            if embeddings:
                merged_timeline = TargetSelector._merge_similar_ids(id_timeline, embeddings)
                return TargetSelector._select_most_frequent(merged_timeline)
            else:
                return TargetSelector._select_most_frequent(id_timeline)
        else:
            raise ValueError(f"지원하지 않는 트래킹 모드: {mode}")
    
    @staticmethod
    def _select_first_person(id_timeline: List[Optional[int]]) -> Optional[int]:
        """첫 번째로 등장하는 인물 선택"""
        return next((tid for tid in id_timeline if tid is not None), None)
    
    @staticmethod
    def _select_most_frequent(id_timeline: List[Optional[int]]) -> Optional[int]:
        """가장 많이 등장하는 인물 선택"""
        if not id_timeline:
            return None
            
        # None이 아닌 ID들만 카운트
        valid_ids = [tid for tid in id_timeline if tid is not None]
        if not valid_ids:
            return None
            
        # 가장 빈번한 ID 반환
        counter = Counter(valid_ids)
        most_common_id, count = counter.most_common(1)[0]
        return most_common_id
    
    @staticmethod
    def get_target_stats(id_timeline: List[Optional[int]], target_id: int) -> Dict[str, Any]:
        """타겟 ID에 대한 통계 정보 반환"""
        total_frames = len(id_timeline)
        target_frames = sum(1 for tid in id_timeline if tid == target_id)
        
        return {
            "target_id": target_id,
            "total_frames": total_frames,
            "target_frames": target_frames,
            "coverage_ratio": target_frames / total_frames if total_frames > 0 else 0
        }
    
    @staticmethod
    def _merge_similar_ids(id_timeline: List[Optional[int]], embeddings: Dict, similarity_threshold: float = 0.75) -> List[Optional[int]]:
        """
        유사한 ID들을 병합하여 같은 사람을 하나의 ID로 통합
        
        Args:
            id_timeline: 원본 ID 타임라인
            embeddings: ID별 임베딩 딕셔너리
            similarity_threshold: ID 병합 임계값
            
        Returns:
            병합된 ID 타임라인
        """
        if not embeddings:
            return id_timeline
            
        # 유니크한 ID 목록 추출
        unique_ids = list(set(tid for tid in id_timeline if tid is not None))
        if len(unique_ids) <= 1:
            return id_timeline
            
        # ID 병합 맵 생성
        merge_map = {}
        for i, id1 in enumerate(unique_ids):
            if id1 in merge_map:
                continue
                
            merge_map[id1] = id1  # 자기 자신으로 초기화
            
            for j, id2 in enumerate(unique_ids[i+1:], i+1):
                if id2 in merge_map:
                    continue
                    
                # 임베딩 유사도 계산
                if id1 in embeddings and id2 in embeddings:
                    emb1 = embeddings[id1]
                    emb2 = embeddings[id2]
                    
                    similarity = F.cosine_similarity(emb1, emb2, dim=1).item()
                    
                    if similarity >= similarity_threshold:
                        merge_map[id2] = id1  # id2를 id1로 병합
        
        # 타임라인에 병합 맵 적용
        merged_timeline = []
        for tid in id_timeline:
            if tid is None:
                merged_timeline.append(None)
            else:
                merged_timeline.append(merge_map.get(tid, tid))
                
        return merged_timeline