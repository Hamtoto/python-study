# -*- coding: utf-8 -*-
"""
SmartEmbeddingManager 클래스
LRU 캐시 + 시간 기반 정리를 조합한 임베딩 관리자
"""
import time
from collections import OrderedDict
from config import EMBEDDING_MAX_SIZE, EMBEDDING_TTL_SECONDS


class SmartEmbeddingManager:
    """LRU 캐시 + 시간 기반 정리를 조합한 임베딩 관리자"""
    
    def __init__(self, max_size=EMBEDDING_MAX_SIZE, ttl_seconds=EMBEDDING_TTL_SECONDS):
        self.embeddings = OrderedDict() # {track_id: (embedding, timestamp)}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.last_cleaned_timestamp = time.time()
        
    def _remove_old_embeddings(self):
        """TTL 기반 오래된 임베딩 정리 (내부용)"""
        current_time = time.time()
        expired_ids = []
        
        for track_id, (emb, timestamp) in list(self.embeddings.items()): # items()를 복사하여 순회
            if current_time - timestamp > self.ttl_seconds:
                expired_ids.append(track_id)
        
        for track_id in expired_ids:
            del self.embeddings[track_id]
            # print(f"  만료된 ID 정리: {track_id}") # 디버그용
        self.last_cleaned_timestamp = current_time

    def _remove_lru_embedding(self):
        """LRU 기반 가장 오래된 임베딩 정리 (내부용)"""
        if self.embeddings:
            oldest_id = next(iter(self.embeddings)) # OrderedDict의 첫 번째 항목이 가장 오래된 것
            del self.embeddings[oldest_id]
            # print(f"  LRU 정리: {oldest_id}") # 디버그용

    def get_embedding(self, track_id):
        """임베딩 조회 (LRU 업데이트)"""
        if track_id in self.embeddings:
            emb, timestamp = self.embeddings.pop(track_id) # 제거 후 다시 추가하여 LRU 업데이트
            self.embeddings[track_id] = (emb, time.time())
            return emb
        return None
    
    def add_embedding(self, track_id, emb):
        """임베딩 추가 (크기 제한 + TTL 정리)"""
        # 주기적 정리 (너무 자주 호출되지 않도록)
        if time.time() - self.last_cleaned_timestamp > self.ttl_seconds / 2:
            self._remove_old_embeddings()
        
        if track_id in self.embeddings:
            # 기존 ID 업데이트: 제거 후 다시 추가하여 LRU 업데이트
            del self.embeddings[track_id]
        elif len(self.embeddings) >= self.max_size:
            # 새 ID 추가 시 크기 제한 초과하면 LRU 제거
            self._remove_lru_embedding()
        
        self.embeddings[track_id] = (emb, time.time())
    
    def get_all_embeddings(self):
        """모든 임베딩 반환 (유사도 계산용) - 임베딩 텐서만 포함"""
        return {track_id: emb for track_id, (emb, timestamp) in self.embeddings.items()}
    
    def get_stats(self):
        """통계 정보 반환"""
        return {
            'count': len(self.embeddings),
            'ids': list(self.embeddings.keys()),
            'last_cleaned': self.last_cleaned_timestamp
        }
