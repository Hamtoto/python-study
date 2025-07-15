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