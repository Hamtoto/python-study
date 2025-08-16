"""
얼굴 임베딩 캐시 및 관리 시스템.

ConditionalReID 시스템에서 사용할 얼굴 임베딩을 효율적으로 관리합니다.
- 트랙별 임베딩 히스토리 관리
- 메모리 효율적 캐시 시스템
- 임베딩 품질 평가 및 필터링
- 유사도 계산 및 매칭
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from .tracking_structures import Track
from ..inference.reid_model import ReIDModel
from ..utils.logger import UnifiedLogger


@dataclass
class EmbeddingRecord:
    """개별 임베딩 레코드."""
    
    embedding: np.ndarray        # 임베딩 벡터
    timestamp: float            # 생성 시간
    frame_id: int              # 프레임 ID
    confidence: float          # 임베딩 품질 신뢰도
    bbox: Tuple[float, float, float, float]  # 바운딩 박스
    track_id: int              # 트랙 ID
    quality_score: float       # 이미지 품질 점수
    
    def __repr__(self):
        return f"EmbeddingRecord(track={self.track_id}, frame={self.frame_id}, conf={self.confidence:.2f})"


@dataclass
class MatchingResult:
    """임베딩 매칭 결과."""
    
    source_track_id: int       # 소스 트랙 ID
    matched_track_id: int      # 매칭된 트랙 ID
    similarity: float          # 유사도 점수 (0.0 ~ 1.0)
    confidence: float          # 매칭 신뢰도
    embedding_count: int       # 사용된 임베딩 수
    match_type: str           # 매칭 타입 ("single", "ensemble", "historical")
    
    def __repr__(self):
        return f"MatchingResult({self.source_track_id}->{self.matched_track_id}, sim={self.similarity:.2f})"


class EmbeddingManager:
    """
    얼굴 임베딩 관리자 클래스.
    
    트랙별 임베딩 히스토리를 관리하고 효율적인 유사도 계산을 제공합니다.
    """
    
    def __init__(self,
                 max_embeddings_per_track: int = 20,       # 트랙당 최대 임베딩 수
                 cache_size_limit_mb: int = 100,           # 캐시 크기 제한 (MB)
                 similarity_threshold: float = 0.7,        # 유사도 임계값
                 quality_threshold: float = 0.5,           # 임베딩 품질 임계값
                 cleanup_interval: int = 1000,             # 정리 주기 (프레임)
                 use_threading: bool = True):               # 멀티스레딩 사용 여부
        """
        임베딩 관리자를 초기화합니다.
        
        Args:
            max_embeddings_per_track: 트랙당 저장할 최대 임베딩 수
            cache_size_limit_mb: 전체 캐시 크기 제한 (MB)
            similarity_threshold: 유사도 매칭 임계값
            quality_threshold: 임베딩 품질 필터링 임계값
            cleanup_interval: 캐시 정리 주기 (프레임 수)
            use_threading: 백그라운드 처리를 위한 멀티스레딩 사용
        """
        # 설정
        self.max_embeddings_per_track = max_embeddings_per_track
        self.cache_size_limit_bytes = cache_size_limit_mb * 1024 * 1024
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold
        self.cleanup_interval = cleanup_interval
        self.use_threading = use_threading
        
        # 임베딩 저장소
        self.track_embeddings = defaultdict(lambda: deque(maxlen=max_embeddings_per_track))
        self.embedding_cache = {}  # 빠른 검색을 위한 캐시
        
        # 메모리 관리
        self.current_memory_usage = 0
        self.frame_counter = 0
        self.last_cleanup_frame = 0
        
        # 멀티스레딩
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2) if use_threading else None
        
        # 통계
        self.stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'quality_filtered': 0,
            'memory_cleanups': 0,
            'total_matches': 0,
            'successful_matches': 0
        }
        
        self.logger = UnifiedLogger("EmbeddingManager")
        self.logger.info(f"임베딩 관리자 초기화: max_per_track={max_embeddings_per_track}, "
                        f"cache_limit={cache_size_limit_mb}MB, sim_threshold={similarity_threshold}")
    
    def add_embedding(self, 
                     track: Track, 
                     embedding: np.ndarray, 
                     face_image: Optional[np.ndarray] = None) -> bool:
        """
        트랙에 새로운 임베딩을 추가합니다.
        
        Args:
            track: 트랙 객체
            embedding: 임베딩 벡터
            face_image: 얼굴 이미지 (품질 평가용, 선택적)
            
        Returns:
            bool: 추가 성공 여부
        """
        try:
            with self.lock:
                # 품질 평가
                quality_score = self._evaluate_embedding_quality(embedding, face_image, track)
                
                if quality_score < self.quality_threshold:
                    self.stats['quality_filtered'] += 1
                    self.logger.debug(f"임베딩 품질 필터링: Track {track.track_id}, "
                                     f"품질={quality_score:.2f} < {self.quality_threshold}")
                    return False
                
                # 임베딩 레코드 생성
                record = EmbeddingRecord(
                    embedding=embedding.copy(),
                    timestamp=time.time(),
                    frame_id=track.frame_id,
                    confidence=track.score,
                    bbox=tuple(track.tlbr),
                    track_id=track.track_id,
                    quality_score=quality_score
                )
                
                # 트랙별 저장소에 추가
                self.track_embeddings[track.track_id].append(record)
                
                # 캐시에도 추가 (빠른 접근용)
                cache_key = f"{track.track_id}_{track.frame_id}"
                self.embedding_cache[cache_key] = record
                
                # 메모리 사용량 업데이트
                embedding_size = embedding.nbytes + 200  # 메타데이터 추정 크기
                self.current_memory_usage += embedding_size
                
                # 통계 업데이트
                self.stats['total_embeddings'] += 1
                
                # 주기적 정리
                self.frame_counter += 1
                if self.frame_counter - self.last_cleanup_frame >= self.cleanup_interval:
                    if self.use_threading and self.executor:
                        self.executor.submit(self._cleanup_cache)
                    else:
                        self._cleanup_cache()
                
                self.logger.debug(f"임베딩 추가: Track {track.track_id}, "
                                 f"품질={quality_score:.2f}, 메모리={self.current_memory_usage/(1024*1024):.1f}MB")
                
                return True
                
        except Exception as e:
            self.logger.error(f"임베딩 추가 실패: {e}")
            return False
    
    def get_track_embeddings(self, track_id: int, limit: Optional[int] = None) -> List[EmbeddingRecord]:
        """
        특정 트랙의 임베딩들을 반환합니다.
        
        Args:
            track_id: 트랙 ID
            limit: 반환할 최대 임베딩 수
            
        Returns:
            List[EmbeddingRecord]: 임베딩 레코드 리스트
        """
        with self.lock:
            embeddings = list(self.track_embeddings[track_id])
            
            if limit and len(embeddings) > limit:
                # 품질 순으로 정렬하여 상위 임베딩 반환
                embeddings.sort(key=lambda x: x.quality_score, reverse=True)
                embeddings = embeddings[:limit]
            
            return embeddings
    
    def find_similar_embeddings(self, 
                               query_embedding: np.ndarray,
                               exclude_track_id: Optional[int] = None,
                               top_k: int = 5) -> List[Tuple[EmbeddingRecord, float]]:
        """
        유사한 임베딩들을 찾습니다.
        
        Args:
            query_embedding: 쿼리 임베딩
            exclude_track_id: 제외할 트랙 ID
            top_k: 반환할 상위 결과 수
            
        Returns:
            List[Tuple[EmbeddingRecord, float]]: (임베딩 레코드, 유사도) 튜플 리스트
        """
        similar_embeddings = []
        
        with self.lock:
            for track_id, embeddings in self.track_embeddings.items():
                if exclude_track_id and track_id == exclude_track_id:
                    continue
                
                for record in embeddings:
                    # 유사도 계산
                    similarity = self._calculate_cosine_similarity(query_embedding, record.embedding)
                    
                    if similarity >= self.similarity_threshold:
                        similar_embeddings.append((record, similarity))
        
        # 유사도 순으로 정렬
        similar_embeddings.sort(key=lambda x: x[1], reverse=True)
        
        return similar_embeddings[:top_k]
    
    def match_tracks_by_embeddings(self, 
                                  source_track_id: int, 
                                  candidate_track_ids: List[int]) -> List[MatchingResult]:
        """
        임베딩을 사용하여 트랙들을 매칭합니다.
        
        Args:
            source_track_id: 소스 트랙 ID
            candidate_track_ids: 후보 트랙 ID들
            
        Returns:
            List[MatchingResult]: 매칭 결과 리스트
        """
        try:
            with self.lock:
                # 소스 트랙 임베딩 가져오기
                source_embeddings = self.get_track_embeddings(source_track_id)
                if not source_embeddings:
                    return []
                
                results = []
                
                for candidate_id in candidate_track_ids:
                    if candidate_id == source_track_id:
                        continue
                    
                    candidate_embeddings = self.get_track_embeddings(candidate_id)
                    if not candidate_embeddings:
                        continue
                    
                    # 다양한 매칭 방법 시도
                    match_result = self._perform_embedding_matching(
                        source_embeddings, candidate_embeddings, source_track_id, candidate_id
                    )
                    
                    if match_result and match_result.similarity >= self.similarity_threshold:
                        results.append(match_result)
                
                # 유사도 순으로 정렬
                results.sort(key=lambda x: x.similarity, reverse=True)
                
                # 통계 업데이트
                self.stats['total_matches'] += 1
                if results:
                    self.stats['successful_matches'] += 1
                
                return results
                
        except Exception as e:
            self.logger.error(f"임베딩 매칭 실패: {e}")
            return []
    
    def _perform_embedding_matching(self,
                                  source_embeddings: List[EmbeddingRecord],
                                  candidate_embeddings: List[EmbeddingRecord],
                                  source_track_id: int,
                                  candidate_track_id: int) -> Optional[MatchingResult]:
        """실제 임베딩 매칭을 수행합니다."""
        
        # 1. 최고 품질 임베딩 간 매칭
        best_source = max(source_embeddings, key=lambda x: x.quality_score)
        best_candidate = max(candidate_embeddings, key=lambda x: x.quality_score)
        
        single_similarity = self._calculate_cosine_similarity(
            best_source.embedding, best_candidate.embedding
        )
        
        # 2. 앙상블 매칭 (상위 임베딩들의 평균)
        ensemble_similarity = self._calculate_ensemble_similarity(
            source_embeddings[:5], candidate_embeddings[:5]
        )
        
        # 3. 시간 가중 매칭 (최근 임베딩에 더 높은 가중치)
        temporal_similarity = self._calculate_temporal_weighted_similarity(
            source_embeddings, candidate_embeddings
        )
        
        # 최종 유사도는 가중 평균
        final_similarity = (
            single_similarity * 0.4 +
            ensemble_similarity * 0.4 +
            temporal_similarity * 0.2
        )
        
        # 신뢰도 계산
        confidence = self._calculate_matching_confidence(
            source_embeddings, candidate_embeddings, final_similarity
        )
        
        return MatchingResult(
            source_track_id=source_track_id,
            matched_track_id=candidate_track_id,
            similarity=final_similarity,
            confidence=confidence,
            embedding_count=len(source_embeddings) + len(candidate_embeddings),
            match_type="ensemble"
        )
    
    def _calculate_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """코사인 유사도를 계산합니다."""
        # L2 정규화가 되어있다면 내적이 코사인 유사도
        similarity = np.dot(emb1, emb2)
        
        # 수치적 안정성
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # [0, 1] 범위로 변환
        return (similarity + 1.0) / 2.0
    
    def _calculate_ensemble_similarity(self,
                                    embeddings1: List[EmbeddingRecord],
                                    embeddings2: List[EmbeddingRecord]) -> float:
        """앙상블 유사도를 계산합니다."""
        if not embeddings1 or not embeddings2:
            return 0.0
        
        # 각 그룹의 평균 임베딩 계산 (품질 가중)
        def weighted_mean(embeddings):
            if not embeddings:
                return None
            
            weights = np.array([emb.quality_score for emb in embeddings])
            weights = weights / np.sum(weights)  # 정규화
            
            weighted_embs = [emb.embedding * w for emb, w in zip(embeddings, weights)]
            mean_emb = np.sum(weighted_embs, axis=0)
            
            # L2 정규화
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-6:
                mean_emb = mean_emb / norm
                
            return mean_emb
        
        mean_emb1 = weighted_mean(embeddings1)
        mean_emb2 = weighted_mean(embeddings2)
        
        if mean_emb1 is None or mean_emb2 is None:
            return 0.0
        
        return self._calculate_cosine_similarity(mean_emb1, mean_emb2)
    
    def _calculate_temporal_weighted_similarity(self,
                                             embeddings1: List[EmbeddingRecord],
                                             embeddings2: List[EmbeddingRecord]) -> float:
        """시간 가중 유사도를 계산합니다."""
        if not embeddings1 or not embeddings2:
            return 0.0
        
        current_time = time.time()
        similarities = []
        weights = []
        
        # 모든 쌍에 대해 유사도 계산
        for emb1 in embeddings1[-3:]:  # 최근 3개
            for emb2 in embeddings2[-3:]:  # 최근 3개
                similarity = self._calculate_cosine_similarity(emb1.embedding, emb2.embedding)
                
                # 시간 가중치 (최근일수록 높은 가중치)
                time_weight1 = np.exp(-(current_time - emb1.timestamp) / 10.0)  # 10초 감쇠
                time_weight2 = np.exp(-(current_time - emb2.timestamp) / 10.0)
                combined_weight = (time_weight1 + time_weight2) / 2.0
                
                # 품질 가중치도 반영
                quality_weight = (emb1.quality_score + emb2.quality_score) / 2.0
                
                final_weight = combined_weight * quality_weight
                
                similarities.append(similarity)
                weights.append(final_weight)
        
        if not similarities:
            return 0.0
        
        # 가중 평균
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            return np.average(similarities, weights=weights)
        else:
            return np.mean(similarities)
    
    def _calculate_matching_confidence(self,
                                     embeddings1: List[EmbeddingRecord],
                                     embeddings2: List[EmbeddingRecord],
                                     similarity: float) -> float:
        """매칭 신뢰도를 계산합니다."""
        # 임베딩 수에 따른 신뢰도
        count_factor = min((len(embeddings1) + len(embeddings2)) / 10.0, 1.0)
        
        # 평균 품질에 따른 신뢰도
        avg_quality1 = np.mean([emb.quality_score for emb in embeddings1])
        avg_quality2 = np.mean([emb.quality_score for emb in embeddings2])
        quality_factor = (avg_quality1 + avg_quality2) / 2.0
        
        # 유사도에 따른 신뢰도
        similarity_factor = similarity
        
        # 가중 평균
        confidence = (count_factor * 0.3 + quality_factor * 0.4 + similarity_factor * 0.3)
        
        return confidence
    
    def _evaluate_embedding_quality(self,
                                  embedding: np.ndarray,
                                  face_image: Optional[np.ndarray],
                                  track: Track) -> float:
        """임베딩 품질을 평가합니다."""
        quality_factors = []
        
        # 1. 임베딩 자체의 품질
        # L2 노름이 적절한 범위에 있는지 확인
        l2_norm = np.linalg.norm(embedding)
        if 0.8 <= l2_norm <= 1.2:  # L2 정규화 가정
            quality_factors.append(1.0)
        else:
            quality_factors.append(max(0.3, 1.0 - abs(1.0 - l2_norm)))
        
        # 2. 임베딩 분산 (너무 균일하지 않아야 함)
        embedding_var = np.var(embedding)
        if embedding_var > 0.01:  # 충분한 분산
            quality_factors.append(1.0)
        else:
            quality_factors.append(max(0.3, embedding_var * 100))
        
        # 3. 트랙 신뢰도
        quality_factors.append(track.score)
        
        # 4. 트랙 안정성 (나이와 히트 스트릭)
        age_factor = min(track.age / 10.0, 1.0)
        hit_factor = min(track.hit_streak / 5.0, 1.0)
        stability_factor = (age_factor + hit_factor) / 2.0
        quality_factors.append(stability_factor)
        
        # 5. 이미지 품질 (제공된 경우)
        if face_image is not None:
            image_quality = self._evaluate_image_quality(face_image)
            quality_factors.append(image_quality)
        
        # 평균 품질 점수
        return np.mean(quality_factors)
    
    def _evaluate_image_quality(self, image: np.ndarray) -> float:
        """이미지 품질을 평가합니다."""
        if image is None or image.size == 0:
            return 0.0
        
        try:
            # 1. 이미지 크기 (너무 작지 않아야 함)
            h, w = image.shape[:2]
            size_score = min(min(h, w) / 64.0, 1.0)  # 64px 이상 권장
            
            # 2. 선명도 (라플라시안 분산)
            gray = image if len(image.shape) == 2 else image[:,:,0]
            laplacian_var = np.var(np.gradient(gray.astype(float)))
            sharpness_score = min(laplacian_var / 100.0, 1.0)
            
            # 3. 밝기 적정성
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
            
            # 종합 점수
            return (size_score * 0.4 + sharpness_score * 0.4 + brightness_score * 0.2)
            
        except Exception:
            return 0.5  # 기본값
    
    def _cleanup_cache(self):
        """캐시를 정리합니다."""
        with self.lock:
            try:
                # 메모리 사용량이 임계값을 초과하는 경우
                if self.current_memory_usage > self.cache_size_limit_bytes:
                    self.logger.info(f"캐시 정리 시작: {self.current_memory_usage/(1024*1024):.1f}MB")
                    
                    # 오래된 임베딩 제거
                    current_time = time.time()
                    removed_count = 0
                    
                    for track_id in list(self.track_embeddings.keys()):
                        embeddings = self.track_embeddings[track_id]
                        
                        # 24시간 이상 된 임베딩 제거
                        old_embeddings = []
                        for emb in embeddings:
                            if current_time - emb.timestamp > 24 * 3600:  # 24시간
                                old_embeddings.append(emb)
                        
                        for old_emb in old_embeddings:
                            if old_emb in embeddings:
                                embeddings.remove(old_emb)
                                removed_count += 1
                        
                        # 빈 트랙 제거
                        if not embeddings:
                            del self.track_embeddings[track_id]
                    
                    # 캐시도 정리
                    old_cache_keys = []
                    for key, record in self.embedding_cache.items():
                        if current_time - record.timestamp > 24 * 3600:
                            old_cache_keys.append(key)
                    
                    for key in old_cache_keys:
                        del self.embedding_cache[key]
                    
                    # 메모리 사용량 재계산
                    self._recalculate_memory_usage()
                    
                    self.stats['memory_cleanups'] += 1
                    self.last_cleanup_frame = self.frame_counter
                    
                    self.logger.info(f"캐시 정리 완료: {removed_count}개 임베딩 제거, "
                                   f"메모리={self.current_memory_usage/(1024*1024):.1f}MB")
                
            except Exception as e:
                self.logger.error(f"캐시 정리 실패: {e}")
    
    def _recalculate_memory_usage(self):
        """메모리 사용량을 재계산합니다."""
        total_usage = 0
        
        for track_embeddings in self.track_embeddings.values():
            for record in track_embeddings:
                total_usage += record.embedding.nbytes + 200  # 메타데이터 추정
        
        self.current_memory_usage = total_usage
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보를 반환합니다."""
        with self.lock:
            stats = self.stats.copy()
            
            stats.update({
                'active_tracks': len(self.track_embeddings),
                'total_embedding_records': sum(len(embs) for embs in self.track_embeddings.values()),
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'cache_size': len(self.embedding_cache),
                'avg_embeddings_per_track': (
                    sum(len(embs) for embs in self.track_embeddings.values()) / 
                    max(len(self.track_embeddings), 1)
                )
            })
            
            if stats['total_matches'] > 0:
                stats['match_success_rate'] = stats['successful_matches'] / stats['total_matches']
            else:
                stats['match_success_rate'] = 0.0
            
            return stats
    
    def reset(self):
        """관리자를 초기 상태로 리셋합니다."""
        with self.lock:
            self.track_embeddings.clear()
            self.embedding_cache.clear()
            self.current_memory_usage = 0
            self.frame_counter = 0
            self.last_cleanup_frame = 0
            
            # 통계는 유지
            self.logger.info("임베딩 관리자 리셋 완료")
    
    def __del__(self):
        """소멸자 - 스레드풀 정리."""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def __repr__(self):
        return (f"EmbeddingManager(tracks={len(self.track_embeddings)}, "
                f"embeddings={sum(len(e) for e in self.track_embeddings.values())}, "
                f"memory={self.current_memory_usage/(1024*1024):.1f}MB)")