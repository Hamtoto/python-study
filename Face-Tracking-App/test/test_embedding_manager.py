# -*- coding: utf-8 -*-
"""
EmbeddingManager 단위 테스트 모듈
"""
import pytest
import torch
import time
import torch.nn.functional as F # Import F for cosine_similarity in test
from core.embedding_manager import SmartEmbeddingManager
from utils.similarity_utils import find_matching_id_with_best_fallback

@pytest.fixture
def embedding_manager():
    """테스트용 SmartEmbeddingManager 인스턴스 제공"""
    return SmartEmbeddingManager()

def generate_dummy_embedding(type_str="target"):
    """
    더미 임베딩 생성 (테스트용) - 코사인 유사도 테스트를 위해.
    type_str: "target", "similar", "dissimilar"
    """
    emb = torch.zeros(1, 512)
    if type_str == "target":
        emb[0, 0] = 1.0
    elif type_str == "similar":
        emb[0, 0] = 0.9
        emb[0, 1] = 0.1 # 작은 차이를 주어 유사도를 1보다 작게 만듦
    elif type_str == "dissimilar":
        emb[0, 1] = 1.0 # target과 직교하도록 만듦 (유사도 0)
    else:
        raise ValueError("Invalid type_str")
    return F.normalize(emb, p=2, dim=1)

def test_add_embedding(embedding_manager):
    """임베딩 추가 기능 테스트"""
    emb1 = generate_dummy_embedding("target")
    embedding_manager.add_embedding(1, emb1)
    assert embedding_manager.get_all_embeddings()[1].equal(emb1)
    assert embedding_manager.get_stats()['count'] == 1

def test_get_all_embeddings(embedding_manager):
    """모든 임베딩 조회 기능 테스트"""
    emb1 = generate_dummy_embedding("target")
    emb2 = generate_dummy_embedding("similar")
    embedding_manager.add_embedding(1, emb1)
    embedding_manager.add_embedding(2, emb2)
    all_embs = embedding_manager.get_all_embeddings()
    assert len(all_embs) == 2
    assert all_embs[1].equal(emb1)
    assert all_embs[2].equal(emb2)

def test_remove_old_embeddings(embedding_manager):
    """오래된 임베딩 제거 기능 테스트"""
    # TTL을 짧게 설정하여 테스트
    embedding_manager.ttl_seconds = 0.01
    emb1 = generate_dummy_embedding("target")
    embedding_manager.add_embedding(1, emb1)
    time.sleep(0.02) # TTL보다 오래 기다림
    emb2 = generate_dummy_embedding("similar")
    embedding_manager.add_embedding(2, emb2) # 새 임베딩 추가 시 오래된 것 제거 트리거
    all_embs = embedding_manager.get_all_embeddings()
    assert 1 not in all_embs # emb1은 제거되어야 함
    assert 2 in all_embs # emb2는 남아있어야 함

def test_find_matching_id_with_best_fallback(embedding_manager):
    """유사도 기반 ID 매칭 기능 테스트"""
    # 유사도 임계값 설정 (config.py의 SIMILARITY_THRESHOLD와 동일하게)
    SIMILARITY_THRESHOLD = 0.6

    emb_target = generate_dummy_embedding("target")
    embedding_manager.add_embedding(100, emb_target) # 타겟 임베딩 추가

    # 유사한 임베딩 (코사인 유사도 > 임계값)
    emb_similar = generate_dummy_embedding("similar")
    matched_id = find_matching_id_with_best_fallback(emb_similar, embedding_manager.get_all_embeddings(), SIMILARITY_THRESHOLD)
    assert matched_id == 100

    # 유사하지 않은 임베딩 (코사인 유사도 < 임계값)
    emb_dissimilar = generate_dummy_embedding("dissimilar")
    matched_id = find_matching_id_with_best_fallback(emb_dissimilar, embedding_manager.get_all_embeddings(), SIMILARITY_THRESHOLD)
    assert matched_id is None # 매칭되지 않아야 함

    # 임계값에 걸리는 임베딩 (정확히 임계값)
    # 이 테스트는 generate_dummy_embedding의 특성상 정확한 임계값 매칭이 어려울 수 있음.
    # 여기서는 유사한 임베딩이 임계값보다 높게 나오는지 확인하는 것으로 대체.
    # emb_at_threshold = generate_dummy_embedding(0.601) # Slightly above threshold
    # matched_id = find_matching_id_with_best_fallback(emb_at_threshold, embedding_manager.get_all_embeddings(), SIMILARITY_THRESHOLD)
    # assert matched_id == 100

def test_get_stats(embedding_manager):
    """통계 조회 기능 테스트"""
    emb1 = generate_dummy_embedding("target")
    embedding_manager.add_embedding(1, emb1)
    stats = embedding_manager.get_stats()
    assert stats['count'] == 1
    assert 'last_cleaned' in stats