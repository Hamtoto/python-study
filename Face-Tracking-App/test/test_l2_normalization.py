"""
L2 정규화 기능 단위 테스트
"""
import unittest
import torch
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.face_tracker.utils.similarity import (
    cosine_similarity_l2_normalized,
    find_matching_id_with_l2_normalization,
    find_matching_id_with_best_fallback_enhanced,
    calculate_face_similarity
)
from src.face_tracker.processing.selector import TargetSelector


class TestL2Normalization(unittest.TestCase):
    """L2 정규화 관련 기능 테스트"""
    
    def setUp(self):
        """테스트용 임베딩 데이터 준비"""
        # 512차원 임베딩 벡터 (FaceNet 표준)
        self.emb1 = torch.randn(1, 512)
        self.emb2 = torch.randn(1, 512) 
        self.emb3 = self.emb1 * 2.0  # 같은 방향, 다른 크기
        
        # 테스트용 임베딩 딕셔너리
        self.test_embeddings = {
            1: torch.randn(1, 512),
            2: torch.randn(1, 512),
            3: torch.randn(1, 512)
        }
    
    def test_cosine_similarity_l2_normalized(self):
        """L2 정규화된 코사인 유사도 계산 테스트"""
        # 동일한 벡터는 유사도 1.0
        sim_same = cosine_similarity_l2_normalized(self.emb1, self.emb1)
        self.assertAlmostEqual(sim_same, 1.0, places=5)
        
        # 다른 방향의 벡터는 유사도 < 1.0
        sim_diff = cosine_similarity_l2_normalized(self.emb1, self.emb2)
        self.assertLess(sim_diff, 1.0)
        self.assertGreater(sim_diff, -1.0)
        
        # 같은 방향, 다른 크기의 벡터는 유사도 1.0 (L2 정규화 효과)
        sim_scaled = cosine_similarity_l2_normalized(self.emb1, self.emb3)
        self.assertAlmostEqual(sim_scaled, 1.0, places=5)
    
    def test_find_matching_id_with_l2_normalization(self):
        """L2 정규화 적용 ID 매칭 테스트"""
        # 높은 임계값으로 매칭 없음 테스트
        result = find_matching_id_with_l2_normalization(
            self.emb1, self.test_embeddings, threshold=0.99
        )
        self.assertIsNone(result)
        
        # 낮은 임계값으로 매칭 있음 테스트 
        result = find_matching_id_with_l2_normalization(
            self.emb1, self.test_embeddings, threshold=0.1
        )
        # 랜덤 임베딩이므로 매칭될 수도 있고 안될 수도 있음
        if result is not None:
            self.assertIn(result, self.test_embeddings.keys())
        
        # 확실한 매칭을 위해 동일한 임베딩 추가
        test_embeddings_with_match = self.test_embeddings.copy()
        test_embeddings_with_match[99] = self.emb1  # 동일한 임베딩 추가
        
        result_match = find_matching_id_with_l2_normalization(
            self.emb1, test_embeddings_with_match, threshold=0.1
        )
        self.assertIsNotNone(result_match)
        self.assertEqual(result_match, 99)
    
    def test_find_matching_id_enhanced_compatibility(self):
        """향상된 함수의 하위 호환성 테스트"""
        # L2 정규화 사용
        result_l2 = find_matching_id_with_best_fallback_enhanced(
            self.emb1, self.test_embeddings, threshold=0.5, use_l2_norm=True
        )
        
        # L2 정규화 미사용 (기존 방식)
        result_legacy = find_matching_id_with_best_fallback_enhanced(
            self.emb1, self.test_embeddings, threshold=0.5, use_l2_norm=False
        )
        
        # 둘 다 유효한 결과여야 함 (None이거나 유효한 ID)
        if result_l2 is not None:
            self.assertIn(result_l2, self.test_embeddings.keys())
        if result_legacy is not None:
            self.assertIn(result_legacy, self.test_embeddings.keys())
    
    def test_calculate_face_similarity_interface(self):
        """통합 유사도 계산 인터페이스 테스트"""
        # L2 정규화 사용
        sim_l2 = calculate_face_similarity(self.emb1, self.emb2, use_l2_norm=True)
        
        # L2 정규화 미사용
        sim_legacy = calculate_face_similarity(self.emb1, self.emb2, use_l2_norm=False)
        
        # 모두 0~1 범위의 유효한 값
        self.assertGreaterEqual(sim_l2, -1.0)
        self.assertLessEqual(sim_l2, 1.0)
        self.assertGreaterEqual(sim_legacy, -1.0)
        self.assertLessEqual(sim_legacy, 1.0)
    
    def test_target_selector_dual_speakers_l2(self):
        """TargetSelector의 DUAL 모드 L2 정규화 테스트"""
        # 테스트 데이터 준비
        voice_segments = [(0.0, 2.0), (3.0, 5.0)]
        id_timeline = [1, 1, 2, 2, 1] * 10  # 50프레임
        fps = 25
        
        # L2 정규화 사용
        result_l2 = TargetSelector.select_dual_speakers(
            voice_segments, id_timeline, fps, 
            embeddings=self.test_embeddings, use_l2_norm=True
        )
        
        # L2 정규화 미사용
        result_legacy = TargetSelector.select_dual_speakers(
            voice_segments, id_timeline, fps,
            embeddings=self.test_embeddings, use_l2_norm=False
        )
        
        # 결과 구조 검증
        self.assertIn('speaker_a', result_l2)
        self.assertIn('speaker_b', result_l2)
        self.assertIn('speaker_a', result_legacy)
        self.assertIn('speaker_b', result_legacy)
        
        # 각 speaker의 결과가 리스트인지 확인
        self.assertIsInstance(result_l2['speaker_a'], list)
        self.assertIsInstance(result_l2['speaker_b'], list)
    
    def test_l2_normalization_scale_invariance(self):
        """L2 정규화의 크기 불변성 테스트"""
        # 동일한 방향, 다른 크기의 벡터들
        base_vector = torch.randn(1, 512)
        scaled_vectors = [base_vector * scale for scale in [0.5, 1.0, 2.0, 10.0]]
        
        # 모든 스케일된 벡터들 간의 L2 정규화 유사도는 1.0이어야 함
        for i, vec1 in enumerate(scaled_vectors):
            for j, vec2 in enumerate(scaled_vectors):
                sim = cosine_similarity_l2_normalized(vec1, vec2)
                self.assertAlmostEqual(sim, 1.0, places=5, 
                    msg=f"Scale {i} vs {j} should have similarity 1.0")
    
    def test_empty_embeddings_handling(self):
        """빈 임베딩 딕셔너리 처리 테스트"""
        empty_embeddings = {}
        
        result = find_matching_id_with_l2_normalization(
            self.emb1, empty_embeddings, threshold=0.5
        )
        self.assertIsNone(result)
        
        result_enhanced = find_matching_id_with_best_fallback_enhanced(
            self.emb1, empty_embeddings, threshold=0.5, use_l2_norm=True
        )
        self.assertIsNone(result_enhanced)


class TestL2NormalizationPerformance(unittest.TestCase):
    """L2 정규화 성능 테스트"""
    
    def test_performance_comparison(self):
        """기존 방식 vs L2 정규화 방식 성능 비교"""
        import time
        
        # 대량 테스트 데이터 생성
        emb1 = torch.randn(1, 512)
        test_embeddings = {i: torch.randn(1, 512) for i in range(100)}
        
        # 기존 방식 시간 측정
        start_time = time.time()
        for _ in range(100):
            find_matching_id_with_best_fallback_enhanced(
                emb1, test_embeddings, use_l2_norm=False
            )
        legacy_time = time.time() - start_time
        
        # L2 정규화 방식 시간 측정
        start_time = time.time()
        for _ in range(100):
            find_matching_id_with_best_fallback_enhanced(
                emb1, test_embeddings, use_l2_norm=True
            )
        l2_time = time.time() - start_time
        
        # L2 정규화 방식이 기존 방식보다 크게 느리지 않아야 함 (2배 이내)
        performance_ratio = l2_time / legacy_time
        self.assertLess(performance_ratio, 2.0, 
            f"L2 normalization is too slow: {performance_ratio:.2f}x slower")
        
        print(f"성능 비교 - 기존: {legacy_time:.4f}s, L2: {l2_time:.4f}s, 비율: {performance_ratio:.2f}x")


if __name__ == '__main__':
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA 사용 불가, CPU로 테스트 진행")
    
    # 테스트 실행
    unittest.main(verbosity=2)