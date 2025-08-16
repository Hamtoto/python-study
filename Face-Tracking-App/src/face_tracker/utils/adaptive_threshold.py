"""
동적 임계값 자동 계산 모듈
이미 메모리에 로드된 임베딩 데이터를 활용하여 각 비디오에 최적화된 임계값을 실시간 계산
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from src.face_tracker.utils.similarity import calculate_face_similarity
from src.face_tracker.utils.logging import logger


def calculate_optimal_threshold(
    embeddings_dict: Dict[int, List[torch.Tensor]], 
    use_l2_norm: bool = True,
    min_same_samples: int = 5,
    min_different_samples: int = 5,
    safety_range: Tuple[float, float] = (0.6, 0.95)
) -> Tuple[Optional[float], str, Dict[str, Any]]:
    """
    이미 메모리에 있는 임베딩으로 최적 임계값 실시간 계산
    
    Args:
        embeddings_dict: {person_id: [embeddings]} 형태의 임베딩 딕셔너리
        use_l2_norm: L2 정규화 사용 여부
        min_same_samples: 같은 사람 유사도 최소 샘플 수
        min_different_samples: 다른 사람 유사도 최소 샘플 수
        safety_range: 안전 범위 (min_threshold, max_threshold)
        
    Returns:
        tuple: (optimal_threshold, confidence, statistics)
    """
    
    if len(embeddings_dict) < 2:
        return None, "insufficient_ids", {}
    
    same_person_similarities = []
    different_person_similarities = []
    
    person_ids = list(embeddings_dict.keys())
    
    # 1. 같은 사람끼리 유사도 수집 (ID 내부)
    for person_id, embedding_list in embeddings_dict.items():
        if len(embedding_list) >= 2:
            for i in range(len(embedding_list)):
                for j in range(i + 1, len(embedding_list)):
                    try:
                        similarity = calculate_face_similarity(
                            embedding_list[i], 
                            embedding_list[j], 
                            use_l2_norm=use_l2_norm
                        )
                        same_person_similarities.append(similarity)
                    except Exception as e:
                        logger.warning(f"같은 사람 유사도 계산 오류: {e}")
                        continue
    
    # 2. 다른 사람끼리 유사도 수집 (ID 간)
    for i in range(len(person_ids)):
        for j in range(i + 1, len(person_ids)):
            try:
                # 각 ID의 대표 임베딩 (첫 번째) 사용
                emb1 = embeddings_dict[person_ids[i]][0]
                emb2 = embeddings_dict[person_ids[j]][0]
                
                similarity = calculate_face_similarity(emb1, emb2, use_l2_norm=use_l2_norm)
                different_person_similarities.append(similarity)
            except Exception as e:
                logger.warning(f"다른 사람 유사도 계산 오류: {e}")
                continue
    
    # 3. 데이터 충분성 검증
    if (len(same_person_similarities) < min_same_samples or 
        len(different_person_similarities) < min_different_samples):
        return None, "insufficient_samples", {
            'same_person_count': len(same_person_similarities),
            'different_person_count': len(different_person_similarities)
        }
    
    # 4. 통계 분석
    same_array = np.array(same_person_similarities)
    different_array = np.array(different_person_similarities)
    
    same_mean = np.mean(same_array)
    same_std = np.std(same_array)
    same_min = np.min(same_array)
    same_max = np.max(same_array)
    
    different_mean = np.mean(different_array)
    different_std = np.std(different_array)
    different_min = np.min(different_array)
    different_max = np.max(different_array)
    
    # 5. 최적 임계값 계산
    if same_min > different_max:
        # 명확하게 구분 가능한 경우
        optimal_threshold = (same_min + different_max) / 2
        confidence = "high"
        confidence_score = 0.9
    elif same_mean - same_std > different_mean + different_std:
        # 분포가 약간 겹치지만 구분 가능
        optimal_threshold = (same_mean - same_std + different_mean + different_std) / 2
        confidence = "medium"
        confidence_score = 0.7
    else:
        # 분포가 많이 겹침, 보수적 접근
        # 같은 사람 분포의 하위 15% 지점 사용
        optimal_threshold = np.percentile(same_array, 15)
        confidence = "low"
        confidence_score = 0.5
    
    # 6. 안전 범위 제한
    optimal_threshold = np.clip(optimal_threshold, safety_range[0], safety_range[1])
    
    # 7. 통계 정보 생성
    statistics = {
        'same_person': {
            'count': len(same_person_similarities),
            'mean': float(same_mean),
            'std': float(same_std),
            'min': float(same_min),
            'max': float(same_max),
            'percentile_10': float(np.percentile(same_array, 10)),
            'percentile_90': float(np.percentile(same_array, 90))
        },
        'different_person': {
            'count': len(different_person_similarities),
            'mean': float(different_mean),
            'std': float(different_std),
            'min': float(different_min),
            'max': float(different_max),
            'percentile_10': float(np.percentile(different_array, 10)),
            'percentile_90': float(np.percentile(different_array, 90))
        },
        'separation_quality': float(same_min - different_max),  # 양수면 완전 분리
        'confidence_score': confidence_score,
        'total_ids': len(embeddings_dict),
        'total_embeddings': sum(len(embs) for embs in embeddings_dict.values())
    }
    
    return optimal_threshold, confidence, statistics


def log_threshold_optimization(
    optimal_threshold: Optional[float],
    confidence: str,
    statistics: Dict[str, Any],
    current_threshold: float,
    use_l2_norm: bool
) -> None:
    """임계값 최적화 결과를 로그로 출력"""
    
    if optimal_threshold is None:
        logger.warning(f"🔧 임계값 최적화 실패: {confidence}")
        if 'same_person_count' in statistics:
            logger.info(f"   - 같은 사람 샘플: {statistics['same_person_count']}개")
            logger.info(f"   - 다른 사람 샘플: {statistics['different_person_count']}개")
        logger.info(f"   - 기본값 사용: {current_threshold:.3f}")
        return
    
    logger.info("=" * 60)
    logger.info("🔧 실시간 임계값 최적화")
    logger.info("=" * 60)
    
    # 기본 정보
    logger.info(f"📊 임베딩 데이터: {statistics['total_ids']}개 ID, "
                f"총 {statistics['total_embeddings']}개 임베딩")
    
    # 같은 사람 통계
    same_stats = statistics['same_person']
    logger.info(f"👥 같은 사람 유사도: {same_stats['mean']:.3f} ± {same_stats['std']:.3f} "
                f"({same_stats['count']}개)")
    logger.info(f"   범위: {same_stats['min']:.3f} ~ {same_stats['max']:.3f}")
    
    # 다른 사람 통계
    diff_stats = statistics['different_person']
    logger.info(f"🚫 다른 사람 유사도: {diff_stats['mean']:.3f} ± {diff_stats['std']:.3f} "
                f"({diff_stats['count']}개)")
    logger.info(f"   범위: {diff_stats['min']:.3f} ~ {diff_stats['max']:.3f}")
    
    # 분리 품질
    separation = statistics['separation_quality']
    if separation > 0:
        logger.info(f"🎯 분리 품질: 완전 분리 (+{separation:.3f})")
    else:
        logger.info(f"⚠️ 분리 품질: 분포 겹침 ({separation:.3f})")
    
    # 최적화 결과
    improvement = abs(optimal_threshold - current_threshold) / current_threshold * 100
    logger.info(f"💡 최적 임계값: {optimal_threshold:.3f} (신뢰도: {confidence})")
    logger.info(f"📏 기존 → 신규: {current_threshold:.3f} → {optimal_threshold:.3f} "
                f"({improvement:+.1f}%)")
    logger.info(f"🔍 L2 정규화: {'적용' if use_l2_norm else '미적용'}")
    logger.info("=" * 60)


def should_apply_adaptive_threshold(
    confidence: str, 
    optimal_threshold: float, 
    current_threshold: float,
    min_confidence_level: str = "medium",
    min_improvement_percent: float = 2.0
) -> bool:
    """
    동적 임계값을 적용할지 여부 결정
    
    Args:
        confidence: 신뢰도 레벨 ("high", "medium", "low")
        optimal_threshold: 계산된 최적 임계값
        current_threshold: 현재 사용 중인 임계값
        min_confidence_level: 최소 신뢰도 레벨
        min_improvement_percent: 최소 개선율 (%)
        
    Returns:
        bool: 적용 여부
    """
    confidence_levels = {"high": 3, "medium": 2, "low": 1}
    
    # 신뢰도 확인
    if confidence_levels.get(confidence, 0) < confidence_levels.get(min_confidence_level, 2):
        return False
    
    # 개선율 확인
    improvement = abs(optimal_threshold - current_threshold) / current_threshold * 100
    if improvement < min_improvement_percent:
        return False
    
    return True