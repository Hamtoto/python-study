"""
트래커 관련 유틸리티 함수들
"""
from config import BASE_REINIT_INTERVAL


def get_adaptive_reinit_interval(success_rate, base_interval=BASE_REINIT_INTERVAL):
    """
    성공률에 따른 적응적 재초기화 간격 계산
    
    Args:
        success_rate: 트래커 성공률 (0.0 ~ 1.0)
        base_interval: 기본 재초기화 간격 (프레임 수)
    
    Returns:
        int: 계산된 재초기화 간격
    """
    if success_rate > 0.95:      # 매우 좋음
        return base_interval * 2     # 80프레임
    elif success_rate > 0.8:     # 좋음  
        return base_interval         # 40프레임
    elif success_rate > 0.6:     # 보통
        return base_interval // 2    # 20프레임
    else:                        # 나쁨
        return base_interval // 4    # 10프레임