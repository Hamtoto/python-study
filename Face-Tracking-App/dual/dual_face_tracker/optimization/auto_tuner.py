"""
자동 성능 튜닝 시스템

실행 중 시스템 성능을 모니터링하고 자동으로 파라미터를 최적화하는 시스템
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import logger
from ..utils.exceptions import OptimizationError


class TuningMode(Enum):
    """튜닝 모드"""
    CONSERVATIVE = "conservative"    # 안전한 최적화
    BALANCED = "balanced"           # 균형잡힌 최적화
    AGGRESSIVE = "aggressive"       # 공격적 최적화


@dataclass
class TuningParameter:
    """튜닝 파라미터 정의"""
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    impact_weight: float = 1.0  # 성능 영향도 가중치
    
    def can_increase(self) -> bool:
        """증가 가능 여부"""
        return self.current_value < self.max_value
    
    def can_decrease(self) -> bool:
        """감소 가능 여부"""
        return self.current_value > self.min_value
    
    def increase(self):
        """값 증가"""
        if self.can_increase():
            if isinstance(self.current_value, int):
                self.current_value = min(self.max_value, self.current_value + self.step_size)
            else:
                self.current_value = min(self.max_value, self.current_value * self.step_size)
    
    def decrease(self):
        """값 감소"""
        if self.can_decrease():
            if isinstance(self.current_value, int):
                self.current_value = max(self.min_value, self.current_value - self.step_size)
            else:
                self.current_value = max(self.min_value, self.current_value / self.step_size)


@dataclass
class PerformanceMetrics:
    """성능 지표"""
    timestamp: float
    fps: float
    gpu_utilization: float
    memory_usage_mb: float
    processing_latency: float
    error_rate: float
    throughput_score: float = 0  # 종합 처리량 점수
    
    def calculate_score(self) -> float:
        """종합 성능 점수 계산 (0-100)"""
        # FPS 점수 (30fps를 기준으로 정규화)
        fps_score = min(100, (self.fps / 30) * 100)
        
        # GPU 활용률 점수 (70-90%가 최적)
        if 70 <= self.gpu_utilization <= 90:
            gpu_score = 100
        elif self.gpu_utilization < 70:
            gpu_score = (self.gpu_utilization / 70) * 100
        else:
            gpu_score = max(0, 100 - (self.gpu_utilization - 90) * 2)
        
        # 메모리 효율 점수 (80% 이하가 안전)
        memory_percent = self.memory_usage_mb / 32000 * 100  # 32GB 기준
        if memory_percent <= 80:
            memory_score = 100
        else:
            memory_score = max(0, 100 - (memory_percent - 80) * 5)
        
        # 지연시간 점수 (50ms 이하가 좋음)
        if self.processing_latency <= 50:
            latency_score = 100
        else:
            latency_score = max(0, 100 - (self.processing_latency - 50))
        
        # 에러율 점수 (0%가 최적)
        error_score = max(0, 100 - self.error_rate * 100)
        
        # 가중 평균
        self.throughput_score = (
            fps_score * 0.3 +
            gpu_score * 0.25 +
            memory_score * 0.2 +
            latency_score * 0.15 +
            error_score * 0.1
        )
        
        return self.throughput_score


class AutoTuner:
    """
    자동 성능 튜너
    
    기능:
    - 실시간 성능 지표 기반 파라미터 자동 조정
    - 배치 크기 최적화
    - GPU 메모리 사용량 최적화
    - 인코딩 품질/속도 균형 조정
    - A/B 테스트를 통한 최적값 탐색
    - 성능 기록 및 분석
    """
    
    def __init__(self, 
                 mode: TuningMode = TuningMode.BALANCED,
                 tuning_interval: float = 30.0,
                 min_samples: int = 5):
        """
        Args:
            mode: 튜닝 모드
            tuning_interval: 튜닝 실행 간격 (초)
            min_samples: 튜닝 전 최소 샘플 수
        """
        self.mode = mode
        self.tuning_interval = tuning_interval
        self.min_samples = min_samples
        
        # 튜닝 파라미터들
        self.parameters = self._init_parameters()
        
        # 성능 히스토리 (최근 100개)
        self.performance_history = deque(maxlen=100)
        
        # 튜닝 기록
        self.tuning_history: List[Dict[str, Any]] = []
        
        # 최적 설정 기록
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: float = 0
        
        # 현재 튜닝 상태
        self.is_tuning = False
        self.last_tuning_time = 0
        self.tuning_round = 0
        
        # 안정성 체크
        self.consecutive_improvements = 0
        self.consecutive_degradations = 0
        
        logger.info(f"⚙️ AutoTuner 초기화 완료 (모드: {mode.value})")
    
    def _init_parameters(self) -> Dict[str, TuningParameter]:
        """튜닝 파라미터 초기화"""
        params = {}
        
        # 배치 크기
        params['batch_size'] = TuningParameter(
            name='batch_size',
            current_value=4,
            min_value=1,
            max_value=16,
            step_size=1,
            impact_weight=0.8
        )
        
        # CUDA 스트림 수
        params['cuda_streams'] = TuningParameter(
            name='cuda_streams',
            current_value=4,
            min_value=2,
            max_value=8,
            step_size=1,
            impact_weight=0.6
        )
        
        # 메모리 풀 크기 (MB)
        params['memory_pool_mb'] = TuningParameter(
            name='memory_pool_mb',
            current_value=2048,
            min_value=512,
            max_value=8192,
            step_size=512,
            impact_weight=0.4
        )
        
        # 인코딩 품질 (CRF 값)
        params['encoding_crf'] = TuningParameter(
            name='encoding_crf',
            current_value=23,
            min_value=18,
            max_value=28,
            step_size=1,
            impact_weight=0.3
        )
        
        # 프레임 스킵 비율 (0.0 = 스킵 없음)
        params['frame_skip_ratio'] = TuningParameter(
            name='frame_skip_ratio',
            current_value=0.0,
            min_value=0.0,
            max_value=0.5,
            step_size=0.1,
            impact_weight=0.5
        )
        
        return params
    
    def add_performance_sample(self, 
                             fps: float,
                             gpu_util: float,
                             memory_mb: float,
                             latency_ms: float,
                             error_rate: float = 0.0):
        """성능 샘플 추가"""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            fps=fps,
            gpu_utilization=gpu_util,
            memory_usage_mb=memory_mb,
            processing_latency=latency_ms,
            error_rate=error_rate
        )
        
        # 성능 점수 계산
        score = metrics.calculate_score()
        
        self.performance_history.append(metrics)
        
        logger.debug(f"📊 성능 샘플 추가: FPS={fps:.1f}, GPU={gpu_util:.1f}%, "
                    f"메모리={memory_mb:.0f}MB, 점수={score:.1f}")
        
        # 최고 성능 기록 업데이트
        if score > self.best_score:
            self.best_score = score
            self.best_config = self.get_current_config()
            logger.info(f"🏆 최고 성능 갱신: {score:.1f}점")
        
        # 자동 튜닝 트리거
        self._check_auto_tuning()
    
    def _check_auto_tuning(self):
        """자동 튜닝 필요성 체크"""
        current_time = time.time()
        
        # 간격 체크
        if current_time - self.last_tuning_time < self.tuning_interval:
            return
        
        # 최소 샘플 수 체크
        if len(self.performance_history) < self.min_samples:
            return
        
        # 튜닝 실행
        self.run_auto_tuning()
    
    def run_auto_tuning(self):
        """자동 튜닝 실행"""
        if self.is_tuning:
            return
        
        self.is_tuning = True
        self.last_tuning_time = time.time()
        self.tuning_round += 1
        
        logger.info(f"🔧 자동 튜닝 시작 (라운드 {self.tuning_round})")
        
        try:
            # 현재 성능 분석
            current_perf = self._analyze_current_performance()
            
            # 튜닝 전략 결정
            tuning_actions = self._decide_tuning_actions(current_perf)
            
            # 튜닝 실행
            improvements = self._apply_tuning_actions(tuning_actions)
            
            # 결과 기록
            self._record_tuning_result(current_perf, tuning_actions, improvements)
            
            logger.info(f"✅ 자동 튜닝 완료: {len(improvements)}개 개선")
            
        except Exception as e:
            logger.error(f"❌ 자동 튜닝 실패: {e}")
        finally:
            self.is_tuning = False
    
    def _analyze_current_performance(self) -> Dict[str, float]:
        """현재 성능 분석"""
        if not self.performance_history:
            return {}
        
        recent_samples = list(self.performance_history)[-self.min_samples:]
        
        # 평균 성능 지표 계산
        avg_fps = sum(m.fps for m in recent_samples) / len(recent_samples)
        avg_gpu = sum(m.gpu_utilization for m in recent_samples) / len(recent_samples)
        avg_memory = sum(m.memory_usage_mb for m in recent_samples) / len(recent_samples)
        avg_latency = sum(m.processing_latency for m in recent_samples) / len(recent_samples)
        avg_error = sum(m.error_rate for m in recent_samples) / len(recent_samples)
        avg_score = sum(m.calculate_score() for m in recent_samples) / len(recent_samples)
        
        # 성능 변화 트렌드 분석
        if len(recent_samples) >= 3:
            early_score = sum(m.calculate_score() for m in recent_samples[:2]) / 2
            late_score = sum(m.calculate_score() for m in recent_samples[-2:]) / 2
            trend = late_score - early_score
        else:
            trend = 0
        
        return {
            'avg_fps': avg_fps,
            'avg_gpu_util': avg_gpu,
            'avg_memory_mb': avg_memory,
            'avg_latency': avg_latency,
            'avg_error_rate': avg_error,
            'avg_score': avg_score,
            'trend': trend
        }
    
    def _decide_tuning_actions(self, current_perf: Dict[str, float]) -> List[Tuple[str, str]]:
        """튜닝 액션 결정"""
        actions = []
        
        avg_fps = current_perf.get('avg_fps', 0)
        avg_gpu = current_perf.get('avg_gpu_util', 0)
        avg_memory = current_perf.get('avg_memory_mb', 0)
        avg_latency = current_perf.get('avg_latency', 0)
        
        # 낮은 FPS 문제
        if avg_fps < 20:
            if avg_gpu < 70:  # GPU 활용률이 낮으면
                actions.append(('batch_size', 'increase'))
                actions.append(('cuda_streams', 'increase'))
            else:  # GPU가 포화되면
                actions.append(('encoding_crf', 'increase'))  # 품질 낮춰서 속도 향상
                actions.append(('frame_skip_ratio', 'increase'))  # 프레임 스킵
        
        # 높은 메모리 사용량
        memory_percent = avg_memory / 32000 * 100
        if memory_percent > 85:
            actions.append(('batch_size', 'decrease'))
            actions.append(('memory_pool_mb', 'decrease'))
        
        # 높은 지연시간
        if avg_latency > 100:
            actions.append(('batch_size', 'decrease'))
            actions.append(('frame_skip_ratio', 'increase'))
        
        # GPU 활용률이 너무 낮음
        if avg_gpu < 50:
            actions.append(('batch_size', 'increase'))
            actions.append(('cuda_streams', 'increase'))
        
        # 모드별 추가 액션
        if self.mode == TuningMode.AGGRESSIVE:
            if avg_fps > 40:  # 성능이 좋으면 품질 향상
                actions.append(('encoding_crf', 'decrease'))
        elif self.mode == TuningMode.CONSERVATIVE:
            # 안정성 우선 - 메모리 사용량 낮춤
            if memory_percent > 70:
                actions.append(('batch_size', 'decrease'))
        
        return actions
    
    def _apply_tuning_actions(self, actions: List[Tuple[str, str]]) -> List[str]:
        """튜닝 액션 적용"""
        improvements = []
        
        for param_name, action in actions:
            if param_name not in self.parameters:
                continue
            
            param = self.parameters[param_name]
            old_value = param.current_value
            
            if action == 'increase' and param.can_increase():
                param.increase()
                improvements.append(f"{param_name}: {old_value} → {param.current_value}")
                logger.info(f"📈 {param_name} 증가: {old_value} → {param.current_value}")
                
            elif action == 'decrease' and param.can_decrease():
                param.decrease()
                improvements.append(f"{param_name}: {old_value} → {param.current_value}")
                logger.info(f"📉 {param_name} 감소: {old_value} → {param.current_value}")
        
        return improvements
    
    def _record_tuning_result(self, 
                            current_perf: Dict[str, float],
                            actions: List[Tuple[str, str]],
                            improvements: List[str]):
        """튜닝 결과 기록"""
        result = {
            'round': self.tuning_round,
            'timestamp': time.time(),
            'performance_before': current_perf,
            'actions_taken': actions,
            'improvements': improvements,
            'config_after': self.get_current_config()
        }
        
        self.tuning_history.append(result)
        
        # 연속 개선/악화 추적
        if improvements:
            self.consecutive_improvements += 1
            self.consecutive_degradations = 0
        else:
            self.consecutive_degradations += 1
            self.consecutive_improvements = 0
    
    def get_current_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return {name: param.current_value for name, param in self.parameters.items()}
    
    def apply_config(self, config: Dict[str, Any]):
        """설정 적용"""
        for name, value in config.items():
            if name in self.parameters:
                self.parameters[name].current_value = value
                logger.info(f"⚙️ {name} 설정: {value}")
    
    def get_optimal_batch_size(self) -> int:
        """현재 상황에 맞는 최적 배치 크기 반환"""
        return self.parameters['batch_size'].current_value
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """현재 상황에 맞는 권장 설정"""
        if self.best_config and self.best_score > 80:
            return self.best_config.copy()
        
        return self.get_current_config()
    
    def reset_to_defaults(self):
        """기본값으로 초기화"""
        defaults = {
            'batch_size': 4,
            'cuda_streams': 4,
            'memory_pool_mb': 2048,
            'encoding_crf': 23,
            'frame_skip_ratio': 0.0
        }
        
        self.apply_config(defaults)
        logger.info("🔄 기본 설정으로 초기화")
    
    def save_tuning_history(self, file_path: str):
        """튜닝 기록 저장"""
        history_data = {
            'mode': self.mode.value,
            'tuning_rounds': self.tuning_round,
            'best_score': self.best_score,
            'best_config': self.best_config,
            'current_config': self.get_current_config(),
            'tuning_history': self.tuning_history
        }
        
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"💾 튜닝 기록 저장: {file_path}")
    
    def load_tuning_history(self, file_path: str):
        """튜닝 기록 로드"""
        try:
            with open(file_path, 'r') as f:
                history_data = json.load(f)
            
            self.tuning_round = history_data.get('tuning_rounds', 0)
            self.best_score = history_data.get('best_score', 0)
            self.best_config = history_data.get('best_config')
            self.tuning_history = history_data.get('tuning_history', [])
            
            # 최적 설정 적용
            if self.best_config:
                self.apply_config(self.best_config)
            
            logger.info(f"📂 튜닝 기록 로드: {file_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ 튜닝 기록 로드 실패: {e}")
    
    def get_tuning_stats(self) -> Dict[str, Any]:
        """튜닝 통계 반환"""
        return {
            'tuning_rounds': self.tuning_round,
            'best_score': self.best_score,
            'current_score': self.performance_history[-1].calculate_score() if self.performance_history else 0,
            'consecutive_improvements': self.consecutive_improvements,
            'consecutive_degradations': self.consecutive_degradations,
            'total_samples': len(self.performance_history),
            'is_tuning': self.is_tuning,
            'mode': self.mode.value
        }
    
    def print_tuning_summary(self):
        """튜닝 요약 출력"""
        stats = self.get_tuning_stats()
        config = self.get_current_config()
        
        print(f"""
⚙️ 자동 튜닝 요약:
   • 튜닝 라운드: {stats['tuning_rounds']}회
   • 최고 성능: {stats['best_score']:.1f}점
   • 현재 성능: {stats['current_score']:.1f}점
   • 연속 개선: {stats['consecutive_improvements']}회
   • 모드: {stats['mode']}

🎛️ 현재 설정:
   • 배치 크기: {config['batch_size']}
   • CUDA 스트림: {config['cuda_streams']}
   • 메모리 풀: {config['memory_pool_mb']}MB
   • 인코딩 CRF: {config['encoding_crf']}
   • 프레임 스킵: {config['frame_skip_ratio']:.1f}
        """)


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 AutoTuner 테스트 시작...")
    
    tuner = AutoTuner(mode=TuningMode.BALANCED, tuning_interval=5.0)
    
    # 가짜 성능 데이터 시뮬레이션
    import random
    
    for i in range(10):
        fps = random.uniform(15, 35)
        gpu_util = random.uniform(40, 90)
        memory_mb = random.uniform(8000, 25000)
        latency = random.uniform(30, 120)
        
        tuner.add_performance_sample(fps, gpu_util, memory_mb, latency)
        
        print(f"📊 샘플 {i+1}: FPS={fps:.1f}, GPU={gpu_util:.1f}%")
        time.sleep(1)
    
    tuner.print_tuning_summary()
    
    print("✅ 테스트 완료!")