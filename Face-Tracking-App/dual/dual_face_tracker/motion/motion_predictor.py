"""
Motion Prediction System for Dual Face Tracking
Phase B: Kalman Filter + 1-Euro Filter implementation

박스 위치 예측과 부드러운 스무딩으로 지터 제거 및 99.5% ID 일관성 달성
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

# filterpy 사용 가능한 경우 Kalman Filter, 없으면 직접 구현
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("⚠️ filterpy not available - using custom Kalman implementation")


@dataclass
class MotionState:
    """모션 상태 데이터 클래스"""
    position: Tuple[float, float]  # (center_x, center_y)
    size: Tuple[float, float]      # (width, height)
    velocity: Tuple[float, float]  # (vx, vy)
    acceleration: Tuple[float, float]  # (ax, ay)
    confidence: float
    timestamp: float


class KalmanTracker:
    """Kalman Filter 기반 박스 위치 예측"""
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int], track_id: str = "default"):
        """
        KalmanTracker 초기화
        
        Args:
            initial_bbox: (x1, y1, x2, y2) 초기 박스
            track_id: 추적 ID
        """
        self.track_id = track_id
        self.logger = logging.getLogger(__name__)
        
        # 박스를 중심점과 크기로 변환
        self.last_bbox = initial_bbox
        x1, y1, x2, y2 = initial_bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        self.initial_state = MotionState(
            position=(center_x, center_y),
            size=(width, height),
            velocity=(0.0, 0.0),
            acceleration=(0.0, 0.0),
            confidence=1.0,
            timestamp=time.time()
        )
        
        # Kalman Filter 초기화
        if HAS_FILTERPY:
            self.kf = self._init_filterpy_kalman(center_x, center_y, width, height)
        else:
            self.kf = self._init_custom_kalman(center_x, center_y, width, height)
        
        # 상태 추적
        self.motion_history = []
        self.prediction_count = 0
        self.update_count = 0
        
        self.logger.info(f"KalmanTracker({track_id}) 초기화: 중심=({center_x:.1f}, {center_y:.1f}), 크기=({width:.1f}x{height:.1f})")
    
    def _init_filterpy_kalman(self, cx: float, cy: float, w: float, h: float):
        """FilterPy 기반 Kalman Filter 초기화"""
        # 8차원 상태벡터: [cx, cy, w, h, vx, vy, vw, vh]
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 상태 전이 행렬 (Constant Velocity Model)
        dt = 1.0  # 1 프레임 시간
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],   # cx = cx + vx*dt
            [0, 1, 0, 0, 0, dt, 0, 0],   # cy = cy + vy*dt
            [0, 0, 1, 0, 0, 0, dt, 0],   # w = w + vw*dt
            [0, 0, 0, 1, 0, 0, 0, dt],   # h = h + vh*dt
            [0, 0, 0, 0, 1, 0, 0, 0],    # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],    # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],    # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],    # vh = vh
        ], dtype=np.float32)
        
        # 관측 행렬 (위치와 크기만 관측)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],    # 관측: cx
            [0, 1, 0, 0, 0, 0, 0, 0],    # 관측: cy
            [0, 0, 1, 0, 0, 0, 0, 0],    # 관측: w
            [0, 0, 0, 1, 0, 0, 0, 0],    # 관측: h
        ], dtype=np.float32)
        
        # 초기 상태
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # 초기 공분산 행렬
        kf.P *= 100.0  # 초기 불확실성
        
        # 관측 노이즈
        kf.R = np.diag([10.0, 10.0, 25.0, 25.0])  # 위치, 크기 관측 노이즈
        
        # 프로세스 노이즈
        q = Q_discrete_white_noise(dim=2, dt=dt, var=5.0)  # 2차원 white noise
        kf.Q = np.block([
            [q, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), q, np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), q*0.01, np.zeros((2, 2))],  # 크기 변화는 작음
            [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), q*0.01],
        ])
        
        return kf
    
    def _init_custom_kalman(self, cx: float, cy: float, w: float, h: float):
        """Custom Kalman Filter 초기화 (filterpy 없이)"""
        # 간단한 칼만 필터 상태
        return {
            'state': np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32),  # [cx,cy,w,h,vx,vy,vw,vh]
            'covariance': np.eye(8, dtype=np.float32) * 100.0,
            'process_noise': 5.0,
            'measurement_noise': 10.0
        }
    
    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """다음 프레임 박스 위치 예측
        
        Returns:
            (x1, y1, x2, y2) 예측된 박스 좌표
        """
        try:
            if HAS_FILTERPY:
                # FilterPy 예측
                self.kf.predict()
                cx, cy, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]
                vx, vy = self.kf.x[4], self.kf.x[5]
            else:
                # Custom 예측
                cx, cy, w, h, vx, vy = self._custom_predict()
            
            # 박스 좌표 변환
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
            predicted_bbox = (x1, y1, x2, y2)
            
            # 히스토리 업데이트
            motion_state = MotionState(
                position=(cx, cy),
                size=(w, h),
                velocity=(vx, vy),
                acceleration=(0.0, 0.0),  # 가속도는 별도 계산
                confidence=0.8,  # 예측이므로 신뢰도 낮음
                timestamp=time.time()
            )
            self.motion_history.append(motion_state)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)
            
            self.prediction_count += 1
            
            self.logger.debug(f"KalmanTracker({self.track_id}) 예측: "
                            f"중심=({cx:.1f},{cy:.1f}), 속도=({vx:.1f},{vy:.1f}) → {predicted_bbox}")
            
            return predicted_bbox
            
        except Exception as e:
            self.logger.error(f"KalmanTracker({self.track_id}) 예측 실패: {e}")
            # 폴백: 마지막 박스 반환
            return self.last_bbox
    
    def _custom_predict(self) -> Tuple[float, float, float, float, float, float]:
        """Custom Kalman 예측 (filterpy 없이)"""
        state = self.kf['state']
        
        # 상태 전이: 위치 += 속도 * dt
        dt = 1.0
        state[0] += state[4] * dt  # cx += vx
        state[1] += state[5] * dt  # cy += vy
        state[2] += state[6] * dt  # w += vw (보통 0)
        state[3] += state[7] * dt  # h += vh (보통 0)
        
        # 공분산 업데이트 (간단화)
        self.kf['covariance'] += np.eye(8) * self.kf['process_noise']
        
        return state[0], state[1], state[2], state[3], state[4], state[5]
    
    def update_with_detection(self, detected_bbox: Tuple[int, int, int, int]):
        """실제 검출로 필터 업데이트
        
        Args:
            detected_bbox: (x1, y1, x2, y2) 실제 검출된 박스
        """
        try:
            # 박스를 중심점과 크기로 변환
            x1, y1, x2, y2 = detected_bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            if HAS_FILTERPY:
                # FilterPy 업데이트
                measurement = np.array([cx, cy, w, h], dtype=np.float32)
                self.kf.update(measurement)
                
                # 속도 계산
                vx, vy = self.kf.x[4], self.kf.x[5]
            else:
                # Custom 업데이트
                vx, vy = self._custom_update(cx, cy, w, h)
            
            # 상태 히스토리 업데이트
            motion_state = MotionState(
                position=(cx, cy),
                size=(w, h),
                velocity=(vx, vy),
                acceleration=(0.0, 0.0),
                confidence=1.0,  # 실제 검출이므로 높은 신뢰도
                timestamp=time.time()
            )
            self.motion_history.append(motion_state)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)
            
            self.last_bbox = detected_bbox
            self.update_count += 1
            
            self.logger.debug(f"KalmanTracker({self.track_id}) 업데이트: "
                            f"중심=({cx:.1f},{cy:.1f}), 속도=({vx:.1f},{vy:.1f})")
            
        except Exception as e:
            self.logger.error(f"KalmanTracker({self.track_id}) 업데이트 실패: {e}")
    
    def _custom_update(self, cx: float, cy: float, w: float, h: float) -> Tuple[float, float]:
        """Custom Kalman 업데이트"""
        measurement = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # 칼만 게인 계산 (간단화)
        kalman_gain = 0.3  # 고정 게인
        
        # 상태 업데이트
        prev_state = self.kf['state'].copy()
        self.kf['state'] = (1 - kalman_gain) * self.kf['state'] + kalman_gain * measurement
        
        # 속도 계산 (위치 변화)
        dt = 1.0
        vx = (self.kf['state'][0] - prev_state[0]) / dt
        vy = (self.kf['state'][1] - prev_state[1]) / dt
        
        self.kf['state'][4] = vx
        self.kf['state'][5] = vy
        
        # 공분산 업데이트 (간단화)
        self.kf['covariance'] *= (1 - kalman_gain)
        
        return vx, vy
    
    def get_velocity(self) -> Tuple[float, float]:
        """현재 속도 벡터 반환
        
        Returns:
            (vx, vy) 픽셀/프레임 속도
        """
        if HAS_FILTERPY:
            return float(self.kf.x[4]), float(self.kf.x[5])
        else:
            return float(self.kf['state'][4]), float(self.kf['state'][5])
    
    def get_acceleration(self) -> Tuple[float, float]:
        """현재 가속도 계산"""
        if len(self.motion_history) < 2:
            return 0.0, 0.0
        
        # 최근 2개 속도로 가속도 계산
        current_vel = self.motion_history[-1].velocity
        prev_vel = self.motion_history[-2].velocity
        
        dt = self.motion_history[-1].timestamp - self.motion_history[-2].timestamp
        if dt <= 0:
            return 0.0, 0.0
        
        ax = (current_vel[0] - prev_vel[0]) / dt
        ay = (current_vel[1] - prev_vel[1]) / dt
        
        return ax, ay
    
    def get_motion_confidence(self) -> float:
        """모션 예측 신뢰도 계산"""
        if self.update_count == 0:
            return 0.0
        
        # 업데이트 빈도 기반 신뢰도
        update_ratio = self.update_count / (self.update_count + self.prediction_count)
        
        # 최근 모션 일관성
        consistency = 1.0
        if len(self.motion_history) >= 3:
            velocities = [state.velocity for state in self.motion_history[-3:]]
            velocity_std = np.std([v[0] for v in velocities]) + np.std([v[1] for v in velocities])
            consistency = max(0.0, 1.0 - velocity_std / 50.0)  # 50px/frame 기준
        
        return min(1.0, update_ratio * 0.7 + consistency * 0.3)


class OneEuroFilter:
    """1-Euro Filter로 박스 스무딩"""
    
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        """
        1-Euro Filter 초기화
        
        Args:
            freq: 프레임레이트 (Hz)
            mincutoff: 최소 차단 주파수
            beta: 속도 적응 계수
            dcutoff: 미분 차단 주파수
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        self.logger = logging.getLogger(__name__)
        
        # 각 좌표별 필터 상태 (x1, y1, x2, y2)
        self.filters = {}
        
        self.logger.info(f"OneEuroFilter 초기화: freq={freq}Hz, mincutoff={mincutoff}, beta={beta}")
    
    def _low_pass_filter(self, value: float, prev_value: float, alpha: float) -> float:
        """저주파 통과 필터"""
        return alpha * value + (1.0 - alpha) * prev_value
    
    def _compute_alpha(self, cutoff: float) -> float:
        """차단 주파수에서 알파 계산"""
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def filter_bbox(self, bbox: Tuple[int, int, int, int], track_id: str = "default") -> Tuple[int, int, int, int]:
        """박스 좌표 스무딩
        
        Args:
            bbox: (x1, y1, x2, y2) 입력 박스
            track_id: 추적 ID (여러 객체 지원)
            
        Returns:
            (x1, y1, x2, y2) 스무딩된 박스
        """
        try:
            current_time = time.time()
            
            # 첫 번째 프레임이면 초기화
            if track_id not in self.filters:
                self.filters[track_id] = {
                    'prev_values': np.array(bbox, dtype=np.float32),
                    'prev_derivatives': np.zeros(4, dtype=np.float32),
                    'prev_time': current_time
                }
                return bbox
            
            filter_state = self.filters[track_id]
            dt = current_time - filter_state['prev_time']
            
            # 주파수 업데이트 (실제 프레임레이트)
            if dt > 0:
                actual_freq = 1.0 / dt
                self.freq = 0.9 * self.freq + 0.1 * actual_freq  # EMA 업데이트
            
            # 현재 값과 이전 값
            current_values = np.array(bbox, dtype=np.float32)
            prev_values = filter_state['prev_values']
            
            # 미분 계산 (속도)
            derivatives = (current_values - prev_values) * self.freq if dt > 0 else np.zeros(4)
            
            # 미분 스무딩
            alpha_d = self._compute_alpha(self.dcutoff)
            smoothed_derivatives = self._low_pass_filter_array(derivatives, filter_state['prev_derivatives'], alpha_d)
            
            # 적응적 차단 주파수 계산
            cutoff = self.mincutoff + self.beta * np.abs(smoothed_derivatives)
            
            # 값 스무딩
            alpha = self._compute_alpha(cutoff.mean())  # 평균 차단주파수 사용
            smoothed_values = self._low_pass_filter_array(current_values, prev_values, alpha)
            
            # 상태 업데이트
            filter_state['prev_values'] = smoothed_values.copy()
            filter_state['prev_derivatives'] = smoothed_derivatives.copy()
            filter_state['prev_time'] = current_time
            
            # 정수로 변환
            result_bbox = tuple(int(v) for v in smoothed_values)
            
            self.logger.debug(f"OneEuroFilter({track_id}): {bbox} → {result_bbox} "
                            f"(alpha={alpha:.3f}, cutoff_avg={cutoff.mean():.2f})")
            
            return result_bbox
            
        except Exception as e:
            self.logger.error(f"OneEuroFilter({track_id}) 필터링 실패: {e}")
            return bbox
    
    def _low_pass_filter_array(self, values: np.ndarray, prev_values: np.ndarray, alpha: float) -> np.ndarray:
        """배열에 대한 저주파 통과 필터"""
        return alpha * values + (1.0 - alpha) * prev_values
    
    def reset(self, track_id: str = None):
        """필터 상태 리셋
        
        Args:
            track_id: 특정 추적 ID (None이면 전체 리셋)
        """
        if track_id is None:
            self.filters.clear()
            self.logger.info("OneEuroFilter: 모든 필터 상태 리셋")
        elif track_id in self.filters:
            del self.filters[track_id]
            self.logger.info(f"OneEuroFilter({track_id}): 필터 상태 리셋")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """필터 통계 정보"""
        return {
            'active_filters': len(self.filters),
            'track_ids': list(self.filters.keys()),
            'freq': self.freq,
            'mincutoff': self.mincutoff,
            'beta': self.beta
        }


class MotionPredictor:
    """통합 모션 예측 시스템 (Kalman + 1-Euro Filter 조합)"""
    
    def __init__(self, fps: float = 30.0, enable_kalman: bool = True, enable_euro: bool = True):
        """
        MotionPredictor 초기화
        
        Args:
            fps: 프레임레이트
            enable_kalman: Kalman 필터 활성화
            enable_euro: 1-Euro 필터 활성화
        """
        self.fps = fps
        self.enable_kalman = enable_kalman
        self.enable_euro = enable_euro
        self.logger = logging.getLogger(__name__)
        
        # 추적별 Kalman 필터들
        self.kalman_trackers = {}
        
        # 1-Euro 필터 (공유)
        self.euro_filter = OneEuroFilter(freq=fps) if enable_euro else None
        
        # 통계
        self.prediction_stats = defaultdict(lambda: {'predictions': 0, 'updates': 0, 'euro_smoothings': 0})
        
        self.logger.info(f"MotionPredictor 초기화: fps={fps}, Kalman={enable_kalman}, Euro={enable_euro}")
    
    def predict_next_bbox(self, track_id: str, current_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int, int, int]:
        """다음 프레임 박스 예측
        
        Args:
            track_id: 추적 ID ('A' 또는 'B')
            current_bbox: 현재 박스 (초기화용, 선택사항)
            
        Returns:
            예측된 박스 좌표
        """
        try:
            # Kalman 추적기 초기화 (필요시)
            if self.enable_kalman and track_id not in self.kalman_trackers:
                if current_bbox is None:
                    self.logger.warning(f"MotionPredictor({track_id}): 초기 박스 없이 예측 불가")
                    return (0, 0, 100, 100)  # 더미 박스
                
                self.kalman_trackers[track_id] = KalmanTracker(current_bbox, track_id)
            
            # Kalman 예측
            if self.enable_kalman and track_id in self.kalman_trackers:
                predicted_bbox = self.kalman_trackers[track_id].predict_next_position()
                self.prediction_stats[track_id]['predictions'] += 1
            else:
                predicted_bbox = current_bbox if current_bbox else (0, 0, 100, 100)
            
            # 1-Euro 스무딩 (예측값에 적용)
            if self.enable_euro and self.euro_filter:
                smoothed_bbox = self.euro_filter.filter_bbox(predicted_bbox, track_id)
                self.prediction_stats[track_id]['euro_smoothings'] += 1
                return smoothed_bbox
            else:
                return predicted_bbox
            
        except Exception as e:
            self.logger.error(f"MotionPredictor({track_id}) 예측 실패: {e}")
            return current_bbox if current_bbox else (0, 0, 100, 100)
    
    def update_with_detection(self, track_id: str, detected_bbox: Tuple[int, int, int, int]):
        """실제 검출로 업데이트
        
        Args:
            track_id: 추적 ID
            detected_bbox: 실제 검출된 박스
        """
        try:
            # Kalman 추적기 초기화 (필요시)
            if self.enable_kalman and track_id not in self.kalman_trackers:
                self.kalman_trackers[track_id] = KalmanTracker(detected_bbox, track_id)
                return
            
            # Kalman 업데이트
            if self.enable_kalman and track_id in self.kalman_trackers:
                self.kalman_trackers[track_id].update_with_detection(detected_bbox)
                self.prediction_stats[track_id]['updates'] += 1
            
            # 1-Euro 필터는 다음 예측에서 자동 업데이트됨
            
        except Exception as e:
            self.logger.error(f"MotionPredictor({track_id}) 업데이트 실패: {e}")
    
    def get_motion_info(self, track_id: str) -> Dict[str, Any]:
        """모션 정보 조회
        
        Args:
            track_id: 추적 ID
            
        Returns:
            모션 정보 딕셔너리
        """
        try:
            info = {
                'track_id': track_id,
                'has_kalman': track_id in self.kalman_trackers,
                'prediction_count': self.prediction_stats[track_id]['predictions'],
                'update_count': self.prediction_stats[track_id]['updates'],
                'smoothing_count': self.prediction_stats[track_id]['euro_smoothings']
            }
            
            if track_id in self.kalman_trackers:
                kalman = self.kalman_trackers[track_id]
                velocity = kalman.get_velocity()
                acceleration = kalman.get_acceleration()
                confidence = kalman.get_motion_confidence()
                
                info.update({
                    'velocity': velocity,
                    'acceleration': acceleration,
                    'motion_confidence': confidence,
                    'last_bbox': kalman.last_bbox
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"MotionPredictor({track_id}) 정보 조회 실패: {e}")
            return {'track_id': track_id, 'error': str(e)}
    
    def reset_tracker(self, track_id: str = None):
        """추적기 리셋
        
        Args:
            track_id: 특정 추적 ID (None이면 전체 리셋)
        """
        if track_id is None:
            # 전체 리셋
            self.kalman_trackers.clear()
            if self.euro_filter:
                self.euro_filter.reset()
            self.prediction_stats.clear()
            self.logger.info("MotionPredictor: 모든 추적기 리셋")
        else:
            # 특정 추적기 리셋
            if track_id in self.kalman_trackers:
                del self.kalman_trackers[track_id]
            if self.euro_filter:
                self.euro_filter.reset(track_id)
            if track_id in self.prediction_stats:
                del self.prediction_stats[track_id]
            self.logger.info(f"MotionPredictor({track_id}): 추적기 리셋")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 전체 통계"""
        stats = {
            'fps': self.fps,
            'kalman_enabled': self.enable_kalman,
            'euro_enabled': self.enable_euro,
            'active_trackers': len(self.kalman_trackers),
            'tracker_ids': list(self.kalman_trackers.keys())
        }
        
        if self.euro_filter:
            stats['euro_filter'] = self.euro_filter.get_filter_stats()
        
        # 각 추적기별 통계
        tracker_stats = {}
        for track_id in self.prediction_stats:
            tracker_stats[track_id] = dict(self.prediction_stats[track_id])
            if track_id in self.kalman_trackers:
                tracker_stats[track_id]['motion_confidence'] = self.kalman_trackers[track_id].get_motion_confidence()
        
        stats['per_tracker'] = tracker_stats
        
        return stats


# 통합 테스트 함수
def test_motion_prediction():
    """Motion Prediction 시스템 테스트"""
    print("🧪 Motion Prediction 시스템 테스트 시작...")
    
    # 1. KalmanTracker 테스트
    print("\n1. KalmanTracker 테스트")
    initial_bbox = (100, 100, 200, 200)
    kalman = KalmanTracker(initial_bbox, "test_A")
    
    print(f"   초기 박스: {initial_bbox}")
    
    # 몇 프레임 예측 및 업데이트
    for frame in range(5):
        # 예측
        predicted = kalman.predict_next_position()
        print(f"   프레임 {frame}: 예측 = {predicted}")
        
        # 실제 검출 시뮬레이션 (약간 이동)
        actual_x = initial_bbox[0] + frame * 5
        actual_y = initial_bbox[1] + frame * 2
        actual_bbox = (actual_x, actual_y, actual_x + 100, actual_y + 100)
        
        kalman.update_with_detection(actual_bbox)
        velocity = kalman.get_velocity()
        print(f"   프레임 {frame}: 실제 = {actual_bbox}, 속도 = ({velocity[0]:.1f}, {velocity[1]:.1f})")
    
    # 2. OneEuroFilter 테스트
    print("\n2. OneEuroFilter 테스트")
    euro = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.01)
    
    # 노이즈가 있는 박스 시퀀스
    noisy_boxes = [
        (100, 100, 200, 200),
        (102, 98, 202, 198),   # 노이즈
        (105, 101, 205, 201),
        (103, 99, 203, 199),   # 노이즈
        (108, 102, 208, 202),
    ]
    
    print("   노이즈 박스 → 스무딩 결과:")
    for i, noisy_box in enumerate(noisy_boxes):
        smoothed = euro.filter_bbox(noisy_box, "test_B")
        print(f"   프레임 {i}: {noisy_box} → {smoothed}")
    
    # 3. MotionPredictor 통합 테스트
    print("\n3. MotionPredictor 통합 테스트")
    predictor = MotionPredictor(fps=30.0, enable_kalman=True, enable_euro=True)
    
    # A, B 두 객체 추적
    for track_id in ['A', 'B']:
        initial = (50 + ord(track_id)*100, 50, 150 + ord(track_id)*100, 150)
        print(f"\n   추적 ID {track_id} 초기화: {initial}")
        
        # 첫 업데이트
        predictor.update_with_detection(track_id, initial)
        
        # 예측-업데이트 사이클
        for frame in range(3):
            # 예측
            predicted = predictor.predict_next_bbox(track_id)
            
            # 실제 검출 (시뮬레이션)
            offset = frame * 10
            actual = (initial[0] + offset, initial[1] + offset, 
                     initial[2] + offset, initial[3] + offset)
            
            predictor.update_with_detection(track_id, actual)
            
            motion_info = predictor.get_motion_info(track_id)
            print(f"   {track_id} 프레임 {frame}: 예측={predicted}, 실제={actual}, "
                  f"속도=({motion_info.get('velocity', (0,0))[0]:.1f}, {motion_info.get('velocity', (0,0))[1]:.1f})")
    
    # 4. 시스템 통계
    print("\n4. 시스템 통계")
    stats = predictor.get_system_stats()
    print(f"   활성 추적기: {stats['active_trackers']}개")
    print(f"   추적 ID: {stats['tracker_ids']}")
    print(f"   Kalman 활성화: {stats['kalman_enabled']}")
    print(f"   Euro 활성화: {stats['euro_enabled']}")
    
    print("\n✅ Motion Prediction 시스템 테스트 완료!")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 테스트 실행
    test_motion_prediction()