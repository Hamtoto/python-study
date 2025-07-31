"""
TrackerMonitor 클래스
트래커 성능 모니터링 클래스
"""
from src.face_tracker.config import TRACKER_MONITOR_WINDOW_SIZE


class TrackerMonitor:
    """트래커 성능 모니터링 클래스"""
    
    def __init__(self, window_size=TRACKER_MONITOR_WINDOW_SIZE):
        self.success_history = []
        self.window_size = window_size
        
    def update(self, success):
        """성공/실패 기록 업데이트"""
        self.success_history.append(success)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
            
    def get_success_rate(self):
        """성공률 계산"""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)