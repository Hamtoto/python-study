"""
비디오 처리 성능 리포트 생성기
"""
import os
import psutil
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import cpu_count

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
from src.face_tracker.utils.logging import logger
from src.face_tracker.config import BATCH_SIZE_ANALYZE, BATCH_SIZE_ID_TIMELINE


class PerformanceReporter:
    """비디오 처리 성능 측정 및 리포트 생성"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.start_time = time.time()
        
        # 서울 시간대로 시작 시간 기록
        if PYTZ_AVAILABLE:
            seoul_tz = pytz.timezone('Asia/Seoul')
            self.start_datetime = datetime.now(seoul_tz)
        else:
            # UTC+9 시간대 생성 (서울)
            seoul_tz = timezone(timedelta(hours=9))
            self.start_datetime = datetime.now(seoul_tz)
        
        self.stages = {}
        self.cpu_cores_used = 0
        self.total_frames = 0
        self.segments_count = 0
        self.threshold_optimization = None  # 임계값 최적화 정보
        
        # DUAL_SPLIT 모드 전용 통계
        self.dual_split_stats = None
        
    def start_stage(self, stage_name: str):
        """단계 시작 시간 기록"""
        self.stages[stage_name] = {'start': time.time()}
        
    def end_stage(self, stage_name: str, **metrics):
        """단계 종료 시간 및 추가 메트릭 기록"""
        if stage_name in self.stages:
            self.stages[stage_name]['end'] = time.time()
            self.stages[stage_name]['duration'] = self.stages[stage_name]['end'] - self.stages[stage_name]['start']
            self.stages[stage_name].update(metrics)
    
    def set_processing_info(self, total_frames: int, segments_count: int, cpu_cores: int):
        """처리 정보 설정"""
        self.total_frames = total_frames
        self.segments_count = segments_count
        self.cpu_cores_used = cpu_cores
    
    def set_threshold_optimization_info(self, applied_threshold: float, 
                                      original_threshold: float, 
                                      confidence: str, 
                                      statistics: dict):
        """임계값 최적화 정보 설정"""
        self.threshold_optimization = {
            'applied_threshold': applied_threshold,
            'original_threshold': original_threshold,
            'confidence': confidence,
            'statistics': statistics,
            'improvement_percent': abs(applied_threshold - original_threshold) / original_threshold * 100
        }
    
    def set_dual_split_stats(self, processed_frames: int, total_frames: int, segments: int):
        """DUAL_SPLIT 모드 전용 통계 설정"""
        self.dual_split_stats = {
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'segments': segments
        }
        # 기본 처리 정보도 업데이트
        self.total_frames = total_frames
        self.segments_count = segments
    
    def generate_report(self):
        """상세 성능 리포트 생성 및 출력"""
        total_time = time.time() - self.start_time
        total_minutes = total_time / 60
        total_hours = total_time / 3600
        
        # 종료 시간 계산 (서울 시간대)
        if PYTZ_AVAILABLE:
            seoul_tz = pytz.timezone('Asia/Seoul')
            end_datetime = datetime.now(seoul_tz)
        else:
            # UTC+9 시간대 생성 (서울)
            seoul_tz = timezone(timedelta(hours=9))
            end_datetime = datetime.now(seoul_tz)
        
        # 리포트 헤더
        report_lines = [
            "",
            "=" * 80,
            f"📊 {self.video_name} 처리 완료 리포트",
            "=" * 80,
        ]
        
        # 시간 정보
        report_lines.extend([
            f"🕐 시작 시간: {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"🕐 종료 시간: {end_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            ""
        ])
        
        # 기본 정보
        report_lines.extend([
            f"🎬 영상: {self.video_name}",
            f"⏱️  총 처리시간: {self._format_duration(total_time)}",
            f"🖼️  총 프레임 수: {self.total_frames:,}",
            f"📦 생성된 세그먼트: {self.segments_count}개",
            f"🖥️  사용된 CPU 코어: {self.cpu_cores_used}/{cpu_count()}개",
            ""
        ])
        
        # 배치 크기 정보
        report_lines.extend([
            "⚙️ 설정 정보:",
            f"   • 얼굴 분석 배치 크기: {BATCH_SIZE_ANALYZE}",
            f"   • 얼굴 인식 배치 크기: {BATCH_SIZE_ID_TIMELINE}",
            ""
        ])
        
        # 단계별 성능
        if self.stages:
            report_lines.append("📈 단계별 성능:")
            total_stage_time = 0
            
            for stage_name, metrics in self.stages.items():
                if 'duration' in metrics:
                    duration = metrics['duration']
                    total_stage_time += duration
                    percentage = (duration / total_time) * 100
                    
                    stage_line = f"   • {stage_name}: {self._format_duration(duration)} ({percentage:.1f}%)"
                    
                    # 추가 메트릭 표시
                    if 'frames' in metrics:
                        fps = metrics['frames'] / duration if duration > 0 else 0
                        stage_line += f" - {fps:.1f} FPS"
                    
                    if 'batch_size' in metrics:
                        stage_line += f" - 배치: {metrics['batch_size']}"
                        
                    report_lines.append(stage_line)
            
            # 기타 시간 (초기화, 정리 등)
            other_time = total_time - total_stage_time
            if other_time > 0:
                other_percentage = (other_time / total_time) * 100
                report_lines.append(f"   • 기타 (초기화/정리): {self._format_duration(other_time)} ({other_percentage:.1f}%)")
            
            report_lines.append("")
        
        # 성능 평가
        fps_overall = self.total_frames / total_time if total_time > 0 else 0
        report_lines.extend([
            "🎯 성능 지표:",
            f"   • 총 처리 시간: {self._format_duration(total_time)}",
            f"   • 전체 처리 속도: {fps_overall:.1f} FPS",
            f"   • 프레임당 평균 시간: {(total_time/self.total_frames)*1000:.2f}ms" if self.total_frames > 0 else "   • 프레임당 평균 시간: N/A",
        ])
        
        # 메모리 사용량 (현재 시점)
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            report_lines.append(f"   • 메모리 사용량: {memory_mb:.1f} MB")
        except:
            pass
        
        # 임계값 최적화 정보
        if self.threshold_optimization:
            opt = self.threshold_optimization
            report_lines.extend([
                "",
                "🔧 동적 임계값 최적화:",
                f"   • 계산된 임계값: {opt['applied_threshold']:.3f}",
                f"   • 기본 임계값: {opt['original_threshold']:.3f}",
                f"   • 최적화 신뢰도: {opt['confidence']}",
                f"   • 개선율: {opt['improvement_percent']:+.1f}%"
            ])
            
            if 'statistics' in opt and opt['statistics']:
                stats = opt['statistics']
                if 'same_person' in stats and 'different_person' in stats:
                    same = stats['same_person']
                    diff = stats['different_person']
                    report_lines.extend([
                        f"   • 같은 사람 유사도: {same['mean']:.3f} ± {same['std']:.3f} ({same['count']}개)",
                        f"   • 다른 사람 유사도: {diff['mean']:.3f} ± {diff['std']:.3f} ({diff['count']}개)"
                    ])
        
        report_lines.extend([
            "",
            "=" * 80,
            ""
        ])
        
        # 콘솔에 출력
        for line in report_lines:
            print(line)
        
        # 로그에도 간단한 요약 기록
        logger.success(f"{self.video_name} 리포트: {self._format_duration(total_time)}, {fps_overall:.1f}FPS, {self.segments_count}세그먼트")
    
    def _format_duration(self, seconds: float) -> str:
        """시간을 읽기 쉬운 형태로 포맷"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}분 {secs:.1f}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}시간 {minutes}분 {secs:.1f}초"


# 전역 리포터 인스턴스 (현재 처리 중인 비디오용)
current_reporter = None

def start_video_report(video_name: str) -> PerformanceReporter:
    """비디오 처리 리포트 시작"""
    global current_reporter
    current_reporter = PerformanceReporter(video_name)
    return current_reporter

def get_current_reporter() -> PerformanceReporter:
    """현재 리포터 인스턴스 반환"""
    return current_reporter