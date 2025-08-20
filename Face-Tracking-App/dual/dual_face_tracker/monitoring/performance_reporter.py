"""
성능 리포트 생성 시스템

비디오 처리 완료 후 상세한 성능 분석 리포트를 생성하는 시스템
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

from ..utils.logger import logger


@dataclass
class ProcessingStage:
    """파이프라인 단계별 성능 데이터"""
    name: str
    start_time: float
    end_time: float
    duration: float
    fps: float
    frames_processed: int
    errors: int = 0
    memory_peak_mb: float = 0
    
    @property
    def duration_seconds(self) -> float:
        return self.duration
    
    @property
    def duration_minutes(self) -> float:
        return self.duration / 60
    
    @property
    def percent_of_total(self) -> float:
        return 0  # 전체 시간 대비 비율 (외부에서 계산)


@dataclass 
class VideoProcessingResult:
    """단일 비디오 처리 결과"""
    video_path: str
    video_size_mb: float
    duration_seconds: float
    total_frames: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    output_size_mb: float = 0
    
    @property
    def fps_achieved(self) -> float:
        if self.processing_time > 0:
            return self.total_frames / self.processing_time
        return 0
    
    @property
    def real_time_factor(self) -> float:
        """실시간 대비 처리 속도 (1.0 = 실시간)"""
        if self.processing_time > 0:
            return self.duration_seconds / self.processing_time
        return 0


@dataclass
class BatchProcessingResult:
    """배치 처리 결과"""
    batch_id: int
    videos: List[VideoProcessingResult]
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_count(self) -> int:
        return sum(1 for v in self.videos if v.success)
    
    @property
    def total_videos(self) -> int:
        return len(self.videos)
    
    @property
    def success_rate(self) -> float:
        if self.total_videos > 0:
            return (self.success_count / self.total_videos) * 100
        return 0


class PerformanceReporter:
    """
    성능 리포트 생성기
    
    기능:
    - 비디오 처리 결과 수집 및 분석
    - 파이프라인 단계별 성능 측정
    - 배치 처리 성능 분석
    - 상세한 텍스트/JSON 리포트 생성
    - 성능 트렌드 분석
    """
    
    def __init__(self, report_dir: str = "performance_reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
        # 처리 결과 저장
        self.video_results: List[VideoProcessingResult] = []
        self.batch_results: List[BatchProcessingResult] = []
        self.pipeline_stages: List[ProcessingStage] = []
        
        # 세션 정보
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 성능 통계
        self.stats = {
            'total_videos_processed': 0,
            'total_videos_successful': 0,
            'total_processing_time': 0,
            'total_input_size_mb': 0,
            'total_output_size_mb': 0,
            'gpu_peak_utilization': 0,
            'memory_peak_usage_mb': 0
        }
        
        logger.info(f"📊 PerformanceReporter 초기화 (세션: {self.session_id})")
    
    def start_stage(self, stage_name: str) -> int:
        """파이프라인 단계 시작"""
        stage_id = len(self.pipeline_stages)
        stage = ProcessingStage(
            name=stage_name,
            start_time=time.time(),
            end_time=0,
            duration=0,
            fps=0,
            frames_processed=0
        )
        self.pipeline_stages.append(stage)
        
        logger.debug(f"🔄 단계 시작: {stage_name}")
        return stage_id
    
    def end_stage(self, stage_id: int, frames_processed: int = 0, errors: int = 0):
        """파이프라인 단계 종료"""
        if stage_id >= len(self.pipeline_stages):
            logger.warning(f"⚠️ 유효하지 않은 단계 ID: {stage_id}")
            return
        
        stage = self.pipeline_stages[stage_id]
        stage.end_time = time.time()
        stage.duration = stage.end_time - stage.start_time
        stage.frames_processed = frames_processed
        stage.errors = errors
        
        if stage.duration > 0:
            stage.fps = frames_processed / stage.duration
        
        logger.debug(f"✅ 단계 완료: {stage.name} ({stage.duration:.2f}초, {stage.fps:.1f} FPS)")
    
    def add_video_result(self, video_result: VideoProcessingResult):
        """비디오 처리 결과 추가"""
        self.video_results.append(video_result)
        
        # 통계 업데이트
        self.stats['total_videos_processed'] += 1
        self.stats['total_input_size_mb'] += video_result.video_size_mb
        self.stats['total_processing_time'] += video_result.processing_time
        
        if video_result.success:
            self.stats['total_videos_successful'] += 1
            self.stats['total_output_size_mb'] += video_result.output_size_mb
        
        logger.info(f"📹 비디오 결과 추가: {Path(video_result.video_path).name} "
                   f"({'✅' if video_result.success else '❌'})")
    
    def add_batch_result(self, batch_result: BatchProcessingResult):
        """배치 처리 결과 추가"""
        self.batch_results.append(batch_result)
        
        # 배치 내 비디오 결과들도 개별 추가
        for video_result in batch_result.videos:
            self.add_video_result(video_result)
        
        logger.info(f"📦 배치 결과 추가: Batch {batch_result.batch_id} "
                   f"({batch_result.success_count}/{batch_result.total_videos} 성공)")
    
    def update_hardware_stats(self, gpu_util: float, memory_mb: float):
        """하드웨어 통계 업데이트"""
        self.stats['gpu_peak_utilization'] = max(self.stats['gpu_peak_utilization'], gpu_util)
        self.stats['memory_peak_usage_mb'] = max(self.stats['memory_peak_usage_mb'], memory_mb)
    
    def calculate_pipeline_percentages(self):
        """파이프라인 단계별 시간 비율 계산"""
        total_time = sum(stage.duration for stage in self.pipeline_stages)
        
        if total_time > 0:
            for stage in self.pipeline_stages:
                stage.percent_of_total = (stage.duration / total_time) * 100
    
    def generate_text_report(self) -> str:
        """텍스트 형태 상세 리포트 생성"""
        self.calculate_pipeline_percentages()
        
        session_duration = time.time() - self.session_start_time
        total_input_time = sum(v.duration_seconds for v in self.video_results if v.success)
        
        # 성공률 계산
        success_rate = 0
        if self.stats['total_videos_processed'] > 0:
            success_rate = (self.stats['total_videos_successful'] / 
                          self.stats['total_videos_processed']) * 100
        
        # 평균 성능 계산
        avg_fps = 0
        avg_real_time_factor = 0
        if self.stats['total_videos_successful'] > 0:
            successful_videos = [v for v in self.video_results if v.success]
            avg_fps = sum(v.fps_achieved for v in successful_videos) / len(successful_videos)
            avg_real_time_factor = sum(v.real_time_factor for v in successful_videos) / len(successful_videos)
        
        report = f"""
================================================================================
📊 비디오 처리 성능 리포트 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
🎬 세션 정보:
   • 세션 ID: {self.session_id}
   • 총 실행 시간: {session_duration/60:.1f}분
   • 실제 처리 시간: {self.stats['total_processing_time']/60:.1f}분

📈 처리 결과:
   • 처리 비디오 수: {self.stats['total_videos_processed']}개
   • 성공한 비디오: {self.stats['total_videos_successful']}개
   • 성공률: {success_rate:.1f}%
   • 총 입력 비디오 시간: {total_input_time/60:.1f}분
   • 총 입력 크기: {self.stats['total_input_size_mb']/1024:.2f}GB
   • 총 출력 크기: {self.stats['total_output_size_mb']/1024:.2f}GB

⚡ 성능 지표:
   • 평균 처리 FPS: {avg_fps:.1f}
   • 평균 실시간 배속: {avg_real_time_factor:.1f}x
   • 처리량 효율성: {(total_input_time/60) / (self.stats['total_processing_time']/60):.1f}x

🖥️ 하드웨어 사용량:
   • 최대 GPU 사용률: {self.stats['gpu_peak_utilization']:.1f}%
   • 최대 메모리 사용량: {self.stats['memory_peak_usage_mb']/1024:.2f}GB
"""
        
        # 배치별 결과
        if self.batch_results:
            report += "\n📦 배치별 성능:\n"
            for batch in self.batch_results:
                report += (f"   • 배치 {batch.batch_id}: {batch.duration:.1f}초 "
                          f"({batch.success_count}/{batch.total_videos} 성공, "
                          f"{batch.success_rate:.1f}%)\n")
        
        # 파이프라인 단계별 분석
        if self.pipeline_stages:
            report += "\n🔄 파이프라인 단계별 성능:\n"
            for stage in self.pipeline_stages:
                report += (f"   • {stage.name}: {stage.duration:.1f}초 "
                          f"({stage.percent_of_total:.1f}%) - "
                          f"{stage.fps:.1f} FPS")
                if stage.errors > 0:
                    report += f" - ❌ {stage.errors}개 에러"
                report += "\n"
        
        # 개별 비디오 결과 (처음 10개만)
        if self.video_results:
            report += "\n📹 개별 비디오 결과:\n"
            for i, video in enumerate(self.video_results[:10]):
                status = "✅" if video.success else "❌"
                video_name = Path(video.video_path).name
                report += (f"   {status} {video_name}: {video.processing_time:.1f}초 "
                          f"({video.fps_achieved:.1f} FPS, {video.real_time_factor:.1f}x)\n")
            
            if len(self.video_results) > 10:
                report += f"   ... 및 {len(self.video_results) - 10}개 추가 비디오\n"
        
        # 성능 분석 및 권장사항
        report += "\n💡 성능 분석:\n"
        
        if avg_fps < 30:
            report += "   ⚠️ 낮은 FPS - GPU 사용률이나 배치 크기 최적화 검토 필요\n"
        elif avg_fps > 100:
            report += "   🟢 높은 FPS - 우수한 처리 성능\n"
        else:
            report += "   🟡 보통 FPS - 적절한 처리 성능\n"
        
        if success_rate < 95:
            report += "   ⚠️ 낮은 성공률 - 에러 복구 메커니즘 점검 필요\n"
        else:
            report += "   🟢 높은 성공률 - 안정적인 처리\n"
        
        if self.stats['gpu_peak_utilization'] < 50:
            report += "   ⚠️ 낮은 GPU 활용률 - 배치 크기나 파이프라인 최적화 검토\n"
        elif self.stats['gpu_peak_utilization'] > 95:
            report += "   🟡 높은 GPU 활용률 - 안정성 모니터링 필요\n"
        else:
            report += "   🟢 적절한 GPU 활용률\n"
        
        report += "================================================================================\n"
        
        return report
    
    def generate_json_report(self) -> Dict[str, Any]:
        """JSON 형태 구조화된 리포트 생성"""
        self.calculate_pipeline_percentages()
        
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.session_start_time).isoformat(),
                'duration_seconds': session_duration,
                'report_generated_at': datetime.now().isoformat()
            },
            'summary': {
                'total_videos_processed': self.stats['total_videos_processed'],
                'total_videos_successful': self.stats['total_videos_successful'],
                'success_rate_percent': (self.stats['total_videos_successful'] / 
                                       max(1, self.stats['total_videos_processed'])) * 100,
                'total_processing_time_seconds': self.stats['total_processing_time'],
                'total_input_size_mb': self.stats['total_input_size_mb'],
                'total_output_size_mb': self.stats['total_output_size_mb']
            },
            'performance': {
                'average_fps': sum(v.fps_achieved for v in self.video_results if v.success) / 
                              max(1, len([v for v in self.video_results if v.success])),
                'average_real_time_factor': sum(v.real_time_factor for v in self.video_results if v.success) / 
                                          max(1, len([v for v in self.video_results if v.success])),
                'gpu_peak_utilization_percent': self.stats['gpu_peak_utilization'],
                'memory_peak_usage_mb': self.stats['memory_peak_usage_mb']
            },
            'pipeline_stages': [asdict(stage) for stage in self.pipeline_stages],
            'batch_results': [asdict(batch) for batch in self.batch_results],
            'video_results': [asdict(video) for video in self.video_results]
        }
    
    def save_reports(self) -> tuple[Path, Path]:
        """리포트를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 텍스트 리포트 저장
        text_file = self.report_dir / f"performance_report_{self.session_id}.txt"
        text_report = self.generate_text_report()
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        # JSON 리포트 저장
        json_file = self.report_dir / f"performance_report_{self.session_id}.json"
        json_report = self.generate_json_report()
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 성능 리포트 저장:")
        logger.info(f"   📄 텍스트: {text_file}")
        logger.info(f"   📊 JSON: {json_file}")
        
        return text_file, json_file
    
    def print_summary(self):
        """콘솔에 요약 출력"""
        text_report = self.generate_text_report()
        print(text_report)
    
    def reset_session(self):
        """새 세션 시작 (데이터 초기화)"""
        self.video_results.clear()
        self.batch_results.clear()
        self.pipeline_stages.clear()
        
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 통계 초기화
        for key in self.stats:
            self.stats[key] = 0
        
        logger.info(f"🔄 새 성능 측정 세션 시작: {self.session_id}")


# 유틸리티 함수들
def create_video_result(video_path: str, 
                       video_size_mb: float,
                       duration_seconds: float,
                       total_frames: int,
                       processing_time: float,
                       success: bool,
                       error_message: Optional[str] = None,
                       output_path: Optional[str] = None,
                       output_size_mb: float = 0) -> VideoProcessingResult:
    """VideoProcessingResult 생성 헬퍼"""
    return VideoProcessingResult(
        video_path=video_path,
        video_size_mb=video_size_mb,
        duration_seconds=duration_seconds,
        total_frames=total_frames,
        processing_time=processing_time,
        success=success,
        error_message=error_message,
        output_path=output_path,
        output_size_mb=output_size_mb
    )


def create_batch_result(batch_id: int,
                       videos: List[VideoProcessingResult],
                       start_time: float,
                       end_time: float) -> BatchProcessingResult:
    """BatchProcessingResult 생성 헬퍼"""
    return BatchProcessingResult(
        batch_id=batch_id,
        videos=videos,
        start_time=start_time,
        end_time=end_time
    )


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 PerformanceReporter 테스트 시작...")
    
    reporter = PerformanceReporter()
    
    # 가짜 데이터 생성
    stage_id = reporter.start_stage("테스트 디코딩")
    time.sleep(1)
    reporter.end_stage(stage_id, frames_processed=100)
    
    video_result = create_video_result(
        video_path="test_video.mp4",
        video_size_mb=100.0,
        duration_seconds=60.0,
        total_frames=1800,
        processing_time=30.0,
        success=True,
        output_path="output.mp4",
        output_size_mb=80.0
    )
    reporter.add_video_result(video_result)
    
    reporter.update_hardware_stats(85.0, 2048.0)
    
    # 리포트 생성
    reporter.print_summary()
    reporter.save_reports()
    
    print("✅ 테스트 완료!")