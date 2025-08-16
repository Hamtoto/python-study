"""
프로덕션 테스트 스위트

실제 운영 환경을 시뮬레이션하는 종합적인 테스트 시스템
"""

import asyncio
import time
import random
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import logger
from ..monitoring.hardware_monitor import HardwareMonitor
from ..monitoring.performance_reporter import PerformanceReporter, create_video_result
from ..recovery.recovery_manager import StreamRecoveryManager
from ..recovery.memory_manager import MemoryManager


@dataclass
class TestScenario:
    """테스트 시나리오 정의"""
    name: str
    description: str
    video_count: int
    video_duration_minutes: float
    resolution: str
    concurrent_streams: int
    duration_minutes: float
    expected_errors: bool = False
    stress_test: bool = False


@dataclass
class TestResult:
    """테스트 결과"""
    scenario_name: str
    start_time: float
    end_time: float
    success: bool
    videos_processed: int
    videos_successful: int
    total_processing_time: float
    average_fps: float
    peak_gpu_util: float
    peak_memory_mb: float
    errors_encountered: List[str]
    performance_score: float
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60
    
    @property
    def success_rate(self) -> float:
        if self.videos_processed > 0:
            return (self.videos_successful / self.videos_processed) * 100
        return 0


class ProductionTestSuite:
    """
    프로덕션 테스트 스위트
    
    기능:
    - 다양한 시나리오의 부하 테스트
    - 장시간 안정성 테스트
    - 엣지 케이스 처리 테스트
    - 메모리 누수 감지
    - 성능 회귀 테스트
    - 복구 메커니즘 검증
    """
    
    def __init__(self, 
                 test_data_dir: str = "test_data",
                 results_dir: str = "test_results"):
        self.test_data_dir = Path(test_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 테스트 시나리오들
        self.scenarios = self._define_test_scenarios()
        
        # 테스트 결과
        self.test_results: List[TestResult] = []
        
        # 모니터링 컴포넌트
        self.hardware_monitor: Optional[HardwareMonitor] = None
        self.performance_reporter: Optional[PerformanceReporter] = None
        self.recovery_manager: Optional[StreamRecoveryManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # 테스트 상태
        self.current_test: Optional[str] = None
        self.abort_requested = False
        
        logger.info("🧪 ProductionTestSuite 초기화 완료")
    
    def _define_test_scenarios(self) -> List[TestScenario]:
        """테스트 시나리오 정의"""
        return [
            TestScenario(
                name="basic_functionality",
                description="기본 기능 테스트 (4개 비디오, 단일 배치)",
                video_count=4,
                video_duration_minutes=2.0,
                resolution="1080p",
                concurrent_streams=4,
                duration_minutes=5
            ),
            TestScenario(
                name="stress_test_multiple_batches",
                description="스트레스 테스트 (20개 비디오, 다중 배치)",
                video_count=20,
                video_duration_minutes=3.0,
                resolution="1080p",
                concurrent_streams=4,
                duration_minutes=15,
                stress_test=True
            ),
            TestScenario(
                name="high_resolution_test",
                description="고해상도 테스트 (4K 비디오)",
                video_count=4,
                video_duration_minutes=1.0,
                resolution="4K",
                concurrent_streams=2,  # 메모리 절약
                duration_minutes=8
            ),
            TestScenario(
                name="long_duration_stability",
                description="장시간 안정성 테스트 (30분 연속)",
                video_count=100,
                video_duration_minutes=1.0,
                resolution="720p",
                concurrent_streams=4,
                duration_minutes=30
            ),
            TestScenario(
                name="memory_pressure_test",
                description="메모리 압박 테스트 (대용량 배치)",
                video_count=8,
                video_duration_minutes=5.0,
                resolution="1080p",
                concurrent_streams=8,
                duration_minutes=12,
                stress_test=True
            ),
            TestScenario(
                name="error_recovery_test",
                description="에러 복구 테스트 (의도적 에러 발생)",
                video_count=10,
                video_duration_minutes=2.0,
                resolution="1080p",
                concurrent_streams=4,
                duration_minutes=8,
                expected_errors=True
            ),
            TestScenario(
                name="mixed_workload_test",
                description="혼합 워크로드 테스트 (다양한 해상도)",
                video_count=12,
                video_duration_minutes=2.5,
                resolution="mixed",
                concurrent_streams=4,
                duration_minutes=10
            )
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("🚀 프로덕션 테스트 스위트 시작")
        
        overall_start = time.time()
        self.test_results.clear()
        
        # 모니터링 시작
        self._start_monitoring()
        
        try:
            for scenario in self.scenarios:
                if self.abort_requested:
                    logger.warning("⚠️ 테스트 중단 요청됨")
                    break
                
                logger.info(f"📋 시나리오 시작: {scenario.name}")
                result = await self.run_scenario(scenario)
                self.test_results.append(result)
                
                # 시나리오 간 정리 시간
                if not self.abort_requested:
                    await self._cleanup_between_tests()
        
        except Exception as e:
            logger.error(f"❌ 테스트 스위트 실행 실패: {e}")
        
        finally:
            self._stop_monitoring()
        
        overall_time = time.time() - overall_start
        
        # 결과 분석 및 리포트 생성
        summary = self._generate_test_summary(overall_time)
        self._save_test_results(summary)
        
        logger.info(f"✅ 프로덕션 테스트 완료 ({overall_time/60:.1f}분)")
        
        return summary
    
    async def run_scenario(self, scenario: TestScenario) -> TestResult:
        """개별 시나리오 실행"""
        self.current_test = scenario.name
        start_time = time.time()
        
        # 결과 초기화
        result = TestResult(
            scenario_name=scenario.name,
            start_time=start_time,
            end_time=0,
            success=False,
            videos_processed=0,
            videos_successful=0,
            total_processing_time=0,
            average_fps=0,
            peak_gpu_util=0,
            peak_memory_mb=0,
            errors_encountered=[],
            performance_score=0
        )
        
        try:
            logger.info(f"🎬 시나리오 실행: {scenario.description}")
            
            # 테스트 비디오 생성/준비
            test_videos = await self._prepare_test_videos(scenario)
            
            # 시나리오별 실행
            if scenario.name == "error_recovery_test":
                await self._run_error_recovery_test(scenario, test_videos, result)
            elif scenario.name == "long_duration_stability":
                await self._run_stability_test(scenario, test_videos, result)
            elif scenario.name == "memory_pressure_test":
                await self._run_memory_pressure_test(scenario, test_videos, result)
            else:
                await self._run_standard_test(scenario, test_videos, result)
            
            result.success = True
            logger.info(f"✅ 시나리오 성공: {scenario.name}")
            
        except Exception as e:
            result.errors_encountered.append(str(e))
            logger.error(f"❌ 시나리오 실패: {scenario.name} - {e}")
        
        finally:
            result.end_time = time.time()
            self.current_test = None
        
        return result
    
    async def _prepare_test_videos(self, scenario: TestScenario) -> List[str]:
        """테스트 비디오 준비"""
        # 실제 구현에서는 테스트 비디오를 생성하거나 기존 비디오를 사용
        # 여기서는 가상의 비디오 경로를 반환
        
        videos = []
        for i in range(scenario.video_count):
            if scenario.resolution == "mixed":
                resolution = random.choice(["720p", "1080p", "4K"])
            else:
                resolution = scenario.resolution
            
            video_path = f"test_video_{resolution}_{i+1}.mp4"
            videos.append(video_path)
        
        logger.debug(f"📁 테스트 비디오 준비: {len(videos)}개")
        return videos
    
    async def _run_standard_test(self, 
                               scenario: TestScenario, 
                               test_videos: List[str], 
                               result: TestResult):
        """표준 테스트 실행"""
        start_time = time.time()
        
        # 가상의 비디오 처리 시뮬레이션
        for i, video_path in enumerate(test_videos):
            if self.abort_requested:
                break
            
            try:
                # 처리 시뮬레이션 (실제로는 비디오 처리 함수 호출)
                processing_time = await self._simulate_video_processing(
                    video_path, scenario.video_duration_minutes
                )
                
                result.videos_processed += 1
                result.videos_successful += 1
                result.total_processing_time += processing_time
                
                # 성능 지표 수집
                if self.hardware_monitor:
                    status = self.hardware_monitor.get_current_status()
                    result.peak_gpu_util = max(result.peak_gpu_util, status.get('gpu_util', 0))
                    result.peak_memory_mb = max(result.peak_memory_mb, status.get('memory_used_gb', 0) * 1024)
                
                logger.debug(f"✅ 비디오 처리 완료: {Path(video_path).name}")
                
            except Exception as e:
                result.errors_encountered.append(f"{video_path}: {str(e)}")
                result.videos_processed += 1
                logger.warning(f"⚠️ 비디오 처리 실패: {video_path} - {e}")
        
        # 평균 FPS 계산
        if result.total_processing_time > 0:
            total_frames = result.videos_successful * scenario.video_duration_minutes * 60 * 30  # 30fps 가정
            result.average_fps = total_frames / result.total_processing_time
        
        # 성능 점수 계산
        result.performance_score = self._calculate_performance_score(result, scenario)
    
    async def _run_error_recovery_test(self, 
                                     scenario: TestScenario, 
                                     test_videos: List[str], 
                                     result: TestResult):
        """에러 복구 테스트"""
        logger.info("🛡️ 에러 복구 테스트 시작")
        
        if not self.recovery_manager:
            self.recovery_manager = StreamRecoveryManager()
        
        for i, video_path in enumerate(test_videos):
            try:
                # 의도적으로 에러 발생 (일부 비디오)
                if i % 3 == 1:  # 매 3번째마다 에러 발생
                    raise Exception(f"Simulated error for {video_path}")
                
                processing_time = await self._simulate_video_processing(
                    video_path, scenario.video_duration_minutes
                )
                
                result.videos_processed += 1
                result.videos_successful += 1
                result.total_processing_time += processing_time
                
            except Exception as e:
                # 복구 매니저를 통한 처리
                recovery_result = await self.recovery_manager.attempt_recovery(
                    error=e, video_path=video_path, attempt=1
                )
                
                result.videos_processed += 1
                if recovery_result:
                    result.videos_successful += 1
                
                result.errors_encountered.append(f"{video_path}: {str(e)} (복구: {recovery_result})")
        
        # 복구 통계 출력
        if self.recovery_manager:
            self.recovery_manager.print_recovery_summary()
    
    async def _run_stability_test(self, 
                                scenario: TestScenario, 
                                test_videos: List[str], 
                                result: TestResult):
        """장시간 안정성 테스트"""
        logger.info("⏳ 장시간 안정성 테스트 시작")
        
        test_start = time.time()
        round_count = 0
        
        while time.time() - test_start < scenario.duration_minutes * 60:
            if self.abort_requested:
                break
            
            round_count += 1
            logger.info(f"🔄 안정성 테스트 라운드 {round_count}")
            
            # 비디오 배치 처리
            batch_videos = test_videos[:4]  # 4개씩 처리
            
            for video_path in batch_videos:
                try:
                    processing_time = await self._simulate_video_processing(
                        video_path, scenario.video_duration_minutes
                    )
                    
                    result.videos_processed += 1
                    result.videos_successful += 1
                    result.total_processing_time += processing_time
                    
                except Exception as e:
                    result.errors_encountered.append(f"Round {round_count} - {video_path}: {str(e)}")
                    result.videos_processed += 1
            
            # 메모리 누수 체크
            if self.memory_manager and round_count % 5 == 0:
                leak_detected = self.memory_manager.detect_memory_leak()
                if leak_detected:
                    result.errors_encountered.append(f"Memory leak detected at round {round_count}")
            
            await asyncio.sleep(1)  # 라운드 간 잠시 대기
        
        logger.info(f"✅ 안정성 테스트 완료: {round_count}라운드")
    
    async def _run_memory_pressure_test(self, 
                                      scenario: TestScenario, 
                                      test_videos: List[str], 
                                      result: TestResult):
        """메모리 압박 테스트"""
        logger.info("💾 메모리 압박 테스트 시작")
        
        if not self.memory_manager:
            self.memory_manager = MemoryManager(threshold_percent=60.0)  # 더 낮은 임계값
        
        # 대용량 배치로 처리하여 메모리 압박 유도
        batch_size = scenario.concurrent_streams
        
        for i in range(0, len(test_videos), batch_size):
            batch = test_videos[i:i+batch_size]
            
            logger.info(f"📦 메모리 압박 배치 {i//batch_size + 1}: {len(batch)}개 비디오")
            
            # 배치 동시 처리 시뮬레이션
            tasks = []
            for video_path in batch:
                task = self._simulate_video_processing(video_path, scenario.video_duration_minutes)
                tasks.append(task)
            
            try:
                processing_times = await asyncio.gather(*tasks)
                
                for j, processing_time in enumerate(processing_times):
                    result.videos_processed += 1
                    result.videos_successful += 1
                    result.total_processing_time += processing_time
                
            except Exception as e:
                result.errors_encountered.append(f"Batch {i//batch_size + 1}: {str(e)}")
                result.videos_processed += len(batch)
            
            # 강제 메모리 정리
            if self.memory_manager:
                self.memory_manager.check_and_cleanup(force=True)
    
    async def _simulate_video_processing(self, 
                                       video_path: str, 
                                       duration_minutes: float) -> float:
        """비디오 처리 시뮬레이션"""
        # 실제 처리 시간을 시뮬레이션 (실제로는 GPU 처리 함수 호출)
        
        # 해상도에 따른 처리 시간 조정
        if "4K" in video_path:
            base_time = duration_minutes * 60 * 0.8  # 4K는 실시간보다 빠름
        elif "1080p" in video_path:
            base_time = duration_minutes * 60 * 0.3  # 1080p는 실시간의 30%
        else:  # 720p
            base_time = duration_minutes * 60 * 0.2  # 720p는 실시간의 20%
        
        # 변동성 추가
        processing_time = base_time * random.uniform(0.8, 1.2)
        
        # 실제 대기 (테스트 시뮬레이션)
        await asyncio.sleep(min(processing_time / 10, 2.0))  # 시뮬레이션용 단축
        
        return processing_time
    
    def _calculate_performance_score(self, result: TestResult, scenario: TestScenario) -> float:
        """성능 점수 계산"""
        # 성공률 점수 (40%)
        success_score = result.success_rate
        
        # 속도 점수 (30%) - 실시간 대비 얼마나 빠른지
        expected_time = scenario.video_count * scenario.video_duration_minutes * 60
        if result.total_processing_time > 0:
            speed_ratio = expected_time / result.total_processing_time
            speed_score = min(100, speed_ratio * 20)  # 5배 빠르면 100점
        else:
            speed_score = 0
        
        # GPU 활용률 점수 (20%)
        if 70 <= result.peak_gpu_util <= 90:
            gpu_score = 100
        elif result.peak_gpu_util < 70:
            gpu_score = result.peak_gpu_util / 70 * 100
        else:
            gpu_score = max(0, 100 - (result.peak_gpu_util - 90) * 2)
        
        # 에러 처리 점수 (10%)
        if len(result.errors_encountered) == 0:
            error_score = 100
        else:
            error_score = max(0, 100 - len(result.errors_encountered) * 10)
        
        total_score = (
            success_score * 0.4 +
            speed_score * 0.3 +
            gpu_score * 0.2 +
            error_score * 0.1
        )
        
        return total_score
    
    def _start_monitoring(self):
        """모니터링 시작"""
        try:
            self.hardware_monitor = HardwareMonitor(
                log_dir=str(self.results_dir / "monitoring"),
                interval=10.0
            )
            self.hardware_monitor.start_monitoring()
            
            self.performance_reporter = PerformanceReporter(
                report_dir=str(self.results_dir / "performance")
            )
            
            if not self.memory_manager:
                self.memory_manager = MemoryManager()
            self.memory_manager.start_monitoring()
            
            logger.info("📊 모니터링 시작")
            
        except Exception as e:
            logger.warning(f"⚠️ 모니터링 시작 실패: {e}")
    
    def _stop_monitoring(self):
        """모니터링 중지"""
        try:
            if self.hardware_monitor:
                self.hardware_monitor.stop_monitoring()
            
            if self.performance_reporter:
                self.performance_reporter.save_reports()
            
            if self.memory_manager:
                self.memory_manager.stop_monitoring()
            
            logger.info("📊 모니터링 중지")
            
        except Exception as e:
            logger.warning(f"⚠️ 모니터링 중지 실패: {e}")
    
    async def _cleanup_between_tests(self):
        """테스트 간 정리"""
        logger.debug("🧹 테스트 간 정리 중...")
        
        # GPU 메모리 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 시스템 메모리 정리
        import gc
        gc.collect()
        
        # 잠시 대기
        await asyncio.sleep(2.0)
    
    def _generate_test_summary(self, overall_time: float) -> Dict[str, Any]:
        """테스트 요약 생성"""
        total_videos = sum(r.videos_processed for r in self.test_results)
        total_successful = sum(r.videos_successful for r in self.test_results)
        total_errors = sum(len(r.errors_encountered) for r in self.test_results)
        
        avg_performance_score = 0
        if self.test_results:
            avg_performance_score = sum(r.performance_score for r in self.test_results) / len(self.test_results)
        
        scenarios_passed = sum(1 for r in self.test_results if r.success)
        
        return {
            'test_summary': {
                'total_scenarios': len(self.scenarios),
                'scenarios_passed': scenarios_passed,
                'overall_success_rate': (scenarios_passed / len(self.scenarios)) * 100 if self.scenarios else 0,
                'total_execution_time_minutes': overall_time / 60,
                'average_performance_score': avg_performance_score
            },
            'video_processing': {
                'total_videos_processed': total_videos,
                'total_videos_successful': total_successful,
                'video_success_rate': (total_successful / total_videos) * 100 if total_videos > 0 else 0,
                'total_errors': total_errors
            },
            'scenario_results': [asdict(result) for result in self.test_results],
            'timestamp': time.time()
        }
    
    def _save_test_results(self, summary: Dict[str, Any]):
        """테스트 결과 저장"""
        # JSON 결과 저장
        import json
        
        results_file = self.results_dir / f"test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 텍스트 리포트 생성
        report_file = self.results_dir / f"test_report_{int(time.time())}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._format_test_report(summary))
        
        logger.info(f"💾 테스트 결과 저장: {results_file}")
        logger.info(f"📋 테스트 리포트: {report_file}")
    
    def _format_test_report(self, summary: Dict[str, Any]) -> str:
        """테스트 리포트 포맷팅"""
        report = f"""
================================================================================
🧪 프로덕션 테스트 스위트 결과 리포트
================================================================================
📅 실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
⏱️ 총 실행 시간: {summary['test_summary']['total_execution_time_minutes']:.1f}분

📊 전체 요약:
   • 실행 시나리오: {summary['test_summary']['scenarios_passed']}/{summary['test_summary']['total_scenarios']}개 성공
   • 전체 성공률: {summary['test_summary']['overall_success_rate']:.1f}%
   • 평균 성능 점수: {summary['test_summary']['average_performance_score']:.1f}점

🎬 비디오 처리 통계:
   • 처리된 비디오: {summary['video_processing']['total_videos_processed']}개
   • 성공한 비디오: {summary['video_processing']['total_videos_successful']}개
   • 비디오 성공률: {summary['video_processing']['video_success_rate']:.1f}%
   • 발생한 에러: {summary['video_processing']['total_errors']}개

📋 시나리오별 결과:
"""
        
        for result_data in summary['scenario_results']:
            report += f"""
   🎯 {result_data['scenario_name']}:
      • 성공: {'✅' if result_data['success'] else '❌'}
      • 처리 시간: {result_data['duration_minutes']:.1f}분
      • 비디오 성공률: {result_data['success_rate']:.1f}%
      • 성능 점수: {result_data['performance_score']:.1f}점
      • 에러 수: {len(result_data['errors_encountered'])}개"""
        
        report += "\n================================================================================\n"
        
        return report
    
    def abort_tests(self):
        """테스트 중단"""
        self.abort_requested = True
        logger.warning("⚠️ 테스트 중단 요청")
    
    def get_test_status(self) -> Dict[str, Any]:
        """현재 테스트 상태"""
        return {
            'current_test': self.current_test,
            'completed_tests': len(self.test_results),
            'total_tests': len(self.scenarios),
            'abort_requested': self.abort_requested,
            'is_running': self.current_test is not None
        }


if __name__ == "__main__":
    # 테스트 코드
    async def main():
        print("🧪 ProductionTestSuite 테스트 시작...")
        
        test_suite = ProductionTestSuite()
        
        # 기본 기능 테스트만 실행
        basic_scenario = test_suite.scenarios[0]  # basic_functionality
        result = await test_suite.run_scenario(basic_scenario)
        
        print(f"테스트 결과: {result.scenario_name}")
        print(f"성공: {result.success}")
        print(f"성공률: {result.success_rate:.1f}%")
        print(f"성능 점수: {result.performance_score:.1f}점")
        
        print("✅ 테스트 완료!")
    
    asyncio.run(main())