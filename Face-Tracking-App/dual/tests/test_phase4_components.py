#!/usr/bin/env python3
"""
Phase 4 컴포넌트 테스트 스크립트

Phase 4에서 구현된 모니터링, 복구, 최적화 시스템들을 테스트
"""

import sys
import asyncio
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dual_face_tracker.monitoring.hardware_monitor import HardwareMonitor
from dual_face_tracker.monitoring.performance_reporter import (
    PerformanceReporter, create_video_result
)
from dual_face_tracker.recovery.recovery_manager import StreamRecoveryManager
from dual_face_tracker.recovery.memory_manager import MemoryManager
from dual_face_tracker.recovery.error_handlers import ErrorHandlerRegistry
from dual_face_tracker.optimization.auto_tuner import AutoTuner, TuningMode
from dual_face_tracker.optimization.production_test_suite import ProductionTestSuite
from dual_face_tracker.utils.exceptions import GPUMemoryError, NVENCSessionError


class Phase4ComponentTester:
    """Phase 4 컴포넌트 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.test_dir = Path("phase4_test_results")
        self.test_dir.mkdir(exist_ok=True)
        
    async def run_all_tests(self):
        """모든 Phase 4 컴포넌트 테스트 실행"""
        print("🚀 Phase 4 컴포넌트 테스트 시작...")
        print("=" * 80)
        
        # 개별 컴포넌트 테스트
        await self.test_hardware_monitor()
        await self.test_performance_reporter()
        await self.test_recovery_manager()
        await self.test_memory_manager()
        await self.test_error_handlers()
        await self.test_auto_tuner()
        await self.test_production_suite()
        
        # 통합 테스트
        await self.test_integration()
        
        # 결과 요약
        self.print_test_summary()
        
        print("✅ Phase 4 컴포넌트 테스트 완료!")
    
    async def test_hardware_monitor(self):
        """하드웨어 모니터 테스트"""
        print("\n📊 [1/7] HardwareMonitor 테스트...")
        
        try:
            monitor = HardwareMonitor(
                log_dir=str(self.test_dir / "monitoring"),
                interval=1.0
            )
            
            # 5초간 모니터링
            monitor.start_monitoring()
            await asyncio.sleep(5)
            monitor.stop_monitoring()
            
            # 상태 확인
            status = monitor.get_current_status()
            
            self.test_results['hardware_monitor'] = {
                'success': True,
                'samples_collected': status.get('total_samples', 0),
                'gpu_util': status.get('gpu_util', 0),
                'memory_percent': status.get('memory_percent', 0)
            }
            
            print(f"   ✅ 모니터링 샘플: {status.get('total_samples', 0)}개")
            print(f"   📊 GPU 사용률: {status.get('gpu_util', 0)}%")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['hardware_monitor'] = {'success': False, 'error': str(e)}
    
    async def test_performance_reporter(self):
        """성능 리포터 테스트"""
        print("\n📋 [2/7] PerformanceReporter 테스트...")
        
        try:
            reporter = PerformanceReporter(
                report_dir=str(self.test_dir / "performance")
            )
            
            # 가짜 성능 데이터 추가
            stage_id = reporter.start_stage("테스트 디코딩")
            await asyncio.sleep(1)
            reporter.end_stage(stage_id, frames_processed=300)
            
            # 가짜 비디오 결과 추가
            video_result = create_video_result(
                video_path="test_video.mp4",
                video_size_mb=100.0,
                duration_seconds=60.0,
                total_frames=1800,
                processing_time=20.0,
                success=True,
                output_path="output.mp4",
                output_size_mb=80.0
            )
            reporter.add_video_result(video_result)
            
            # 하드웨어 통계 업데이트
            reporter.update_hardware_stats(75.0, 4096.0)
            
            # 리포트 생성
            text_file, json_file = reporter.save_reports()
            
            self.test_results['performance_reporter'] = {
                'success': True,
                'stages_recorded': len(reporter.pipeline_stages),
                'videos_recorded': len(reporter.video_results),
                'text_report': str(text_file),
                'json_report': str(json_file)
            }
            
            print(f"   ✅ 파이프라인 단계: {len(reporter.pipeline_stages)}개")
            print(f"   🎬 비디오 결과: {len(reporter.video_results)}개")
            print(f"   📄 리포트 생성: {text_file.name}")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['performance_reporter'] = {'success': False, 'error': str(e)}
    
    async def test_recovery_manager(self):
        """복구 매니저 테스트"""
        print("\n🛡️ [3/7] StreamRecoveryManager 테스트...")
        
        try:
            manager = StreamRecoveryManager()
            
            # 가짜 처리 함수
            async def mock_processor(video_path, **kwargs):
                if "fail" in video_path:
                    raise GPUMemoryError("Test GPU memory error")
                return {"success": True, "output": f"processed_{Path(video_path).name}"}
            
            # 성공 케이스
            result1 = await manager.process_with_recovery("success_video.mp4", mock_processor)
            
            # 실패 후 복구 케이스
            result2 = await manager.process_with_recovery("fail_video.mp4", mock_processor)
            
            stats = manager.get_recovery_stats()
            
            self.test_results['recovery_manager'] = {
                'success': True,
                'total_errors': stats['total_errors'],
                'total_recoveries': stats['total_recoveries'],
                'success_rate': stats['recovery_success_rate'],
                'current_batch_size': stats['current_batch_size']
            }
            
            print(f"   ✅ 총 에러: {stats['total_errors']}개")
            print(f"   🔧 복구 성공: {stats['total_recoveries']}개")
            print(f"   📊 복구 성공률: {stats['recovery_success_rate']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['recovery_manager'] = {'success': False, 'error': str(e)}
    
    async def test_memory_manager(self):
        """메모리 매니저 테스트"""
        print("\n💾 [4/7] MemoryManager 테스트...")
        
        try:
            manager = MemoryManager(threshold_percent=50.0, monitoring_interval=1.0)
            
            # 메모리 상태 체크
            info = manager.get_memory_info()
            
            # 정리 테스트
            cleanup_success = manager.check_and_cleanup(force=True)
            
            # 배치 크기 권장
            recommended_batch = manager.get_recommended_batch_size(
                base_batch_size=8, 
                memory_per_item_mb=500
            )
            
            stats = manager.get_memory_stats()
            
            self.test_results['memory_manager'] = {
                'success': True,
                'current_memory_mb': stats['current_allocated_mb'],
                'total_cleanups': stats['total_cleanups'],
                'recommended_batch': recommended_batch,
                'cleanup_success': cleanup_success
            }
            
            print(f"   ✅ 현재 GPU 메모리: {stats['current_allocated_mb']:.1f}MB")
            print(f"   🧹 총 정리 횟수: {stats['total_cleanups']}회")
            print(f"   📏 권장 배치 크기: {recommended_batch}")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['memory_manager'] = {'success': False, 'error': str(e)}
    
    async def test_error_handlers(self):
        """에러 핸들러 테스트"""
        print("\n🛡️ [5/7] ErrorHandlerRegistry 테스트...")
        
        try:
            registry = ErrorHandlerRegistry()
            
            # 다양한 에러 테스트
            test_errors = [
                GPUMemoryError("GPU memory allocation failed"),
                NVENCSessionError("NVENC session limit exceeded"),
                ValueError("General error for testing")
            ]
            
            handled_count = 0
            for error in test_errors:
                context = {"timestamp": time.time(), "batch_size": 4}
                result = await registry.handle_error(error, context)
                if result.get('handled', False):
                    handled_count += 1
            
            stats = registry.get_handling_stats()
            
            self.test_results['error_handlers'] = {
                'success': True,
                'total_errors': stats['total_errors'],
                'handled_errors': stats['handled_errors'],
                'success_rate': stats['success_rate'],
                'registered_handlers': stats['registered_handlers']
            }
            
            print(f"   ✅ 처리된 에러: {stats['handled_errors']}/{stats['total_errors']}개")
            print(f"   📊 처리 성공률: {stats['success_rate']:.1f}%")
            print(f"   🔧 등록된 핸들러: {stats['registered_handlers']}개")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['error_handlers'] = {'success': False, 'error': str(e)}
    
    async def test_auto_tuner(self):
        """자동 튜너 테스트"""
        print("\n⚙️ [6/7] AutoTuner 테스트...")
        
        try:
            tuner = AutoTuner(mode=TuningMode.BALANCED, tuning_interval=2.0)
            
            # 가짜 성능 데이터 추가
            import random
            for i in range(8):
                fps = random.uniform(15, 35)
                gpu_util = random.uniform(40, 90)
                memory_mb = random.uniform(8000, 25000)
                latency = random.uniform(30, 120)
                
                tuner.add_performance_sample(fps, gpu_util, memory_mb, latency)
                await asyncio.sleep(0.5)
            
            # 현재 설정 확인
            config = tuner.get_current_config()
            stats = tuner.get_tuning_stats()
            
            self.test_results['auto_tuner'] = {
                'success': True,
                'tuning_rounds': stats['tuning_rounds'],
                'best_score': stats['best_score'],
                'current_batch_size': config['batch_size'],
                'total_samples': stats['total_samples']
            }
            
            print(f"   ✅ 튜닝 라운드: {stats['tuning_rounds']}회")
            print(f"   🏆 최고 성능: {stats['best_score']:.1f}점")
            print(f"   📏 현재 배치 크기: {config['batch_size']}")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['auto_tuner'] = {'success': False, 'error': str(e)}
    
    async def test_production_suite(self):
        """프로덕션 테스트 스위트 테스트"""
        print("\n🧪 [7/7] ProductionTestSuite 테스트...")
        
        try:
            test_suite = ProductionTestSuite(
                test_data_dir=str(self.test_dir / "test_data"),
                results_dir=str(self.test_dir / "test_results")
            )
            
            # 기본 기능 시나리오만 테스트
            basic_scenario = test_suite.scenarios[0]  # basic_functionality
            result = await test_suite.run_scenario(basic_scenario)
            
            self.test_results['production_suite'] = {
                'success': result.success,
                'scenario_name': result.scenario_name,
                'videos_processed': result.videos_processed,
                'videos_successful': result.videos_successful,
                'success_rate': result.success_rate,
                'performance_score': result.performance_score
            }
            
            print(f"   ✅ 시나리오: {result.scenario_name}")
            print(f"   🎬 처리 성공: {result.videos_successful}/{result.videos_processed}")
            print(f"   📊 성능 점수: {result.performance_score:.1f}점")
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            self.test_results['production_suite'] = {'success': False, 'error': str(e)}
    
    async def test_integration(self):
        """통합 테스트"""
        print("\n🔗 통합 테스트...")
        
        try:
            # 여러 컴포넌트 동시 실행 테스트
            with HardwareMonitor(log_dir=str(self.test_dir / "integration"), interval=2.0) as monitor:
                recovery_manager = StreamRecoveryManager()
                memory_manager = MemoryManager()
                
                # 5초간 실행
                await asyncio.sleep(5)
                
                # 상태 확인
                monitor_status = monitor.get_current_status()
                recovery_stats = recovery_manager.get_recovery_stats()
                memory_stats = memory_manager.get_memory_stats()
                
                integration_success = (
                    monitor_status.get('total_samples', 0) > 0 and
                    recovery_stats['total_errors'] >= 0 and
                    memory_stats['current_allocated_mb'] >= 0
                )
                
                self.test_results['integration'] = {
                    'success': integration_success,
                    'monitor_samples': monitor_status.get('total_samples', 0),
                    'components_active': 3
                }
                
                print(f"   ✅ 컴포넌트 통합: {'성공' if integration_success else '실패'}")
                print(f"   📊 모니터링 샘플: {monitor_status.get('total_samples', 0)}개")
            
        except Exception as e:
            print(f"   ❌ 통합 테스트 에러: {e}")
            self.test_results['integration'] = {'success': False, 'error': str(e)}
    
    def print_test_summary(self):
        """테스트 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 Phase 4 컴포넌트 테스트 요약")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        print(f"📋 전체 테스트: {successful_tests}/{total_tests}개 성공")
        print(f"📊 성공률: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\n🔍 개별 테스트 결과:")
        
        for test_name, result in self.test_results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
            
            if not result.get('success', False) and 'error' in result:
                print(f"      └─ 에러: {result['error']}")
        
        # 성과 요약
        print(f"\n🏆 주요 성과:")
        
        if 'hardware_monitor' in self.test_results and self.test_results['hardware_monitor']['success']:
            samples = self.test_results['hardware_monitor']['samples_collected']
            print(f"   📊 하드웨어 모니터링: {samples}개 샘플 수집")
        
        if 'recovery_manager' in self.test_results and self.test_results['recovery_manager']['success']:
            rate = self.test_results['recovery_manager']['success_rate']
            print(f"   🛡️ 복구 시스템: {rate:.1f}% 성공률")
        
        if 'auto_tuner' in self.test_results and self.test_results['auto_tuner']['success']:
            score = self.test_results['auto_tuner']['best_score']
            print(f"   ⚙️ 자동 튜닝: {score:.1f}점 최고 성능")
        
        if 'production_suite' in self.test_results and self.test_results['production_suite']['success']:
            score = self.test_results['production_suite']['performance_score']
            print(f"   🧪 프로덕션 테스트: {score:.1f}점 성능 점수")
        
        print(f"\n💾 테스트 결과 저장: {self.test_dir}")


async def main():
    """메인 실행 함수"""
    print("🚀 Phase 4 컴포넌트 테스트 시작")
    print("이 테스트는 Phase 4에서 구현된 모든 운영화 컴포넌트를 검증합니다.")
    print()
    
    tester = Phase4ComponentTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())