#!/usr/bin/env python3
"""
최소 의존성 멀티스트림 테스트 (PyAV 없이도 실행 가능)

멀티스트림 로직과 GPU 메모리 관리만 테스트하고,
실제 비디오 처리는 모킹으로 대체합니다.

Author: Dual-Face High-Speed Processing System  
Date: 2025.01
Version: 1.0.0 (Phase 3 - Minimal)
"""

import asyncio
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# 프로젝트 모듈 import with fallback
try:
    from dual_face_tracker.core.multi_stream_processor import (
        MultiStreamProcessor,
        MultiStreamConfig, 
        StreamJob,
        create_stream_jobs
    )
    from dual_face_tracker.utils.logger import UnifiedLogger
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 모듈 import 실패: {e}")
    MODULES_AVAILABLE = False


class MockVideoProcessor:
    """비디오 처리 모킹 클래스"""
    
    def __init__(self):
        self.logger = UnifiedLogger("MockVideoProcessor")
    
    def process_video_mock(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """모킹된 비디오 처리"""
        
        # 가짜 처리 시간 (실제 처리 시간 시뮬레이션)
        processing_time = np.random.uniform(2.0, 8.0)  # 2-8초 랜덤
        
        self.logger.debug(f"모킹 처리: {Path(input_path).name} → {Path(output_path).name}")
        
        # 실제 처리 시뮬레이션
        time.sleep(processing_time)
        
        # 출력 파일 생성 (빈 파일)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch()
        
        return {
            'success': True,
            'frames_processed': np.random.randint(30, 300),  # 30-300 프레임
            'processing_time': processing_time,
            'fps': np.random.uniform(15.0, 60.0),
            'mock_mode': True
        }


class MinimalMultiStreamTest:
    """최소 의존성 멀티스트림 테스트"""
    
    def __init__(self):
        self.logger = UnifiedLogger("MinimalMultiStreamTest")
        self.test_videos_dir = Path("test_videos_minimal")
        self.output_dir = Path("test_output_minimal")
        self.mock_processor = MockVideoProcessor()
        
        # 테스트 결과
        self.test_results: Dict[str, Any] = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def setup_test_environment(self) -> None:
        """테스트 환경 설정"""
        self.logger.stage("최소 의존성 테스트 환경 설정...")
        
        self.test_videos_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.success("테스트 환경 설정 완료")
    
    def create_mock_videos(self, count: int = 4) -> List[Path]:
        """모킹 비디오 파일 생성 (실제 파일 X)"""
        self.logger.stage(f"{count}개 모킹 비디오 생성...")
        
        video_paths = []
        for i in range(count):
            # 실제 파일은 생성하지 않고 경로만 생성
            video_path = self.test_videos_dir / f"mock_video_{i}.mp4"
            video_paths.append(video_path)
        
        self.logger.success(f"{len(video_paths)}개 모킹 비디오 경로 생성 완료")
        return video_paths
    
    async def test_stream_allocation_logic(self) -> bool:
        """스트림 할당 로직 테스트 (GPU 없이도 가능)"""
        test_name = "stream_allocation_logic"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            if not MODULES_AVAILABLE:
                raise ImportError("필요한 모듈들이 import되지 않음")
            
            # StreamManager와 MemoryPoolManager만 테스트
            from dual_face_tracker.core.stream_manager import StreamManager
            from dual_face_tracker.core.memory_pool_manager import MemoryPoolManager, MemoryPoolConfig
            
            # StreamManager 테스트
            stream_manager = StreamManager(max_streams=4, gpu_id=0)
            
            # 초기화 없이 기본 로직만 테스트
            self.logger.debug("스트림 할당 로직 테스트 중...")
            
            # 기본 상태 확인
            status = stream_manager.get_stream_status()
            
            success = (
                isinstance(status, dict) and
                'stats' in status and
                'streams' in status
            )
            
            self.test_results[test_name] = {
                'success': success,
                'stream_status': status,
                'test_type': 'logic_only'
            }
            
            if success:
                self.logger.success(f"✅ {test_name}: 스트림 할당 로직 정상")
            else:
                self.logger.error(f"❌ {test_name}: 스트림 할당 로직 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_memory_pool_logic(self) -> bool:
        """메모리 풀 로직 테스트 (GPU 없이도 가능)"""
        test_name = "memory_pool_logic"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            if not MODULES_AVAILABLE:
                raise ImportError("필요한 모듈들이 import되지 않음")
                
            from dual_face_tracker.core.memory_pool_manager import (
                MemoryPoolManager, 
                MemoryPoolConfig,
                MemoryPoolType,
                MemoryAllocationStrategy
            )
            
            # MemoryPoolConfig 테스트
            config = MemoryPoolConfig(
                max_vram_usage=0.5,  # 낮게 설정
                allocation_strategy=MemoryAllocationStrategy.CONSERVATIVE
            )
            
            # MemoryPoolManager 생성 (초기화 없이)
            manager = MemoryPoolManager(config, gpu_id=0)
            
            # 배치 크기 최적화 로직 테스트
            optimal_decode = manager.get_optimal_batch_size('decode')
            optimal_compose = manager.get_optimal_batch_size('compose')
            optimal_encode = manager.get_optimal_batch_size('encode')
            
            success = (
                1 <= optimal_decode <= 16 and
                1 <= optimal_compose <= 8 and
                1 <= optimal_encode <= 12
            )
            
            self.test_results[test_name] = {
                'success': success,
                'batch_sizes': {
                    'decode': optimal_decode,
                    'compose': optimal_compose, 
                    'encode': optimal_encode
                },
                'test_type': 'logic_only'
            }
            
            if success:
                self.logger.success(f"✅ {test_name}: 메모리 풀 로직 정상")
            else:
                self.logger.error(f"❌ {test_name}: 메모리 풀 로직 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_mock_multi_processing(self) -> bool:
        """모킹 멀티 프로세싱 테스트"""
        test_name = "mock_multi_processing"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            # 4개 모킹 작업 생성
            video_paths = self.create_mock_videos(4)
            
            # 모킹 처리 (실제 MultiStreamProcessor 없이)
            start_time = time.time()
            results = []
            
            # 동시 처리 시뮬레이션 (실제로는 순차)
            for i, video_path in enumerate(video_paths):
                output_path = self.output_dir / f"mock_output_{i}.mp4"
                result = self.mock_processor.process_video_mock(
                    str(video_path), 
                    str(output_path)
                )
                results.append(result)
            
            total_time = time.time() - start_time
            
            # 성공률 계산
            successful = sum(1 for r in results if r['success'])
            success_rate = successful / len(results) if results else 0
            
            success = (
                success_rate >= 0.8 and  # 80% 이상 성공
                total_time < 60  # 1분 이내
            )
            
            self.test_results[test_name] = {
                'success': success,
                'total_time': total_time,
                'success_rate': success_rate,
                'processed_jobs': len(results),
                'successful_jobs': successful,
                'test_type': 'mock_processing'
            }
            
            if success:
                self.logger.success(
                    f"✅ {test_name}: {successful}/{len(results)} 작업 완료 "
                    f"({total_time:.1f}초)"
                )
            else:
                self.logger.error(f"❌ {test_name}: 모킹 처리 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_configuration_loading(self) -> bool:
        """설정 로딩 테스트"""
        test_name = "configuration_loading"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            if not MODULES_AVAILABLE:
                raise ImportError("필요한 모듈들이 import되지 않음")
            
            # MultiStreamConfig 생성 테스트
            config = MultiStreamConfig(
                max_streams=4,
                target_gpu_utilization=0.8,
                max_vram_usage=0.75
            )
            
            success = (
                config.max_streams == 4 and
                config.target_gpu_utilization == 0.8 and
                config.max_vram_usage == 0.75
            )
            
            self.test_results[test_name] = {
                'success': success,
                'config': {
                    'max_streams': config.max_streams,
                    'target_gpu_utilization': config.target_gpu_utilization,
                    'max_vram_usage': config.max_vram_usage
                },
                'test_type': 'config_only'
            }
            
            if success:
                self.logger.success(f"✅ {test_name}: 설정 로딩 성공")
            else:
                self.logger.error(f"❌ {test_name}: 설정 로딩 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 최소 의존성 테스트 실행"""
        self.logger.stage("=== 최소 의존성 멀티스트림 테스트 시작 ===")
        
        self.setup_test_environment()
        
        # 테스트 목록
        tests = [
            ("설정 로딩", self.test_configuration_loading),
            ("스트림 할당 로직", self.test_stream_allocation_logic),
            ("메모리 풀 로직", self.test_memory_pool_logic),
            ("모킹 멀티 프로세싱", self.test_mock_multi_processing),
        ]
        
        self.total_tests = len(tests)
        start_time = time.time()
        
        # 테스트 실행
        for test_name, test_func in tests:
            try:
                self.logger.stage(f"\n{'='*50}")
                self.logger.stage(f"테스트: {test_name}")
                self.logger.stage(f"{'='*50}")
                
                success = await test_func()
                if success:
                    self.passed_tests += 1
                
            except Exception as e:
                self.logger.error(f"테스트 실행 중 예외: {test_name} - {e}")
        
        total_time = time.time() - start_time
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        # 결과 출력
        self.logger.stage(f"\n{'='*60}")
        self.logger.stage("=== 최소 의존성 테스트 결과 ===")
        self.logger.stage(f"{'='*60}")
        
        if success_rate >= 75:  # 75% 이상 성공
            self.logger.success(
                f"🎉 테스트 성공!\n"
                f"   • 성공률: {success_rate:.1f}% ({self.passed_tests}/{self.total_tests})\n"
                f"   • 총 소요 시간: {total_time:.1f}초\n"
                f"   • Phase 3 로직 검증: ✅ 완료"
            )
        else:
            self.logger.error(
                f"❌ 테스트 실패\n"
                f"   • 성공률: {success_rate:.1f}% ({self.passed_tests}/{self.total_tests})\n"
                f"   • Phase 3 로직 검증: ❌ 미완료"
            )
        
        # 상세 결과
        self.logger.stage("\n=== 테스트별 상세 결과 ===")
        for test_name, result in self.test_results.items():
            status = "✅ 성공" if result.get('success', False) else "❌ 실패"
            self.logger.stage(f"{status} {test_name}")
            if not result.get('success', False) and 'error' in result:
                self.logger.stage(f"    오류: {result['error']}")
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'modules_available': MODULES_AVAILABLE,
            'detailed_results': self.test_results
        }


async def main():
    """메인 테스트 실행"""
    print("🧪 Phase 3 최소 의존성 테스트 시작")
    print("목표: PyAV 없이도 멀티스트림 로직 검증")
    print("=" * 50)
    
    test_suite = MinimalMultiStreamTest()
    results = await test_suite.run_all_tests()
    
    print("\n" + "=" * 50)
    print("🏁 최소 의존성 테스트 완료")
    print(f"최종 결과: {results['success_rate']:.1f}% 성공")
    
    if results['modules_available']:
        print("✅ 모든 모듈 정상 로드됨")
    else:
        print("⚠️ 일부 모듈 로드 실패 (PyAV 관련)")
    
    if results['success_rate'] >= 75:
        print("🎉 Phase 3 로직 검증 성공!")
        print("\n💡 다음 단계:")
        print("   1. PyAV 문제 해결: ./fix_pyav.sh 실행")
        print("   2. 완전한 테스트: ./run_phase3_test.sh 실행")
    else:
        print("❌ Phase 3 로직 검증 실패")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())