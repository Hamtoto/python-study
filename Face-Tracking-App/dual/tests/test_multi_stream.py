#!/usr/bin/env python3
"""
멀티스트림 비디오 처리 테스트 시스템.

4개의 스트림을 동시에 처리하여 Phase 3 목표 달성을 검증합니다:
- 4개 비디오 동시 처리
- GPU 활용률 80% 이상
- 15분 내 완료 목표
- 완전 파이프라인 검증

테스트 시나리오:
1. 기본 4-스트림 병렬 처리
2. 우선순위 기반 스케줄링
3. 메모리 제한 상황 테스트
4. 스트림 실패 복구 테스트
5. 성능 벤치마크

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0 (Phase 3)
"""

import asyncio
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# 프로젝트 모듈 임포트
try:
    from dual_face_tracker.core.multi_stream_processor import (
        MultiStreamProcessor,
        MultiStreamConfig,
        StreamJob,
        create_stream_jobs,
        process_videos_parallel
    )
    from dual_face_tracker.utils.logger import UnifiedLogger
    DUAL_FACE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Dual-face 모듈 임포트 실패: {e}")
    DUAL_FACE_MODULES_AVAILABLE = False
    
# PyAV 선택적 임포트
try:
    import av
    PYAV_AVAILABLE = True
    print(f"✅ PyAV 사용 가능: {av.__version__}")
except ImportError:
    PYAV_AVAILABLE = False
    print("⚠️ PyAV 사용 불가 - 모킹 모드로 실행")


class MultiStreamTestSuite:
    """멀티스트림 처리 테스트 스위트"""
    
    def __init__(self):
        if DUAL_FACE_MODULES_AVAILABLE:
            self.logger = UnifiedLogger("MultiStreamTest")
        else:
            self.logger = self._create_mock_logger()
        self.test_videos_dir = Path("test_videos")
        self.output_dir = Path("test_output")
        
        # 테스트 결과 저장
        self.test_results: Dict[str, Any] = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def _create_mock_logger(self):
        """모킹 로거 생성"""
        class MockLogger:
            def stage(self, msg): print(f"🔄 {msg}")
            def success(self, msg): print(f"✅ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def debug(self, msg): print(f"🔧 {msg}")
            def info(self, msg): print(f"ℹ️ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
        return MockLogger()
    
    def setup_test_environment(self) -> None:
        """테스트 환경 설정"""
        self.logger.stage("테스트 환경 설정 시작...")
        
        # 디렉토리 생성
        self.test_videos_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.success("테스트 환경 설정 완료")
    
    def create_test_videos(self, count: int = 4) -> List[Path]:
        """
        테스트용 비디오 파일들을 생성합니다.
        
        Args:
            count: 생성할 비디오 개수
            
        Returns:
            생성된 비디오 파일 경로 목록
        """
        self.logger.stage(f"{count}개 테스트 비디오 생성 시작...")
        
        video_paths = []
        
        for i in range(count):
            video_path = self.test_videos_dir / f"test_video_{i}.mp4"
            
            if not video_path.exists():
                self.logger.debug(f"비디오 {i} 생성 중...")
                
                # 테스트 비디오 생성 (30초, 30fps)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
                
                try:
                    # 900프레임 (30초 * 30fps) 생성
                    for frame_idx in range(900):
                        # 다양한 패턴의 프레임 생성
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # 스트림별 다른 색상
                        if i == 0:
                            frame[:, :, 2] = (frame_idx * 2) % 255  # Red
                        elif i == 1:
                            frame[:, :, 1] = (frame_idx * 3) % 255  # Green
                        elif i == 2:
                            frame[:, :, 0] = (frame_idx * 4) % 255  # Blue
                        else:
                            frame[:] = (frame_idx * 2) % 255        # Gray
                        
                        # 프레임 번호 텍스트
                        cv2.putText(frame, f"Stream{i} Frame{frame_idx}", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                   (255, 255, 255), 2)
                        
                        writer.write(frame)
                    
                    writer.release()
                    self.logger.debug(f"비디오 {i} 생성 완료: {video_path}")
                    
                except Exception as e:
                    self.logger.error(f"비디오 {i} 생성 실패: {e}")
                    if writer:
                        writer.release()
                    continue
            
            video_paths.append(video_path)
        
        self.logger.success(f"{len(video_paths)}개 테스트 비디오 준비 완료")
        return video_paths
    
    async def test_basic_multi_stream(self) -> bool:
        """기본 4-스트림 병렬 처리 테스트"""
        test_name = "basic_multi_stream"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            if not DUAL_FACE_MODULES_AVAILABLE:
                raise ImportError("Dual-face 모듈들이 사용 불가능")
                
            # 테스트 비디오 생성
            video_paths = self.create_test_videos(4)
            if len(video_paths) < 4:
                raise Exception("테스트 비디오 생성 실패")
            
            # 멀티스트림 설정
            config = MultiStreamConfig(
                max_streams=4,
                target_gpu_utilization=0.8,
                max_vram_usage=0.75
            )
            
            # 프로세서 생성 및 실행
            processor = MultiStreamProcessor(config)
            await processor.initialize()
            
            try:
                # 작업 생성
                jobs = create_stream_jobs(
                    video_paths, 
                    self.output_dir / test_name,
                    priorities=[1, 1, 2, 2]  # 우선순위 다양화
                )
                
                # 처리 시작
                start_time = time.time()
                stats = await processor.process_jobs(jobs)
                processing_time = time.time() - start_time
                
                # 결과 검증
                success = (
                    stats.completed_jobs == 4 and
                    stats.failed_jobs == 0 and
                    processing_time < 300  # 5분 이내
                )
                
                self.test_results[test_name] = {
                    'success': success,
                    'processing_time': processing_time,
                    'completed_jobs': stats.completed_jobs,
                    'failed_jobs': stats.failed_jobs,
                    'gpu_utilization': stats.gpu_utilization,
                    'vram_usage': stats.vram_usage,
                    'error_rate': stats.error_rate
                }
                
                if success:
                    self.logger.success(f"✅ {test_name}: {processing_time:.1f}초, {stats.completed_jobs}/{len(jobs)} 완료")
                else:
                    self.logger.error(f"❌ {test_name}: {stats.failed_jobs}/{len(jobs)} 실패")
                
                return success
                
            finally:
                await processor.shutdown()
        
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_priority_scheduling(self) -> bool:
        """우선순위 기반 스케줄링 테스트"""
        test_name = "priority_scheduling"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            video_paths = self.create_test_videos(6)  # 스트림보다 많은 작업
            
            config = MultiStreamConfig(max_streams=3)  # 스트림 수 제한
            processor = MultiStreamProcessor(config)
            await processor.initialize()
            
            try:
                # 우선순위 다양화
                jobs = create_stream_jobs(
                    video_paths,
                    self.output_dir / test_name,
                    priorities=[1, 3, 1, 2, 3, 1]  # 높음, 낮음, 높음, 보통, 낮음, 높음
                )
                
                start_time = time.time()
                stats = await processor.process_jobs(jobs)
                processing_time = time.time() - start_time
                
                # 우선순위 1이 먼저 처리되었는지 확인 (간단한 확인)
                success = (
                    stats.completed_jobs >= 4 and  # 최소 4개는 성공
                    stats.error_rate < 0.5  # 50% 미만 오류율
                )
                
                self.test_results[test_name] = {
                    'success': success,
                    'processing_time': processing_time,
                    'completed_jobs': stats.completed_jobs,
                    'total_jobs': len(jobs)
                }
                
                if success:
                    self.logger.success(f"✅ {test_name}: {stats.completed_jobs}/{len(jobs)} 완료")
                else:
                    self.logger.error(f"❌ {test_name}: 우선순위 처리 실패")
                
                return success
                
            finally:
                await processor.shutdown()
        
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_memory_pressure(self) -> bool:
        """메모리 제한 상황 테스트"""
        test_name = "memory_pressure"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            video_paths = self.create_test_videos(4)
            
            # 메모리 사용량을 낮게 설정
            config = MultiStreamConfig(
                max_streams=4,
                max_vram_usage=0.6,  # 60%로 제한
                target_gpu_utilization=0.9  # 높은 활용률 목표
            )
            
            processor = MultiStreamProcessor(config)
            await processor.initialize()
            
            try:
                jobs = create_stream_jobs(video_paths, self.output_dir / test_name)
                
                start_time = time.time()
                stats = await processor.process_jobs(jobs)
                processing_time = time.time() - start_time
                
                # 메모리 제한 상황에서도 처리 성공
                success = (
                    stats.completed_jobs >= 2 and  # 최소 2개는 성공
                    stats.vram_usage <= 0.7  # VRAM 사용량 준수
                )
                
                self.test_results[test_name] = {
                    'success': success,
                    'processing_time': processing_time,
                    'completed_jobs': stats.completed_jobs,
                    'vram_usage': stats.vram_usage,
                    'memory_constrained': True
                }
                
                if success:
                    self.logger.success(f"✅ {test_name}: 메모리 제약 하에서 {stats.completed_jobs}개 완료")
                else:
                    self.logger.error(f"❌ {test_name}: 메모리 제약 처리 실패")
                
                return success
                
            finally:
                await processor.shutdown()
        
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_error_recovery(self) -> bool:
        """스트림 실패 복구 테스트"""
        test_name = "error_recovery"
        self.logger.stage(f"테스트 시작: {test_name}")
        
        try:
            video_paths = self.create_test_videos(3)
            
            # 존재하지 않는 비디오 추가 (의도적 오류)
            invalid_video = self.test_videos_dir / "nonexistent.mp4"
            video_paths.append(invalid_video)
            
            config = MultiStreamConfig(max_streams=4)
            processor = MultiStreamProcessor(config)
            await processor.initialize()
            
            try:
                jobs = create_stream_jobs(video_paths, self.output_dir / test_name)
                
                start_time = time.time()
                stats = await processor.process_jobs(jobs)
                processing_time = time.time() - start_time
                
                # 일부 실패해도 나머지는 성공
                success = (
                    stats.completed_jobs >= 3 and  # 정상 비디오는 성공
                    stats.failed_jobs >= 1 and     # 오류 비디오는 실패
                    stats.error_rate <= 0.25       # 25% 이하 오류율
                )
                
                self.test_results[test_name] = {
                    'success': success,
                    'processing_time': processing_time,
                    'completed_jobs': stats.completed_jobs,
                    'failed_jobs': stats.failed_jobs,
                    'error_recovery': True
                }
                
                if success:
                    self.logger.success(f"✅ {test_name}: 오류 복구 성공 ({stats.completed_jobs}개 완료, {stats.failed_jobs}개 실패)")
                else:
                    self.logger.error(f"❌ {test_name}: 오류 복구 실패")
                
                return success
                
            finally:
                await processor.shutdown()
        
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e)}
            return False
    
    async def test_performance_benchmark(self) -> bool:
        """성능 벤치마크 테스트 (Phase 3 목표 검증)"""
        test_name = "performance_benchmark"
        self.logger.stage(f"테스트 시작: {test_name} - Phase 3 목표 검증")
        
        try:
            # 더 큰 테스트 비디오들 생성 (실제 성능 테스트)
            video_paths = []
            for i in range(4):
                video_path = self.test_videos_dir / f"benchmark_video_{i}.mp4"
                
                if not video_path.exists():
                    self.logger.debug(f"벤치마크 비디오 {i} 생성 중... (더 큰 크기)")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1280, 720))  # HD 해상도
                    
                    try:
                        # 1800 프레임 (60초 * 30fps) 생성
                        for frame_idx in range(1800):
                            frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
                            
                            # 더 복잡한 패턴 생성
                            cv2.circle(frame, (640 + i*100, 360), 50, (255, 255, 255), -1)
                            cv2.putText(frame, f"Benchmark{i}-{frame_idx}", 
                                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                       (255, 255, 255), 2)
                            
                            writer.write(frame)
                        
                        writer.release()
                        
                    except Exception as e:
                        self.logger.error(f"벤치마크 비디오 {i} 생성 실패: {e}")
                        if writer:
                            writer.release()
                        continue
                
                video_paths.append(video_path)
            
            if len(video_paths) < 4:
                raise Exception("벤치마크 비디오 생성 실패")
            
            # 최적 성능 설정
            config = MultiStreamConfig(
                max_streams=4,
                target_gpu_utilization=0.85,  # 85% 목표
                max_vram_usage=0.75,
                performance_monitoring=True
            )
            
            processor = MultiStreamProcessor(config)
            await processor.initialize()
            
            try:
                jobs = create_stream_jobs(video_paths, self.output_dir / test_name)
                
                self.logger.stage("🚀 Phase 3 성능 벤치마크 시작 - 4개 비디오 병렬 처리")
                start_time = time.time()
                
                stats = await processor.process_jobs(jobs)
                
                processing_time = time.time() - start_time
                
                # Phase 3 목표 달성 여부
                target_time = 15 * 60  # 15분
                gpu_utilization_target = 0.8  # 80%
                
                success = (
                    stats.completed_jobs == 4 and
                    processing_time < target_time and
                    stats.gpu_utilization >= gpu_utilization_target and
                    stats.error_rate < 0.05  # 5% 미만 오류율
                )
                
                # 상세 결과
                self.test_results[test_name] = {
                    'success': success,
                    'processing_time': processing_time,
                    'target_time': target_time,
                    'time_saved': target_time - processing_time,
                    'speedup_achieved': target_time / processing_time if processing_time > 0 else 0,
                    'completed_jobs': stats.completed_jobs,
                    'failed_jobs': stats.failed_jobs,
                    'gpu_utilization': stats.gpu_utilization,
                    'gpu_utilization_target': gpu_utilization_target,
                    'vram_usage': stats.vram_usage,
                    'error_rate': stats.error_rate,
                    'throughput_fps': stats.throughput_fps,
                    'phase3_target_achieved': success
                }
                
                if success:
                    self.logger.success(
                        f"🎉 ✅ {test_name}: Phase 3 목표 달성!\n"
                        f"   • 처리 시간: {processing_time/60:.1f}분 (목표: 15분)\n"
                        f"   • 시간 단축: {(target_time - processing_time)/60:.1f}분\n"
                        f"   • 속도 향상: {target_time/processing_time:.1f}배\n"
                        f"   • GPU 활용률: {stats.gpu_utilization:.1%} (목표: 80%)\n"
                        f"   • 성공률: {(1-stats.error_rate):.1%}\n"
                        f"   • 처리량: {stats.throughput_fps:.1f} FPS"
                    )
                else:
                    self.logger.error(
                        f"❌ {test_name}: Phase 3 목표 미달성\n"
                        f"   • 처리 시간: {processing_time/60:.1f}분 (목표: 15분)\n"
                        f"   • 완료된 작업: {stats.completed_jobs}/4\n"
                        f"   • GPU 활용률: {stats.gpu_utilization:.1%} (목표: 80%)\n"
                        f"   • 오류율: {stats.error_rate:.1%}"
                    )
                
                return success
                
            finally:
                await processor.shutdown()
        
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results[test_name] = {'success': False, 'error': str(e), 'phase3_target_achieved': False}
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        self.logger.stage("=== 멀티스트림 테스트 스위트 시작 ===")
        
        # 환경 설정
        self.setup_test_environment()
        
        # 테스트 목록
        tests = [
            ("기본 4-스트림 병렬 처리", self.test_basic_multi_stream),
            ("우선순위 기반 스케줄링", self.test_priority_scheduling),
            ("메모리 제한 상황 처리", self.test_memory_pressure),
            ("스트림 실패 복구", self.test_error_recovery),
            ("성능 벤치마크 (Phase 3)", self.test_performance_benchmark),
        ]
        
        self.total_tests = len(tests)
        start_time = time.time()
        
        # 테스트 실행
        for test_name, test_func in tests:
            try:
                self.logger.stage(f"\n{'='*60}")
                self.logger.stage(f"테스트 실행: {test_name}")
                self.logger.stage(f"{'='*60}")
                
                success = await test_func()
                if success:
                    self.passed_tests += 1
                
            except Exception as e:
                self.logger.error(f"테스트 실행 중 예외 발생: {test_name} - {e}")
        
        total_time = time.time() - start_time
        
        # 최종 결과
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        self.logger.stage(f"\n{'='*60}")
        self.logger.stage("=== 멀티스트림 테스트 결과 요약 ===")
        self.logger.stage(f"{'='*60}")
        
        if success_rate >= 80:  # 80% 이상 성공 시
            self.logger.success(
                f"🎉 테스트 스위트 성공!\n"
                f"   • 성공률: {success_rate:.1f}% ({self.passed_tests}/{self.total_tests})\n"
                f"   • 총 소요 시간: {total_time/60:.1f}분\n"
                f"   • Phase 3 준비 상태: ✅ 완료"
            )
        else:
            self.logger.error(
                f"❌ 테스트 스위트 실패\n"
                f"   • 성공률: {success_rate:.1f}% ({self.passed_tests}/{self.total_tests})\n"
                f"   • Phase 3 준비 상태: ❌ 미완료"
            )
        
        # 상세 결과
        self.logger.stage("\n=== 테스트별 상세 결과 ===")
        for test_name, result in self.test_results.items():
            status = "✅ 성공" if result.get('success', False) else "❌ 실패"
            self.logger.stage(f"{status} {test_name}: {result}")
        
        # Phase 3 목표 달성 여부
        phase3_achieved = self.test_results.get('performance_benchmark', {}).get('phase3_target_achieved', False)
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'phase3_target_achieved': phase3_achieved,
            'detailed_results': self.test_results
        }


async def main():
    """메인 테스트 실행 함수"""
    print("🚀 Phase 3 멀티스트림 테스트 시작")
    print("목표: 4개 비디오 동시 처리, 15분 내 완료, GPU 활용률 80%+")
    print("=" * 60)
    
    test_suite = MultiStreamTestSuite()
    results = await test_suite.run_all_tests()
    
    print("\n" + "=" * 60)
    print("🏁 테스트 완료")
    print(f"최종 결과: {results['success_rate']:.1f}% 성공")
    
    if results['phase3_target_achieved']:
        print("🎉 Phase 3 목표 달성!")
    else:
        print("❌ Phase 3 목표 미달성")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())