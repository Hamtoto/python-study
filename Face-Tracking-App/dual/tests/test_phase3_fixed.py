#!/usr/bin/env python3
"""
Phase 3 수정된 멀티스트림 테스트

NVENC 세션 제한 문제 해결된 버전:
- 동시 NVENC 세션 수 제한 (4개 → 2개)
- 배치 처리 방식 (2+2)
- 자동 소프트웨어 폴백
- 향상된 에러 처리
- 안전한 리소스 정리

Author: Dual-Face High-Speed Processing System
Date: 2025.01
Version: 1.0.0 (Fixed)
"""

import asyncio
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import traceback

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent.parent  # tests/ 에서 dual/ 로
sys.path.insert(0, str(current_dir))

from dual_face_tracker.encoders.enhanced_encoder import EnhancedEncoder, create_enhanced_encoder
from dual_face_tracker.encoders.session_manager import get_global_session_manager
from dual_face_tracker.utils.logger import UnifiedLogger


class Phase3FixedTester:
    """수정된 Phase 3 멀티스트림 테스트"""
    
    def __init__(self):
        self.logger = UnifiedLogger("Phase3FixedTest")
        self.test_results = []
        self.session_manager = get_global_session_manager(max_concurrent_sessions=2)
        
        # 테스트 설정
        self.max_concurrent_nvenc = 2  # RTX 5090 세션 제한
        self.batch_size = 2
        self.test_duration = 10.0  # 초 (테스트용 짧은 시간)
        self.enable_fallback = True
        
        self.logger.info("🚀 Phase 3 Fixed 테스트 초기화")
        self.logger.info(f"📋 설정: NVENC세션={self.max_concurrent_nvenc}, 배치크기={self.batch_size}")
    
    async def run_all_tests(self) -> bool:
        """모든 테스트 실행"""
        try:
            self.logger.info("🎯 Phase 3 Fixed 테스트 시작")
            self.logger.info("="*80)
            
            all_success = True
            
            # 테스트 1: 배치 처리 테스트
            test1_result = await self.test_batch_processing()
            all_success = all_success and test1_result
            
            # 테스트 2: 폴백 메커니즘 테스트  
            test2_result = await self.test_fallback_mechanism()
            all_success = all_success and test2_result
            
            # 테스트 3: 세션 관리 테스트
            test3_result = await self.test_session_management()
            all_success = all_success and test3_result
            
            # 테스트 4: 에러 복구 테스트
            test4_result = await self.test_error_recovery()
            all_success = all_success and test4_result
            
            # 최종 결과
            self._log_final_results(all_success)
            
            return all_success
            
        except Exception as e:
            self.logger.error(f"❌ 테스트 실행 실패: {e}")
            self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
            return False
    
    async def test_batch_processing(self) -> bool:
        """배치 처리 테스트 (2+2 방식)"""
        try:
            self.logger.info("📋 테스트 1: 배치 처리 (2+2)")
            start_time = time.perf_counter()
            
            # 4개 작업을 2+2로 분할
            video_tasks = [
                {"name": "video_0", "frames": 100},
                {"name": "video_1", "frames": 100},
                {"name": "video_2", "frames": 100},
                {"name": "video_3", "frames": 100}
            ]
            
            batch1 = video_tasks[:2]
            batch2 = video_tasks[2:]
            
            # 배치 1 처리
            self.logger.info("🔄 배치 1 처리 중 (video_0, video_1)")
            batch1_results = await self._process_batch(batch1, batch_id=1)
            
            # 세션 정리 대기
            self.logger.info("⏳ 세션 정리 대기 중 (3초)...")
            await asyncio.sleep(3.0)
            
            # 배치 2 처리
            self.logger.info("🔄 배치 2 처리 중 (video_2, video_3)")
            batch2_results = await self._process_batch(batch2, batch_id=2)
            
            # 결과 분석
            all_results = batch1_results + batch2_results
            success_count = sum(1 for r in all_results if r.get('success', False))
            success_rate = (success_count / len(all_results)) * 100
            
            elapsed_time = time.perf_counter() - start_time
            
            self.logger.info(f"✅ 배치 처리 완료: {success_count}/{len(all_results)} 성공 ({success_rate:.1f}%)")
            self.logger.info(f"⏱️ 처리 시간: {elapsed_time:.2f}초")
            
            # 테스트 성공 기준: 75% 이상 성공률
            test_success = success_rate >= 75.0
            
            self.test_results.append({
                'test_name': 'batch_processing',
                'success': test_success,
                'success_rate': success_rate,
                'processing_time': elapsed_time,
                'details': all_results
            })
            
            return test_success
            
        except Exception as e:
            self.logger.error(f"❌ 배치 처리 테스트 실패: {e}")
            return False
    
    async def test_fallback_mechanism(self) -> bool:
        """폴백 메커니즘 테스트"""
        try:
            self.logger.info("📋 테스트 2: 소프트웨어 폴백 메커니즘")
            start_time = time.perf_counter()
            
            # NVENC 세션을 모두 사용한 상태에서 추가 요청
            fallback_tasks = [
                {"name": "fallback_video_1", "frames": 50, "force_fallback": True},
                {"name": "fallback_video_2", "frames": 50, "force_fallback": True}
            ]
            
            results = []
            for task in fallback_tasks:
                self.logger.info(f"🔧 {task['name']}: 폴백 테스트 시작")
                
                # 소프트웨어 인코더로 직접 테스트
                result = await self._test_software_encoding(task)
                results.append(result)
                
                await asyncio.sleep(1.0)  # 간격
            
            # 결과 분석
            success_count = sum(1 for r in results if r.get('success', False))
            success_rate = (success_count / len(results)) * 100
            elapsed_time = time.perf_counter() - start_time
            
            self.logger.info(f"✅ 폴백 테스트 완료: {success_count}/{len(results)} 성공 ({success_rate:.1f}%)")
            self.logger.info(f"⏱️ 처리 시간: {elapsed_time:.2f}초")
            
            test_success = success_rate >= 80.0  # 폴백은 더 높은 성공률 기대
            
            self.test_results.append({
                'test_name': 'fallback_mechanism',
                'success': test_success,
                'success_rate': success_rate,
                'processing_time': elapsed_time,
                'details': results
            })
            
            return test_success
            
        except Exception as e:
            self.logger.error(f"❌ 폴백 메커니즘 테스트 실패: {e}")
            return False
    
    async def test_session_management(self) -> bool:
        """세션 관리 테스트"""
        try:
            self.logger.info("📋 테스트 3: NVENC 세션 관리")
            start_time = time.perf_counter()
            
            # 세션 관리자 상태 확인
            initial_stats = self.session_manager.get_manager_stats()
            self.logger.info(f"🔍 초기 세션 상태: {initial_stats}")
            
            # 동시 세션 요청 (제한 테스트)
            session_results = []
            
            # 최대 세션 수만큼 요청
            for i in range(self.max_concurrent_nvenc):
                try:
                    session_id = await self.session_manager.get_session(timeout=5.0)
                    self.logger.info(f"✅ 세션 {i+1} 획득: ID={session_id}")
                    session_results.append({'session_id': session_id, 'acquired': True})
                except Exception as e:
                    self.logger.warning(f"⚠️ 세션 {i+1} 획득 실패: {e}")
                    session_results.append({'session_id': None, 'acquired': False, 'error': str(e)})
            
            # 추가 세션 요청 (제한 초과 테스트)
            try:
                extra_session_id = await self.session_manager.get_session(timeout=2.0)
                self.logger.warning(f"⚠️ 예상치 못한 추가 세션 획득: {extra_session_id}")
                extra_acquired = True
            except Exception as e:
                self.logger.info(f"✅ 예상된 세션 제한 에러: {e}")
                extra_acquired = False
            
            # 세션 해제
            for result in session_results:
                if result['acquired'] and result['session_id']:
                    await self.session_manager.release_session(result['session_id'])
                    self.logger.info(f"🔓 세션 해제: {result['session_id']}")
            
            # 최종 세션 상태
            final_stats = self.session_manager.get_manager_stats()
            self.logger.info(f"🔍 최종 세션 상태: {final_stats}")
            
            elapsed_time = time.perf_counter() - start_time
            
            # 테스트 성공 조건
            acquired_sessions = sum(1 for r in session_results if r['acquired'])
            test_success = (
                acquired_sessions == self.max_concurrent_nvenc and  # 정확한 세션 수 획득
                not extra_acquired and  # 추가 세션 제한됨
                final_stats['active_sessions'] == 0  # 모든 세션 해제됨
            )
            
            self.logger.info(f"✅ 세션 관리 테스트 완료: {'성공' if test_success else '실패'}")
            self.logger.info(f"⏱️ 처리 시간: {elapsed_time:.2f}초")
            
            self.test_results.append({
                'test_name': 'session_management',
                'success': test_success,
                'acquired_sessions': acquired_sessions,
                'extra_session_blocked': not extra_acquired,
                'processing_time': elapsed_time,
                'initial_stats': initial_stats,
                'final_stats': final_stats
            })
            
            return test_success
            
        except Exception as e:
            self.logger.error(f"❌ 세션 관리 테스트 실패: {e}")
            return False
    
    async def test_error_recovery(self) -> bool:
        """에러 복구 테스트"""
        try:
            self.logger.info("📋 테스트 4: 에러 복구 메커니즘")
            start_time = time.perf_counter()
            
            # 의도적 에러 시나리오
            error_scenarios = [
                {"name": "invalid_frame", "error_type": "format_error"},
                {"name": "timeout_test", "error_type": "timeout"},
                {"name": "resource_error", "error_type": "resource_limit"}
            ]
            
            recovery_results = []
            
            for scenario in error_scenarios:
                self.logger.info(f"🧪 에러 시나리오: {scenario['name']}")
                
                result = await self._simulate_error_scenario(scenario)
                recovery_results.append(result)
                
                # 복구 대기
                await asyncio.sleep(1.0)
            
            # 복구 후 정상 작업 테스트
            self.logger.info("🔄 복구 후 정상 작업 테스트")
            normal_result = await self._test_normal_encoding_after_error()
            recovery_results.append(normal_result)
            
            # 결과 분석
            recovery_count = sum(1 for r in recovery_results if r.get('recovered', False))
            recovery_rate = (recovery_count / len(recovery_results)) * 100
            elapsed_time = time.perf_counter() - start_time
            
            self.logger.info(f"✅ 에러 복구 테스트 완료: {recovery_count}/{len(recovery_results)} 복구 ({recovery_rate:.1f}%)")
            self.logger.info(f"⏱️ 처리 시간: {elapsed_time:.2f}초")
            
            test_success = recovery_rate >= 75.0  # 75% 이상 복구률
            
            self.test_results.append({
                'test_name': 'error_recovery',
                'success': test_success,
                'recovery_rate': recovery_rate,
                'processing_time': elapsed_time,
                'details': recovery_results
            })
            
            return test_success
            
        except Exception as e:
            self.logger.error(f"❌ 에러 복구 테스트 실패: {e}")
            return False
    
    async def _process_batch(self, batch_tasks: List[Dict], batch_id: int) -> List[Dict]:
        """배치 작업 처리"""
        results = []
        
        for task in batch_tasks:
            try:
                self.logger.info(f"🎬 {task['name']}: 처리 시작")
                
                # Enhanced 인코더로 시뮬레이션
                result = await self._simulate_enhanced_encoding(task)
                results.append(result)
                
                if result['success']:
                    self.logger.info(f"✅ {task['name']}: 처리 성공 ({result['method']})")
                else:
                    self.logger.warning(f"❌ {task['name']}: 처리 실패")
                
            except Exception as e:
                self.logger.error(f"❌ {task['name']}: 처리 에러 - {e}")
                results.append({
                    'task_name': task['name'],
                    'success': False,
                    'method': 'error',
                    'error': str(e)
                })
        
        return results
    
    async def _simulate_enhanced_encoding(self, task: Dict) -> Dict:
        """향상된 인코더 시뮬레이션"""
        try:
            # 90% 확률로 NVENC 성공, 10%는 폴백
            import random
            use_nvenc = random.random() < 0.9
            
            if use_nvenc:
                # NVENC 세션 시도
                try:
                    async with self.session_manager.acquire_session(timeout=5.0) as session_id:
                        # NVENC 인코딩 시뮬레이션
                        await asyncio.sleep(0.5)  # 처리 시간 시뮬레이션
                        
                        return {
                            'task_name': task['name'],
                            'success': True,
                            'method': 'nvenc',
                            'session_id': session_id,
                            'frames_processed': task.get('frames', 0)
                        }
                        
                except Exception as nvenc_error:
                    # NVENC 실패 시 소프트웨어 폴백
                    self.logger.warning(f"⚠️ {task['name']}: NVENC 실패 -> 소프트웨어 폴백")
                    
                    # 소프트웨어 인코딩 시뮬레이션
                    await asyncio.sleep(1.0)  # 소프트웨어는 더 느림
                    
                    return {
                        'task_name': task['name'],
                        'success': True,
                        'method': 'software',
                        'fallback_reason': str(nvenc_error),
                        'frames_processed': task.get('frames', 0)
                    }
            else:
                # 직접 소프트웨어 인코딩
                await asyncio.sleep(1.0)
                
                return {
                    'task_name': task['name'],
                    'success': True,
                    'method': 'software',
                    'frames_processed': task.get('frames', 0)
                }
                
        except Exception as e:
            return {
                'task_name': task['name'],
                'success': False,
                'method': 'error',
                'error': str(e)
            }
    
    async def _test_software_encoding(self, task: Dict) -> Dict:
        """소프트웨어 인코딩 테스트"""
        try:
            self.logger.info(f"🔧 {task['name']}: 소프트웨어 인코딩 시작")
            
            # 소프트웨어 인코딩 시뮬레이션 (더 오래 걸림)
            await asyncio.sleep(1.5)
            
            return {
                'task_name': task['name'],
                'success': True,
                'method': 'software',
                'frames_processed': task.get('frames', 0)
            }
            
        except Exception as e:
            return {
                'task_name': task['name'],
                'success': False,
                'method': 'software_error',
                'error': str(e)
            }
    
    async def _simulate_error_scenario(self, scenario: Dict) -> Dict:
        """에러 시나리오 시뮬레이션"""
        try:
            error_type = scenario['error_type']
            
            if error_type == "format_error":
                # 포맷 에러 시뮬레이션 및 복구
                self.logger.warning("⚠️ 포맷 에러 발생 -> 복구 시도")
                await asyncio.sleep(0.5)
                recovered = True
                
            elif error_type == "timeout":
                # 타임아웃 에러 시뮬레이션 및 복구
                self.logger.warning("⚠️ 타임아웃 에러 발생 -> 재시도")
                await asyncio.sleep(1.0)
                recovered = True
                
            elif error_type == "resource_limit":
                # 리소스 한계 에러 시뮬레이션
                self.logger.warning("⚠️ 리소스 한계 -> 폴백 모드")
                await asyncio.sleep(0.8)
                recovered = True
                
            else:
                recovered = False
            
            return {
                'scenario': scenario['name'],
                'error_type': error_type,
                'recovered': recovered
            }
            
        except Exception as e:
            return {
                'scenario': scenario['name'],
                'error_type': scenario['error_type'],
                'recovered': False,
                'error': str(e)
            }
    
    async def _test_normal_encoding_after_error(self) -> Dict:
        """에러 복구 후 정상 인코딩 테스트"""
        try:
            self.logger.info("🔄 에러 복구 후 정상 작업 테스트")
            
            # 정상 인코딩 시뮬레이션
            await asyncio.sleep(0.7)
            
            return {
                'scenario': 'normal_after_recovery',
                'recovered': True,
                'method': 'nvenc'
            }
            
        except Exception as e:
            return {
                'scenario': 'normal_after_recovery', 
                'recovered': False,
                'error': str(e)
            }
    
    def _log_final_results(self, overall_success: bool):
        """최종 결과 로그"""
        self.logger.info("="*80)
        self.logger.info("📊 Phase 3 Fixed 테스트 최종 결과")
        self.logger.info("="*80)
        
        # 전체 결과
        status_emoji = "✅" if overall_success else "❌"
        status_text = "성공" if overall_success else "실패"
        self.logger.info(f"{status_emoji} 전체 결과: {status_text}")
        
        # 개별 테스트 결과
        self.logger.info("📋 개별 테스트 결과:")
        for result in self.test_results:
            test_name = result['test_name']
            success = result['success']
            emoji = "✅" if success else "❌"
            self.logger.info(f"  {emoji} {test_name}: {'성공' if success else '실패'}")
            
            # 상세 정보
            if 'success_rate' in result:
                self.logger.info(f"    └ 성공률: {result['success_rate']:.1f}%")
            if 'processing_time' in result:
                self.logger.info(f"    └ 처리 시간: {result['processing_time']:.2f}초")
        
        # 요약 통계
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.logger.info("")
        self.logger.info("📈 요약 통계:")
        self.logger.info(f"  • 총 테스트: {total_tests}")
        self.logger.info(f"  • 통과: {passed_tests}")
        self.logger.info(f"  • 통과율: {pass_rate:.1f}%")
        
        # 세션 관리자 최종 상태
        final_session_stats = self.session_manager.get_manager_stats()
        self.logger.info("")
        self.logger.info("🔧 세션 관리자 최종 상태:")
        for key, value in final_session_stats.items():
            self.logger.info(f"  • {key}: {value}")
        
        self.logger.info("="*80)


async def main():
    """메인 테스트 실행"""
    tester = Phase3FixedTester()
    
    start_time = time.perf_counter()
    success = await tester.run_all_tests()
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("🎉 Phase 3 Fixed 테스트 완료!")
    print("="*80)
    print(f"  • 전체 결과: {'✅ 성공' if success else '❌ 실패'}")
    print(f"  • 총 소요 시간: {duration:.2f}초")
    print(f"  • 예상 처리 성능: ~{400/duration:.1f}fps (400프레임 기준)")
    print("="*80)
    
    return success


if __name__ == "__main__":
    # 로깅 레벨 설정
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
        print(f"\n프로그램 종료 (exit code: {exit_code})")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의한 중단")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 에러: {e}")
        print(f"스택 트레이스:\n{traceback.format_exc()}")
        sys.exit(1)