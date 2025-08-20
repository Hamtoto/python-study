#!/usr/bin/env python3
"""
ByteTrack + FaceDetector 통합 테스트.

FaceDetector에서 감지된 얼굴들을 ByteTracker로 추적하여
전체 파이프라인이 정상 작동하는지 테스트합니다.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import cv2

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_face_tracker.inference import FaceDetector, ONNXRuntimeEngine
from dual_face_tracker.core import ByteTracker, ByteTrackConfig, Detection
from dual_face_tracker.utils.logger import setup_dual_face_logger


class ByteTrackIntegrationTest:
    """ByteTrack과 FaceDetector의 통합 테스트 클래스."""
    
    def __init__(self):
        """테스트 환경 초기화."""
        self.logger = setup_dual_face_logger("INFO")
        self.logger.info("🧪 ByteTrack 통합 테스트 시작")
        
        # 모델 경로 설정
        self.model_path = Path("models/yolov8n.onnx")
        
        # 테스트 이미지 경로  
        self.test_images_dir = Path("test_images")
        
        # 결과 저장 디렉토리
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 컴포넌트 초기화
        self.face_detector = None
        self.byte_tracker = None
        
        # 테스트 통계
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    def setup_components(self) -> bool:
        """FaceDetector와 ByteTracker 컴포넌트를 초기화합니다."""
        try:
            self.logger.info("🔧 컴포넌트 초기화 시작")
            
            # 1. FaceDetector 초기화
            if not self.model_path.exists():
                self.logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            self.face_detector = FaceDetector(
                model_path=str(self.model_path),
                confidence_threshold=0.3
            )
            self.logger.info("✅ FaceDetector 초기화 완료")
            
            # 2. ByteTracker 초기화 (얼굴 추적용 설정)
            config = ByteTrackConfig.for_face_tracking()
            self.byte_tracker = ByteTracker(**config.to_dict())
            self.logger.info("✅ ByteTracker 초기화 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            return False
    
    def convert_face_detector_output(self, detections) -> List[Detection]:
        """FaceDetector의 출력을 ByteTrack Detection 형태로 변환합니다."""
        from dual_face_tracker.core.tracking_structures import Detection as ByteDetection
        
        byte_detections = []
        
        for det in detections:
            # FaceDetector의 Detection 객체에서 정보 추출
            if hasattr(det, 'bbox') and hasattr(det, 'confidence'):
                bbox = det.bbox  # (x1, y1, x2, y2)
                confidence = det.confidence
                class_id = getattr(det, 'class_id', 0)
            else:
                # 딕셔너리 형태인 경우
                bbox = tuple(det['bbox'])
                confidence = det['confidence'] 
                class_id = det.get('class_id', 0)
            
            detection = ByteDetection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id
            )
            byte_detections.append(detection)
        
        return byte_detections
    
    def test_single_frame_processing(self) -> bool:
        """단일 프레임 처리 테스트."""
        test_name = "single_frame_processing"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # 테스트 이미지 생성 (640x640 랜덤 이미지)
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # FaceDetector로 얼굴 감지
            start_time = time.time()
            detections = self.face_detector.detect(test_image)
            detection_time = (time.time() - start_time) * 1000
            
            # ByteTracker 형태로 변환
            byte_detections = self.convert_face_detector_output(detections)
            
            # ByteTracker로 추적
            start_time = time.time()
            tracks = self.byte_tracker.update(byte_detections)
            tracking_time = (time.time() - start_time) * 1000
            
            # 결과 검증
            total_time = detection_time + tracking_time
            success = total_time < 10.0  # 10ms 미만이면 성공
            
            result = {
                'test_name': test_name,
                'success': success,
                'detection_count': len(detections),
                'track_count': len(tracks),
                'detection_time_ms': detection_time,
                'tracking_time_ms': tracking_time,
                'total_time_ms': total_time
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - {total_time:.2f}ms")
            else:
                self.logger.warning(f"⚠️ {test_name} 시간 초과 - {total_time:.2f}ms")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results['test_details'].append({
                'test_name': test_name,
                'success': False,
                'error': str(e)
            })
            return False
    
    def test_multi_frame_tracking(self) -> bool:
        """다중 프레임 추적 일관성 테스트."""
        test_name = "multi_frame_tracking"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # ByteTracker 리셋
            self.byte_tracker.reset()
            
            frame_count = 10
            track_histories = {}
            
            for frame_idx in range(frame_count):
                # 시뮬레이션: 2개의 고정 얼굴이 약간씩 움직임
                base_detections = [
                    {
                        'bbox': [100 + frame_idx * 2, 100 + frame_idx, 200 + frame_idx * 2, 200 + frame_idx],
                        'confidence': 0.8
                    },
                    {
                        'bbox': [400 + frame_idx * 3, 150 + frame_idx * 2, 500 + frame_idx * 3, 250 + frame_idx * 2], 
                        'confidence': 0.7
                    }
                ]
                
                # Detection 변환
                byte_detections = []
                for det in base_detections:
                    detection = Detection(
                        bbox=tuple(det['bbox']),
                        confidence=det['confidence'],
                        class_id=0
                    )
                    byte_detections.append(detection)
                
                # 추적 업데이트
                tracks = self.byte_tracker.update(byte_detections)
                
                # 트랙 히스토리 기록
                for track in tracks:
                    if track.track_id not in track_histories:
                        track_histories[track.track_id] = []
                    track_histories[track.track_id].append({
                        'frame': frame_idx,
                        'bbox': track.tlbr.tolist(),
                        'confidence': track.score
                    })
            
            # 결과 분석
            consistent_tracks = 0
            for track_id, history in track_histories.items():
                if len(history) >= 5:  # 최소 5프레임 이상 추적되면 일관성 있음
                    consistent_tracks += 1
            
            success = consistent_tracks >= 2  # 2개 트랙이 일관성 있게 추적되면 성공
            
            result = {
                'test_name': test_name,
                'success': success,
                'total_tracks': len(track_histories),
                'consistent_tracks': consistent_tracks,
                'frame_count': frame_count,
                'track_histories': track_histories
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - {consistent_tracks}개 일관성 트랙")
            else:
                self.logger.warning(f"⚠️ {test_name} 부족 - {consistent_tracks}개 일관성 트랙")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results['test_details'].append({
                'test_name': test_name,
                'success': False,
                'error': str(e)
            })
            return False
    
    def test_memory_usage(self) -> bool:
        """메모리 사용량 테스트."""
        test_name = "memory_usage"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            import torch
            import gc
            
            # 시작 메모리 측정
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()
            else:
                start_memory = 0
            
            # 100 프레임 처리 시뮬레이션
            self.byte_tracker.reset()
            
            for frame_idx in range(100):
                # 랜덤 detection 생성 (1-5개)
                num_detections = np.random.randint(1, 6)
                detections = []
                
                for _ in range(num_detections):
                    x1 = np.random.randint(0, 500)
                    y1 = np.random.randint(0, 500)
                    x2 = x1 + np.random.randint(50, 100)
                    y2 = y1 + np.random.randint(50, 100)
                    conf = np.random.uniform(0.3, 0.9)
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=0
                    )
                    detections.append(detection)
                
                # 추적 업데이트
                tracks = self.byte_tracker.update(detections)
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_increase = end_memory - start_memory
            else:
                memory_increase = 0
            
            # 메모리 증가가 100MB 미만이면 성공
            memory_limit = 100 * 1024 * 1024  # 100MB
            success = memory_increase < memory_limit
            
            result = {
                'test_name': test_name,
                'success': success,
                'start_memory_mb': start_memory / (1024*1024),
                'end_memory_mb': end_memory / (1024*1024) if torch.cuda.is_available() else 0,
                'memory_increase_mb': memory_increase / (1024*1024),
                'frames_processed': 100
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - 메모리 증가 {memory_increase/(1024*1024):.1f}MB")
            else:
                self.logger.warning(f"⚠️ {test_name} 메모리 초과 - {memory_increase/(1024*1024):.1f}MB")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results['test_details'].append({
                'test_name': test_name,
                'success': False,
                'error': str(e)
            })
            return False
    
    def test_performance_benchmark(self) -> bool:
        """성능 벤치마크 테스트."""
        test_name = "performance_benchmark"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            self.byte_tracker.reset()
            
            # 1000 프레임 처리 시간 측정
            total_times = []
            detection_times = []
            tracking_times = []
            
            for frame_idx in range(1000):
                # 테스트 이미지 생성 (작은 이미지로 빠른 테스트)
                test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                
                # Detection 시간 측정
                start_time = time.perf_counter()
                detections = self.face_detector.detect(test_image)
                detection_time = (time.perf_counter() - start_time) * 1000
                detection_times.append(detection_time)
                
                # Tracking 시간 측정
                byte_detections = self.convert_face_detector_output(detections)
                start_time = time.perf_counter()
                tracks = self.byte_tracker.update(byte_detections)
                tracking_time = (time.perf_counter() - start_time) * 1000
                tracking_times.append(tracking_time)
                
                total_times.append(detection_time + tracking_time)
            
            # 통계 계산
            avg_total_time = np.mean(total_times)
            avg_detection_time = np.mean(detection_times)
            avg_tracking_time = np.mean(tracking_times)
            
            max_total_time = np.max(total_times)
            fps = 1000.0 / avg_total_time if avg_total_time > 0 else 0
            
            # 목표: 평균 5ms 미만 (200 FPS 이상)
            success = avg_total_time < 5.0
            
            result = {
                'test_name': test_name,
                'success': success,
                'avg_total_time_ms': avg_total_time,
                'avg_detection_time_ms': avg_detection_time,
                'avg_tracking_time_ms': avg_tracking_time,
                'max_total_time_ms': max_total_time,
                'estimated_fps': fps,
                'frames_tested': 1000
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - 평균 {avg_total_time:.2f}ms ({fps:.1f} FPS)")
            else:
                self.logger.warning(f"⚠️ {test_name} 목표 미달 - 평균 {avg_total_time:.2f}ms ({fps:.1f} FPS)")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results['test_details'].append({
                'test_name': test_name,
                'success': False,
                'error': str(e)
            })
            return False
    
    def run_all_tests(self):
        """모든 테스트를 실행합니다."""
        self.logger.info("🚀 ByteTrack 통합 테스트 시작")
        
        # 컴포넌트 초기화
        if not self.setup_components():
            self.logger.error("❌ 컴포넌트 초기화 실패 - 테스트 중단")
            return False
        
        # 테스트 목록
        tests = [
            self.test_single_frame_processing,
            self.test_multi_frame_tracking,
            self.test_memory_usage,
            self.test_performance_benchmark,
        ]
        
        # 각 테스트 실행
        for test_func in tests:
            self.test_results['total_tests'] += 1
            
            try:
                if test_func():
                    self.test_results['passed_tests'] += 1
                else:
                    self.test_results['failed_tests'] += 1
            except Exception as e:
                self.logger.error(f"테스트 실행 중 오류: {e}")
                self.test_results['failed_tests'] += 1
        
        # 결과 요약
        self._print_test_summary()
        
        return self.test_results['failed_tests'] == 0
    
    def _print_test_summary(self):
        """테스트 결과 요약을 출력합니다."""
        self.logger.info("=" * 80)
        self.logger.info("📊 ByteTrack 통합 테스트 결과 요약")
        self.logger.info("=" * 80)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        
        self.logger.info(f"총 테스트: {total}개")
        self.logger.info(f"성공: {passed}개")
        self.logger.info(f"실패: {failed}개")
        self.logger.info(f"성공률: {(passed/total*100):.1f}%")
        
        self.logger.info("\n📋 상세 결과:")
        for detail in self.test_results['test_details']:
            status = "✅" if detail['success'] else "❌"
            self.logger.info(f"{status} {detail['test_name']}")
            
            if 'total_time_ms' in detail:
                self.logger.info(f"   - 처리 시간: {detail.get('total_time_ms', 0):.2f}ms")
            
            if 'estimated_fps' in detail:
                self.logger.info(f"   - 예상 FPS: {detail.get('estimated_fps', 0):.1f}")
        
        if failed == 0:
            self.logger.info("\n🎉 모든 테스트가 성공했습니다!")
        else:
            self.logger.warning(f"\n⚠️ {failed}개 테스트가 실패했습니다.")


def main():
    """메인 실행 함수."""
    try:
        # DevContainer 환경 확인
        if not os.path.exists("/.dockerenv"):
            print("⚠️ 경고: DevContainer 환경이 아닐 수 있습니다.")
            print("   이 테스트는 DevContainer에서 실행해야 합니다.")
        
        # 테스트 실행
        test_runner = ByteTrackIntegrationTest()
        success = test_runner.run_all_tests()
        
        # 종료 코드 반환
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ 테스트가 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"❌ 테스트 실행 중 심각한 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)