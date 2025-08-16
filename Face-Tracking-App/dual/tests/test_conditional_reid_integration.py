#!/usr/bin/env python3
"""
ConditionalReID + ByteTracker + FaceDetector 통합 테스트.

전체 파이프라인이 정상 작동하는지 테스트합니다:
FaceDetector -> ByteTracker -> ConditionalReID
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

from dual_face_tracker.inference import FaceDetector, ReIDModel, ReIDModelConfig
from dual_face_tracker.core import (
    ByteTracker, ByteTrackConfig, ConditionalReID, 
    IDSwapDetector, EmbeddingManager
)
from dual_face_tracker.utils.logger import setup_dual_face_logger


class ConditionalReIDIntegrationTest:
    """ConditionalReID와 전체 파이프라인의 통합 테스트 클래스."""
    
    def __init__(self):
        """테스트 환경 초기화."""
        self.logger = setup_dual_face_logger("INFO")
        self.logger.info("🧪 ConditionalReID 통합 테스트 시작")
        
        # 모델 경로 설정
        self.model_path = Path("models/yolov8n.onnx")
        
        # 결과 저장 디렉토리
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 컴포넌트 초기화
        self.face_detector = None
        self.byte_tracker = None
        self.conditional_reid = None
        
        # 테스트 통계
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    def setup_components(self) -> bool:
        """모든 컴포넌트를 초기화합니다."""
        try:
            self.logger.info("🔧 전체 파이프라인 컴포넌트 초기화")
            
            # 1. FaceDetector 초기화
            if not self.model_path.exists():
                self.logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            self.face_detector = FaceDetector(
                model_path=str(self.model_path),
                confidence_threshold=0.3
            )
            self.logger.info("✅ FaceDetector 초기화 완료")
            
            # 2. ByteTracker 초기화
            config = ByteTrackConfig.for_face_tracking()
            self.byte_tracker = ByteTracker(**config.to_dict())
            self.logger.info("✅ ByteTracker 초기화 완료")
            
            # 3. ConditionalReID 초기화
            reid_config = ReIDModelConfig.for_face_reid()
            self.conditional_reid = ConditionalReID(
                reid_model_config=reid_config,
                activation_threshold=0.5,  # 테스트용으로 낮은 임계값
                target_activation_rate=0.2  # 테스트용으로 높은 목표
            )
            self.logger.info("✅ ConditionalReID 초기화 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            return False
    
    def convert_face_detector_output(self, detections) -> List:
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
    
    def test_basic_pipeline_flow(self) -> bool:
        """기본 파이프라인 플로우 테스트."""
        test_name = "basic_pipeline_flow"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # 테스트 이미지 생성 (640x640 랜덤 이미지)
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 1단계: FaceDetector
            start_time = time.time()
            detections = self.face_detector.detect(test_image)
            detection_time = (time.time() - start_time) * 1000
            
            # 2단계: ByteTracker
            byte_detections = self.convert_face_detector_output(detections)
            start_time = time.time()
            tracks = self.byte_tracker.update(byte_detections)
            tracking_time = (time.time() - start_time) * 1000
            
            # 3단계: ConditionalReID
            start_time = time.time()
            reid_result = self.conditional_reid.process_frame(tracks, test_image)
            reid_time = (time.time() - start_time) * 1000
            
            # 결과 검증
            total_time = detection_time + tracking_time + reid_time
            success = total_time < 20.0  # 20ms 미만이면 성공
            
            result = {
                'test_name': test_name,
                'success': success,
                'detection_count': len(detections),
                'track_count': len(tracks),
                'reid_activated': reid_result.reid_activated,
                'detection_time_ms': detection_time,
                'tracking_time_ms': tracking_time,
                'reid_time_ms': reid_time,
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
    
    def test_reid_activation_simulation(self) -> bool:
        """ReID 활성화 시뮬레이션 테스트."""
        test_name = "reid_activation_simulation"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # ByteTracker 리셋
            self.byte_tracker.reset()
            self.conditional_reid.reset()
            
            # 시뮬레이션: 의도적으로 ID 스왑 상황 생성
            activation_count = 0
            total_frames = 50
            
            for frame_idx in range(total_frames):
                # 기본 2개 얼굴
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
                
                # 프레임 20에서 의도적인 위치 점프 (스왑 시뮬레이션)
                if frame_idx == 20:
                    # 두 얼굴의 위치를 크게 변경
                    base_detections[0]['bbox'] = [400, 150, 500, 250]  # 큰 점프
                    base_detections[1]['bbox'] = [100, 100, 200, 200]  # 큰 점프
                
                # Detection 변환
                byte_detections = []
                from dual_face_tracker.core.tracking_structures import Detection
                for det in base_detections:
                    detection = Detection(
                        bbox=tuple(det['bbox']),
                        confidence=det['confidence'],
                        class_id=0
                    )
                    byte_detections.append(detection)
                
                # 전체 파이프라인 실행
                tracks = self.byte_tracker.update(byte_detections)
                
                # 테스트 이미지 생성
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # ConditionalReID 처리
                reid_result = self.conditional_reid.process_frame(tracks, test_image)
                
                if reid_result.reid_activated:
                    activation_count += 1
                    self.logger.debug(f"프레임 {frame_idx}: ReID 활성화 "
                                     f"(위험도={reid_result.swap_detection_result.overall_risk_score:.2f})")
            
            # 결과 분석
            activation_rate = activation_count / total_frames
            # 스왑 시뮬레이션이 있었으므로 일부 활성화가 있어야 함
            success = 0.02 <= activation_rate <= 0.5  # 2-50% 활성화율이면 성공
            
            result = {
                'test_name': test_name,
                'success': success,
                'total_frames': total_frames,
                'activation_count': activation_count,
                'activation_rate': activation_rate,
                'expected_range': '2-50%'
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - 활성화율 {activation_rate:.1%}")
            else:
                self.logger.warning(f"⚠️ {test_name} 부적절한 활성화율 - {activation_rate:.1%}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results['test_details'].append({
                'test_name': test_name,
                'success': False,
                'error': str(e)
            })
            return False
    
    def test_reid_model_functionality(self) -> bool:
        """ReID 모델 기능 테스트."""
        test_name = "reid_model_functionality"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # ReID 모델 직접 테스트
            reid_model = self.conditional_reid.reid_model
            
            # 테스트 이미지 2개 생성
            image1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            image2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # 임베딩 추출
            start_time = time.time()
            embedding1 = reid_model.extract_embedding(image1)
            embedding2 = reid_model.extract_embedding(image2)
            extraction_time = (time.time() - start_time) * 1000
            
            # 유사도 계산
            similarity = reid_model.calculate_similarity(embedding1, embedding2)
            
            # 배치 처리 테스트
            images = [image1, image2]
            start_time = time.time()
            batch_embeddings = reid_model.extract_embeddings_batch(images)
            batch_time = (time.time() - start_time) * 1000
            
            # 결과 검증
            success = (
                len(embedding1) == reid_model.embedding_dim and
                len(embedding2) == reid_model.embedding_dim and
                0.0 <= similarity <= 1.0 and
                len(batch_embeddings) == 2 and
                extraction_time < 10.0  # 10ms 미만
            )
            
            result = {
                'test_name': test_name,
                'success': success,
                'embedding_dim': len(embedding1),
                'similarity': similarity,
                'extraction_time_ms': extraction_time,
                'batch_time_ms': batch_time,
                'model_type': reid_model.use_mock_model
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - 유사도={similarity:.2f}")
            else:
                self.logger.warning(f"⚠️ {test_name} 실패 - 검증 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} 실패: {e}")
            self.test_results['test_details'].append({
                'test_name': test_name,
                'success': False,
                'error': str(e)
            })
            return False
    
    def test_embedding_manager_functionality(self) -> bool:
        """임베딩 관리자 기능 테스트."""
        test_name = "embedding_manager_functionality"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            embedding_manager = self.conditional_reid.embedding_manager
            reid_model = self.conditional_reid.reid_model
            
            # Mock Track 생성
            from dual_face_tracker.core.tracking_structures import Track, Detection
            
            mock_detection1 = Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.8,
                class_id=0
            )
            
            mock_detection2 = Detection(
                bbox=(300, 150, 400, 250), 
                confidence=0.7,
                class_id=0
            )
            
            # Track 객체 생성 시뮬레이션
            class MockTrack:
                def __init__(self, track_id, detection):
                    self.track_id = track_id
                    self.detection = detection
                    self.frame_id = 1
                    self.score = detection.confidence
                    self.age = 5
                    self.hit_streak = 3
                    self.tlbr = np.array(detection.bbox)
                    
                @property
                def center_point(self):
                    x1, y1, x2, y2 = self.detection.bbox
                    return ((x1 + x2) / 2, (y1 + y2) / 2)
                
                @property
                def is_active(self):
                    return True
            
            track1 = MockTrack(1, mock_detection1)
            track2 = MockTrack(2, mock_detection2)
            
            # 임베딩 생성 및 추가
            test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            embedding1 = reid_model.extract_embedding(test_image)
            embedding2 = reid_model.extract_embedding(test_image)
            
            # 임베딩 추가
            success1 = embedding_manager.add_embedding(track1, embedding1, test_image)
            success2 = embedding_manager.add_embedding(track2, embedding2, test_image)
            
            # 임베딩 검색
            retrieved_embeddings = embedding_manager.get_track_embeddings(1)
            
            # 매칭 테스트
            matches = embedding_manager.match_tracks_by_embeddings(1, [2])
            
            # 통계 확인
            stats = embedding_manager.get_statistics()
            
            # 결과 검증
            success = (
                success1 and success2 and
                len(retrieved_embeddings) > 0 and
                stats['total_embeddings'] >= 2
            )
            
            result = {
                'test_name': test_name,
                'success': success,
                'embeddings_added': success1 and success2,
                'retrieved_count': len(retrieved_embeddings),
                'match_count': len(matches),
                'total_embeddings': stats['total_embeddings']
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - {stats['total_embeddings']}개 임베딩")
            else:
                self.logger.warning(f"⚠️ {test_name} 실패")
            
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
        """전체 파이프라인 성능 벤치마크."""
        test_name = "performance_benchmark"
        self.logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # 컴포넌트 리셋
            self.byte_tracker.reset()
            self.conditional_reid.reset()
            
            # 100 프레임 처리 시간 측정
            total_times = []
            reid_activation_count = 0
            
            for frame_idx in range(100):
                # 테스트 이미지 생성 (작은 이미지로 빠른 테스트)
                test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                
                # 전체 파이프라인 시간 측정
                start_time = time.perf_counter()
                
                # 1단계: Detection
                detections = self.face_detector.detect(test_image)
                
                # 2단계: Tracking
                byte_detections = self.convert_face_detector_output(detections)
                tracks = self.byte_tracker.update(byte_detections)
                
                # 3단계: ConditionalReID
                reid_result = self.conditional_reid.process_frame(tracks, test_image)
                
                total_time = (time.perf_counter() - start_time) * 1000
                total_times.append(total_time)
                
                if reid_result.reid_activated:
                    reid_activation_count += 1
            
            # 통계 계산
            avg_time = np.mean(total_times)
            max_time = np.max(total_times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            activation_rate = reid_activation_count / 100.0
            
            # 목표: 평균 8ms 미만 (125 FPS 이상)
            success = avg_time < 8.0
            
            result = {
                'test_name': test_name,
                'success': success,
                'avg_time_ms': avg_time,
                'max_time_ms': max_time,
                'estimated_fps': fps,
                'reid_activation_count': reid_activation_count,
                'activation_rate': activation_rate,
                'frames_tested': 100
            }
            
            self.test_results['test_details'].append(result)
            
            if success:
                self.logger.info(f"✅ {test_name} 성공 - 평균 {avg_time:.2f}ms ({fps:.1f} FPS)")
            else:
                self.logger.warning(f"⚠️ {test_name} 목표 미달 - 평균 {avg_time:.2f}ms ({fps:.1f} FPS)")
            
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
        self.logger.info("🚀 ConditionalReID 통합 테스트 시작")
        
        # 컴포넌트 초기화
        if not self.setup_components():
            self.logger.error("❌ 컴포넌트 초기화 실패 - 테스트 중단")
            return False
        
        # 테스트 목록
        tests = [
            self.test_basic_pipeline_flow,
            self.test_reid_model_functionality,
            self.test_embedding_manager_functionality,
            self.test_reid_activation_simulation,
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
        self.logger.info("📊 ConditionalReID 통합 테스트 결과 요약")
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
                
            if 'activation_rate' in detail:
                self.logger.info(f"   - ReID 활성화율: {detail.get('activation_rate', 0):.1%}")
        
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
        test_runner = ConditionalReIDIntegrationTest()
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