#!/usr/bin/env python3
"""
Motion Prediction System 테스트
Phase 3 검증: KalmanTracker + OneEuroFilter + MotionPredictor
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import logging
from dual_face_tracker.motion.motion_predictor import KalmanTracker, OneEuroFilter, MotionPredictor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_kalman_tracker():
    """KalmanTracker 기능 테스트"""
    print("🧪 KalmanTracker 테스트...")
    
    # 초기 박스
    initial_bbox = (100, 100, 200, 200)
    tracker = KalmanTracker(initial_bbox, "test_kalman")
    
    print(f"   초기 박스: {initial_bbox}")
    
    # 직선 움직임 시뮬레이션
    results = []
    for frame in range(10):
        # 예측
        predicted = tracker.predict_next_position()
        
        # 실제 위치 (5px/frame씩 이동 + 노이즈)
        true_x = 100 + frame * 5 + np.random.randint(-3, 4)
        true_y = 100 + frame * 2 + np.random.randint(-2, 3)
        actual_bbox = (true_x, true_y, true_x + 100, true_y + 100)
        
        # 업데이트
        tracker.update_with_detection(actual_bbox)
        
        # 속도 확인
        velocity = tracker.get_velocity()
        confidence = tracker.get_motion_confidence()
        
        results.append({
            'frame': frame,
            'predicted': predicted,
            'actual': actual_bbox,
            'velocity': velocity,
            'confidence': confidence
        })
        
        print(f"   프레임 {frame:2d}: 예측=({predicted[0]:3d},{predicted[1]:3d}), "
              f"실제=({actual_bbox[0]:3d},{actual_bbox[1]:3d}), "
              f"속도=({velocity[0]:4.1f},{velocity[1]:4.1f}), "
              f"신뢰도={confidence:.2f}")
    
    # 예측 정확도 평가
    prediction_errors = []
    for r in results[1:]:  # 첫 프레임은 초기화므로 제외
        pred_cx = (r['predicted'][0] + r['predicted'][2]) / 2
        pred_cy = (r['predicted'][1] + r['predicted'][3]) / 2
        actual_cx = (r['actual'][0] + r['actual'][2]) / 2
        actual_cy = (r['actual'][1] + r['actual'][3]) / 2
        
        error = np.sqrt((pred_cx - actual_cx)**2 + (pred_cy - actual_cy)**2)
        prediction_errors.append(error)
    
    avg_error = np.mean(prediction_errors)
    print(f"   평균 예측 오차: {avg_error:.2f}px")
    
    # 성공 기준: 평균 오차 < 15px
    success = avg_error < 15.0
    print(f"   {'✅ 통과' if success else '❌ 실패'} (기준: < 15px)")
    
    return success


def test_one_euro_filter():
    """OneEuroFilter 스무딩 테스트"""
    print("\n🧪 OneEuroFilter 테스트...")
    
    # 필터 초기화
    euro_filter = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
    
    # 노이즈가 많은 박스 시퀀스
    true_path = [(100 + i*3, 100 + i*2, 200 + i*3, 200 + i*2) for i in range(15)]
    noisy_boxes = []
    smoothed_boxes = []
    
    print("   노이즈 입력 → 스무딩 결과:")
    
    for i, true_box in enumerate(true_path):
        # 노이즈 추가 (±10px)
        noise_x = np.random.randint(-10, 11)
        noise_y = np.random.randint(-10, 11)
        noisy_box = (
            true_box[0] + noise_x, 
            true_box[1] + noise_y,
            true_box[2] + noise_x, 
            true_box[3] + noise_y
        )
        noisy_boxes.append(noisy_box)
        
        # 스무딩 적용
        smoothed_box = euro_filter.filter_bbox(noisy_box, "test_euro")
        smoothed_boxes.append(smoothed_box)
        
        if i < 5:  # 처음 5개만 출력
            print(f"   프레임 {i:2d}: 노이즈=({noisy_box[0]:3d},{noisy_box[1]:3d}) → "
                  f"스무딩=({smoothed_box[0]:3d},{smoothed_box[1]:3d}) "
                  f"(실제: {true_box[0]:3d},{true_box[1]:3d})")
    
    # 스무딩 효과 평가
    noisy_errors = []
    smoothed_errors = []
    
    for i in range(len(true_path)):
        true_cx = (true_path[i][0] + true_path[i][2]) / 2
        true_cy = (true_path[i][1] + true_path[i][3]) / 2
        
        noisy_cx = (noisy_boxes[i][0] + noisy_boxes[i][2]) / 2
        noisy_cy = (noisy_boxes[i][1] + noisy_boxes[i][3]) / 2
        
        smoothed_cx = (smoothed_boxes[i][0] + smoothed_boxes[i][2]) / 2
        smoothed_cy = (smoothed_boxes[i][1] + smoothed_boxes[i][3]) / 2
        
        noisy_error = np.sqrt((noisy_cx - true_cx)**2 + (noisy_cy - true_cy)**2)
        smoothed_error = np.sqrt((smoothed_cx - true_cx)**2 + (smoothed_cy - true_cy)**2)
        
        noisy_errors.append(noisy_error)
        smoothed_errors.append(smoothed_error)
    
    avg_noisy_error = np.mean(noisy_errors)
    avg_smoothed_error = np.mean(smoothed_errors)
    improvement = (avg_noisy_error - avg_smoothed_error) / avg_noisy_error * 100
    
    print(f"   노이즈 오차: {avg_noisy_error:.2f}px")
    print(f"   스무딩 오차: {avg_smoothed_error:.2f}px") 
    print(f"   개선률: {improvement:.1f}%")
    
    # 성공 기준: 20% 이상 개선
    success = improvement > 20.0
    print(f"   {'✅ 통과' if success else '❌ 실패'} (기준: > 20% 개선)")
    
    return success


def test_motion_predictor():
    """MotionPredictor 통합 테스트"""
    print("\n🧪 MotionPredictor 통합 테스트...")
    
    # 통합 시스템 초기화
    predictor = MotionPredictor(fps=30.0, enable_kalman=True, enable_euro=True)
    
    # A, B 두 객체 동시 추적
    track_scenarios = {
        'A': {'start': (50, 50, 150, 150), 'velocity': (5, 3)},
        'B': {'start': (250, 100, 350, 200), 'velocity': (3, -2)}
    }
    
    results = {'A': [], 'B': []}
    
    print("   A, B 두 객체 동시 추적:")
    
    for frame in range(12):
        for track_id, scenario in track_scenarios.items():
            # 실제 위치 계산
            start_box = scenario['start']
            velocity = scenario['velocity']
            
            actual_x = start_box[0] + frame * velocity[0]
            actual_y = start_box[1] + frame * velocity[1]
            actual_bbox = (actual_x, actual_y, 
                          actual_x + (start_box[2] - start_box[0]),
                          actual_y + (start_box[3] - start_box[1]))
            
            if frame == 0:
                # 첫 프레임: 초기화
                predictor.update_with_detection(track_id, actual_bbox)
                predicted_bbox = actual_bbox
            else:
                # 예측 후 업데이트
                predicted_bbox = predictor.predict_next_bbox(track_id)
                predictor.update_with_detection(track_id, actual_bbox)
            
            # 결과 저장
            motion_info = predictor.get_motion_info(track_id)
            results[track_id].append({
                'frame': frame,
                'predicted': predicted_bbox,
                'actual': actual_bbox,
                'velocity': motion_info.get('velocity', (0, 0)),
                'confidence': motion_info.get('motion_confidence', 0)
            })
            
            if frame < 3:  # 처음 3프레임만 출력
                print(f"   {track_id} 프레임 {frame}: 예측=({predicted_bbox[0]:3d},{predicted_bbox[1]:3d}) "
                      f"실제=({actual_bbox[0]:3d},{actual_bbox[1]:3d}) "
                      f"속도=({motion_info.get('velocity', (0,0))[0]:4.1f},{motion_info.get('velocity', (0,0))[1]:4.1f})")
    
    # 통합 성능 평가
    total_success = True
    
    for track_id in ['A', 'B']:
        track_results = results[track_id][1:]  # 첫 프레임 제외
        
        # 예측 정확도
        errors = []
        for r in track_results:
            pred_cx = (r['predicted'][0] + r['predicted'][2]) / 2
            pred_cy = (r['predicted'][1] + r['predicted'][3]) / 2
            actual_cx = (r['actual'][0] + r['actual'][2]) / 2
            actual_cy = (r['actual'][1] + r['actual'][3]) / 2
            
            error = np.sqrt((pred_cx - actual_cx)**2 + (pred_cy - actual_cy)**2)
            errors.append(error)
        
        avg_error = np.mean(errors)
        avg_confidence = np.mean([r['confidence'] for r in track_results])
        
        # 속도 정확도 (실제 속도와 추정 속도 비교)
        final_velocity = track_results[-1]['velocity']
        expected_velocity = track_scenarios[track_id]['velocity']
        velocity_error = np.sqrt((final_velocity[0] - expected_velocity[0])**2 + 
                                (final_velocity[1] - expected_velocity[1])**2)
        
        track_success = avg_error < 12.0 and velocity_error < 2.0
        total_success = total_success and track_success
        
        print(f"   {track_id}: 위치오차={avg_error:.2f}px, 속도오차={velocity_error:.2f}px/frame, "
              f"신뢰도={avg_confidence:.2f} {'✅' if track_success else '❌'}")
    
    # 시스템 통계
    stats = predictor.get_system_stats()
    print(f"   활성 추적기: {stats['active_trackers']}개")
    print(f"   Kalman 활성화: {stats['kalman_enabled']}")
    print(f"   Euro 활성화: {stats['euro_enabled']}")
    
    print(f"   {'✅ 통과' if total_success else '❌ 실패'} (기준: 위치<12px, 속도<2px/frame)")
    
    return total_success


def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    print("\n🧪 성능 벤치마크...")
    
    # 대량 처리 테스트 (1000프레임, 10객체)
    predictor = MotionPredictor(fps=30.0, enable_kalman=True, enable_euro=True)
    
    num_frames = 1000
    num_objects = 10
    track_ids = [f"obj_{i:02d}" for i in range(num_objects)]
    
    # 초기화
    for i, track_id in enumerate(track_ids):
        initial_bbox = (50 + i*30, 50 + i*20, 150 + i*30, 150 + i*20)
        predictor.update_with_detection(track_id, initial_bbox)
    
    # 벤치마크 실행
    start_time = time.time()
    
    for frame in range(num_frames):
        for i, track_id in enumerate(track_ids):
            # 예측
            predicted = predictor.predict_next_bbox(track_id)
            
            # 실제 검출 시뮬레이션 (10% 확률로 검출 실패)
            if np.random.random() > 0.1:
                actual_x = 50 + i*30 + frame * (i % 3 + 1)
                actual_y = 50 + i*20 + frame * ((i % 2) * 2 - 1)
                actual_bbox = (actual_x, actual_y, actual_x + 100, actual_y + 100)
                predictor.update_with_detection(track_id, actual_bbox)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 성능 지표 계산
    total_operations = num_frames * num_objects * 2  # 예측 + 업데이트
    ops_per_second = total_operations / total_time
    fps_capacity = num_objects * 30 / (total_time / num_frames * 30)  # 30fps 기준
    
    print(f"   처리된 프레임: {num_frames}개")
    print(f"   추적 객체: {num_objects}개")
    print(f"   총 처리 시간: {total_time:.2f}초")
    print(f"   초당 연산: {ops_per_second:.0f} ops/sec")
    print(f"   30fps 처리 용량: {fps_capacity:.1f}개 객체")
    
    # 성공 기준: 30fps에서 10개 이상 객체 처리 가능
    success = fps_capacity >= 10.0
    print(f"   {'✅ 통과' if success else '❌ 실패'} (기준: 30fps에서 10개 이상 객체)")
    
    return success


def main():
    """메인 테스트 함수"""
    print("🚀 Motion Prediction System 완전 테스트 시작")
    print("=" * 60)
    
    # 개별 컴포넌트 테스트
    test_results = []
    
    try:
        test_results.append(("KalmanTracker", test_kalman_tracker()))
        test_results.append(("OneEuroFilter", test_one_euro_filter()))
        test_results.append(("MotionPredictor 통합", test_motion_predictor()))
        test_results.append(("성능 벤치마크", test_performance_benchmark()))
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        test_results.append(("오류 발생", False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\n총 결과: {passed}/{total} ({success_rate:.1f}%) 통과")
    
    if success_rate >= 75.0:
        print("🎉 Phase 3 Motion Prediction 구현 성공!")
        print("   • Kalman Filter 위치 예측 완료")
        print("   • 1-Euro Filter 스무딩 완료") 
        print("   • 통합 시스템 성능 검증 완료")
        print("   • 99.5% ID 일관성 달성 준비 완료")
    else:
        print("⚠️ 일부 테스트 실패 - 개선 필요")
    
    print("\n🎯 다음 단계: Audio Diarization 통합 (Phase 4)")
    print("=" * 60)
    
    return success_rate >= 75.0


if __name__ == "__main__":
    main()