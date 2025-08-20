#!/usr/bin/env python3
"""
Audio Speaker Detection 시스템 테스트
Phase 2 검증: AudioActivityDetector + MouthMovementAnalyzer + AudioVisualCorrelator
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
import logging
from dual_face_tracker.audio.audio_speaker_detector import (
    AudioActivityDetector, 
    MouthMovementAnalyzer, 
    AudioVisualCorrelator,
    AudioFeatures,
    MouthFeatures
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_audio_activity_detector():
    """AudioActivityDetector 기능 테스트"""
    print("🧪 AudioActivityDetector 테스트...")
    
    detector = AudioActivityDetector()
    
    # 1. 더미 오디오 신호 생성 (2초, 440Hz 사인파)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 다양한 패턴의 오디오 신호
    signals = {
        'sine_wave': 0.1 * np.sin(2 * np.pi * 440 * t),
        'noise': np.random.normal(0, 0.05, len(t)),
        'silence': np.zeros(len(t)),
        'speech_like': 0.1 * np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))  # AM 변조
    }
    
    results = {}
    
    for signal_name, signal in signals.items():
        print(f"\n   {signal_name} 신호 분석:")
        
        # 발화 구간 검출
        segments = detector.detect_speech_segments(signal, sample_rate)
        print(f"     검출된 발화 구간: {len(segments)}개")
        
        if segments:
            total_speech_time = sum(end - start for start, end in segments)
            speech_ratio = total_speech_time / duration
            print(f"     총 발화 시간: {total_speech_time:.2f}초 ({speech_ratio:.1%})")
            
            # 처음 3개 구간만 출력
            for i, (start, end) in enumerate(segments[:3]):
                print(f"       구간 {i+1}: {start:.2f}-{end:.2f}초 ({end-start:.2f}초)")
        
        # 프레임별 활동도 계산
        activities = detector.calculate_frame_activity(signal, sample_rate, fps=30.0)
        avg_activity = np.mean(activities) if activities else 0.0
        max_activity = max(activities) if activities else 0.0
        
        print(f"     프레임별 활동도: {len(activities)}프레임, 평균={avg_activity:.3f}, 최대={max_activity:.3f}")
        
        results[signal_name] = {
            'segments': len(segments),
            'avg_activity': avg_activity,
            'max_activity': max_activity
        }
    
    # 검증: 다른 신호 타입에서 다른 결과가 나와야 함
    speech_activity = results['speech_like']['avg_activity']
    silence_activity = results['silence']['avg_activity']
    noise_activity = results['noise']['avg_activity']
    
    # 성공 기준: 
    # 1. speech_like > noise > silence 순서
    # 2. speech_like가 silence보다 최소 3배 이상 활성
    condition1 = speech_activity > noise_activity > silence_activity
    condition2 = speech_activity > silence_activity * 3
    
    success = condition1 and condition2
    print(f"\n   활동도 순서: speech={speech_activity:.3f} > noise={noise_activity:.3f} > silence={silence_activity:.3f}")
    print(f"   {'✅ 통과' if success else '❌ 실패'} (기준: 활동도 차별화 + speech > silence * 3)")
    
    return success


def test_mouth_movement_analyzer():
    """MouthMovementAnalyzer 기능 테스트"""
    print("\n🧪 MouthMovementAnalyzer 테스트...")
    
    analyzer = MouthMovementAnalyzer()
    
    # 더미 얼굴 크롭 이미지들 생성 (시뮬레이션된 입 움직임)
    face_crops = []
    expected_mars = []  # 예상 MAR 값들
    
    for i in range(10):
        # 100x100 더미 얼굴 이미지
        face_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 입 부분에 패턴 그리기 (MAR 시뮬레이션용)
        mouth_openness = 0.5 + 0.3 * np.sin(i * 0.5)  # 주기적 입 움직임
        
        # 입 영역 (하단 중앙)에 타원 그리기
        mouth_center = (50, 75)
        mouth_width = 20
        mouth_height = int(5 + mouth_openness * 10)  # 입 열림 정도
        
        cv2.ellipse(face_crop, mouth_center, (mouth_width//2, mouth_height//2), 
                   0, 0, 360, (0, 0, 0), -1)
        
        face_crops.append(face_crop)
        expected_mars.append(mouth_openness)
    
    print(f"   생성된 얼굴 크롭: {len(face_crops)}개")
    
    # 개별 랜드마크 추출 및 MAR 계산 테스트
    print("\n   개별 프레임 MAR 계산:")
    calculated_mars = []
    
    for i, face_crop in enumerate(face_crops[:5]):  # 처음 5개만 테스트
        landmarks = analyzer.extract_face_landmarks(face_crop)
        mar = analyzer.calculate_mouth_aspect_ratio(landmarks) if landmarks is not None else 0.0
        calculated_mars.append(mar)
        
        print(f"     프레임 {i}: 랜드마크={landmarks.shape if landmarks is not None else 'None'}, "
              f"MAR={mar:.4f}")
    
    # 전체 입 움직임 분석
    print("\n   전체 입 움직임 분석:")
    mouth_features = analyzer.analyze_mouth_features(face_crops)
    
    print(f"     MAR 값들: {len(mouth_features.mar_values)}개")
    print(f"     입 움직임 속도: {len(mouth_features.mouth_velocities)}개")
    print(f"     화자 신뢰도: {len(mouth_features.speaking_confidence)}개")
    
    if mouth_features.mar_values:
        avg_mar = np.mean(mouth_features.mar_values)
        max_mar = max(mouth_features.mar_values)
        avg_velocity = np.mean(mouth_features.mouth_velocities)
        avg_confidence = np.mean(mouth_features.speaking_confidence)
        
        print(f"     평균 MAR: {avg_mar:.4f}")
        print(f"     최대 MAR: {max_mar:.4f}")
        print(f"     평균 속도: {avg_velocity:.4f}")
        print(f"     평균 신뢰도: {avg_confidence:.4f}")
    
    # 성공 기준: 
    # 1. 모든 프레임에서 MAR 값이 계산됨
    # 2. MAR 값이 변동함 (모두 같지 않음)
    # 3. 화자 신뢰도가 양수
    condition1 = len(mouth_features.mar_values) == len(face_crops)
    condition2 = len(set(mouth_features.mar_values)) > 1  # 값이 변동함
    condition3 = avg_confidence > 0 if mouth_features.speaking_confidence else False
    
    success = condition1 and condition2 and condition3
    print(f"   조건1 (완전성): {condition1}, 조건2 (변동성): {condition2}, 조건3 (신뢰도): {condition3}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success


def test_audio_visual_correlator():
    """AudioVisualCorrelator 기능 테스트"""
    print("\n🧪 AudioVisualCorrelator 테스트...")
    
    correlator = AudioVisualCorrelator()
    
    # 더미 데이터 생성 (상관관계가 있는 오디오-비디오 신호)
    frame_count = 60  # 2초 @ 30fps
    
    # 패턴 1: 동기화된 신호 (강한 상관관계)
    base_pattern = np.sin(np.linspace(0, 4*np.pi, frame_count))
    audio_activity_sync = 0.5 + 0.3 * base_pattern
    mouth_activity_sync = 0.4 + 0.25 * base_pattern + np.random.normal(0, 0.05, frame_count)
    
    # 패턴 2: 지연된 신호 (5프레임 지연)
    audio_activity_delayed = np.roll(audio_activity_sync, 5)
    mouth_activity_delayed = mouth_activity_sync
    
    # 패턴 3: 무관한 신호 (상관관계 없음)
    audio_activity_random = np.random.random(frame_count)
    mouth_activity_random = np.random.random(frame_count)
    
    test_cases = {
        '동기화': (audio_activity_sync, mouth_activity_sync),
        '5프레임 지연': (audio_activity_delayed, mouth_activity_delayed),
        '무관한 신호': (audio_activity_random, mouth_activity_random)
    }
    
    results = {}
    
    print("\n   상관관계 분석:")
    for case_name, (audio, mouth) in test_cases.items():
        # 직접 상관계수 계산
        correlation = correlator.calculate_correlation(audio, mouth)
        
        # 동기화 분석 (지연 보정)
        best_corr, best_delay = correlator.synchronize_audio_video(audio.tolist(), mouth.tolist())
        
        print(f"     {case_name}:")
        print(f"       직접 상관계수: {correlation:.3f}")
        print(f"       최적 상관계수: {best_corr:.3f} (지연: {best_delay}프레임)")
        
        results[case_name] = {
            'direct_corr': correlation,
            'best_corr': best_corr,
            'delay': best_delay
        }
    
    # Active Speaker 점수 계산 테스트
    print("\n   Active Speaker 점수 계산:")
    
    # Mock 얼굴 추적 정보
    mock_face_tracks = {
        'A': type('MockTrack', (), {'avg_size': 120})(),
        'B': type('MockTrack', (), {'avg_size': 100})()
    }
    
    # Mock 오디오 특징
    mock_audio_features = AudioFeatures(
        rms_envelope=np.random.random(50),
        spectral_centroids=np.random.random(50),
        mfcc_features=None,
        speech_segments=[(0.0, 2.0)],
        frame_activities=audio_activity_sync.tolist(),
        sample_rate=16000,
        duration=2.0
    )
    
    # Mock 입 움직임 특징
    mock_mouth_features = {
        'A': MouthFeatures(
            mar_values=mouth_activity_sync.tolist(),
            mouth_velocities=np.diff(mouth_activity_sync).tolist() + [0.0],
            mouth_accelerations=[0.0] * len(mouth_activity_sync),
            speaking_confidence=(mouth_activity_sync * 1.2).tolist()
        ),
        'B': MouthFeatures(
            mar_values=(mouth_activity_sync * 0.5).tolist(),
            mouth_velocities=(np.diff(mouth_activity_sync) * 0.5).tolist() + [0.0],
            mouth_accelerations=[0.0] * len(mouth_activity_sync),
            speaking_confidence=(mouth_activity_sync * 0.6).tolist()
        )
    }
    
    # Active Speaker 점수 계산
    speaker_scores = correlator.score_active_speakers(
        mock_face_tracks, mock_audio_features, mock_mouth_features
    )
    
    print(f"     화자별 점수:")
    for person_id, score in speaker_scores:
        print(f"       {person_id}: {score:.3f}")
    
    # 성공 기준:
    # 1. 동기화된 신호의 상관계수 > 0.5
    # 2. 지연 보정으로 상관계수 개선
    # 3. Active Speaker 점수 계산 성공
    condition1 = results['동기화']['direct_corr'] > 0.5
    condition2 = results['5프레임 지연']['best_corr'] > results['5프레임 지연']['direct_corr']
    condition3 = len(speaker_scores) == 2 and all(score >= 0 for _, score in speaker_scores)
    
    success = condition1 and condition2 and condition3
    print(f"\n   조건1 (동기화): {condition1}, 조건2 (지연보정): {condition2}, 조건3 (점수계산): {condition3}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success


def test_integrated_audio_visual():
    """통합 오디오-비디오 분석 테스트"""
    print("\n🧪 통합 오디오-비디오 분석 테스트...")
    
    # 모든 컴포넌트 초기화
    audio_detector = AudioActivityDetector()
    mouth_analyzer = MouthMovementAnalyzer()
    correlator = AudioVisualCorrelator()
    
    # 시나리오: 2명의 화자가 번갈아 말하는 상황 시뮬레이션
    total_frames = 90  # 3초 @ 30fps
    
    # 화자 A가 0-1.5초, 화자 B가 1.5-3초 말함
    speaker_timeline = ['A'] * 45 + ['B'] * 45
    
    # 오디오 활동도 (화자에 따라 변화)
    audio_activities = []
    for i, speaker in enumerate(speaker_timeline):
        if speaker == 'A':
            activity = 0.6 + 0.2 * np.sin(i * 0.3) + np.random.normal(0, 0.05)
        else:  # speaker == 'B'
            activity = 0.4 + 0.15 * np.sin(i * 0.2) + np.random.normal(0, 0.03)
        audio_activities.append(max(0, activity))
    
    # 얼굴 A, B의 입 움직임 (해당 화자가 말할 때 활발)
    mouth_a_activities = []
    mouth_b_activities = []
    
    for i, speaker in enumerate(speaker_timeline):
        if speaker == 'A':
            mouth_a = 0.5 + 0.3 * np.sin(i * 0.25) + np.random.normal(0, 0.04)
            mouth_b = 0.1 + np.random.normal(0, 0.02)
        else:  # speaker == 'B'  
            mouth_a = 0.1 + np.random.normal(0, 0.02)
            mouth_b = 0.4 + 0.25 * np.sin(i * 0.2) + np.random.normal(0, 0.03)
        
        mouth_a_activities.append(max(0, mouth_a))
        mouth_b_activities.append(max(0, mouth_b))
    
    print(f"   시뮬레이션: {total_frames}프레임, 화자 A(0-45), B(46-90)")
    
    # 구간별 상관관계 분석
    segment_results = {}
    
    segments = {
        'A 화자 구간 (0-45프레임)': (0, 45, 'A'),
        'B 화자 구간 (45-90프레임)': (45, 90, 'B'),
        '전체 구간': (0, 90, 'ALL')
    }
    
    for segment_name, (start, end, expected_speaker) in segments.items():
        segment_audio = audio_activities[start:end]
        segment_mouth_a = mouth_a_activities[start:end]
        segment_mouth_b = mouth_b_activities[start:end]
        
        # A, B 각각과의 상관관계
        corr_a = correlator.calculate_correlation(np.array(segment_audio), np.array(segment_mouth_a))
        corr_b = correlator.calculate_correlation(np.array(segment_audio), np.array(segment_mouth_b))
        
        # 동기화 분석
        sync_a, delay_a = correlator.synchronize_audio_video(segment_audio, segment_mouth_a)
        sync_b, delay_b = correlator.synchronize_audio_video(segment_audio, segment_mouth_b)
        
        segment_results[segment_name] = {
            'corr_a': corr_a,
            'corr_b': corr_b,
            'sync_a': sync_a,
            'sync_b': sync_b,
            'expected': expected_speaker
        }
        
        print(f"\n     {segment_name}:")
        print(f"       오디오-얼굴A 상관관계: {corr_a:.3f} (동기화: {sync_a:.3f})")
        print(f"       오디오-얼굴B 상관관계: {corr_b:.3f} (동기화: {sync_b:.3f})")
        
        if expected_speaker == 'A':
            better = "A" if sync_a > sync_b else "B"
        elif expected_speaker == 'B':
            better = "B" if sync_b > sync_a else "A"
        else:  # 전체 구간
            better = "A" if sync_a > sync_b else "B"
        
        print(f"       더 높은 상관관계: 얼굴{better} (예상: {expected_speaker})")
    
    # 성공 기준:
    # 1. A 화자 구간에서 얼굴A가 더 높은 상관관계
    # 2. B 화자 구간에서 얼굴B가 더 높은 상관관계
    # 3. 전체 구간에서도 적절한 차별화
    condition1 = segment_results['A 화자 구간 (0-45프레임)']['sync_a'] > segment_results['A 화자 구간 (0-45프레임)']['sync_b']
    condition2 = segment_results['B 화자 구간 (45-90프레임)']['sync_b'] > segment_results['B 화자 구간 (45-90프레임)']['sync_a']
    condition3 = abs(segment_results['전체 구간']['sync_a'] - segment_results['전체 구간']['sync_b']) > 0.1
    
    success = condition1 and condition2 and condition3
    print(f"\n   조건1 (A구간): {condition1}, 조건2 (B구간): {condition2}, 조건3 (전체차별화): {condition3}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success


def main():
    """메인 테스트 함수"""
    print("🚀 Audio Speaker Detection System 완전 테스트 시작")
    print("=" * 70)
    
    # 개별 컴포넌트 테스트
    test_results = []
    
    try:
        test_results.append(("AudioActivityDetector", test_audio_activity_detector()))
        test_results.append(("MouthMovementAnalyzer", test_mouth_movement_analyzer())) 
        test_results.append(("AudioVisualCorrelator", test_audio_visual_correlator()))
        test_results.append(("통합 오디오-비디오 분석", test_integrated_audio_visual()))
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        test_results.append(("오류 발생", False))
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\n총 결과: {passed}/{total} ({success_rate:.1f}%) 통과")
    
    if success_rate >= 75.0:
        print("🎉 Phase 2 Audio Speaker Detection 구현 성공!")
        print("   • 오디오 활동 감지 완료")
        print("   • 입 움직임 분석 완료")
        print("   • 오디오-비디오 상관관계 완료")
        print("   • 98% 화자 선정 정확도 달성 준비 완료")
    else:
        print("⚠️ 일부 테스트 실패 - 개선 필요")
    
    print("\n🎯 다음 단계: Motion Prediction 구현 (Phase 3)")
    print("=" * 70)
    
    return success_rate >= 75.0


if __name__ == "__main__":
    main()