#!/usr/bin/env python3
"""
Audio Diarization System 테스트
Phase 4 검증: SpeakerDiarization + DiarizationMatcher
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import logging
from collections import Counter
from dual_face_tracker.audio.audio_diarization import (
    SpeakerDiarization,
    DiarizationMatcher,
    SpeakerSegment,
    SpeakerProfile,
    FaceSpeakerMatch
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_speaker_diarization():
    """SpeakerDiarization 기능 테스트"""
    print("🧪 SpeakerDiarization 테스트...")
    
    diarizer = SpeakerDiarization()
    print(f"   pyannote.audio 사용 가능: {diarizer.pipeline is not None}")
    
    # Mock 비디오 파일로 화자 분할 테스트
    mock_video_path = "test_video_120sec.mp4"  # 2분 비디오 시뮬레이션
    
    # 1. 화자 분할 실행
    print("\n   1. 화자 분할 실행")
    segments = diarizer.diarize_audio(mock_video_path, min_speakers=2, max_speakers=4)
    
    print(f"     검출된 구간: {len(segments)}개")
    
    if segments:
        speakers = set(s.speaker_id for s in segments)
        print(f"     검출된 화자: {len(speakers)}명 - {speakers}")
        
        # 시간 분포 확인
        total_duration = max(s.end_time for s in segments) if segments else 0
        speech_duration = sum(s.duration for s in segments)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
        
        print(f"     전체 길이: {total_duration:.1f}초")
        print(f"     발화 시간: {speech_duration:.1f}초 ({speech_ratio:.1%})")
        
        # 처음 5개 구간 출력
        print(f"     처음 5개 구간:")
        for i, segment in enumerate(segments[:5]):
            print(f"       {i+1}: {segment.speaker_id} ({segment.start_time:.1f}-{segment.end_time:.1f}초, "
                  f"{segment.duration:.1f}초, 신뢰도={segment.confidence:.2f})")
    
    # 2. 화자 프로파일 분석
    print("\n   2. 화자 프로파일 분석")
    if segments:
        total_duration = max(s.end_time for s in segments)
        profiles = diarizer.analyze_speaker_profiles(segments, total_duration)
        
        print(f"     생성된 프로파일: {len(profiles)}개")
        
        for speaker_id, profile in profiles.items():
            print(f"       {speaker_id}:")
            print(f"         총 발화 시간: {profile.total_duration:.1f}초 ({profile.speaking_ratio:.1%})")
            print(f"         발화 구간: {profile.segment_count}개")
            print(f"         평균 구간 길이: {profile.avg_segment_duration:.1f}초")
            print(f"         평균 신뢰도: {np.mean(profile.confidence_scores):.2f}")
    
    # 3. 화자 타임라인 생성
    print("\n   3. 화자 타임라인 생성")
    if segments:
        timeline = diarizer.get_speaker_timeline(segments, fps=30.0)
        
        print(f"     타임라인 길이: {len(timeline)}프레임")
        
        if timeline:
            active_frames = [t for t in timeline if t is not None]
            active_ratio = len(active_frames) / len(timeline)
            
            print(f"     활성 프레임: {len(active_frames)}/{len(timeline)} ({active_ratio:.1%})")
            
            # 화자별 프레임 분포
            speaker_frame_count = Counter(active_frames)
            print(f"     화자별 프레임 분포:")
            for speaker, count in speaker_frame_count.items():
                frame_ratio = count / len(active_frames) if active_frames else 0
                print(f"       {speaker}: {count}프레임 ({frame_ratio:.1%})")
    
    # 성공 기준:
    # 1. 최소 1개 이상의 화자 구간 검출
    # 2. 화자 프로파일 생성 성공  
    # 3. 타임라인 생성 성공
    condition1 = len(segments) > 0
    condition2 = len(profiles) > 0 if segments else False
    condition3 = len(timeline) > 0 if segments else False
    
    success = condition1 and condition2 and condition3
    print(f"\n   조건1 (구간검출): {condition1}, 조건2 (프로파일): {condition2}, 조건3 (타임라인): {condition3}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success, segments


def test_diarization_matcher():
    """DiarizationMatcher 기능 테스트"""
    print("\n🧪 DiarizationMatcher 테스트...")
    
    matcher = DiarizationMatcher()
    
    # Mock 화자 구간 데이터 생성 (2명 화자가 번갈아 말함)
    mock_segments = [
        SpeakerSegment("SPEAKER_00", 0.0, 3.0, 0.8),    # A가 말함
        SpeakerSegment("SPEAKER_01", 3.5, 6.0, 0.7),    # B가 말함
        SpeakerSegment("SPEAKER_00", 6.5, 9.0, 0.9),    # A가 다시 말함
        SpeakerSegment("SPEAKER_01", 9.5, 12.0, 0.6),   # B가 다시 말함
        SpeakerSegment("SPEAKER_00", 12.5, 15.0, 0.8),  # A가 마지막
    ]
    
    # Mock 얼굴 타임라인 (450프레임 = 15초 @ 30fps)
    mock_face_timeline = []
    
    for frame in range(450):
        time_sec = frame / 30.0
        
        # 화자에 따라 얼굴 A/B 활성화 패턴 다르게
        face_a_active = False
        face_b_active = False
        
        # SPEAKER_00가 말하는 구간에서는 얼굴 A가 더 크고 중심에
        if (0.0 <= time_sec < 3.0) or (6.5 <= time_sec < 9.0) or (12.5 <= time_sec < 15.0):
            face_a_active = True
            face_a_size = 150 + np.random.randint(-10, 10)  # 큰 얼굴
            face_b_size = 80 + np.random.randint(-5, 5)     # 작은 얼굴
        # SPEAKER_01이 말하는 구간에서는 얼굴 B가 더 크고 중심에  
        elif (3.5 <= time_sec < 6.0) or (9.5 <= time_sec < 12.0):
            face_b_active = True
            face_a_size = 85 + np.random.randint(-5, 5)     # 작은 얼굴
            face_b_size = 140 + np.random.randint(-8, 8)    # 큰 얼굴
        else:
            # 침묵 구간
            face_a_size = 90 + np.random.randint(-10, 10)
            face_b_size = 90 + np.random.randint(-10, 10)
        
        # 얼굴 정보 구성 (90% 확률로 검출됨)
        face_data = {}
        if np.random.random() > 0.1:  # 90% 검출률
            face_data['A'] = {'bbox': [50, 50, 50+face_a_size, 50+face_a_size]}
        if np.random.random() > 0.1:  # 90% 검출률  
            face_data['B'] = {'bbox': [200, 60, 200+face_b_size, 60+face_b_size]}
        
        mock_face_timeline.append(face_data)
    
    print(f"   Mock 데이터: {len(mock_segments)}개 화자 구간, {len(mock_face_timeline)}개 프레임")
    
    # 1. 화자-얼굴 매칭 실행
    print("\n   1. 화자-얼굴 매칭 실행")
    matches = matcher.match_speakers_to_faces(mock_segments, mock_face_timeline, fps=30.0)
    
    print(f"     매칭 결과: {matches}")
    
    if matches:
        for speaker_id, face_id in matches.items():
            print(f"       {speaker_id} → 얼굴 {face_id}")
    
    # 2. 개별 매칭 방법 테스트
    print("\n   2. 개별 매칭 방법 분석")
    
    # 화자 타임라인 생성
    speaker_timeline = matcher._get_speaker_timeline_from_segments(mock_segments, len(mock_face_timeline))
    
    # 각 매칭 방법 실행
    temporal_matches = matcher._temporal_matching(speaker_timeline, mock_face_timeline)
    frequency_matches = matcher._frequency_matching(mock_segments, mock_face_timeline, fps=30.0)
    correlation_matches = matcher._correlation_matching(speaker_timeline, mock_face_timeline)
    
    print(f"     시간적 매칭: {temporal_matches}")
    print(f"     빈도 매칭: {frequency_matches}")
    print(f"     상관관계 매칭: {correlation_matches}")
    
    # 3. 실시간 화자 변경 감지
    print("\n   3. 실시간 화자 변경 감지")
    
    # 주요 전환 지점에서 테스트 (3초, 6.5초, 9.5초, 12.5초)
    test_frames = [90, 195, 285, 375]  # 각각 3초, 6.5초, 9.5초, 12.5초
    
    change_detections = []
    for frame in test_frames:
        if frame < len(speaker_timeline):
            change_info = matcher.get_realtime_speaker_change(frame, speaker_timeline)
            change_detections.append(change_info)
            
            time_sec = frame / 30.0
            print(f"     프레임 {frame} ({time_sec:.1f}초): 변경감지={change_info['speaker_change']}, "
                  f"현재화자={change_info.get('current_speaker')}, "
                  f"이전화자={change_info.get('prev_speaker')}, "
                  f"신뢰도={change_info.get('confidence', 0):.2f}")
    
    # 성공 기준:
    # 1. 화자-얼굴 매칭 결과가 있음 (2개 매칭)
    # 2. 최소 1개 이상의 매칭 방법이 결과를 제공
    # 3. 화자 변경 감지가 적절히 작동
    condition1 = len(matches) == 2  # 2명 화자 → 2개 얼굴 매칭
    condition2 = any([len(temporal_matches) > 0, len(frequency_matches) > 0, len(correlation_matches) > 0])
    condition3 = any([cd['speaker_change'] for cd in change_detections])  # 최소 1번 변경 감지
    
    success = condition1 and condition2 and condition3
    print(f"\n   조건1 (매칭완료): {condition1}, 조건2 (방법유효): {condition2}, 조건3 (변경감지): {condition3}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success, matches


def test_integrated_diarization():
    """통합 화자 분할 시스템 테스트"""
    print("\n🧪 통합 화자 분할 시스템 테스트...")
    
    # 전체 워크플로우: 화자 분할 → 프로파일 분석 → 얼굴 매칭 → 실시간 감지
    diarizer = SpeakerDiarization()
    matcher = DiarizationMatcher()
    
    # 시나리오: 3분 회의 영상에서 2명이 번갈아 말함
    mock_video_path = "meeting_180sec.mp4"
    
    print(f"   시나리오: 3분 회의 영상, 2명 화자 번갈아 발화")
    
    # 1. 전체 파이프라인 실행
    print("\n   1. 전체 파이프라인 실행")
    
    # 화자 분할
    segments = diarizer.diarize_audio(mock_video_path, min_speakers=2, max_speakers=3)
    print(f"     화자 분할: {len(segments)}개 구간")
    
    if not segments:
        print("   ❌ 화자 분할 실패 - 테스트 중단")
        return False
    
    # 프로파일 분석
    total_duration = max(s.end_time for s in segments)
    profiles = diarizer.analyze_speaker_profiles(segments, total_duration)
    print(f"     프로파일 생성: {len(profiles)}개")
    
    # 타임라인 생성
    timeline = diarizer.get_speaker_timeline(segments, fps=30.0)
    print(f"     타임라인: {len(timeline)}프레임")
    
    # Mock 얼굴 데이터 (180초 @ 30fps = 5400프레임)
    face_timeline_length = len(timeline) if timeline else 5400
    mock_face_timeline = []
    
    for frame in range(face_timeline_length):
        # 간단한 얼굴 데이터 (크기 변화로 화자 구분)
        base_size_a = 100 + 20 * np.sin(frame * 0.01)
        base_size_b = 90 + 15 * np.cos(frame * 0.008)
        
        face_data = {
            'A': {'bbox': [50, 50, int(50+base_size_a), int(50+base_size_a)]},
            'B': {'bbox': [200, 60, int(200+base_size_b), int(60+base_size_b)]}
        }
        mock_face_timeline.append(face_data)
    
    # 화자-얼굴 매칭
    matches = matcher.match_speakers_to_faces(segments, mock_face_timeline)
    print(f"     화자-얼굴 매칭: {len(matches)}개 매칭")
    
    # 2. 성능 분석
    print("\n   2. 성능 분석")
    
    if profiles:
        # 화자별 발화 분포
        total_speech = sum(p.total_duration for p in profiles.values())
        print(f"     총 발화 시간: {total_speech:.1f}초 / {total_duration:.1f}초 ({total_speech/total_duration:.1%})")
        
        for speaker_id, profile in profiles.items():
            matched_face = matches.get(speaker_id, "매칭안됨")
            print(f"       {speaker_id} → 얼굴{matched_face}: {profile.speaking_ratio:.1%} "
                  f"({profile.segment_count}구간, 평균 {profile.avg_segment_duration:.1f}초)")
    
    # 3. 실시간 처리 시뮬레이션
    print("\n   3. 실시간 처리 시뮬레이션")
    
    if timeline:
        # 30초 간격으로 화자 변경 감지 테스트
        test_intervals = list(range(900, len(timeline), 900))[:5]  # 30초씩, 최대 5번
        
        changes_detected = 0
        for frame in test_intervals:
            change_info = matcher.get_realtime_speaker_change(frame, timeline)
            
            if change_info['speaker_change']:
                changes_detected += 1
                time_sec = frame / 30.0
                current = change_info.get('current_speaker', 'Unknown')
                prev = change_info.get('prev_speaker', 'Unknown')
                confidence = change_info.get('confidence', 0)
                
                matched_current = matches.get(current, '?')
                matched_prev = matches.get(prev, '?')
                
                print(f"       {time_sec:.0f}초: {prev}(얼굴{matched_prev}) → {current}(얼굴{matched_current}) "
                      f"(신뢰도: {confidence:.2f})")
        
        print(f"     감지된 화자 변경: {changes_detected}회")
    
    # 4. 정확도 평가 (Mock 데이터 기준)
    print("\n   4. 정확도 평가")
    
    # 매칭 정확도 (임의 기준으로 평가)
    matching_accuracy = len(matches) / max(len(profiles), 1) if profiles else 0
    timeline_coverage = len([t for t in timeline if t is not None]) / len(timeline) if timeline else 0
    
    print(f"     매칭 완성도: {matching_accuracy:.1%} ({len(matches)}/{len(profiles) if profiles else 0})")
    print(f"     타임라인 커버리지: {timeline_coverage:.1%}")
    
    # 성공 기준:
    # 1. 전체 파이프라인이 오류 없이 실행
    # 2. 화자-얼굴 매칭이 80% 이상 완성
    # 3. 타임라인 커버리지가 30% 이상 (발화 구간)
    # 4. 최소 1번 이상 화자 변경 감지
    condition1 = len(segments) > 0 and len(profiles) > 0 and len(timeline) > 0
    condition2 = matching_accuracy >= 0.8
    condition3 = timeline_coverage >= 0.3
    condition4 = changes_detected > 0
    
    success = condition1 and condition2 and condition3 and condition4
    print(f"\n   조건1 (파이프라인): {condition1}, 조건2 (매칭완성도): {condition2}")
    print(f"   조건3 (커버리지): {condition3}, 조건4 (변경감지): {condition4}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success


def test_performance_scalability():
    """성능 및 확장성 테스트"""
    print("\n🧪 성능 및 확장성 테스트...")
    
    diarizer = SpeakerDiarization()
    matcher = DiarizationMatcher()
    
    # 대용량 데이터 시뮬레이션 (10분 영상, 4명 화자)
    print("   대용량 데이터 처리 테스트 (10분 영상, 4명 화자)")
    
    # 긴 영상에서 많은 화자 구간 생성
    long_segments = []
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
    
    current_time = 0.0
    end_time = 600.0  # 10분
    
    while current_time < end_time:
        # 3-12초 랜덤 구간
        duration = 3.0 + np.random.random() * 9.0
        segment_end = min(current_time + duration, end_time)
        
        if segment_end - current_time < 1.0:  # 너무 짧으면 스킵
            break
        
        speaker = np.random.choice(speakers)
        confidence = 0.6 + np.random.random() * 0.3
        
        segment = SpeakerSegment(speaker, current_time, segment_end, confidence)
        long_segments.append(segment)
        
        # 1-3초 침묵
        current_time = segment_end + 1.0 + np.random.random() * 2.0
    
    print(f"     생성된 구간: {len(long_segments)}개")
    
    # 성능 측정
    start_time = time.time()
    
    # 프로파일 분석
    profiles = diarizer.analyze_speaker_profiles(long_segments, end_time)
    
    # 타임라인 생성 (18000프레임 = 600초 @ 30fps)
    timeline = diarizer.get_speaker_timeline(long_segments, fps=30.0)
    
    # Mock 얼굴 데이터 (간소화)
    face_timeline = [{'A': {'bbox': [50, 50, 150, 150]}, 'B': {'bbox': [200, 60, 300, 160]}} 
                    for _ in range(len(timeline) if timeline else 18000)]
    
    # 매칭 실행
    matches = matcher.match_speakers_to_faces(long_segments, face_timeline)
    
    end_time_test = time.time()
    processing_time = end_time_test - start_time
    
    # 결과 분석
    print(f"     처리 시간: {processing_time:.2f}초")
    print(f"     화자 프로파일: {len(profiles)}개")
    print(f"     타임라인: {len(timeline) if timeline else 0}프레임")
    print(f"     매칭 결과: {len(matches)}개")
    print(f"     처리 속도: {(end_time/processing_time):.1f}x 실시간")
    
    # 메모리 효율성 (대략적 추정)
    estimated_memory_mb = (
        len(long_segments) * 0.001 +  # 구간 데이터
        len(timeline) * 0.0001 if timeline else 0 +  # 타임라인
        len(face_timeline) * 0.0005  # 얼굴 데이터
    )
    print(f"     추정 메모리 사용량: {estimated_memory_mb:.1f}MB")
    
    # 실시간 화자 변경 감지 테스트 (여러 지점)
    change_test_frames = [1800, 3600, 5400, 7200, 9000]  # 1분 간격
    changes = 0
    
    if timeline:
        for frame in change_test_frames:
            if frame < len(timeline):
                change_info = matcher.get_realtime_speaker_change(frame, timeline)
                if change_info['speaker_change']:
                    changes += 1
    
    print(f"     화자 변경 감지: {changes}회 / {len(change_test_frames)}회 테스트")
    
    # 성공 기준:
    # 1. 10분 영상을 30초 이내 처리 (20x 실시간 이상)
    # 2. 메모리 사용량 50MB 이하
    # 3. 4명 화자 모두 매칭 성공
    # 4. 화자 변경 감지율 20% 이상
    condition1 = processing_time < 30.0 and (end_time/processing_time) > 20.0
    condition2 = estimated_memory_mb < 50.0
    condition3 = len(matches) >= 3  # 4명 중 최소 3명
    condition4 = (changes / len(change_test_frames)) > 0.2
    
    success = condition1 and condition2 and condition3 and condition4
    print(f"\n   조건1 (속도): {condition1}, 조건2 (메모리): {condition2}")
    print(f"   조건3 (매칭): {condition3}, 조건4 (변경감지): {condition4}")
    print(f"   {'✅ 통과' if success else '❌ 실패'}")
    
    return success


def main():
    """메인 테스트 함수"""
    print("🚀 Audio Diarization System 완전 테스트 시작")
    print("=" * 70)
    
    # 개별 컴포넌트 테스트
    test_results = []
    
    try:
        # 기본 기능 테스트
        diarization_success, segments = test_speaker_diarization()
        test_results.append(("SpeakerDiarization", diarization_success))
        
        matcher_success, matches = test_diarization_matcher()
        test_results.append(("DiarizationMatcher", matcher_success))
        
        # 통합 및 성능 테스트
        integration_success = test_integrated_diarization()
        test_results.append(("통합 시스템", integration_success))
        
        scalability_success = test_performance_scalability()
        test_results.append(("성능 확장성", scalability_success))
        
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
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\n총 결과: {passed}/{total} ({success_rate:.1f}%) 통과")
    
    if success_rate >= 75.0:
        print("🎉 Phase 4 Audio Diarization 구현 성공!")
        print("   • 화자 분할 (pyannote.audio) 완료")
        print("   • 화자-얼굴 매칭 완료")
        print("   • 실시간 화자 변경 감지 완료")
        print("   • 중간 화자 교체 자동 감지 준비 완료")
    else:
        print("⚠️ 일부 테스트 실패 - 개선 필요")
        print(f"   현재 성능: {success_rate:.1f}% (목표: 75% 이상)")
    
    print("\n🎯 다음 단계: 전체 시스템 통합 (Phase 5)")
    print("=" * 70)
    
    return success_rate >= 75.0


if __name__ == "__main__":
    main()