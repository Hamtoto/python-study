"""
Audio Diarization System for Dual Face Tracking
Phase C: Speaker Diarization + Face-Speaker Matching

pyannote.audio 기반 화자 분할로 중간 화자 교체 감지 및 얼굴-화자 매칭
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import json

# pyannote.audio 사용 가능한 경우
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.core.io import AudioFile
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False
    print("⚠️ pyannote.audio not available - using mock diarization")

# 오디오 처리
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️ librosa not available - limited audio processing")


@dataclass
class SpeakerSegment:
    """화자 구간 데이터 클래스"""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    duration: float = 0.0
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class SpeakerProfile:
    """화자 프로파일 데이터 클래스"""
    speaker_id: str
    total_duration: float
    segment_count: int
    avg_segment_duration: float
    speaking_ratio: float
    time_segments: List[Tuple[float, float]]
    confidence_scores: List[float]


@dataclass
class FaceSpeakerMatch:
    """얼굴-화자 매칭 결과"""
    face_id: str  # 'A' or 'B'
    speaker_id: str
    confidence: float
    match_type: str  # 'temporal', 'frequency', 'correlation'
    supporting_evidence: Dict[str, float]


class SpeakerDiarization:
    """pyannote.audio 기반 화자 분할"""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        SpeakerDiarization 초기화
        
        Args:
            model_name: pyannote.audio 모델 이름
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.pipeline = None
        
        # pyannote.audio 파이프라인 초기화
        if HAS_PYANNOTE:
            try:
                self.pipeline = self._load_pretrained_model()
                if self.pipeline:
                    self.logger.info(f"✅ pyannote.audio 모델 로드 완료: {model_name}")
                else:
                    self.logger.warning("⚠️ pyannote.audio 모델 로드 실패 - Mock 모드로 동작")
            except Exception as e:
                self.logger.warning(f"pyannote.audio 초기화 실패: {e} - Mock 모드로 동작")
                self.pipeline = None
        else:
            self.logger.info("pyannote.audio 미사용 - Mock 모드로 동작")
    
    def _load_pretrained_model(self) -> Optional[Any]:
        """사전 훈련된 diarization 모델 로드"""
        try:
            # Hugging Face 토큰이 필요할 수 있음
            pipeline = Pipeline.from_pretrained(self.model_name)
            return pipeline
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            # 인증 토큰이 필요한 경우
            self.logger.info("💡 Hugging Face 토큰이 필요할 수 있습니다:")
            self.logger.info("   huggingface-cli login")
            self.logger.info("   또는 환경변수: HF_TOKEN=your_token")
            return None
    
    def diarize_audio(self, video_path: str, min_speakers: int = 2, max_speakers: int = 4) -> List[SpeakerSegment]:
        """오디오에서 화자별 구간 추출
        
        Args:
            video_path: 비디오 파일 경로
            min_speakers: 최소 화자 수
            max_speakers: 최대 화자 수
            
        Returns:
            화자별 구간 리스트
        """
        try:
            self.logger.info(f"🎤 화자 분할 시작: {video_path}")
            
            if self.pipeline:
                return self._pyannote_diarize(video_path, min_speakers, max_speakers)
            else:
                return self._mock_diarize(video_path, min_speakers, max_speakers)
                
        except Exception as e:
            self.logger.error(f"화자 분할 실패: {e}")
            return self._mock_diarize(video_path, min_speakers, max_speakers)
    
    def _pyannote_diarize(self, video_path: str, min_speakers: int, max_speakers: int) -> List[SpeakerSegment]:
        """pyannote.audio로 실제 화자 분할"""
        try:
            # 오디오 파일 처리
            audio_file = AudioFile(video_path)
            
            # 화자 분할 실행
            diarization = self.pipeline(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
            
            # 결과 변환
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                    confidence=0.8  # pyannote는 신뢰도를 직접 제공하지 않음
                )
                segments.append(segment)
            
            # 시간순 정렬
            segments.sort(key=lambda x: x.start_time)
            
            self.logger.info(f"✅ pyannote 화자 분할 완료: {len(segments)}개 구간, "
                           f"{len(set(s.speaker_id for s in segments))}명 화자")
            
            return segments
            
        except Exception as e:
            self.logger.error(f"pyannote 화자 분할 실패: {e}")
            return []
    
    def _mock_diarize(self, video_path: str, min_speakers: int, max_speakers: int) -> List[SpeakerSegment]:
        """Mock 화자 분할 (pyannote 없이)"""
        try:
            # 비디오 길이 추정
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 60.0
            cap.release()
            
            # Mock 화자 구간 생성 (교대로 말하는 패턴)
            segments = []
            speakers = [f"SPEAKER_{i:02d}" for i in range(min_speakers)]
            
            # 3-8초 구간으로 나누어 교대로 할당
            current_time = 0.0
            speaker_idx = 0
            
            while current_time < duration:
                # 3-8초 랜덤 구간
                segment_duration = 3.0 + np.random.random() * 5.0
                end_time = min(current_time + segment_duration, duration)
                
                if end_time - current_time < 0.5:  # 너무 짧으면 스킵
                    break
                
                segment = SpeakerSegment(
                    speaker_id=speakers[speaker_idx],
                    start_time=current_time,
                    end_time=end_time,
                    confidence=0.7 + np.random.random() * 0.2
                )
                segments.append(segment)
                
                # 다음 화자로 전환 (70% 확률로 교대, 30% 확률로 계속)
                if np.random.random() < 0.7:
                    speaker_idx = (speaker_idx + 1) % len(speakers)
                
                # 0.5-2초 침묵 구간
                silence_duration = 0.5 + np.random.random() * 1.5
                current_time = end_time + silence_duration
            
            self.logger.info(f"✅ Mock 화자 분할 완료: {len(segments)}개 구간, "
                           f"{len(speakers)}명 화자 (시뮬레이션)")
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Mock 화자 분할 실패: {e}")
            return []
    
    def analyze_speaker_profiles(self, segments: List[SpeakerSegment], total_duration: float) -> Dict[str, SpeakerProfile]:
        """화자별 프로파일 분석
        
        Args:
            segments: 화자 구간 리스트
            total_duration: 전체 영상 길이
            
        Returns:
            화자 ID별 프로파일 딕셔너리
        """
        try:
            profiles = {}
            
            # 화자별 구간 그룹화
            speaker_segments = defaultdict(list)
            for segment in segments:
                speaker_segments[segment.speaker_id].append(segment)
            
            # 각 화자별 프로파일 계산
            for speaker_id, speaker_segments_list in speaker_segments.items():
                # 기본 통계
                total_speaking_duration = sum(s.duration for s in speaker_segments_list)
                segment_count = len(speaker_segments_list)
                avg_segment_duration = total_speaking_duration / segment_count if segment_count > 0 else 0.0
                speaking_ratio = total_speaking_duration / total_duration if total_duration > 0 else 0.0
                
                # 시간 구간과 신뢰도
                time_segments = [(s.start_time, s.end_time) for s in speaker_segments_list]
                confidence_scores = [s.confidence for s in speaker_segments_list]
                
                profile = SpeakerProfile(
                    speaker_id=speaker_id,
                    total_duration=total_speaking_duration,
                    segment_count=segment_count,
                    avg_segment_duration=avg_segment_duration,
                    speaking_ratio=speaking_ratio,
                    time_segments=time_segments,
                    confidence_scores=confidence_scores
                )
                
                profiles[speaker_id] = profile
            
            self.logger.info(f"화자 프로파일 분석 완료: {len(profiles)}명")
            for speaker_id, profile in profiles.items():
                self.logger.info(f"  {speaker_id}: {profile.speaking_ratio:.1%} ({profile.total_duration:.1f}초, "
                               f"{profile.segment_count}구간)")
            
            return profiles
            
        except Exception as e:
            self.logger.error(f"화자 프로파일 분석 실패: {e}")
            return {}
    
    def get_speaker_timeline(self, segments: List[SpeakerSegment], fps: float = 30.0) -> List[Optional[str]]:
        """프레임별 화자 타임라인 생성
        
        Args:
            segments: 화자 구간 리스트
            fps: 프레임레이트
            
        Returns:
            프레임별 화자 ID 리스트 (None은 침묵)
        """
        try:
            if not segments:
                return []
            
            # 전체 길이 계산
            total_duration = max(s.end_time for s in segments)
            total_frames = int(total_duration * fps) + 1
            
            # 프레임별 화자 할당
            timeline = [None] * total_frames
            
            for segment in segments:
                start_frame = int(segment.start_time * fps)
                end_frame = int(segment.end_time * fps)
                
                for frame_idx in range(start_frame, min(end_frame, total_frames)):
                    timeline[frame_idx] = segment.speaker_id
            
            self.logger.info(f"화자 타임라인 생성 완료: {total_frames}프레임, "
                           f"{len([t for t in timeline if t is not None])}프레임 화자 활동")
            
            return timeline
            
        except Exception as e:
            self.logger.error(f"화자 타임라인 생성 실패: {e}")
            return []


class DiarizationMatcher:
    """화자 구간과 얼굴 추적 매칭"""
    
    def __init__(self):
        """DiarizationMatcher 초기화"""
        self.logger = logging.getLogger(__name__)
        self.matching_history = []
    
    def match_speakers_to_faces(self, diar_segments: List[SpeakerSegment], 
                               face_timeline: List[Dict[str, Any]], 
                               fps: float = 30.0) -> Dict[str, str]:
        """화자 ID와 얼굴 ID 매칭
        
        Args:
            diar_segments: 화자 분할 결과
            face_timeline: 프레임별 얼굴 추적 결과 [{'A': face_info, 'B': face_info}, ...]
            fps: 프레임레이트
            
        Returns:
            {'SPEAKER_00': 'A', 'SPEAKER_01': 'B'} 매칭 결과
        """
        try:
            self.logger.info(f"🔗 화자-얼굴 매칭 시작: {len(diar_segments)}개 화자 구간, "
                           f"{len(face_timeline)}개 프레임")
            
            if not diar_segments or not face_timeline:
                return {}
            
            # 화자 타임라인 생성
            speaker_timeline = self._get_speaker_timeline_from_segments(diar_segments, len(face_timeline))
            
            # 여러 방법으로 매칭 시도
            temporal_matches = self._temporal_matching(speaker_timeline, face_timeline)
            frequency_matches = self._frequency_matching(diar_segments, face_timeline, fps)
            correlation_matches = self._correlation_matching(speaker_timeline, face_timeline)
            
            # 투표 기반 최종 매칭
            final_matches = self._vote_based_matching(temporal_matches, frequency_matches, correlation_matches)
            
            self.logger.info(f"✅ 화자-얼굴 매칭 완료: {final_matches}")
            return final_matches
            
        except Exception as e:
            self.logger.error(f"화자-얼굴 매칭 실패: {e}")
            return {}
    
    def _get_speaker_timeline_from_segments(self, segments: List[SpeakerSegment], total_frames: int) -> List[Optional[str]]:
        """화자 구간에서 프레임별 타임라인 생성"""
        timeline = [None] * total_frames
        fps = 30.0  # 기본 프레임레이트
        
        for segment in segments:
            start_frame = int(segment.start_time * fps)
            end_frame = int(segment.end_time * fps)
            
            for frame_idx in range(start_frame, min(end_frame, total_frames)):
                timeline[frame_idx] = segment.speaker_id
        
        return timeline
    
    def _temporal_matching(self, speaker_timeline: List[Optional[str]], 
                          face_timeline: List[Dict[str, Any]]) -> Dict[str, str]:
        """시간적 동시 등장 기반 매칭"""
        try:
            # 화자별로 얼굴 A/B와 동시 등장 횟수 계산
            speaker_face_cooccurrence = defaultdict(lambda: {'A': 0, 'B': 0})
            
            min_length = min(len(speaker_timeline), len(face_timeline))
            
            for frame_idx in range(min_length):
                current_speaker = speaker_timeline[frame_idx]
                face_frame = face_timeline[frame_idx]
                
                if current_speaker and isinstance(face_frame, dict):
                    # 얼굴 A/B가 있는지 확인
                    if 'A' in face_frame and face_frame['A'] is not None:
                        speaker_face_cooccurrence[current_speaker]['A'] += 1
                    if 'B' in face_frame and face_frame['B'] is not None:
                        speaker_face_cooccurrence[current_speaker]['B'] += 1
            
            # 각 화자를 더 많이 동시 등장한 얼굴에 매칭
            temporal_matches = {}
            used_faces = set()
            
            # 동시 등장 횟수 기준으로 정렬
            speakers_sorted = sorted(speaker_face_cooccurrence.keys(), 
                                   key=lambda s: max(speaker_face_cooccurrence[s]['A'], 
                                                   speaker_face_cooccurrence[s]['B']), 
                                   reverse=True)
            
            for speaker in speakers_sorted:
                counts = speaker_face_cooccurrence[speaker]
                if counts['A'] > counts['B'] and 'A' not in used_faces:
                    temporal_matches[speaker] = 'A'
                    used_faces.add('A')
                elif counts['B'] > counts['A'] and 'B' not in used_faces:
                    temporal_matches[speaker] = 'B'
                    used_faces.add('B')
                elif 'A' not in used_faces:
                    temporal_matches[speaker] = 'A'
                    used_faces.add('A')
                elif 'B' not in used_faces:
                    temporal_matches[speaker] = 'B'
                    used_faces.add('B')
            
            self.logger.debug(f"시간적 매칭: {temporal_matches}")
            return temporal_matches
            
        except Exception as e:
            self.logger.error(f"시간적 매칭 실패: {e}")
            return {}
    
    def _frequency_matching(self, segments: List[SpeakerSegment], 
                           face_timeline: List[Dict[str, Any]], fps: float) -> Dict[str, str]:
        """발화 빈도 vs 얼굴 크기/중심도 매칭"""
        try:
            # 화자별 발화 통계
            speaker_stats = {}
            for segment in segments:
                if segment.speaker_id not in speaker_stats:
                    speaker_stats[segment.speaker_id] = {
                        'total_duration': 0.0,
                        'segment_count': 0,
                        'avg_confidence': 0.0
                    }
                
                stats = speaker_stats[segment.speaker_id]
                stats['total_duration'] += segment.duration
                stats['segment_count'] += 1
                stats['avg_confidence'] = (stats['avg_confidence'] * (stats['segment_count'] - 1) + segment.confidence) / stats['segment_count']
            
            # 얼굴별 평균 크기/중심도 계산
            face_stats = {'A': [], 'B': []}
            
            for frame_data in face_timeline:
                if isinstance(frame_data, dict):
                    for face_id in ['A', 'B']:
                        if face_id in frame_data and frame_data[face_id] is not None:
                            face_info = frame_data[face_id]
                            # 얼굴 크기 추정 (박스가 있다고 가정)
                            if hasattr(face_info, 'bbox') or isinstance(face_info, dict):
                                if hasattr(face_info, 'bbox'):
                                    bbox = face_info.bbox
                                elif isinstance(face_info, dict) and 'bbox' in face_info:
                                    bbox = face_info['bbox']
                                else:
                                    bbox = [0, 0, 100, 100]  # 기본값
                                
                                if len(bbox) >= 4:
                                    size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                    face_stats[face_id].append(size)
            
            # 얼굴별 평균 크기
            face_avg_sizes = {}
            for face_id in ['A', 'B']:
                if face_stats[face_id]:
                    face_avg_sizes[face_id] = np.mean(face_stats[face_id])
                else:
                    face_avg_sizes[face_id] = 0
            
            # 매칭: 더 많이 말하는 화자 → 더 큰 얼굴
            speakers_by_duration = sorted(speaker_stats.keys(), 
                                        key=lambda s: speaker_stats[s]['total_duration'], 
                                        reverse=True)
            faces_by_size = sorted(face_avg_sizes.keys(), 
                                 key=lambda f: face_avg_sizes[f], 
                                 reverse=True)
            
            frequency_matches = {}
            for i, speaker in enumerate(speakers_by_duration):
                if i < len(faces_by_size):
                    frequency_matches[speaker] = faces_by_size[i]
            
            self.logger.debug(f"빈도 매칭: {frequency_matches} "
                            f"(화자 발화시간: {[(s, f'{speaker_stats[s]['total_duration']:.1f}s') for s in speakers_by_duration]})")
            return frequency_matches
            
        except Exception as e:
            self.logger.error(f"빈도 매칭 실패: {e}")
            return {}
    
    def _correlation_matching(self, speaker_timeline: List[Optional[str]], 
                             face_timeline: List[Dict[str, Any]]) -> Dict[str, str]:
        """화자 활동과 얼굴 움직임 상관관계 매칭"""
        try:
            # 간단한 상관관계: 화자가 말할 때 얼굴 변화 감지
            # (실제로는 오디오-비디오 상관관계 사용해야 함)
            
            # 기본 매칭 (임시)
            unique_speakers = list(set([s for s in speaker_timeline if s is not None]))
            face_ids = ['A', 'B']
            
            correlation_matches = {}
            for i, speaker in enumerate(unique_speakers):
                if i < len(face_ids):
                    correlation_matches[speaker] = face_ids[i]
            
            self.logger.debug(f"상관관계 매칭: {correlation_matches}")
            return correlation_matches
            
        except Exception as e:
            self.logger.error(f"상관관계 매칭 실패: {e}")
            return {}
    
    def _vote_based_matching(self, temporal: Dict[str, str], 
                            frequency: Dict[str, str], 
                            correlation: Dict[str, str]) -> Dict[str, str]:
        """투표 기반 최종 매칭 결정"""
        try:
            # 모든 화자 수집
            all_speakers = set(list(temporal.keys()) + list(frequency.keys()) + list(correlation.keys()))
            
            final_matches = {}
            
            for speaker in all_speakers:
                votes = {}
                
                # 각 방법별 투표
                if speaker in temporal:
                    face = temporal[speaker]
                    votes[face] = votes.get(face, 0) + 2  # 시간적 매칭은 가중치 높음
                
                if speaker in frequency:
                    face = frequency[speaker]
                    votes[face] = votes.get(face, 0) + 1.5
                
                if speaker in correlation:
                    face = correlation[speaker]
                    votes[face] = votes.get(face, 0) + 1
                
                # 최다 득표 얼굴 선택
                if votes:
                    best_face = max(votes.keys(), key=lambda f: votes[f])
                    final_matches[speaker] = best_face
            
            # 중복 제거 (한 얼굴에 여러 화자 할당된 경우)
            face_to_speaker = {}
            conflicts = defaultdict(list)
            
            for speaker, face in final_matches.items():
                if face in face_to_speaker:
                    conflicts[face].append(speaker)
                    conflicts[face].append(face_to_speaker[face])
                else:
                    face_to_speaker[face] = speaker
            
            # 충돌 해결 (더 많이 말하는 화자 우선)
            for face, conflicting_speakers in conflicts.items():
                # frequency 매칭에서 더 높은 우선순위를 가진 화자 선택
                best_speaker = conflicting_speakers[0]  # 임시로 첫 번째 선택
                
                # 다른 화자들은 남은 얼굴에 할당
                for speaker in conflicting_speakers[1:]:
                    available_faces = ['A', 'B']
                    used_faces = set(face_to_speaker.keys())
                    remaining_faces = [f for f in available_faces if f not in used_faces]
                    
                    if remaining_faces:
                        face_to_speaker[remaining_faces[0]] = speaker
                
                face_to_speaker[face] = best_speaker
            
            # 결과 뒤집기 (speaker -> face)
            result = {speaker: face for face, speaker in face_to_speaker.items()}
            
            self.logger.info(f"투표 기반 최종 매칭: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"투표 기반 매칭 실패: {e}")
            return {}
    
    def get_realtime_speaker_change(self, current_frame: int, speaker_timeline: List[Optional[str]], 
                                   window_size: int = 90) -> Dict[str, Any]:
        """실시간 화자 변경 감지 (3초 윈도우 @ 30fps)
        
        Args:
            current_frame: 현재 프레임 번호
            speaker_timeline: 프레임별 화자 타임라인
            window_size: 분석 윈도우 크기 (기본 90프레임 = 3초)
            
        Returns:
            화자 변경 정보
        """
        try:
            if current_frame < window_size or current_frame >= len(speaker_timeline):
                return {'speaker_change': False, 'current_speaker': None}
            
            # 현재 윈도우와 이전 윈도우
            current_window = speaker_timeline[current_frame-window_size//2:current_frame+window_size//2]
            prev_window = speaker_timeline[current_frame-window_size:current_frame-window_size//2]
            
            # 각 윈도우의 주요 화자
            current_speakers = [s for s in current_window if s is not None]
            prev_speakers = [s for s in prev_window if s is not None]
            
            current_main = Counter(current_speakers).most_common(1)
            prev_main = Counter(prev_speakers).most_common(1)
            
            current_speaker = current_main[0][0] if current_main else None
            prev_speaker = prev_main[0][0] if prev_main else None
            
            speaker_change = (current_speaker != prev_speaker) and (current_speaker is not None)
            
            result = {
                'speaker_change': speaker_change,
                'current_speaker': current_speaker,
                'prev_speaker': prev_speaker,
                'confidence': current_main[0][1] / len(current_speakers) if current_main and current_speakers else 0.0
            }
            
            if speaker_change:
                self.logger.info(f"🔄 화자 변경 감지 (프레임 {current_frame}): {prev_speaker} → {current_speaker}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"실시간 화자 변경 감지 실패: {e}")
            return {'speaker_change': False, 'current_speaker': None}


# 통합 테스트 함수
def test_audio_diarization():
    """Audio Diarization 시스템 테스트"""
    print("🧪 Audio Diarization 시스템 테스트 시작...")
    
    # 1. SpeakerDiarization 테스트
    print("\n1. SpeakerDiarization 테스트")
    diarizer = SpeakerDiarization()
    
    print(f"   pyannote.audio 사용 가능: {diarizer.pipeline is not None}")
    
    # Mock 비디오로 화자 분할 테스트
    mock_video_path = "test_video.mp4"  # 실제로는 존재하지 않음
    segments = diarizer.diarize_audio(mock_video_path, min_speakers=2, max_speakers=3)
    
    print(f"   검출된 화자 구간: {len(segments)}개")
    if segments:
        speakers = set(s.speaker_id for s in segments)
        print(f"   화자 ID: {speakers}")
        print(f"   첫 번째 구간: {segments[0].speaker_id} ({segments[0].start_time:.1f}-{segments[0].end_time:.1f}초)")
    
    # 화자 프로파일 분석
    if segments:
        total_duration = max(s.end_time for s in segments)
        profiles = diarizer.analyze_speaker_profiles(segments, total_duration)
        print(f"   화자 프로파일: {len(profiles)}개")
        
        for speaker_id, profile in profiles.items():
            print(f"     {speaker_id}: {profile.speaking_ratio:.1%} 발화 "
                  f"({profile.total_duration:.1f}초, {profile.segment_count}구간)")
    
    # 2. DiarizationMatcher 테스트
    print("\n2. DiarizationMatcher 테스트")
    matcher = DiarizationMatcher()
    
    # Mock 얼굴 타임라인 생성
    mock_face_timeline = []
    for i in range(100):  # 100프레임
        face_data = {
            'A': {'bbox': [50, 50, 150, 150]} if i % 3 != 0 else None,
            'B': {'bbox': [200, 50, 300, 150]} if i % 4 != 0 else None
        }
        mock_face_timeline.append(face_data)
    
    # 화자-얼굴 매칭 테스트
    if segments:
        matches = matcher.match_speakers_to_faces(segments, mock_face_timeline)
        print(f"   화자-얼굴 매칭 결과: {matches}")
        
        # 실시간 화자 변경 감지 테스트
        speaker_timeline = diarizer.get_speaker_timeline(segments, fps=30.0)
        if speaker_timeline and len(speaker_timeline) > 90:
            change_info = matcher.get_realtime_speaker_change(95, speaker_timeline)
            print(f"   화자 변경 감지 (95프레임): {change_info}")
    
    # 3. 통합 워크플로우 테스트
    print("\n3. 통합 워크플로우 테스트")
    
    if segments and mock_face_timeline:
        # 전체 파이프라인
        print("   1) 화자 분할 → 2) 화자-얼굴 매칭 → 3) 실시간 변경 감지")
        
        # 화자 타임라인 생성
        timeline = diarizer.get_speaker_timeline(segments, fps=30.0)
        print(f"   화자 타임라인: {len(timeline)}프레임")
        
        # 타임라인 분석
        if timeline:
            active_frames = [t for t in timeline if t is not None]
            print(f"   활성 프레임: {len(active_frames)}/{len(timeline)} ({len(active_frames)/len(timeline)*100:.1f}%)")
            
            speaker_distribution = Counter(active_frames)
            print(f"   화자 분포: {dict(speaker_distribution)}")
    
    print("\n✅ Audio Diarization 시스템 테스트 완료!")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 테스트 실행
    test_audio_diarization()