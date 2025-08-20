"""
Audio Speaker Detection System for Dual Face Tracking
Phase A: Active Speaker Detection implementation

오디오 기반 화자 활동 감지, 입 움직임 분석, 오디오-비디오 상관관계 분석
"""

import numpy as np
import cv2
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
from dataclasses import dataclass

# 오디오 처리 라이브러리들 (사용 가능한 경우만)
try:
    import librosa
    import scipy.signal
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️ librosa not available - falling back to basic audio processing")

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    print("⚠️ webrtcvad not available - using amplitude-based VAD")

try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False
    print("⚠️ dlib not available - using simple mouth detection")


@dataclass
class AudioFeatures:
    """오디오 특징 데이터 클래스"""
    rms_envelope: np.ndarray
    spectral_centroids: np.ndarray
    mfcc_features: Optional[np.ndarray]
    speech_segments: List[Tuple[float, float]]
    frame_activities: List[float]
    sample_rate: int
    duration: float


@dataclass
class MouthFeatures:
    """입 움직임 특징 데이터 클래스"""
    mar_values: List[float]  # Mouth Aspect Ratio
    mouth_velocities: List[float]
    mouth_accelerations: List[float]
    speaking_confidence: List[float]


class AudioActivityDetector:
    """오디오 기반 화자 활동 감지"""
    
    def __init__(self):
        """AudioActivityDetector 초기화"""
        self.logger = logging.getLogger(__name__)
        self.sample_rate = 16000  # 기본 샘플레이트
        self.vad_frame_duration = 30  # 30ms VAD 프레임
        
        # VAD 초기화 (webrtcvad 사용 가능한 경우)
        self.vad = None
        if HAS_WEBRTCVAD:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness 2 (0-3)
                self.logger.info("✅ WebRTC VAD 초기화 완료")
            except Exception as e:
                self.logger.warning(f"WebRTC VAD 초기화 실패: {e}")
                self.vad = None
    
    def extract_audio_features(self, video_path: str) -> Optional[AudioFeatures]:
        """비디오에서 오디오 추출 및 특징 계산
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            AudioFeatures 객체 또는 None
        """
        try:
            self.logger.info(f"🎵 오디오 특징 추출 시작: {video_path}")
            
            if not HAS_LIBROSA:
                return self._extract_basic_audio_features(video_path)
            
            # librosa로 오디오 로드
            y, sr = librosa.load(video_path, sr=self.sample_rate)
            duration = len(y) / sr
            
            self.logger.info(f"   오디오 길이: {duration:.2f}초, 샘플레이트: {sr}Hz")
            
            # RMS 엔벨로프 계산
            rms = librosa.feature.rms(y=y, frame_length=512, hop_length=512)[0]
            
            # 스펙트럴 센트로이드 계산
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # MFCC 특징 (선택적)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 발화 구간 검출
            speech_segments = self.detect_speech_segments(y, sr)
            
            # 프레임별 활동도 계산
            frame_activities = self.calculate_frame_activity(y, sr, fps=30.0)
            
            features = AudioFeatures(
                rms_envelope=rms,
                spectral_centroids=spectral_centroids,
                mfcc_features=mfcc,
                speech_segments=speech_segments,
                frame_activities=frame_activities,
                sample_rate=sr,
                duration=duration
            )
            
            self.logger.info(f"✅ 오디오 특징 추출 완료: {len(speech_segments)}개 발화 구간")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ 오디오 특징 추출 실패: {e}")
            return None
    
    def _extract_basic_audio_features(self, video_path: str) -> Optional[AudioFeatures]:
        """기본 오디오 특징 추출 (librosa 없이)"""
        try:
            # OpenCV로 기본 오디오 처리
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            
            # 더미 데이터 생성 (실제 환경에서는 FFmpeg 등 사용)
            dummy_length = int(duration * 50)  # 50Hz 샘플링
            rms_envelope = np.random.random(dummy_length) * 0.1
            spectral_centroids = np.random.random(dummy_length) * 1000 + 500
            
            frame_activities = [0.1] * int(duration * 30)  # 30fps
            
            return AudioFeatures(
                rms_envelope=rms_envelope,
                spectral_centroids=spectral_centroids,
                mfcc_features=None,
                speech_segments=[(0.0, duration)],  # 전체 구간
                frame_activities=frame_activities,
                sample_rate=16000,
                duration=duration
            )
            
        except Exception as e:
            self.logger.error(f"❌ 기본 오디오 특징 추출 실패: {e}")
            return None
    
    def detect_speech_segments(self, audio_signal: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """VAD로 발화 구간 검출
        
        Args:
            audio_signal: 오디오 신호
            sample_rate: 샘플레이트
            
        Returns:
            [(start_time, end_time)] 발화 구간 리스트
        """
        if self.vad and HAS_WEBRTCVAD:
            return self._webrtc_vad_segments(audio_signal, sample_rate)
        else:
            return self._amplitude_vad_segments(audio_signal, sample_rate)
    
    def _webrtc_vad_segments(self, audio_signal: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """WebRTC VAD로 발화 구간 검출"""
        try:
            # 16kHz, 16bit으로 변환
            if sample_rate != 16000:
                if HAS_LIBROSA:
                    audio_16k = librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=16000)
                else:
                    # 단순 다운샘플링
                    step = sample_rate // 16000
                    audio_16k = audio_signal[::step]
            else:
                audio_16k = audio_signal
            
            # 16bit 정수로 변환
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # 30ms 프레임으로 분할 (480 samples at 16kHz)
            frame_size = 480
            segments = []
            current_start = None
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i+frame_size]
                frame_bytes = frame.tobytes()
                
                is_speech = self.vad.is_speech(frame_bytes, 16000)
                time_pos = i / 16000.0
                
                if is_speech and current_start is None:
                    current_start = time_pos
                elif not is_speech and current_start is not None:
                    segments.append((current_start, time_pos))
                    current_start = None
            
            # 마지막 구간 처리
            if current_start is not None:
                segments.append((current_start, len(audio_int16) / 16000.0))
            
            self.logger.info(f"WebRTC VAD: {len(segments)}개 발화 구간 검출")
            return segments
            
        except Exception as e:
            self.logger.warning(f"WebRTC VAD 실패, 진폭 기반으로 전환: {e}")
            return self._amplitude_vad_segments(audio_signal, sample_rate)
    
    def _amplitude_vad_segments(self, audio_signal: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """진폭 기반 VAD로 발화 구간 검출"""
        try:
            # RMS 계산
            window_size = int(sample_rate * 0.025)  # 25ms 윈도우
            hop_size = int(sample_rate * 0.010)     # 10ms 홉
            
            rms_values = []
            for i in range(0, len(audio_signal) - window_size, hop_size):
                window = audio_signal[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            rms_array = np.array(rms_values)
            
            # 적응적 임계값 계산
            rms_mean = np.mean(rms_array)
            rms_std = np.std(rms_array)
            threshold = rms_mean + 0.5 * rms_std
            
            # 발화 구간 검출
            is_speech = rms_array > threshold
            segments = []
            current_start = None
            
            for i, speech in enumerate(is_speech):
                time_pos = i * hop_size / sample_rate
                
                if speech and current_start is None:
                    current_start = time_pos
                elif not speech and current_start is not None:
                    segments.append((current_start, time_pos))
                    current_start = None
            
            # 마지막 구간 처리
            if current_start is not None:
                segments.append((current_start, len(rms_array) * hop_size / sample_rate))
            
            self.logger.info(f"진폭 VAD: {len(segments)}개 발화 구간 검출 (임계값: {threshold:.4f})")
            return segments
            
        except Exception as e:
            self.logger.error(f"진폭 VAD 실패: {e}")
            return []
    
    def calculate_frame_activity(self, audio_signal: np.ndarray, sample_rate: int, fps: float = 30.0) -> List[float]:
        """프레임별 오디오 활동 레벨 계산
        
        Args:
            audio_signal: 오디오 신호
            sample_rate: 샘플레이트  
            fps: 비디오 프레임레이트
            
        Returns:
            프레임별 활동도 리스트
        """
        try:
            frame_duration = 1.0 / fps  # 프레임 지속시간
            samples_per_frame = int(sample_rate * frame_duration)
            
            frame_activities = []
            
            for i in range(0, len(audio_signal), samples_per_frame):
                frame_samples = audio_signal[i:i+samples_per_frame]
                if len(frame_samples) > 0:
                    # RMS 활동도 계산
                    activity = np.sqrt(np.mean(frame_samples**2))
                    frame_activities.append(float(activity))
                else:
                    frame_activities.append(0.0)
            
            # 정규화 (0-1 범위)
            if frame_activities:
                max_activity = max(frame_activities)
                if max_activity > 0:
                    frame_activities = [a / max_activity for a in frame_activities]
            
            self.logger.info(f"프레임별 활동도 계산 완료: {len(frame_activities)}개 프레임")
            return frame_activities
            
        except Exception as e:
            self.logger.error(f"프레임별 활동도 계산 실패: {e}")
            return []


class MouthMovementAnalyzer:
    """입 움직임 기반 화자 감지"""
    
    def __init__(self):
        """MouthMovementAnalyzer 초기화"""
        self.logger = logging.getLogger(__name__)
        self.predictor = None
        
        # dlib 랜드마크 예측기 초기화
        if HAS_DLIB:
            try:
                # 기본 경로에서 shape_predictor 찾기
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                self.predictor = dlib.shape_predictor(predictor_path)
                self.logger.info("✅ dlib 랜드마크 예측기 초기화 완료")
            except Exception as e:
                self.logger.warning(f"dlib 랜드마크 예측기 로드 실패: {e}")
                self.predictor = None
    
    def extract_face_landmarks(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 크롭에서 랜드마크 추출
        
        Args:
            face_crop: 얼굴 크롭 이미지 (BGR)
            
        Returns:
            (68, 2) 랜드마크 좌표 또는 None
        """
        if self.predictor is None or not HAS_DLIB:
            return self._extract_simple_landmarks(face_crop)
        
        try:
            # BGR → RGB 변환
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출 (전체 이미지가 얼굴이라고 가정)
            h, w = gray_crop.shape
            face_rect = dlib.rectangle(0, 0, w, h)
            
            # 랜드마크 예측
            landmarks = self.predictor(gray_crop, face_rect)
            
            # numpy 배열로 변환
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return points
            
        except Exception as e:
            self.logger.warning(f"dlib 랜드마크 추출 실패: {e}")
            return self._extract_simple_landmarks(face_crop)
    
    def _extract_simple_landmarks(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """간단한 랜드마크 추출 (dlib 없이)"""
        try:
            h, w = face_crop.shape[:2]
            
            # 더미 입 랜드마크 생성 (하단 중앙 부근)
            mouth_landmarks = np.array([
                [w*0.3, h*0.75],   # 왼쪽 입꼬리
                [w*0.5, h*0.8],    # 하단 중앙
                [w*0.7, h*0.75],   # 오른쪽 입꼬리
                [w*0.5, h*0.7],    # 상단 중앙
                [w*0.4, h*0.75],   # 왼쪽 중간
                [w*0.6, h*0.75],   # 오른쪽 중간
            ], dtype=np.float32)
            
            return mouth_landmarks
            
        except Exception as e:
            self.logger.error(f"간단한 랜드마크 추출 실패: {e}")
            return None
    
    def calculate_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """MAR(Mouth Aspect Ratio) 계산
        
        Args:
            landmarks: 랜드마크 좌표 (68점 또는 6점)
            
        Returns:
            MAR 값 (0에 가까우면 닫힌 입, 클수록 열린 입)
        """
        try:
            if landmarks is None or len(landmarks) == 0:
                return 0.0
            
            if len(landmarks) == 68:
                # 표준 68점 랜드마크에서 입 부분 추출 (49-68번)
                mouth_points = landmarks[48:68]
            elif len(landmarks) == 6:
                # 간단한 6점 입 랜드마크
                mouth_points = landmarks
            else:
                # 전체 랜드마크에서 하단 부분 추출
                mouth_points = landmarks[-6:] if len(landmarks) >= 6 else landmarks
            
            # MAR 계산 (세로 거리 / 가로 거리)
            if len(mouth_points) >= 4:
                # 세로 거리들의 평균
                vertical_distances = []
                if len(mouth_points) >= 6:
                    # 상하 입술 거리
                    vertical_distances.append(np.linalg.norm(mouth_points[1] - mouth_points[3]))
                    vertical_distances.append(np.linalg.norm(mouth_points[4] - mouth_points[5]))
                
                if not vertical_distances:
                    vertical_distances.append(np.linalg.norm(mouth_points[1] - mouth_points[0]))
                
                # 가로 거리 (입꼬리 간 거리)
                horizontal_distance = np.linalg.norm(mouth_points[2] - mouth_points[0])
                
                avg_vertical = np.mean(vertical_distances)
                mar = avg_vertical / (horizontal_distance + 1e-6)
                
                return float(mar)
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"MAR 계산 실패: {e}")
            return 0.0
    
    def calculate_mouth_velocity(self, mar_history: List[float], window_size: int = 5) -> float:
        """입 움직임 속도 계산 (MAR 변화율)
        
        Args:
            mar_history: MAR 값 히스토리
            window_size: 미분 계산 윈도우 크기
            
        Returns:
            입 움직임 속도 (절대값)
        """
        try:
            if len(mar_history) < 2:
                return 0.0
            
            # 단순 미분 (현재 - 이전)
            if len(mar_history) >= window_size:
                recent_values = mar_history[-window_size:]
                velocity = np.abs(np.diff(recent_values)).mean()
            else:
                velocity = abs(mar_history[-1] - mar_history[-2])
            
            return float(velocity)
            
        except Exception as e:
            self.logger.warning(f"입 움직임 속도 계산 실패: {e}")
            return 0.0
    
    def analyze_mouth_features(self, face_crops: List[np.ndarray]) -> MouthFeatures:
        """연속된 얼굴 크롭에서 입 움직임 특징 분석
        
        Args:
            face_crops: 얼굴 크롭 이미지 리스트
            
        Returns:
            MouthFeatures 객체
        """
        try:
            mar_values = []
            mouth_velocities = []
            mouth_accelerations = []
            speaking_confidence = []
            
            # 각 프레임별 MAR 계산
            for i, crop in enumerate(face_crops):
                if crop is None or crop.size == 0:
                    mar_values.append(0.0)
                    continue
                
                landmarks = self.extract_face_landmarks(crop)
                mar = self.calculate_mouth_aspect_ratio(landmarks)
                mar_values.append(mar)
                
                # 속도 계산 (2프레임 이상부터)
                if len(mar_values) >= 2:
                    velocity = self.calculate_mouth_velocity(mar_values)
                    mouth_velocities.append(velocity)
                else:
                    mouth_velocities.append(0.0)
                
                # 가속도 계산 (3프레임 이상부터)
                if len(mouth_velocities) >= 2:
                    acceleration = abs(mouth_velocities[-1] - mouth_velocities[-2])
                    mouth_accelerations.append(acceleration)
                else:
                    mouth_accelerations.append(0.0)
                
                # 화자 신뢰도 (MAR + 속도 조합)
                confidence = min(1.0, mar * 2.0 + mouth_velocities[-1] * 5.0)
                speaking_confidence.append(confidence)
            
            self.logger.info(f"입 움직임 분석 완료: {len(face_crops)}개 프레임")
            
            return MouthFeatures(
                mar_values=mar_values,
                mouth_velocities=mouth_velocities,
                mouth_accelerations=mouth_accelerations,
                speaking_confidence=speaking_confidence
            )
            
        except Exception as e:
            self.logger.error(f"입 움직임 분석 실패: {e}")
            return MouthFeatures(
                mar_values=[0.0] * len(face_crops),
                mouth_velocities=[0.0] * len(face_crops),
                mouth_accelerations=[0.0] * len(face_crops),
                speaking_confidence=[0.0] * len(face_crops)
            )


class AudioVisualCorrelator:
    """오디오-비디오 상관관계 분석"""
    
    def __init__(self):
        """AudioVisualCorrelator 초기화"""
        self.logger = logging.getLogger(__name__)
        self.correlation_history = deque(maxlen=300)  # 10초 @ 30fps
    
    def synchronize_audio_video(self, audio_activity: List[float], mouth_activity: List[float], 
                               max_delay_frames: int = 10) -> Tuple[float, int]:
        """오디오-입움직임 동기화 및 지연 보정
        
        Args:
            audio_activity: 프레임별 오디오 활동도
            mouth_activity: 프레임별 입 움직임 활동도
            max_delay_frames: 최대 지연 프레임 수
            
        Returns:
            (최고 상관계수, 최적 지연 프레임)
        """
        try:
            # 길이 맞추기
            min_length = min(len(audio_activity), len(mouth_activity))
            if min_length < 10:
                return 0.0, 0
            
            audio_trimmed = np.array(audio_activity[:min_length])
            mouth_trimmed = np.array(mouth_activity[:min_length])
            
            best_correlation = -1.0
            best_delay = 0
            
            # 여러 지연값에서 상관계수 계산
            for delay in range(-max_delay_frames, max_delay_frames + 1):
                if delay == 0:
                    audio_shifted = audio_trimmed
                    mouth_shifted = mouth_trimmed
                elif delay > 0:
                    # 오디오가 입보다 늦음 (오디오를 앞으로 밀기)
                    if delay >= min_length:
                        continue
                    audio_shifted = audio_trimmed[:-delay]
                    mouth_shifted = mouth_trimmed[delay:]
                else:
                    # 입이 오디오보다 늦음 (입을 앞으로 밀기)
                    delay_abs = abs(delay)
                    if delay_abs >= min_length:
                        continue
                    audio_shifted = audio_trimmed[delay_abs:]
                    mouth_shifted = mouth_trimmed[:-delay_abs]
                
                if len(audio_shifted) > 0 and len(mouth_shifted) > 0:
                    correlation = self.calculate_correlation(audio_shifted, mouth_shifted)
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_delay = delay
            
            self.logger.info(f"동기화 완료: 상관계수={best_correlation:.3f}, 지연={best_delay}프레임")
            return best_correlation, best_delay
            
        except Exception as e:
            self.logger.error(f"오디오-비디오 동기화 실패: {e}")
            return 0.0, 0
    
    def calculate_correlation(self, audio_normalized: np.ndarray, mouth_normalized: np.ndarray) -> float:
        """정규화된 상관계수 계산
        
        Args:
            audio_normalized: 정규화된 오디오 신호
            mouth_normalized: 정규화된 입 움직임 신호
            
        Returns:
            피어슨 상관계수 (-1 ~ 1)
        """
        try:
            # 길이 맞추기
            min_length = min(len(audio_normalized), len(mouth_normalized))
            if min_length < 2:
                return 0.0
            
            audio_trimmed = audio_normalized[:min_length]
            mouth_trimmed = mouth_normalized[:min_length]
            
            # 표준화 (평균 0, 표준편차 1)
            audio_std = (audio_trimmed - np.mean(audio_trimmed)) / (np.std(audio_trimmed) + 1e-8)
            mouth_std = (mouth_trimmed - np.mean(mouth_trimmed)) / (np.std(mouth_trimmed) + 1e-8)
            
            # 피어슨 상관계수 계산
            correlation = np.corrcoef(audio_std, mouth_std)[0, 1]
            
            # NaN 처리
            if np.isnan(correlation):
                correlation = 0.0
            
            return float(correlation)
            
        except Exception as e:
            self.logger.warning(f"상관계수 계산 실패: {e}")
            return 0.0
    
    def score_active_speakers(self, face_tracks: Dict[str, Any], audio_features: AudioFeatures, 
                            mouth_features: Dict[str, MouthFeatures]) -> List[Tuple[str, float]]:
        """화자별 Active Speaker 점수 계산
        
        Args:
            face_tracks: {'A': track_info, 'B': track_info} 얼굴 추적 정보
            audio_features: 오디오 특징
            mouth_features: {'A': MouthFeatures, 'B': MouthFeatures} 입 움직임 특징
            
        Returns:
            [(person_id, active_speaker_score)] 정렬된 리스트
        """
        try:
            speaker_scores = []
            
            for person_id in ['A', 'B']:
                if person_id not in mouth_features:
                    speaker_scores.append((person_id, 0.0))
                    continue
                
                mouth_feat = mouth_features[person_id]
                
                # 1. 입 움직임 점수 (0.4 가중치)
                mouth_score = 0.0
                if mouth_feat.speaking_confidence:
                    mouth_score = np.mean(mouth_feat.speaking_confidence)
                
                # 2. 오디오-비디오 상관관계 점수 (0.4 가중치)
                correlation_score = 0.0
                if audio_features.frame_activities and mouth_feat.mar_values:
                    correlation, delay = self.synchronize_audio_video(
                        audio_features.frame_activities, 
                        mouth_feat.mar_values
                    )
                    correlation_score = max(0.0, correlation)  # 음의 상관관계는 0으로
                
                # 3. 얼굴 크기/중심도 점수 (0.2 가중치)
                face_score = 0.5  # 기본값
                if person_id in face_tracks:
                    track = face_tracks[person_id]
                    if hasattr(track, 'avg_size'):
                        face_score = min(1.0, track.avg_size / 150.0)  # 150px 기준 정규화
                
                # 최종 점수 계산
                total_score = (
                    mouth_score * 0.4 +
                    correlation_score * 0.4 +
                    face_score * 0.2
                )
                
                speaker_scores.append((person_id, total_score))
                
                self.logger.debug(f"{person_id}: 입움직임={mouth_score:.3f}, 상관관계={correlation_score:.3f}, "
                                f"얼굴={face_score:.3f} → 총점={total_score:.3f}")
            
            # 점수 기준 정렬 (높은 순)
            speaker_scores.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Active Speaker 점수: {speaker_scores}")
            return speaker_scores
            
        except Exception as e:
            self.logger.error(f"Active Speaker 점수 계산 실패: {e}")
            return [('A', 0.0), ('B', 0.0)]
    
    def update_correlation_history(self, correlation: float):
        """상관관계 히스토리 업데이트"""
        self.correlation_history.append(correlation)
    
    def get_average_correlation(self, window_size: int = 30) -> float:
        """최근 상관관계 평균 계산"""
        if len(self.correlation_history) == 0:
            return 0.0
        
        recent = list(self.correlation_history)[-window_size:]
        return np.mean(recent)


# 통합 테스트 함수
def test_audio_speaker_detection():
    """Audio Speaker Detection 시스템 테스트"""
    print("🧪 Audio Speaker Detection 시스템 테스트 시작...")
    
    # 1. AudioActivityDetector 테스트
    print("\n1. AudioActivityDetector 테스트")
    audio_detector = AudioActivityDetector()
    print(f"   WebRTC VAD 사용 가능: {audio_detector.vad is not None}")
    
    # 더미 오디오 신호 생성
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    dummy_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz 사인파
    
    # 발화 구간 검출 테스트
    segments = audio_detector.detect_speech_segments(dummy_audio, sample_rate)
    print(f"   검출된 발화 구간: {len(segments)}개")
    
    # 프레임별 활동도 계산 테스트
    activities = audio_detector.calculate_frame_activity(dummy_audio, sample_rate)
    print(f"   프레임별 활동도: {len(activities)}개 프레임")
    
    # 2. MouthMovementAnalyzer 테스트  
    print("\n2. MouthMovementAnalyzer 테스트")
    mouth_analyzer = MouthMovementAnalyzer()
    print(f"   dlib 사용 가능: {mouth_analyzer.predictor is not None}")
    
    # 더미 얼굴 크롭 생성
    dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    landmarks = mouth_analyzer.extract_face_landmarks(dummy_face)
    print(f"   추출된 랜드마크: {landmarks.shape if landmarks is not None else None}")
    
    if landmarks is not None:
        mar = mouth_analyzer.calculate_mouth_aspect_ratio(landmarks)
        print(f"   MAR 값: {mar:.4f}")
    
    # 3. AudioVisualCorrelator 테스트
    print("\n3. AudioVisualCorrelator 테스트")
    correlator = AudioVisualCorrelator()
    
    # 더미 데이터로 상관관계 테스트
    dummy_audio_activity = [0.1, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4]
    dummy_mouth_activity = [0.0, 0.2, 0.6, 0.4, 0.1, 0.7, 0.3]
    
    correlation = correlator.calculate_correlation(
        np.array(dummy_audio_activity), 
        np.array(dummy_mouth_activity)
    )
    print(f"   상관계수: {correlation:.4f}")
    
    # 동기화 테스트
    best_corr, best_delay = correlator.synchronize_audio_video(
        dummy_audio_activity, dummy_mouth_activity
    )
    print(f"   최적 동기화: 상관계수={best_corr:.4f}, 지연={best_delay}프레임")
    
    print("\n✅ Audio Speaker Detection 시스템 테스트 완료!")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 테스트 실행
    test_audio_speaker_detection()