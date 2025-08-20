#!/usr/bin/env python3
"""
실제 영상을 사용한 전체 파이프라인 테스트.

FaceDetector -> ByteTracker -> ConditionalReID 전체 파이프라인을
실제 영상으로 테스트합니다.
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

from dual_face_tracker.inference import FaceDetector, ReIDModelConfig
from dual_face_tracker.core import (
    ByteTracker, ByteTrackConfig, ConditionalReID
)
from dual_face_tracker.utils.logger import setup_dual_face_logger


class RealVideoPipelineTest:
    """실제 영상을 사용한 파이프라인 테스트 클래스."""
    
    def __init__(self):
        """테스트 환경 초기화."""
        self.logger = setup_dual_face_logger("INFO")
        self.logger.info("🎬 실제 영상 파이프라인 테스트 시작")
        
        # 경로 설정
        self.videos_dir = Path("/workspace/videos")
        self.model_path = Path("models/yolov8n.onnx")
        self.output_dir = Path("/workspace/test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 컴포넌트
        self.face_detector = None
        self.byte_tracker = None
        self.conditional_reid = None
        
        # 통계
        self.frame_stats = []
        self.reid_activations = []
        
    def setup_components(self) -> bool:
        """파이프라인 컴포넌트를 초기화합니다."""
        try:
            self.logger.info("🔧 파이프라인 컴포넌트 초기화")
            
            # FaceDetector
            if not self.model_path.exists():
                self.logger.error(f"모델 파일 없음: {self.model_path}")
                return False
                
            self.face_detector = FaceDetector(
                model_path=str(self.model_path),
                confidence_threshold=0.5  # 실제 영상용 높은 임계값
            )
            
            # ByteTracker (얼굴 추적용 설정)
            config = ByteTrackConfig.for_face_tracking()
            self.byte_tracker = ByteTracker(**config.to_dict())
            
            # ConditionalReID (더 민감한 설정)
            reid_config = ReIDModelConfig.for_face_reid()
            self.conditional_reid = ConditionalReID(
                reid_model_config=reid_config,
                activation_threshold=0.3,  # 더 낮은 임계값
                target_activation_rate=0.05,  # 5% 목표
                swap_detector_config={
                    'position_threshold': 50.0,  # 더 민감한 위치 변화 감지
                    'size_change_threshold': 0.3,  # 더 민감한 크기 변화 감지
                    'overall_threshold': 0.3  # 더 낮은 종합 임계값
                }
            )
            
            self.logger.info("✅ 모든 컴포넌트 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            return False
    
    def find_video_files(self) -> List[Path]:
        """videos 디렉토리에서 영상 파일들을 찾습니다."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        if not self.videos_dir.exists():
            self.logger.error(f"영상 디렉토리가 존재하지 않습니다: {self.videos_dir}")
            return []
        
        for ext in video_extensions:
            video_files.extend(self.videos_dir.glob(f"**/*{ext}"))
            
        self.logger.info(f"발견된 영상 파일: {len(video_files)}개")
        for video_file in video_files:
            self.logger.info(f"  - {video_file.name}")
            
        return video_files
    
    def convert_detections(self, detections) -> List:
        """FaceDetector 출력을 ByteTracker 입력으로 변환."""
        from dual_face_tracker.core.tracking_structures import Detection as ByteDetection
        
        byte_detections = []
        for det in detections:
            if hasattr(det, 'bbox') and hasattr(det, 'confidence'):
                detection = ByteDetection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=getattr(det, 'class_id', 0)
                )
                byte_detections.append(detection)
        
        return byte_detections
    
    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """단일 영상을 처리합니다."""
        self.logger.info(f"🎬 영상 처리 시작: {video_path.name}")
        
        # 통계 초기화
        self.frame_stats.clear()
        self.reid_activations.clear()
        self.byte_tracker.reset()
        self.conditional_reid.reset()
        
        # 영상 열기
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"영상을 열 수 없습니다: {video_path}")
            return {}
        
        # 영상 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"영상 정보: {width}x{height}, {fps}fps, {total_frames}프레임")
        
        # 결과 영상 준비 (선택적)
        output_video_path = self.output_dir / f"processed_{video_path.name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_detection_time = 0.0
        total_tracking_time = 0.0
        total_reid_time = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 처리 시간 측정
                frame_start_time = time.perf_counter()
                
                # 1. Face Detection
                detection_start = time.perf_counter()
                detections = self.face_detector.detect(frame)
                detection_time = (time.perf_counter() - detection_start) * 1000
                total_detection_time += detection_time
                
                # 2. ByteTracker
                tracking_start = time.perf_counter()
                byte_detections = self.convert_detections(detections)
                tracks = self.byte_tracker.update(byte_detections)
                tracking_time = (time.perf_counter() - tracking_start) * 1000
                total_tracking_time += tracking_time
                
                # 3. ConditionalReID
                reid_start = time.perf_counter()
                reid_result = self.conditional_reid.process_frame(tracks, frame)
                reid_time = (time.perf_counter() - reid_start) * 1000
                total_reid_time += reid_time
                
                frame_total_time = (time.perf_counter() - frame_start_time) * 1000
                
                # 통계 수집
                frame_stat = {
                    'frame_id': frame_count,
                    'detections': len(detections),
                    'tracks': len(tracks),
                    'reid_activated': reid_result.reid_activated,
                    'swap_risk': reid_result.swap_detection_result.overall_risk_score,
                    'detection_time_ms': detection_time,
                    'tracking_time_ms': tracking_time,
                    'reid_time_ms': reid_time,
                    'total_time_ms': frame_total_time
                }
                self.frame_stats.append(frame_stat)
                
                # ReID 활성화 기록
                if reid_result.reid_activated:
                    activation_info = {
                        'frame_id': frame_count,
                        'risk_score': reid_result.swap_detection_result.overall_risk_score,
                        'affected_tracks': list(reid_result.swap_detection_result.affected_track_ids),
                        'indicators': [ind.indicator_type for ind in reid_result.swap_detection_result.indicators],
                        'processing_time_ms': reid_result.processing_time_ms
                    }
                    self.reid_activations.append(activation_info)
                    
                    self.logger.info(f"🔥 프레임 {frame_count}: ReID 활성화! "
                                   f"위험도={reid_result.swap_detection_result.overall_risk_score:.2f}, "
                                   f"트랙={len(reid_result.swap_detection_result.affected_track_ids)}개")
                
                # 결과 영상에 시각화 (간단한 바운딩 박스)
                vis_frame = frame.copy()
                for track in tracks:
                    if track.detection:
                        x1, y1, x2, y2 = map(int, track.detection.bbox)
                        color = (0, 255, 0) if not reid_result.reid_activated else (0, 0, 255)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis_frame, f"ID:{track.track_id}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # ReID 활성화 표시
                if reid_result.reid_activated:
                    cv2.putText(vis_frame, "ReID ACTIVE", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                out_writer.write(vis_frame)
                
                # 진행률 표시 (10프레임마다)
                if frame_count % 10 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"진행률: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # 테스트용으로 최대 300프레임만 처리
                if frame_count >= 300:
                    self.logger.info("테스트를 위해 300프레임에서 중단")
                    break
                    
        except Exception as e:
            self.logger.error(f"영상 처리 중 오류: {e}")
            
        finally:
            cap.release()
            out_writer.release()
        
        # 결과 분석
        result = self.analyze_results(video_path, frame_count, 
                                    total_detection_time, total_tracking_time, total_reid_time)
        
        self.logger.info(f"✅ 영상 처리 완료: {video_path.name}")
        return result
    
    def analyze_results(self, video_path: Path, frame_count: int,
                       total_detection_time: float, total_tracking_time: float, 
                       total_reid_time: float) -> Dict[str, Any]:
        """처리 결과를 분석합니다."""
        
        if frame_count == 0:
            return {}
        
        # 기본 통계
        avg_detection_time = total_detection_time / frame_count
        avg_tracking_time = total_tracking_time / frame_count
        avg_reid_time = total_reid_time / frame_count
        avg_total_time = (total_detection_time + total_tracking_time + total_reid_time) / frame_count
        
        # ReID 통계
        reid_activation_count = len(self.reid_activations)
        activation_rate = (reid_activation_count / frame_count) * 100
        
        # 감지/추적 통계
        total_detections = sum(stat['detections'] for stat in self.frame_stats)
        total_tracks = sum(stat['tracks'] for stat in self.frame_stats)
        avg_detections = total_detections / frame_count if frame_count > 0 else 0
        avg_tracks = total_tracks / frame_count if frame_count > 0 else 0
        
        # 시스템 통계
        conditional_reid_stats = self.conditional_reid.get_statistics()
        bytetrack_stats = self.byte_tracker.get_statistics()
        
        result = {
            'video_name': video_path.name,
            'frames_processed': frame_count,
            'performance': {
                'avg_detection_time_ms': avg_detection_time,
                'avg_tracking_time_ms': avg_tracking_time,
                'avg_reid_time_ms': avg_reid_time,
                'avg_total_time_ms': avg_total_time,
                'estimated_fps': 1000.0 / avg_total_time if avg_total_time > 0 else 0
            },
            'reid_stats': {
                'activation_count': reid_activation_count,
                'activation_rate_percent': activation_rate,
                'activations': self.reid_activations
            },
            'tracking_stats': {
                'avg_detections_per_frame': avg_detections,
                'avg_tracks_per_frame': avg_tracks,
                'total_detections': total_detections,
                'total_tracks': total_tracks
            },
            'system_stats': {
                'conditional_reid': conditional_reid_stats,
                'bytetrack': bytetrack_stats
            }
        }
        
        return result
    
    def print_detailed_results(self, results: Dict[str, Any]):
        """상세 결과를 출력합니다."""
        self.logger.info("=" * 80)
        self.logger.info("📊 실제 영상 파이프라인 테스트 결과")
        self.logger.info("=" * 80)
        
        if not results:
            self.logger.error("결과 데이터가 없습니다.")
            return
        
        perf = results['performance']
        reid = results['reid_stats']
        tracking = results['tracking_stats']
        
        self.logger.info(f"🎬 영상: {results['video_name']}")
        self.logger.info(f"📊 처리된 프레임: {results['frames_processed']}개")
        self.logger.info("")
        
        self.logger.info("⏱️ 성능 지표:")
        self.logger.info(f"   • 얼굴 감지: {perf['avg_detection_time_ms']:.2f}ms")
        self.logger.info(f"   • 추적: {perf['avg_tracking_time_ms']:.2f}ms")
        self.logger.info(f"   • ConditionalReID: {perf['avg_reid_time_ms']:.2f}ms")
        self.logger.info(f"   • 전체 파이프라인: {perf['avg_total_time_ms']:.2f}ms")
        self.logger.info(f"   • 예상 FPS: {perf['estimated_fps']:.1f}")
        self.logger.info("")
        
        self.logger.info("🔄 ReID 활성화 분석:")
        self.logger.info(f"   • 활성화 횟수: {reid['activation_count']}회")
        self.logger.info(f"   • 활성화율: {reid['activation_rate_percent']:.2f}%")
        
        if reid['activations']:
            self.logger.info("   • 활성화 상세:")
            for activation in reid['activations'][:5]:  # 최대 5개만 표시
                self.logger.info(f"     - 프레임 {activation['frame_id']}: "
                               f"위험도={activation['risk_score']:.2f}, "
                               f"지표=[{', '.join(activation['indicators'])}]")
            if len(reid['activations']) > 5:
                self.logger.info(f"     ... 및 {len(reid['activations']) - 5}개 더")
        self.logger.info("")
        
        self.logger.info("👥 추적 통계:")
        self.logger.info(f"   • 평균 감지 수: {tracking['avg_detections_per_frame']:.1f}개/프레임")
        self.logger.info(f"   • 평균 트랙 수: {tracking['avg_tracks_per_frame']:.1f}개/프레임")
        self.logger.info(f"   • 총 감지 횟수: {tracking['total_detections']}회")
        self.logger.info("")
        
        # 성능 평가
        self.logger.info("📈 종합 평가:")
        if perf['avg_total_time_ms'] < 33.3:  # 30 FPS 기준
            self.logger.info("   ✅ 실시간 처리 가능 (30+ FPS)")
        elif perf['avg_total_time_ms'] < 50:  # 20 FPS 기준
            self.logger.info("   ✅ 준실시간 처리 가능 (20+ FPS)")
        else:
            self.logger.info("   ⚠️ 실시간 처리 어려움")
            
        if 2 <= reid['activation_rate_percent'] <= 15:
            self.logger.info("   ✅ 적절한 ReID 활성화율")
        elif reid['activation_rate_percent'] < 2:
            self.logger.info("   ⚠️ ReID 활성화율이 너무 낮음 (스왑 감지 못할 수 있음)")
        else:
            self.logger.info("   ⚠️ ReID 활성화율이 너무 높음 (성능 저하 가능)")
    
    def run_test(self):
        """전체 테스트를 실행합니다."""
        self.logger.info("🚀 실제 영상 파이프라인 테스트 시작")
        
        # 컴포넌트 초기화
        if not self.setup_components():
            self.logger.error("❌ 초기화 실패")
            return False
        
        # 영상 파일 찾기
        video_files = self.find_video_files()
        if not video_files:
            self.logger.error("❌ 처리할 영상 파일이 없습니다.")
            self.logger.info("영상 파일을 다음 경로에 넣어주세요: /workspace/videos/")
            return False
        
        # 각 영상 처리
        all_results = []
        for video_file in video_files[:3]:  # 최대 3개 영상만 처리
            try:
                result = self.process_video(video_file)
                if result:
                    all_results.append(result)
                    self.print_detailed_results(result)
                    
            except Exception as e:
                self.logger.error(f"영상 처리 실패 {video_file.name}: {e}")
        
        if all_results:
            self.logger.info(f"🎉 테스트 완료! {len(all_results)}개 영상 처리됨")
            return True
        else:
            self.logger.error("❌ 처리된 영상이 없습니다.")
            return False


def main():
    """메인 실행 함수."""
    try:
        # DevContainer 환경 확인
        if not os.path.exists("/.dockerenv"):
            print("⚠️ 경고: DevContainer 환경이 아닐 수 있습니다.")
            print("   이 테스트는 DevContainer에서 실행해야 합니다.")
        
        # 테스트 실행
        test_runner = RealVideoPipelineTest()
        success = test_runner.run_test()
        
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