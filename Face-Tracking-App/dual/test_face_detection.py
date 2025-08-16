#!/usr/bin/env python3
"""
얼굴 검출 테스트 스크립트
"""

import cv2
import numpy as np
from pathlib import Path

def test_haar_detection():
    """Haar Cascade 얼굴 검출 테스트"""
    print("🔍 Haar Cascade 얼굴 검출 테스트")
    
    # Haar Cascade 로드
    cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("❌ Haar Cascade 로드 실패")
        return False
    
    print("✅ Haar Cascade 로드 완료")
    
    # 테스트 비디오 열기
    video_path = "tests/videos/2people_sample1.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return False
    
    print(f"✅ 비디오 열기 완료: {video_path}")
    
    # 첫 10개 프레임 테스트
    frame_count = 0
    total_detections = 0
    
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 여러 파라미터로 얼굴 검출 시도
        detection_configs = [
            {"scaleFactor": 1.02, "minNeighbors": 1, "minSize": (10, 10)},
            {"scaleFactor": 1.05, "minNeighbors": 2, "minSize": (15, 15)},
            {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20)},
            {"scaleFactor": 1.2, "minNeighbors": 5, "minSize": (30, 30)},
        ]
        
        frame_detections = 0
        
        for i, config in enumerate(detection_configs):
            faces = face_cascade.detectMultiScale(gray, **config)
            detection_count = len(faces)
            frame_detections = max(frame_detections, detection_count)
            
            print(f"   프레임 {frame_count} - 설정 {i+1}: {detection_count}개 얼굴 검출")
            
            if detection_count > 0:
                for j, (x, y, w, h) in enumerate(faces):
                    print(f"     얼굴 {j+1}: ({x},{y}) {w}x{h}")
        
        total_detections += frame_detections
        print(f"📊 프레임 {frame_count} 최대 검출: {frame_detections}개")
        print()
    
    cap.release()
    
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    print(f"📈 평균 검출 개수: {avg_detections:.1f}개/프레임")
    
    if total_detections > 0:
        print("✅ 얼굴 검출 성공!")
        return True
    else:
        print("❌ 얼굴 검출 실패 - 모든 프레임에서 0개")
        return False

if __name__ == "__main__":
    success = test_haar_detection()
    exit(0 if success else 1)