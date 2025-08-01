# Face-Tracking-App 프로젝트 개요

## 프로젝트 목적
Face-Tracking-App은 GPU 최적화된 비디오 처리 파이프라인으로, MTCNN과 FaceNet (InceptionResnetV1) 모델을 사용하여 얼굴 감지, 인식, 추적을 수행하는 시스템입니다. Producer-Consumer 패턴과 멀티프로세싱 아키텍처를 통해 고성능 비디오 처리를 제공합니다.

## 주요 기능
- 얼굴 감지: MTCNN을 사용한 실시간 얼굴 감지
- 얼굴 인식: FaceNet을 사용한 얼굴 특징 추출 및 인식
- 얼굴 추적: 비디오 프레임 간 얼굴 추적
- 비디오 세그먼트 생성: 특정 인물 중심의 비디오 세그먼트 자동 생성
- 오디오 VAD: FFmpeg 기반 음성 구간 감지

## 현재 성능 지표
- GPU 사용률: 97.3% (목표 95% 초과 달성)
- 처리 속도: 15-20초 per 비디오 (기존 45-60초 대비 67% 단축)
- 배치 처리: 얼굴 감지 256, 얼굴 인식 128
- 하드웨어: RTX 5090 32GB GPU 환경 최적화

## 핵심 기술 스택
- 딥러닝: PyTorch, facenet-pytorch, MTCNN
- 컴퓨터 비전: OpenCV, PIL
- 비디오 처리: FFmpeg (직접 호출)
- 성능 최적화: Producer-Consumer 패턴, 동적 배치 크기, 멀티프로세싱