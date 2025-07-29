# Face-Tracking-App

## 📝 프로젝트 개요

이 프로젝트는 비디오에서 얼굴을 탐지하고, 동일 인물을 식별하여 트래킹하는 딥러닝 기반 애플리케이션입니다. MTCNN과 FaceNet (InceptionResnetV1) 모델을 사용하여 높은 정확도의 얼굴 인식 및 추적 기능을 제공합니다.

주요 목표는 대용량 비디오 파일을 효율적으로 처리하기 위해 GPU 사용률을 극대화하고, 전체 처리 시간을 단축하는 것입니다.

## 🛠️ 기술 스택

*   **언어**: Python
*   **딥러닝/컴퓨터 비전**: PyTorch, facenet-pytorch, MTCNN, OpenCV
*   **비디오/오디오 처리**: MoviePy, FFmpeg
*   **성능 최적화**: Producer-Consumer 패턴, 동적 배치 크기 조정, 멀티프로세싱

## 📂 프로젝트 구조

```
Face-Tracking-App/
├── main.py                # 메인 실행 파일
├── config.py              # 프로젝트 설정 파일
├── requirements.txt       # 의존성 라이브러리 목록
├── refactor.md            # 성능 개선 로드맵 및 기록
├── processors/            # 비디오 처리 핵심 로직
│   ├── video_processor.py   # 전체 처리 파이프라인
│   ├── face_analyzer.py     # 얼굴 탐지 (Producer-Consumer)
│   └── id_timeline_generator.py # 얼굴 ID 생성 (Producer-Consumer)
├── core/                  # 핵심 모델 및 데이터 관리
│   ├── model_manager.py     # 딥러닝 모델 관리 (싱글톤)
│   └── embedding_manager.py # 얼굴 임베딩 벡터 관리
├── utils/                 # 각종 유틸리티 함수
└── videos/                # 입력/출력 비디오 저장
```

## 🚀 실행 방법

1.  **가상환경 활성화**:
    ```bash
    source .venv/bin/activate
    ```

2.  **의존성 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **입력 비디오 준비**:
    `videos/input` 디렉토리에 처리할 비디오 파일을 넣어주세요.

4.  **애플리케이션 실행**:
    ```bash
    python main.py
    ```
    처리 결과는 `videos/output` 디렉토리에 저장됩니다.

## ✅ 개선 과제 (To-Do List)

`refactor.md` 문서와 현재 코드베이스를 기반으로, 프로젝트의 안정성과 성능을 향상시키기 위한 과제는 다음과 같습니다.

### 완료된 과제

-   [x] **GPU 자원 경합 문제 해결**:
    -   멀티프로세스 환경에서 다수의 프로세스가 동시에 GPU에 접근하여 발생하던 CUDA 메모리 오버플로우 문제를 해결했습니다.
    -   **GPU Worker + CPU Pool 아키텍처**를 도입하여, GPU 작업은 단일 전용 프로세스가 순차적으로 처리하고, CPU 집약적 작업(FFmpeg 인코딩)은 별도의 프로세스 풀에서 병렬로 처리하도록 변경했습니다.

### 진행할 과제

#### 우선순위: 높음 (안정성 강화)

-   [ ] **FFmpeg 처리 안정화 및 `moviepy` 의존성 완전 제거**:
    -   `process_cpu_task` 내 FFmpeg 실패 시 `moviepy`를 사용하는 Fallback 로직을 분석하고, FFmpeg가 실패하는 예외 케이스를 처리하여 `moviepy` 의존성을 제거합니다.
-   [ ] **강력한 예외 처리 시스템 구축**:
    -   `GPUMemoryError`, `VideoProcessingError` 등 사용자 정의 예외 클래스를 도입하여, 오류 발생 시 원인 파악 및 대응이 용이하도록 개선합니다.
-   [ ] **입력 파일 검증 강화**:
    -   처리 시작 전, 입력 비디오 파일의 유효성(MIME 타입, 코덱, 해상도 등)을 검증하는 단계를 추가하여 파이프라인의 안정성을 높입니다.

#### 우선순위: 중간 (성능 최적화)

-   [ ] **전체 파이프라인 비동기화**:
    -   현재 동기식으로 동작하는 전체 처리 흐름(얼굴 분석 → ID 타임라인 → 후처리)을 `asyncio` 또는 `Queue` 기반의 비동기 파이프라인으로 전환합니다.
    -   이를 통해 각 처리 단계 간의 대기 시간을 최소화하고, CPU와 GPU 리소스 활용률을 극대화합니다.

#### 우선순위: 낮음 (운영 및 확장성)

-   [ ] **단위 테스트 및 CI/CD 구축**:
    -   `pytest`를 사용하여 핵심 기능에 대한 단위 테스트를 작성하고, GitHub Actions와 연동하여 코드 변경 시 안정성을 보장하는 CI 파이프라인을 구축합니다.
