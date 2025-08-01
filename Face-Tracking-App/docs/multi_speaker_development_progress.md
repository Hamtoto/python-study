# Face-Tracking-App 다중화자 모드 개발 진행상황

## 📋 프로젝트 개요
Face-Tracking-App에 SINGLE(1인)/DUAL(2인) 화자 모드를 추가하여 발화 구간별 화자 분리 기능을 구현

**기준 문서**: 
- `face_tracking_requirements.md` - 요구사항 정의서
- `face_tracking_functional_spec.md` - 기능 정의서
- `SYSTEM_ANALYSIS.md` - 현재 시스템 분석 결과

---

## ✅ 완료된 분석 작업

### 1. 현재 시스템 구조 분석 (완료)
- **기존 VAD 기능**: `get_voice_segments_ffmpeg()` 구현됨 ✅
- **얼굴 인식/추적**: MTCNN + FaceNet 완전 구현 ✅  
- **타겟 선택**: `TargetSelector` 클래스에서 `first_person`/`most_frequent` 지원 ✅
- **GPU 최적화**: Producer-Consumer 패턴으로 97.3% 활용률 달성 ✅

### 2. 요구사항 분석 및 설계 (완료)
- **SINGLE 모드**: 기존 `most_frequent` 모드로 이미 구현완료 ✅
- **DUAL 모드**: 발화 구간별 상위 2명 화자 식별 → **신규 개발 필요** 🔧

---

## 🚧 현재 개발 진행상황

### Phase 1: 분석 및 설계 ✅ (2025-08-01 완료)
- [x] 기존 시스템 구조 파악
- [x] VAD 기능 확인 (이미 구현됨)
- [x] 요구사항 분석 완료
- [x] 구현 계획 수립

### Phase 2: DUAL 모드 핵심 구현 🔧 (진행중)
- [ ] **2-1. TargetSelector 확장** (우선순위: 높음)
  - [ ] `select_dual_speakers()` 메소드 추가
  - [ ] 발화 구간별 상위 2명 화자 식별 로직 구현
  
- [ ] **2-2. Processor 수정** (우선순위: 높음)
  - [ ] DUAL 모드 분기 처리 추가
  - [ ] `speaker_a`/`speaker_b` 폴더 분리 저장 로직
  - [ ] 각 세그먼트별 화자 판별 및 개별 처리

- [ ] **2-3. Config 확장** (우선순위: 중간)
  - [ ] `TRACKING_MODE`에 "dual" 추가
  - [ ] 출력 경로 구조 설정 추가

### Phase 3: 테스트 및 검증 ⏳ (대기중)
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 수행
- [ ] 성능 최적화 검토

---

## 🎯 핵심 구현 알고리즘 (DUAL 모드)

```python
# 의사코드: DUAL 모드 화자 식별 로직
def select_dual_speakers(voice_segments, id_timeline):
    speakers = {"speaker_a": [], "speaker_b": []}
    
    for start_time, end_time in voice_segments:
        # 해당 발화 구간의 프레임 범위 계산
        frame_range = get_frame_range(start_time, end_time, fps)
        
        # 구간 내 face_id별 등장 빈도 카운트
        face_counts = count_faces_in_range(id_timeline, frame_range)
        
        # 상위 2명 선택
        top_2_speakers = get_top_n_speakers(face_counts, n=2)
        
        if len(top_2_speakers) >= 1:
            speakers["speaker_a"].append((start_time, end_time, top_2_speakers[0]))
        if len(top_2_speakers) >= 2:
            speakers["speaker_b"].append((start_time, end_time, top_2_speakers[1]))
    
    return speakers
```

---

## 📁 수정 대상 파일 목록

### 핵심 수정 파일
1. **`src/face_tracker/processing/selector.py`**
   - `select_dual_speakers()` 메소드 추가
   - 발화 구간별 화자 분리 로직

2. **`src/face_tracker/processing/processor.py`**
   - `process_single_video_optimized()` 함수 수정
   - DUAL 모드 분기 처리 및 개별 저장

3. **`src/face_tracker/config.py`**
   - `TRACKING_MODE` 옵션에 "dual" 추가
   - 출력 디렉토리 구조 설정

### 테스트 파일 (신규 생성)
4. **`test/test_dual_mode.py`** (예정)
   - DUAL 모드 단위 테스트

---

## 📊 예상 출력 구조

### 기존 (SINGLE 모드)
```
videos/output/
└── video_name/
    ├── segment_000.mp4
    ├── segment_001.mp4
    └── ...
```

### 신규 (DUAL 모드)
```
videos/output/
├── speaker_a/
│   ├── segment_000.mp4
│   ├── segment_001.mp4
│   └── ...
└── speaker_b/
    ├── segment_000.mp4
    ├── segment_002.mp4
    └── ...
```

---

## ⚡ 성능 목표
- **처리 속도**: 기존 15-20초 성능 유지
- **GPU 활용률**: 97.3% 유지
- **메모리 사용량**: 32GB 중 80% 이내

---

## 🎯 다음 작업 계획

### 즉시 착수 (2025-08-01)
1. `TargetSelector.select_dual_speakers()` 메소드 구현
2. Config에 DUAL 모드 설정 추가

### 이어서 진행
3. Processor에서 DUAL 모드 분기 로직 구현
4. 화자별 개별 저장 시스템 구축

### 마무리 단계  
5. 통합 테스트 및 성능 검증
6. 문서 업데이트

---

## 📝 개발 노트

### 기술적 고려사항
- **기존 아키텍처 유지**: Producer-Consumer 패턴 및 GPU 최적화 구조 보존
- **하위 호환성**: 기존 SINGLE 모드 동작에 영향 없음
- **확장성**: 향후 3인 이상 다중화자 확장 가능한 구조

### 위험 요소 및 대응
- **성능 저하 위험**: 화자 분리 로직 추가로 인한 처리 시간 증가 → 병렬 처리로 최소화
- **메모리 사용량 증가**: 세그먼트 복제 저장 → 임시 파일 관리 최적화
- **화자 식별 정확도**: 발화 구간 내 얼굴 없음 예외 → 로그 기록 후 continue 정책

---

**최종 업데이트**: 2025-08-01  
**다음 체크포인트**: TargetSelector 확장 완료 후  
**예상 완료일**: 2025-08-01 (당일 완료 목표)