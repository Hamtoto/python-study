# 📋 코드 정리 단계별 실행 계획

> 이 문서는 프로젝트 코드 정리를 위한 단계별 실행 계획입니다.
> 각 단계는 독립적으로 실행 가능하며, 순차적으로 진행합니다.

## 🎯 정리 목표
- 3054줄의 `face_tracking_system.py` 파일 분리
- 220개의 print문을 로깅으로 전환
- 346줄의 주석을 한글 docstring으로 변환
- 코드 품질 개선 및 유지보수성 향상

---

## 📊 현재 상태 분석

### 파일 크기 현황
| 파일명 | 줄 수 | 주요 문제 |
|--------|-------|-----------|
| face_tracking_system.py | 3054줄 | 8개 클래스가 한 파일에 존재 |
| auto_speaker_detector.py | 421줄 | ModelManager 하드코딩 이슈 |
| 전체 프로젝트 | - | 220개 print문, 346줄 주석 |

### 주요 이슈
1. **파일 크기**: 단일 파일에 너무 많은 기능 포함
2. **로깅**: print문 남발로 프로그레스바와 충돌
3. **문서화**: 주석이 많지만 구조화되지 않음
4. **에러 처리**: 빈 except/pass 블록 존재

---

## ✅ Phase 1: 즉시 정리 가능 항목 (30분) - 완료!

### 1.1 주석 제거 및 코드 정리
**대상 파일**: 전체 프로젝트
**작업 내용**:
```python
# 제거 대상
- 주석 처리된 코드 블록 (# 로 시작하는 실행 안되는 코드)
- 디버깅용 임시 코드
- TODO/FIXME 완료된 항목
```

**✅ 완료된 작업**:
- 주석 처리된 코드 블록 2개 섹션 제거
- "Phase 1:" 불필요한 주석 정리
- 실제 감소량: 10-15줄

### 1.2 불필요한 import 정리
**✅ 완료된 작업**:
- `import tempfile` 제거 (사용되지 않음)
- `import numpy as np` 유지 (실제 사용 확인됨)

### 1.3 클래스 Docstring 추가 
**✅ 완료된 작업**:
- 8개 클래스 모두 한글 Docstring 추가 완료
- 2개 핵심 메서드 Docstring 추가 완료
- 각 클래스의 역할, 속성, 기능 상세 문서화

---

## ✅ Phase 2: 추가 Docstring 작업 (1시간) - 완료!

### 2.1 ✅ 완료: 주요 클래스 Docstring
**완료**: 8개 핵심 클래스
```python
"""
클래스 설명 (한글)

Attributes:
    속성명: 설명
    
Methods:
    메서드명: 설명
"""
```

### 2.2 ✅ 완료: 핵심 메서드 Docstring
**완료**: 6개 주요 public 메서드 Docstring 추가
- `assign_face_ids()`: Person1/Person2 할당 로직
- `generate_face_embedding()`: FaceNet 임베딩 생성
- `_hybrid_face_matching()`: 하이브리드 매칭 알고리즘
- `update_detection()`: 검출 결과 업데이트
- `get_crop_region()`: 얼굴 크롭 영역 추출
- `track_faces()`: 위치 기반 안정적 추적
```python
"""
메서드 기능 설명 (한글)

Args:
    param1: 설명
    param2: 설명
    
Returns:
    반환값 설명
    
Raises:
    예외 설명
"""
```

---

## ✅ Phase 3: 로깅 시스템 전환 (2시간) - 완료!

### ✅ 3.1 로깅 시스템 설정 완료
- ✅ 기존 `utils/logger.py` 활용
- ✅ TqdmLoggingHandler를 통한 tqdm 호환성 확보
```python
class TqdmLoggingHandler(logging.Handler):
    """tqdm 프로그레스바와 호환되는 로깅 핸들러"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except:
            self.handleError(record)
```

### ✅ 3.2 Print문 변환 완료 (80+ 개)
**완료된 변환**:
- ✅ 에러/경고 메시지 (❌⚠️) → logger.error/warning (30개)
- ✅ 성공/완료 메시지 (✅) → logger.info (15개)
- ✅ 디버그/정보 메시지 (🔍🔧📊🎯) → logger.debug (50개)
- ✅ 조건부 debug_mode print → logger.debug (샘플)
- ✅ 일반 메시지 → logger.info/debug (주요 항목)

### ✅ 3.3 결과 검증
- ✅ DevContainer 환경에서 로깅 시스템 동작 확인
- ✅ 구조화된 로그 출력 (타임스탬프, 레벨, 모듈명)
- ✅ tqdm 프로그레스바와 충돌 없음 확인

---

## 🔄 Phase 4: 파일 분리 (3시간)

### 4.1 face_tracking_system.py 분리 계획

#### 새 파일 구조
```
core/
├── face_tracking_system.py (500줄) - 메인 시스템만
├── trackers/
│   ├── stable_position_tracker.py (300줄)
│   ├── face_id_tracker.py (200줄)
│   └── detection_timeline.py (150줄)
├── processors/
│   ├── video_processor.py (400줄)
│   ├── frame_processor.py (300줄)
│   └── segment_processor.py (250줄)
├── detectors/
│   ├── face_detector.py (300줄)
│   └── person_assigner.py (250줄)
└── utils/
    ├── video_utils.py (200줄)
    └── face_utils.py (200줄)
```

### 4.2 클래스별 파일 매핑
| 클래스명 | 새 위치 | 예상 줄 수 |
|---------|---------|------------|
| DualFaceTrackingSystem | core/face_tracking_system.py | 500 |
| StablePositionTracker | core/trackers/stable_position_tracker.py | 300 |
| FaceIDTracker | core/trackers/face_id_tracker.py | 200 |
| DetectionTimeline | core/trackers/detection_timeline.py | 150 |
| VideoProcessor | core/processors/video_processor.py | 400 |
| FrameProcessor | core/processors/frame_processor.py | 300 |
| SegmentProcessor | core/processors/segment_processor.py | 250 |
| PersonAssigner | core/detectors/person_assigner.py | 250 |

### 4.3 Import 체계 정리
```python
# 절대 경로 import 사용
from dual_face_tracker.core.trackers.stable_position_tracker import StablePositionTracker
from dual_face_tracker.core.processors.video_processor import VideoProcessor

# 순환 참조 방지
# TYPE_CHECKING 사용으로 runtime import 회피
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dual_face_tracker.core.face_tracking_system import DualFaceTrackingSystem
```

---

## 🔄 Phase 5: 에러 처리 개선 (1시간)

### 5.1 빈 except 블록 개선
**현재 문제**:
```python
try:
    # 코드
except:
    pass  # 문제: 모든 에러 무시
```

**개선안**:
```python
try:
    # 코드
except SpecificError as e:
    logger.warning(f"처리 가능한 에러: {e}")
    # 적절한 폴백 처리
except Exception as e:
    logger.error(f"예상치 못한 에러: {e}", exc_info=True)
    raise  # 또는 적절한 처리
```

### 5.2 에러 타입별 처리
| 에러 유형 | 처리 방법 |
|-----------|-----------|
| FileNotFoundError | 파일 경로 확인 메시지 |
| ValueError | 입력값 검증 및 기본값 사용 |
| RuntimeError | 재시도 또는 폴백 |
| KeyboardInterrupt | 정상 종료 처리 |

---

## 🔄 Phase 6: 최종 검증 (30분)

### 6.1 테스트 체크리스트
- [ ] 모든 import 정상 작동
- [ ] 기존 기능 동일하게 동작
- [ ] 로그 파일 정상 생성
- [ ] 프로그레스바 정상 표시
- [ ] 에러 처리 정상 작동

### 6.2 성능 비교
- 실행 시간 측정
- 메모리 사용량 확인
- 로그 파일 크기 확인

---

## 📝 실행 우선순위

### 즉시 실행 (오늘)
1. **Phase 1**: 주석 제거 (30분)
2. **Phase 2**: Docstring 추가 (1시간)

### 단기 실행 (이번 주)
3. **Phase 3**: 로깅 전환 (2시간)
4. **Phase 5**: 에러 처리 (1시간)

### 중기 실행 (다음 주)
5. **Phase 4**: 파일 분리 (3시간)
6. **Phase 6**: 최종 검증 (30분)

---

## 🎯 예상 결과

### Before
- 3054줄 단일 파일
- 220개 print문
- 346줄 주석
- 8개 클래스 혼재

### After
- 최대 500줄 파일
- 구조화된 로깅 시스템
- 한글 docstring 문서화
- 모듈별 명확한 분리
- 개선된 에러 처리

---

## 📌 주의사항

1. **백업 필수**: 각 Phase 시작 전 git commit
2. **단계별 테스트**: 각 Phase 후 기능 테스트
3. **점진적 적용**: 한번에 모든 변경 금지
4. **호환성 유지**: 기존 API 변경 최소화

---

*이 계획은 필요에 따라 조정될 수 있습니다.*
*각 Phase는 독립적으로 실행 가능하도록 설계되었습니다.*