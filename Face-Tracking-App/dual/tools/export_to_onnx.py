#!/usr/bin/env python3
"""
PyTorch 모델을 ONNX 형식으로 변환하는 스크립트
YOLOv8-face PT 파일 → ONNX 변환
"""

import os
import sys
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import argparse

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_face_tracker.utils.logger import UnifiedLogger

logger = UnifiedLogger()

def convert_yolov8_to_onnx(
    pt_path: Path,
    onnx_path: Path,
    input_size: tuple = (640, 640),
    batch_size: int = 1,
    dynamic_batch: bool = True,
    fp16: bool = False
) -> bool:
    """
    YOLOv8 PT 모델을 ONNX로 변환합니다.
    
    Args:
        pt_path: 입력 PT 파일 경로
        onnx_path: 출력 ONNX 파일 경로
        input_size: 입력 이미지 크기 (H, W)
        batch_size: 배치 크기
        dynamic_batch: 동적 배치 크기 지원 여부
        fp16: FP16 정밀도 사용 여부
        
    Returns:
        변환 성공 여부
    """
    try:
        logger.stage(f"🔄 YOLOv8 → ONNX 변환 시작: {pt_path.name}")
        
        # YOLOv8 모델 로드
        # 참고: ultralytics 패키지가 필요할 수 있음
        try:
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            
            # ONNX 내보내기
            logger.info("YOLO 모델 내보내기 중...")
            model.export(
                format='onnx',
                imgsz=input_size,
                batch=batch_size,
                dynamic=dynamic_batch,
                half=fp16,
                simplify=True,
                opset=16  # ONNX opset 버전
            )
            
            # 생성된 ONNX 파일 이동
            expected_onnx = pt_path.with_suffix('.onnx')
            if expected_onnx.exists() and expected_onnx != onnx_path:
                expected_onnx.rename(onnx_path)
                
        except ImportError:
            logger.warning("ultralytics 패키지 없음, 대체 방법 시도...")
            
            # 대체 방법: 직접 모델 로드 및 변환
            model = torch.load(pt_path, map_location='cpu')
            
            # 더미 입력 생성
            dummy_input = torch.randn(batch_size, 3, *input_size)
            
            # 동적 축 설정
            dynamic_axes = None
            if dynamic_batch:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # ONNX 내보내기
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
        
        # ONNX 모델 검증
        logger.info("ONNX 모델 검증 중...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # ONNX Runtime으로 테스트
        logger.info("ONNX Runtime 테스트 중...")
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # 입출력 정보 출력
        logger.info("ONNX 모델 정보:")
        for input_meta in ort_session.get_inputs():
            logger.info(f"  입력: {input_meta.name} - {input_meta.shape} ({input_meta.type})")
        for output_meta in ort_session.get_outputs():
            logger.info(f"  출력: {output_meta.name} - {output_meta.shape} ({output_meta.type})")
        
        # 간단한 추론 테스트
        test_input = torch.randn(1, 3, *input_size).numpy()
        outputs = ort_session.run(None, {'input': test_input})
        logger.success(f"✅ ONNX 변환 및 검증 완료: {onnx_path}")
        
        # 파일 크기 정보
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX 파일 크기: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX 변환 실패: {e}")
        if onnx_path.exists():
            onnx_path.unlink()  # 실패한 파일 삭제
        return False

def convert_scrfd_onnx(onnx_path: Path) -> bool:
    """
    SCRFD ONNX 모델을 검증하고 최적화합니다.
    
    Args:
        onnx_path: SCRFD ONNX 파일 경로
        
    Returns:
        검증 성공 여부
    """
    try:
        logger.stage(f"🔍 SCRFD ONNX 검증: {onnx_path.name}")
        
        # ONNX 모델 로드 및 검증
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # ONNX Runtime으로 테스트
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # 입출력 정보
        logger.info("SCRFD 모델 정보:")
        input_shape = None
        for input_meta in ort_session.get_inputs():
            logger.info(f"  입력: {input_meta.name} - {input_meta.shape} ({input_meta.type})")
            input_shape = input_meta.shape
            
        for output_meta in ort_session.get_outputs():
            logger.info(f"  출력: {output_meta.name} - {output_meta.shape} ({output_meta.type})")
        
        # 간단한 추론 테스트
        if input_shape:
            # SCRFD는 일반적으로 (1, 3, 640, 640) 입력 사용
            test_shape = [1, 3, 640, 640] if len(input_shape) != 4 else input_shape
            test_input = torch.randn(*test_shape).numpy()
            
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: test_input})
            
            logger.success(f"✅ SCRFD 모델 검증 완료")
            logger.info(f"출력 수: {len(outputs)}")
            for i, out in enumerate(outputs):
                logger.debug(f"  출력 {i}: shape={out.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"SCRFD 검증 실패: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="모델을 ONNX로 변환")
    parser.add_argument("--model", type=str, help="변환할 모델 경로")
    parser.add_argument("--output", type=str, help="출력 ONNX 경로")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], help="입력 크기")
    parser.add_argument("--batch", type=int, default=1, help="배치 크기")
    parser.add_argument("--dynamic", action="store_true", help="동적 배치 지원")
    parser.add_argument("--fp16", action="store_true", help="FP16 정밀도")
    parser.add_argument("--all", action="store_true", help="모든 모델 변환")
    
    args = parser.parse_args()
    
    logger.stage("🚀 ONNX 변환 시작")
    
    weights_dir = Path("weights")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if args.all:
        # 모든 다운로드된 모델 변환
        success_count = 0
        total_count = 0
        
        # YOLOv8 PT 파일 변환
        for pt_file in weights_dir.glob("*.pt"):
            total_count += 1
            onnx_path = models_dir / pt_file.with_suffix('.onnx').name
            
            if convert_yolov8_to_onnx(
                pt_file, 
                onnx_path,
                tuple(args.size),
                args.batch,
                args.dynamic,
                args.fp16
            ):
                success_count += 1
        
        # SCRFD ONNX 파일 검증 및 복사
        for onnx_file in weights_dir.glob("*.onnx"):
            total_count += 1
            dest_path = models_dir / onnx_file.name
            
            # 파일 복사
            import shutil
            shutil.copy2(onnx_file, dest_path)
            
            if convert_scrfd_onnx(dest_path):
                success_count += 1
        
        logger.stage(f"📊 변환 결과: {success_count}/{total_count} 성공")
        
        # 생성된 ONNX 파일 목록
        logger.info("\n📁 ONNX 모델 파일:")
        for onnx_file in models_dir.glob("*.onnx"):
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            logger.info(f"  - {onnx_file.name} ({size_mb:.1f} MB)")
            
    else:
        # 단일 모델 변환
        if not args.model:
            logger.error("--model 옵션이 필요합니다")
            return 1
            
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"모델 파일을 찾을 수 없음: {model_path}")
            return 1
            
        output_path = Path(args.output) if args.output else model_path.with_suffix('.onnx')
        
        if model_path.suffix == '.pt':
            success = convert_yolov8_to_onnx(
                model_path,
                output_path,
                tuple(args.size),
                args.batch,
                args.dynamic,
                args.fp16
            )
        elif model_path.suffix == '.onnx':
            success = convert_scrfd_onnx(model_path)
        else:
            logger.error(f"지원하지 않는 파일 형식: {model_path.suffix}")
            return 1
            
        return 0 if success else 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())