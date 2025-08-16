#!/usr/bin/env python3
"""
Face detection 모델 다운로드 스크립트
YOLOv8-face 또는 SCRFD 모델을 다운로드하고 준비합니다.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_face_tracker.utils.logger import UnifiedLogger

logger = UnifiedLogger()

# 모델 다운로드 URL 및 정보
MODELS = {
    "yolov8n-face": {
        "url": "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt",
        "sha256": None,  # 실제 해시값으로 업데이트 필요
        "description": "YOLOv8n face detection model (lightweight)"
    },
    "yolov8s-face": {
        "url": "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8s-face.pt",
        "sha256": None,
        "description": "YOLOv8s face detection model (balanced)"
    },
    "scrfd_500m": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/scrfd_500m_bnkps.onnx",
        "sha256": None,
        "description": "SCRFD 500M parameters model"
    },
    "scrfd_2.5g": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/scrfd_2.5g_bnkps.onnx", 
        "sha256": None,
        "description": "SCRFD 2.5G parameters model (high accuracy)"
    }
}

def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """
    URL에서 파일을 다운로드합니다.
    
    Args:
        url: 다운로드 URL
        dest_path: 저장 경로
        desc: 진행 표시줄 설명
        
    Returns:
        성공 여부
    """
    try:
        # 이미 파일이 존재하는 경우
        if dest_path.exists():
            logger.info(f"✅ {dest_path.name} 이미 존재함")
            return True
            
        # 다운로드 시작
        logger.stage(f"다운로드 시작: {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        # 진행 표시줄과 함께 다운로드
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc or dest_path.name) as pbar:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.success(f"다운로드 완료: {dest_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"다운로드 실패: {e}")
        if dest_path.exists():
            dest_path.unlink()  # 불완전한 파일 삭제
        return False
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def verify_file(file_path: Path, expected_sha256: str = None) -> bool:
    """
    파일의 SHA256 해시를 검증합니다.
    
    Args:
        file_path: 검증할 파일 경로
        expected_sha256: 예상 SHA256 해시값
        
    Returns:
        검증 성공 여부
    """
    if not file_path.exists():
        return False
        
    if expected_sha256 is None:
        logger.warning(f"⚠️ {file_path.name} 해시 검증 건너뜀 (해시값 없음)")
        return True
        
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_hash = sha256_hash.hexdigest()
    if actual_hash == expected_sha256:
        logger.success(f"해시 검증 성공: {file_path.name}")
        return True
    else:
        logger.error(f"해시 불일치: {file_path.name}")
        logger.error(f"  예상: {expected_sha256}")
        logger.error(f"  실제: {actual_hash}")
        return False

def main():
    """메인 함수"""
    logger.stage("🚀 Face Detection 모델 다운로드 시작")
    
    # 모델 저장 디렉토리 설정
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # 사용할 모델 선택 (기본: yolov8n-face)
    selected_models = ["yolov8n-face", "scrfd_500m"]  # 경량 모델 2개 선택
    
    logger.info(f"선택된 모델: {', '.join(selected_models)}")
    
    success_count = 0
    failed_models = []
    
    for model_name in selected_models:
        if model_name not in MODELS:
            logger.warning(f"⚠️ 알 수 없는 모델: {model_name}")
            continue
            
        model_info = MODELS[model_name]
        
        # 파일 확장자 결정
        if model_info["url"].endswith(".onnx"):
            file_ext = ".onnx"
        elif model_info["url"].endswith(".pt"):
            file_ext = ".pt"
        else:
            file_ext = ""
            
        dest_path = weights_dir / f"{model_name}{file_ext}"
        
        logger.info(f"\n📦 {model_name}: {model_info['description']}")
        
        # 다운로드
        if download_file(model_info["url"], dest_path, desc=model_name):
            # 해시 검증
            if verify_file(dest_path, model_info["sha256"]):
                success_count += 1
            else:
                failed_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # 결과 요약
    logger.stage("📊 다운로드 결과")
    logger.info(f"성공: {success_count}/{len(selected_models)}")
    
    if failed_models:
        logger.warning(f"실패한 모델: {', '.join(failed_models)}")
        return 1
    else:
        logger.success("✅ 모든 모델 다운로드 완료!")
        
        # 다운로드된 파일 목록 표시
        logger.info("\n📁 다운로드된 파일:")
        for file_path in weights_dir.glob("*"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  - {file_path.name} ({size_mb:.1f} MB)")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())