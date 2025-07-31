"""
Face-Tracking-App 메인 진입점
"""
from src.face_tracker.processing.processor import process_all_videos
from src.face_tracker.utils.logging import logger


if __name__ == "__main__":
    # 직접 로깅으로 디버깅 (session_context 문제 우회)
    logger.info("Face-Tracking-App 메인 시작")
    print("DEBUG: 메인 함수 진입, 로깅 시작")
    
    try:
        process_all_videos()
        logger.info("Face-Tracking-App 정상 완료")
        print("DEBUG: 처리 완료")
    except Exception as e:
        logger.error(f"Face-Tracking-App 실행 오류: {str(e)}")
        print(f"DEBUG: 오류 발생 - {str(e)}")
        import traceback
        traceback.print_exc()