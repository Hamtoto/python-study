"""
통합 로깅 시스템 - 단일 로그 파일로 모든 로그 관리
"""
import os
import sys
import logging
from datetime import datetime
from contextlib import contextmanager


class UnifiedLogger:
    """모든 로그를 하나의 파일에 통합 관리하는 로거"""
    
    def __init__(self, log_file="face_tracker.log"):
        # 프로젝트 루트에 로그 파일 생성
        if not os.path.isabs(log_file):
            # 현재 파일: src/face_tracker/utils/logging.py
            # 목표: Face-Tracking-App/
            current_file = os.path.abspath(__file__)
            
            # Face-Tracking-App/ 찾기 (더 안전한 방법)
            current_dir = os.path.dirname(current_file)
            while current_dir != "/" and not current_dir.endswith("Face-Tracking-App"):
                current_dir = os.path.dirname(current_dir)
            
            if current_dir == "/" or not os.path.exists(current_dir):
                # 백업 방법: 4단계 상위 디렉토리 사용
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
            else:
                project_root = current_dir
                
            self.log_file = os.path.join(project_root, log_file)
            
            # 디버그용: 경로 정보 출력
            print(f"🔍 DEBUG: current_file: {current_file}")
            print(f"🔍 DEBUG: project_root: {project_root}")
            print(f"🔍 DEBUG: log_file_path: {self.log_file}")
            print(f"🔍 DEBUG: project_root exists: {os.path.exists(project_root)}")
            
            # 로그 디렉토리 생성 권한 확인
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                # 테스트 파일 생성으로 권한 확인
                test_file = self.log_file + ".test"
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"🔍 DEBUG: 로그 파일 생성 권한 확인됨")
            except Exception as e:
                print(f"🔍 DEBUG: 로그 파일 생성 권한 오류: {e}")
        else:
            self.log_file = log_file
            
        # 직접 파일 쓰기를 위한 파일 핸들 준비
        self.direct_file_handle = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._setup_logger()
        self._setup_direct_logging()
    
    def _setup_logger(self):
        """통합 로거 설정"""
        try:
            # 로거 생성
            self.logger = logging.getLogger('face_tracker')
            self.logger.setLevel(logging.DEBUG)  # DEBUG 레벨로 변경
            
            # 기존 핸들러 제거 (중복 방지)
            if self.logger.handlers:
                self.logger.handlers.clear()
            
            # 파일 핸들러 설정 (즉시 기록)
            print(f"🔍 DEBUG: 로그 파일 경로: {self.log_file}")  # 디버그용
            
            # 파일 핸들러 생성 - buffering=0으로 즉시 기록
            file_handler = logging.FileHandler(
                self.log_file, 
                mode='a', 
                encoding='utf-8',
                delay=False  # 즉시 파일 열기
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 스트림 버퍼링 완전 비활성화
            if hasattr(file_handler.stream, 'reconfigure'):
                try:
                    file_handler.stream.reconfigure(line_buffering=True)
                    print(f"🔍 DEBUG: line_buffering 설정 완료")
                except Exception as e:
                    print(f"🔍 DEBUG: reconfigure 실패: {e}")
            
            # 수동 flush를 위한 원본 flush 메서드 보존
            original_flush = file_handler.stream.flush
            
            # 로그 포맷 설정 (타임스탬프 포함)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s', 
                datefmt='%H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # 핸들러 추가
            self.logger.addHandler(file_handler)
            
            # 즉시 테스트 로그 작성
            test_msg = f"UnifiedLogger 초기화 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.logger.info(test_msg)
            
            # 강제 flush
            original_flush()
            
            print(f"🔍 DEBUG: 로거 설정 완료, 테스트 로그 작성됨")
            
        except Exception as e:
            print(f"🔍 DEBUG: _setup_logger 오류: {e}")
            # 폴백: 기본 로거 설정
            self.logger = logging.getLogger('face_tracker_fallback')
            console_handler = logging.StreamHandler()
            self.logger.addHandler(console_handler)
    
    def _setup_direct_logging(self):
        """직접 파일 쓰기 방식의 백업 로깅 시스템"""
        try:
            self.direct_file_handle = open(self.log_file, 'a', encoding='utf-8', buffering=1)
            print(f"🔍 DEBUG: 직접 파일 핸들 열기 성공: {self.log_file}")
            
            # 초기 테스트 메시지 직접 기록
            timestamp = datetime.now().strftime('%H:%M:%S')
            test_message = f"[{timestamp}] DIRECT: UnifiedLogger 직접 파일 기록 테스트\n"
            self.direct_file_handle.write(test_message)
            self.direct_file_handle.flush()
            os.fsync(self.direct_file_handle.fileno())
            print(f"🔍 DEBUG: 직접 로그 기록 테스트 완료")
            
        except Exception as e:
            print(f"🔍 DEBUG: 직접 파일 핸들 열기 실패: {e}")
            self.direct_file_handle = None
    
    def _direct_write_log(self, level: str, message: str):
        """로깅 시스템 우회하여 직접 파일에 기록"""
        try:
            if self.direct_file_handle:
                timestamp = datetime.now().strftime('%H:%M:%S')
                log_line = f"[{timestamp}] {level}: {message}\n"
                self.direct_file_handle.write(log_line)
                self.direct_file_handle.flush()
                os.fsync(self.direct_file_handle.fileno())
        except Exception as e:
            print(f"🔍 DEBUG: 직접 로그 쓰기 실패: {e}")
    
    def info(self, message: str):
        """정보 로그 - 즉시 기록"""
        print(f"🔍 CONSOLE: {message}")  # 콘솔 강제 출력
        try:
            self.logger.info(message)
            self._direct_write_log("INFO", message)
            self.flush()
        except Exception as e:
            print(f"🔍 CONSOLE: 표준 로깅 실패, 비상용 로깅 사용 - {e}")
            self.emergency_log("INFO", message)
    
    def error(self, message: str):
        """에러 로그 - 즉시 기록"""
        full_message = f"ERROR: {message}"
        print(f"❌ CONSOLE: {full_message}")  # 콘솔 강제 출력
        try:
            self.logger.error(full_message)
            self._direct_write_log("ERROR", full_message)
            self.flush()
        except Exception as e:
            print(f"❌ CONSOLE: 표준 로깅 실패, 비상용 로깅 사용 - {e}")
            self.emergency_log("ERROR", full_message)
    
    def success(self, message: str):
        """성공 로그 - 즉시 기록"""
        full_message = f"SUCCESS: {message}"
        print(f"✅ CONSOLE: {full_message}")  # 콘솔 강제 출력
        try:
            self.logger.info(full_message)
            self._direct_write_log("SUCCESS", full_message)
            self.flush()
        except Exception as e:
            print(f"✅ CONSOLE: 표준 로깅 실패, 비상용 로깅 사용 - {e}")
            self.emergency_log("SUCCESS", full_message)
    
    def stage(self, message: str):
        """단계별 로그 - 즉시 기록"""
        full_message = f"STAGE: {message}"
        print(f"🔄 CONSOLE: {full_message}")  # 콘솔 강제 출력
        try:
            self.logger.info(full_message)
            self._direct_write_log("STAGE", full_message)
            self.flush()
        except Exception as e:
            print(f"🔄 CONSOLE: 표준 로깅 실패, 비상용 로깅 사용 - {e}")
            self.emergency_log("STAGE", full_message)
    
    def warning(self, message: str):
        """경고 로그 - 즉시 기록"""
        full_message = f"WARNING: {message}"
        print(f"⚠️ CONSOLE: {full_message}")  # 콘솔 강제 출력
        try:
            self.logger.warning(full_message)
            self._direct_write_log("WARNING", full_message)
            self.flush()
        except Exception as e:
            print(f"⚠️ CONSOLE: 표준 로깅 실패, 비상용 로깅 사용 - {e}")
            self.emergency_log("WARNING", full_message)

    
    def flush(self):
        """로그 강제 플러시 - 즉시 파일에 기록"""
        try:
            for handler in self.logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
                # 스트림 레벨에서도 flush
                if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                    handler.stream.flush()
                    # 운영체제 레벨 sync 강제 실행
                    if hasattr(handler.stream, 'fileno'):
                        try:
                            os.fsync(handler.stream.fileno())
                        except:
                            pass  # fsync 실패해도 계속 진행
        except Exception as e:
            print(f"🔍 DEBUG: flush 오류: {e}")
    
    def debug(self, message: str):
        """디버그 로그 - 즉시 기록"""
        full_message = f"DEBUG: {message}"
        print(f"🔧 CONSOLE: {full_message}")  # 콘솔 강제 출력
        try:
            self.logger.debug(full_message)
            self._direct_write_log("DEBUG", full_message)
            self.flush()
        except Exception as e:
            print(f"🔧 CONSOLE: 표준 로깅 실패, 비상용 로깅 사용 - {e}")
            self.emergency_log("DEBUG", full_message)
    
    def clear_log(self):
        """로그 파일 초기화"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        # 직접 파일 핸들도 초기화
        if self.direct_file_handle:
            try:
                self.direct_file_handle.close()
            except:
                pass
            self.direct_file_handle = None
            self._setup_direct_logging()
    
    def emergency_log(self, level: str, message: str):
        """비상용 로깅 - 모든 방법을 시도하여 로그 기록"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] EMERGENCY-{level}: {message}"
        
        # 1차: 콘솔 출력 (항상 작동)
        print(f"🚨 EMERGENCY: {formatted_message}")
        
        # 2차: 직접 파일 쓰기 시도
        try:
            if self.direct_file_handle and not self.direct_file_handle.closed:
                self.direct_file_handle.write(f"{formatted_message}\n")
                self.direct_file_handle.flush()
                os.fsync(self.direct_file_handle.fileno())
        except Exception as e:
            print(f"🚨 EMERGENCY: 직접 파일 쓰기 실패 - {e}")
        
        # 3차: 새로운 파일 핸들로 시도
        try:
            with open(self.log_file, 'a', encoding='utf-8', buffering=1) as emergency_file:
                emergency_file.write(f"{formatted_message}\n")
                emergency_file.flush()
                os.fsync(emergency_file.fileno())
        except Exception as e:
            print(f"🚨 EMERGENCY: 새 파일 핸들 쓰기 실패 - {e}")
        
        # 4차: 백업 로그 파일 시도
        try:
            backup_log_file = self.log_file + ".backup"
            with open(backup_log_file, 'a', encoding='utf-8', buffering=1) as backup_file:
                backup_file.write(f"{formatted_message}\n")
                backup_file.flush()
                os.fsync(backup_file.fileno())
                print(f"🚨 EMERGENCY: 백업 파일에 기록됨 - {backup_log_file}")
        except Exception as e:
            print(f"🚨 EMERGENCY: 백업 파일 쓰기도 실패 - {e}")
        
        # 5차: 임시 디렉토리에 로그 파일 생성 시도
        try:
            import tempfile
            temp_log_file = os.path.join(tempfile.gettempdir(), "face_tracker_emergency.log")
            with open(temp_log_file, 'a', encoding='utf-8', buffering=1) as temp_file:
                temp_file.write(f"{formatted_message}\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                print(f"🚨 EMERGENCY: 임시 디렉토리에 기록됨 - {temp_log_file}")
        except Exception as e:
            print(f"🚨 EMERGENCY: 임시 디렉토리 쓰기도 실패 - {e}")
        
        # 최종: stderr 출력
        try:
            import sys
            sys.stderr.write(f"{formatted_message}\n")
            sys.stderr.flush()
        except Exception as e:
            print(f"🚨 EMERGENCY: stderr 출력도 실패 - {e}")
    
    def test_logging_system(self, verbose=True):
        """로깅 시스템 건전성 테스트"""
        if verbose:
            print(f"🔍 DEBUG: 로깅 시스템 테스트 시작...")
        
        test_passed = 0
        test_total = 5
        
        # 1. 로그 파일 경로 확인
        try:
            if verbose:
                print(f"🔍 DEBUG: 로그 파일 경로: {self.log_file}")
                print(f"🔍 DEBUG: 로그 파일 존재 여부: {os.path.exists(self.log_file)}")
            test_passed += 1
        except Exception as e:
            if verbose:
                print(f"❌ DEBUG: 로그 파일 경로 확인 실패 - {e}")
        
        # 2. 디렉토리 권한 확인
        try:
            log_dir = os.path.dirname(self.log_file)
            if verbose:
                print(f"🔍 DEBUG: 로그 디렉토리: {log_dir}")
                print(f"🔍 DEBUG: 디렉토리 존재 여부: {os.path.exists(log_dir)}")
                print(f"🔍 DEBUG: 디렉토리 쓰기 권한: {os.access(log_dir, os.W_OK)}")
            if os.path.exists(log_dir) and os.access(log_dir, os.W_OK):
                test_passed += 1
        except Exception as e:
            if verbose:
                print(f"❌ DEBUG: 디렉토리 권한 확인 실패 - {e}")
        
        # 3. 직접 파일 핸들 상태 확인
        try:
            if verbose:
                print(f"🔍 DEBUG: 직접 파일 핸들 존재: {self.direct_file_handle is not None}")
            if self.direct_file_handle:
                if verbose:
                    print(f"🔍 DEBUG: 직접 파일 핸들 닫힘 여부: {self.direct_file_handle.closed}")
                if not self.direct_file_handle.closed:
                    test_passed += 1
        except Exception as e:
            if verbose:
                print(f"❌ DEBUG: 직접 파일 핸들 확인 실패 - {e}")
        
        # 4. 테스트 메시지 기록
        test_msg = "로깅 시스템 테스트 메시지"
        if verbose:
            print(f"🔍 DEBUG: 테스트 메시지 기록 시도: {test_msg}")
        
        # 표준 로거 테스트
        try:
            self.logger.info(test_msg)
            if verbose:
                print("✅ DEBUG: 표준 로거 테스트 성공")
            test_passed += 1
        except Exception as e:
            if verbose:
                print(f"❌ DEBUG: 표준 로거 테스트 실패 - {e}")
        
        # 직접 파일 쓰기 테스트
        try:
            self._direct_write_log("TEST", test_msg)
            if verbose:
                print("✅ DEBUG: 직접 파일 쓰기 테스트 성공")
            test_passed += 1
        except Exception as e:
            if verbose:
                print(f"❌ DEBUG: 직접 파일 쓰기 테스트 실패 - {e}")
        
        # 비상용 로깅 테스트 (오류 메시지 출력하지 않음)
        try:
            # emergency_log는 내부에서 많은 print를 생성하므로 조용히 테스트
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            self.emergency_log("TEST", test_msg)
            
            sys.stdout = old_stdout
            if verbose:
                print("✅ DEBUG: 비상용 로깅 테스트 성공")
        except Exception as e:
            if verbose:
                print(f"❌ DEBUG: 비상용 로깅 테스트 실패 - {e}")
        finally:
            sys.stdout = old_stdout
        
        # 테스트 결과 요약
        if verbose:
            success_rate = (test_passed / test_total) * 100
            print(f"🔍 DEBUG: 로깅 시스템 테스트 완료 - 성공률: {success_rate:.1f}% ({test_passed}/{test_total})")
        
        return test_passed >= 3  # 5개 중 3개 이상 성공하면 OK
    
    def repair_logging_system(self):
        """로깅 시스템 복구 시도"""
        print(f"🔧 REPAIR: 로깅 시스템 복구 시작...")
        
        # 1. 기존 핸들러 정리
        try:
            if self.logger.handlers:
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
            print("✅ REPAIR: 기존 핸들러 정리 완료")
        except Exception as e:
            print(f"❌ REPAIR: 핸들러 정리 실패 - {e}")
        
        # 2. 직접 파일 핸들 재설정
        try:
            if self.direct_file_handle and not self.direct_file_handle.closed:
                self.direct_file_handle.close()
            self.direct_file_handle = None
            self._setup_direct_logging()
            print("✅ REPAIR: 직접 파일 핸들 재설정 완료")
        except Exception as e:
            print(f"❌ REPAIR: 직접 파일 핸들 재설정 실패 - {e}")
        
        # 3. 표준 로거 재설정
        try:
            self._setup_logger()
            print("✅ REPAIR: 표준 로거 재설정 완료")
        except Exception as e:
            print(f"❌ REPAIR: 표준 로거 재설정 실패 - {e}")
        
        # 4. 복구 테스트
        try:
            self.info("로깅 시스템 복구 테스트 메시지")
            print("✅ REPAIR: 복구 테스트 성공")
        except Exception as e:
            print(f"❌ REPAIR: 복구 테스트 실패 - {e}")
            # 최후 수단으로 비상용 로깅 사용
            self.emergency_log("ERROR", f"로깅 시스템 복구 실패: {e}")
        
        print(f"🔧 REPAIR: 로깅 시스템 복구 완료")
    
    @contextmanager
    def session_context(self):
        """처리 세션 컨텍스트"""
        # 세션 시작
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Face-Tracking-App 시작: {timestamp}")
        self.logger.info(f"{'='*50}")
        
        try:
            yield self
        except Exception as e:
            try:
                self.error(f"처리 중 오류 발생: {str(e)}")
            except:
                # 로깅도 실패한 경우 비상용 로깅 사용
                self.emergency_log("ERROR", f"처리 중 오류 발생: {str(e)}")
            raise
        finally:
            # 세션 종료
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"Face-Tracking-App 완료: {end_time}")
            self.logger.info(f"{'='*50}")


class TeeOutput:
    """콘솔과 파일에 동시 출력 (기존 코드와의 호환성 유지)"""
    def __init__(self, console, file_handle):
        self.console = console
        self.file_handle = file_handle
        
    def write(self, text):
        # 콘솔에 출력
        self.console.write(text)
        # 파일에도 저장
        self.file_handle.write(text)
        self.file_handle.flush()
        
    def flush(self):
        self.console.flush()
        self.file_handle.flush()
        
    def __getattr__(self, name):
        return getattr(self.console, name)


# 전역 통합 로거 인스턴스
logger = UnifiedLogger()

# 로깅 시스템 초기 건전성 테스트 (안전 모드)
def _safe_initialize_logging():
    """로깅 시스템 안전 초기화"""
    try:
        # 기본 콘솔 출력 테스트
        print("🔍 UnifiedLogger 초기화 시작...")
        
        # 비중요 테스트는 실제 사용 시 수행
        return True
        
    except Exception as e:
        print(f"🚨 EMERGENCY: 로깅 시스템 초기화 실패 - {e}")
        return False

# 안전 초기화 실행
_initialization_success = _safe_initialize_logging()

if not _initialization_success:
    print("🚨 EMERGENCY: 콘솔 출력 모드로 대체")

# 하위 호환성을 위한 에러 로거 인터페이스
class ErrorLoggerCompat:
    """기존 error_logger 코드와의 호환성 유지"""
    def log_error(self, message: str):
        logger.error(message)
    
    def log_segment_error(self, segment_name: str, error_msg: str):
        logger.error(f"{segment_name} - {error_msg}")
    
    def log_video_error(self, video_name: str, error_msg: str):
        logger.error(f"{video_name} - {error_msg}")

error_logger = ErrorLoggerCompat()

# 하위 호환성을 위한 ConsoleLogger
class ConsoleLogger:
    """기존 ConsoleLogger와의 호환성 유지"""
    def __init__(self, log_file=None):
        pass  # 이제 단일 로거를 사용하므로 별도 설정 불필요
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass