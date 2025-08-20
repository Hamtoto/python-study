"""
Logging utilities for dual-face tracker system.

Provides unified logging configuration with structured formatting
and performance monitoring capabilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def get_logger(name: str, 
               level: int = logging.INFO,
               log_file: Optional[Union[str, Path]] = None,
               console_output: bool = True) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, uses 'dual_face_tracker.log'
        console_output: Whether to output to console
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter with emoji support
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = Path.cwd() / 'dual_face_tracker.log'
    else:
        log_file = Path(log_file)
        
    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class PerformanceLogger:
    """
    Performance logging utility for tracking processing times.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"🔄 {operation} 시작")
        
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and log the duration.
        
        Args:
            operation: Operation name
            
        Returns:
            float: Duration in seconds
        """
        if operation not in self.start_times:
            self.logger.warning(f"⚠️ 타이머 '{operation}'가 시작되지 않았습니다")
            return 0.0
            
        start_time = self.start_times.pop(operation)
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"✅ {operation} 완료 ({duration:.2f}초)")
        return duration
        
    def log_memory_usage(self, operation: str) -> None:
        """Log current memory usage for an operation."""
        try:
            from .cuda_utils import get_gpu_memory_info
            
            memory_info = get_gpu_memory_info()
            total_gb = memory_info['total'] / (1024**3)
            allocated_gb = memory_info['allocated'] / (1024**3)
            utilization = (allocated_gb / total_gb) * 100
            
            self.logger.debug(
                f"🔧 {operation} - GPU 메모리: {allocated_gb:.1f}/{total_gb:.1f}GB ({utilization:.1f}%)"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정보 수집 실패: {e}")


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging purposes.
    
    Args:
        logger: Logger instance to use
    """
    import torch
    import platform
    
    logger.info("🖥️ 시스템 정보:")
    logger.info(f"   - Platform: {platform.platform()}")
    logger.info(f"   - Python: {platform.python_version()}")
    logger.info(f"   - PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"   - CUDA: {torch.version.cuda}")
        logger.info(f"   - GPU: {torch.cuda.get_device_name(0)}")
        
        memory_info = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"   - VRAM: {memory_info:.1f}GB")
    else:
        logger.warning("   - CUDA: 사용 불가")


class UnifiedLogger:
    """
    통합 로거 클래스 - 이모지 기반 구조화된 로깅.
    
    Phase 1에서 사용된 로깅 패턴을 유지하면서 새로운 inference 모듈에 적용.
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = get_logger(name, level=level, console_output=True)
        self.name = name
    
    def stage(self, message: str):
        """🔄 단계 진행 로깅"""
        self.logger.info(f"🔄 {message}")
    
    def success(self, message: str):
        """✅ 성공 로깅"""
        self.logger.info(f"✅ {message}")
    
    def warning(self, message: str):
        """⚠️ 경고 로깅"""
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str):
        """❌ 오류 로깅"""
        self.logger.error(f"❌ {message}")
    
    def debug(self, message: str):
        """🔧 디버그 로깅"""
        self.logger.debug(f"🔧 {message}")
    
    def info(self, message: str):
        """📋 일반 정보 로깅"""
        self.logger.info(f"📋 {message}")


def setup_dual_face_logger(log_level: str = "INFO", 
                          log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup main dual-face tracker logger with standard configuration.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional log file path
        
    Returns:
        logging.Logger: Configured main logger
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    logger = get_logger("dual_face_tracker", level=level, log_file=log_file)
    
    # Log startup message
    logger.info("🚀 Dual-Face Tracker 시스템 초기화")
    log_system_info(logger)
    
    return logger