"""
병렬 FFmpeg 처리 시스템 - Phase 1B CPU 최적화
7950X3D 32스레드 최대 활용을 위한 FFmpeg 병렬 처리
"""
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from multiprocessing import cpu_count

from ..utils.logging import logger
from ..config import AUDIO_CODEC, VIDEO_CODEC


class ParallelFFmpegProcessor:
    """병렬 FFmpeg 처리 시스템"""
    
    def __init__(self, max_workers: int = None):
        """
        Args:
            max_workers: 최대 프로세스 수 (None이면 CPU 코어 기준 자동 설정)
        """
        if max_workers is None:
            # 7950X3D 16코어 기준 최대 16개 프로세스 사용
            # FFmpeg는 CPU 집약적이므로 코어 수만큼 사용
            self.max_workers = min(16, cpu_count())
        else:
            self.max_workers = max_workers
            
        logger.info(f"병렬 FFmpeg 프로세서 초기화 - {self.max_workers}개 프로세스")
    
    def process_audio_sync_batch(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        오디오 동기화 배치 처리
        
        Args:
            tasks: 처리할 작업 리스트
                각 작업: {
                    'seg_cropped': 크롭된 비디오 경로,
                    'seg_input': 원본 오디오가 있는 비디오 경로, 
                    'output_path': 최종 출력 경로,
                    'seg_fname': 세그먼트 파일명
                }
                
        Returns:
            처리 결과 리스트
        """
        if not tasks:
            return []
            
        logger.info(f"병렬 오디오 동기화 시작 - {len(tasks)}개 작업, {self.max_workers}개 프로세스")
        start_time = time.time()
        
        results = []
        
        # ProcessPoolExecutor로 병렬 처리
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_task = {
                executor.submit(self._process_single_audio_sync, task): task
                for task in tasks
            }
            
            # 결과 수집
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    result['task_data'] = task
                    results.append(result)
                except Exception as e:
                    logger.error(f"FFmpeg 처리 오류: {task['seg_fname']} - {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'task_data': task
                    })
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        logger.success(f"병렬 오디오 동기화 완료 - {success_count}/{len(tasks)}개 성공, {elapsed:.1f}초")
        
        return results
    
    @staticmethod
    def _process_single_audio_sync(task: Dict[str, str]) -> Dict[str, Any]:
        """
        단일 오디오 동기화 처리
        
        Args:
            task: 처리할 작업 정보
            
        Returns:
            처리 결과
        """
        seg_cropped = task['seg_cropped']
        seg_input = task['seg_input'] 
        output_path = task['output_path']
        seg_fname = task['seg_fname']
        
        try:
            # FFmpeg 오디오 동기화 명령어
            cmd = [
                'ffmpeg', '-y',
                '-i', seg_cropped,  # 크롭된 비디오
                '-i', seg_input,    # 원본 오디오가 있는 비디오
                '-c:v', 'copy',     # 비디오 재인코딩 없이 복사
                '-c:a', AUDIO_CODEC,
                '-map', '0:v:0',    # 첫 번째 입력의 비디오
                '-map', '1:a:0',    # 두 번째 입력의 오디오
                '-shortest',        # 더 짧은 스트림에 맞춤
                output_path
            ]
            
            # FFmpeg 실행
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            processing_time = time.time() - start_time
            
            # 출력 파일 생성 확인
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                return {
                    'success': True,
                    'processing_time': processing_time,
                    'file_size': file_size,
                    'seg_fname': seg_fname
                }
            else:
                return {
                    'success': False,
                    'error': '출력 파일이 생성되지 않음',
                    'seg_fname': seg_fname
                }
                
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f'FFmpeg 실행 오류: {e.stderr}',
                'seg_fname': seg_fname
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'seg_fname': seg_fname
            }
    
    def process_video_encoding_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        비디오 인코딩 배치 처리
        
        Args:
            tasks: 인코딩 작업 리스트
            
        Returns:
            처리 결과 리스트
        """
        if not tasks:
            return []
            
        logger.info(f"병렬 비디오 인코딩 시작 - {len(tasks)}개 작업")
        start_time = time.time()
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_encoding, task): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    result['task_data'] = task
                    results.append(result)
                except Exception as e:
                    logger.error(f"인코딩 처리 오류: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'task_data': task
                    })
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        logger.success(f"병렬 비디오 인코딩 완료 - {success_count}/{len(tasks)}개 성공, {elapsed:.1f}초")
        
        return results
    
    @staticmethod  
    def _process_single_encoding(task: Dict[str, Any]) -> Dict[str, Any]:
        """단일 비디오 인코딩 처리"""
        # 구체적인 인코딩 로직은 필요에 따라 구현
        pass


def create_parallel_ffmpeg_processor(max_workers: int = None) -> ParallelFFmpegProcessor:
    """
    병렬 FFmpeg 프로세서 생성
    
    Args:
        max_workers: 최대 프로세스 수
        
    Returns:
        ParallelFFmpegProcessor 인스턴스
    """
    return ParallelFFmpegProcessor(max_workers=max_workers)