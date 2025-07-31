"""
GPU 프로세스 풀 관리자 - RTX 5090 극한 병렬화
세그먼트별 독립 GPU 프로세스로 최대 성능 달성
"""

import os
import time
import torch
import psutil
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from typing import List, Dict, Any, Optional
from concurrent.futures import as_completed
import threading

from ..core.models import ModelManager
from ..utils.logging import logger
from ..config import DEVICE
# GPUBatchProcessor는 런타임에 import하여 순환 import 방지


class VRAMMonitor:
    """VRAM 사용량 실시간 모니터링"""
    
    def __init__(self, device = DEVICE):
        # torch.device 객체를 문자열로 변환
        if isinstance(device, torch.device):
            self.device = str(device)
        else:
            self.device = device
        
        # 디바이스 ID 추출
        if ':' in self.device:
            self.device_id = int(self.device.split(':')[1])
        else:
            self.device_id = 0
        
    def get_vram_usage(self) -> tuple:
        """현재 VRAM 사용량 반환 (사용량GB, 전체GB, 사용률%)"""
        try:
            torch.cuda.synchronize()
            used = torch.cuda.memory_allocated(self.device_id) / 1024**3
            total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
            usage_percent = (used / total) * 100
            return used, total, usage_percent
        except Exception as e:
            logger.error(f"VRAM 모니터링 오류: {str(e)}")
            return 0, 32, 0  # RTX 5090 기본값
    
    def get_available_vram(self) -> float:
        """사용 가능한 VRAM 반환 (GB)"""
        used, total, _ = self.get_vram_usage()
        return total - used - 2  # 2GB 안전 여유
    
    def can_allocate(self, required_gb: float) -> bool:
        """필요한 VRAM 할당 가능 여부"""
        available = self.get_available_vram()
        return available >= required_gb


class GPUProcessPool:
    """다중 GPU 프로세스 풀 관리자"""
    
    def __init__(self, device = DEVICE, max_processes: int = None):
        # torch.device 객체를 문자열로 변환
        if isinstance(device, torch.device):
            self.device = str(device)
        else:
            self.device = device
            
        self.vram_monitor = VRAMMonitor(self.device)
        
        # RTX 5090 32GB 기준 최적 프로세스 수 계산
        self.memory_per_process = 2.0  # GB per process (모델 + 배치)
        self.max_processes = max_processes or self._calculate_max_processes()
        
        # 프로세스 관리
        self.active_processes: List[Process] = []
        self.task_queues: List[Queue] = []
        self.result_queues: List[Queue] = []
        self.process_stats = Manager().dict()
        
        logger.info(f"GPU 프로세스 풀 초기화 - 최대 {self.max_processes}개 프로세스")
        
    def _calculate_max_processes(self) -> int:
        """VRAM 기준 최대 프로세스 수 계산"""
        _, total_vram, _ = self.vram_monitor.get_vram_usage()
        
        # 모델 기본 메모리 (4GB) + 안전 여유 (2GB) 제외
        available_for_processes = total_vram - 4 - 2
        max_by_memory = int(available_for_processes / self.memory_per_process)
        
        # 세그먼트 수와 하드웨어 한계 고려
        max_reasonable = min(max_by_memory, 8)  # 실용적 한계
        
        logger.info(f"VRAM {total_vram:.1f}GB 기준 최대 프로세스: {max_reasonable}개")
        return max(1, max_reasonable)
    
    def _create_gpu_worker_process(self, process_id: int, task_queue: Queue, result_queue: Queue):
        """GPU 워커 프로세스 생성"""
        process = Process(
            target=self._gpu_worker_function,
            args=(process_id, task_queue, result_queue, self.device),
            name=f"GPUWorker-{process_id}"
        )
        return process
    
    @staticmethod
    def _gpu_worker_function(process_id: int, task_queue: Queue, result_queue: Queue, device_str: str):
        """GPU 워커 프로세스 실행 함수"""
        try:
            # 런타임에 import하여 순환 import 방지
            from .gpu_batch_processor import GPUBatchProcessor
            
            # 문자열을 torch.device 객체로 변환
            device = torch.device(device_str)
            
            # 프로세스별 독립 GPU 초기화
            processor = GPUBatchProcessor(device)
            processor.initialize_models()
            
            logger.info(f"GPU 워커 {process_id} 시작 - {device}")
            
            while True:
                try:
                    # 작업 수신 (타임아웃 30초)
                    task_data = task_queue.get(timeout=30.0)
                    
                    if task_data == "STOP":
                        break
                    
                    # 세그먼트 처리
                    start_time = time.time()
                    success = processor._process_single_segment_gpu(
                        task_data['seg_input'], 
                        task_data['seg_cropped']
                    )
                    processing_time = time.time() - start_time
                    
                    # 결과 반환
                    result = {
                        'process_id': process_id,
                        'task_data': task_data,
                        'success': success,
                        'processing_time': processing_time,
                        'error': None if success else "GPU 처리 실패"
                    }
                    
                    result_queue.put(result)
                    
                    if success:
                        logger.success(f"워커 {process_id}: {task_data['seg_fname']} 완료 ({processing_time:.1f}초)")
                    else:
                        logger.error(f"워커 {process_id}: {task_data['seg_fname']} 실패")
                        
                except mp.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"워커 {process_id} 처리 오류: {str(e)}")
                    error_result = {
                        'process_id': process_id,
                        'task_data': task_data if 'task_data' in locals() else None,
                        'success': False,
                        'processing_time': 0,
                        'error': str(e)
                    }
                    result_queue.put(error_result)
                    
        except Exception as e:
            logger.error(f"워커 {process_id} 초기화 실패: {str(e)}")
        finally:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"GPU 워커 {process_id} 종료")
    
    def start_pool(self, num_processes: int = None) -> bool:
        """프로세스 풀 시작"""
        try:
            # 프로세스 수 결정
            if num_processes is None:
                num_processes = min(self.max_processes, 5)  # 기본적으로 5개 세그먼트 기준
            
            # VRAM 체크
            required_vram = num_processes * self.memory_per_process
            if not self.vram_monitor.can_allocate(required_vram):
                available = self.vram_monitor.get_available_vram()
                num_processes = max(1, int(available / self.memory_per_process))
                logger.warning(f"VRAM 부족으로 프로세스 수 조정: {num_processes}개")
            
            logger.info(f"GPU 프로세스 풀 시작 - {num_processes}개 프로세스")
            
            # 큐 및 프로세스 생성
            for i in range(num_processes):
                task_queue = Queue(maxsize=5)  # 프로세스별 작업 큐
                result_queue = Queue(maxsize=5)  # 프로세스별 결과 큐
                
                process = self._create_gpu_worker_process(i, task_queue, result_queue)
                
                self.active_processes.append(process)
                self.task_queues.append(task_queue)
                self.result_queues.append(result_queue)
                
                process.start()
                logger.info(f"GPU 프로세스 {i} 시작됨")
            
            # 프로세스 시작 대기
            time.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"프로세스 풀 시작 실패: {str(e)}")
            self.shutdown_pool()
            return False
    
    def process_segments(self, segment_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """세그먼트 병렬 처리"""
        try:
            if not self.active_processes:
                logger.error("활성 프로세스가 없습니다")
                return []
            
            logger.info(f"세그먼트 병렬 처리 시작 - {len(segment_tasks)}개 세그먼트")
            start_time = time.time()
            
            # 작업 분배
            self._distribute_tasks(segment_tasks)
            
            # 결과 수집
            results = self._collect_results(len(segment_tasks))
            
            elapsed = time.time() - start_time
            success_count = sum(1 for r in results if r['success'])
            
            logger.success(f"병렬 처리 완료 - 성공: {success_count}/{len(segment_tasks)}개, {elapsed:.1f}초")
            
            return results
            
        except Exception as e:
            logger.error(f"세그먼트 병렬 처리 오류: {str(e)}")
            return []
    
    def _distribute_tasks(self, segment_tasks: List[Dict[str, Any]]):
        """작업을 프로세스에 분배"""
        num_processes = len(self.active_processes)
        
        for i, task in enumerate(segment_tasks):
            process_idx = i % num_processes  # 라운드 로빈 분배
            
            try:
                self.task_queues[process_idx].put(task, timeout=5.0)
                logger.info(f"작업 분배: {task['seg_fname']} → 프로세스 {process_idx}")
            except Exception as e:
                logger.error(f"작업 분배 실패: {str(e)}")
    
    def _collect_results(self, expected_results: int) -> List[Dict[str, Any]]:
        """결과 수집"""
        results = []
        collected = 0
        timeout_per_result = 120  # 2분 타임아웃
        
        while collected < expected_results:
            # 모든 결과 큐에서 결과 수집
            for result_queue in self.result_queues:
                try:
                    result = result_queue.get(timeout=1.0)
                    results.append(result)
                    collected += 1
                    
                    if result['success']:
                        logger.info(f"결과 수집: 프로세스 {result['process_id']} 성공")
                    else:
                        logger.error(f"결과 수집: 프로세스 {result['process_id']} 실패 - {result['error']}")
                        
                    if collected >= expected_results:
                        break
                        
                except mp.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"결과 수집 오류: {str(e)}")
        
        return results
    
    def shutdown_pool(self):
        """프로세스 풀 종료"""
        try:
            logger.info("GPU 프로세스 풀 종료 시작")
            
            # 모든 프로세스에 종료 신호 전송
            for task_queue in self.task_queues:
                try:
                    task_queue.put("STOP", timeout=1.0)
                except:
                    pass
            
            # 프로세스 종료 대기
            for i, process in enumerate(self.active_processes):
                try:
                    process.join(timeout=10)
                    if process.is_alive():
                        logger.warning(f"프로세스 {i} 강제 종료")
                        process.terminate()
                        process.join(timeout=5)
                except Exception as e:
                    logger.error(f"프로세스 {i} 종료 오류: {str(e)}")
            
            # 리소스 정리
            self.active_processes.clear()
            self.task_queues.clear()
            self.result_queues.clear()
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.success("GPU 프로세스 풀 종료 완료")
            
        except Exception as e:
            logger.error(f"프로세스 풀 종료 오류: {str(e)}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """프로세스 풀 상태 정보"""
        vram_used, vram_total, vram_percent = self.vram_monitor.get_vram_usage()
        
        return {
            'active_processes': len(self.active_processes),
            'max_processes': self.max_processes,
            'vram_used': f"{vram_used:.1f}GB",
            'vram_total': f"{vram_total:.1f}GB", 
            'vram_usage': f"{vram_percent:.1f}%",
            'memory_per_process': f"{self.memory_per_process}GB"
        }


def create_gpu_process_pool(device = DEVICE, max_processes: int = None) -> GPUProcessPool:
    """GPU 프로세스 풀 생성"""
    return GPUProcessPool(device, max_processes)