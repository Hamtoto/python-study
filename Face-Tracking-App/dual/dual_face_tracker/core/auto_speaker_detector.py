#!/usr/bin/env python3
"""
AutoSpeakerDetector: 통계 기반 자동 화자 선정 시스템

기능:
- 전체 영상 5% 샘플링으로 빠른 스캔
- 얼굴 임베딩 기반 클러스터링 (DBSCAN)
- 중요도 점수 계산 (빈도, 크기, 위치, 지속시간)
- 상위 2명 주요 화자 자동 선정

Author: Auto Speaker Detection System v1.0
Date: 2025.08.17
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from PIL import Image
from torchvision import transforms
import time
import logging
from ..utils.logger import get_logger

# Logger 설정
logger = get_logger(__name__, level=logging.INFO)

# 클러스터링 라이브러리
try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    CLUSTERING_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn 없음. 기본 클러스터링 사용")
    CLUSTERING_AVAILABLE = False

# Dual 시스템 전용 모델 import - models.py 삭제됨
try:
    # from models import ModelManager  # models.py 파일이 삭제됨
    MODEL_MANAGER_AVAILABLE = False  # 항상 False로 설정
except ImportError:
    MODEL_MANAGER_AVAILABLE = False


class FaceCluster:
    """얼굴 클러스터 (같은 사람의 얼굴들)"""
    
    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        self.detections = []  # List of face detections
        self.embeddings = []  # List of embeddings
        self.positions = []   # List of (x, y) positions
        self.sizes = []       # List of face areas
        self.timestamps = []  # List of timestamps
        self.confidences = [] # List of detection confidences
        
        # 계산된 통계
        self.representative_embedding = None
        self.average_position = None
        self.importance_score = 0.0
        
    def add_detection(self, detection_data: Dict[str, Any]):
        """검출 데이터 추가"""
        self.detections.append(detection_data)
        
        if detection_data.get('embedding') is not None:
            self.embeddings.append(detection_data['embedding'])
        
        self.positions.append(detection_data['center'])
        self.sizes.append(detection_data['size'])
        self.timestamps.append(detection_data['timestamp'])
        self.confidences.append(detection_data['confidence'])
    
    def calculate_statistics(self):
        """클러스터 통계 계산"""
        if not self.detections:
            return
            
        # 대표 임베딩 (평균)
        if self.embeddings:
            stacked_embeddings = torch.stack(self.embeddings)
            self.representative_embedding = torch.mean(stacked_embeddings, dim=0)
            self.representative_embedding = F.normalize(self.representative_embedding, p=2, dim=0)
        
        # 평균 위치
        if self.positions:
            avg_x = sum(pos[0] for pos in self.positions) / len(self.positions)
            avg_y = sum(pos[1] for pos in self.positions) / len(self.positions)
            self.average_position = (avg_x, avg_y)
    
    def get_stats(self) -> Dict[str, Any]:
        """클러스터 통계 반환"""
        return {
            'cluster_id': self.cluster_id,
            'appearance_count': len(self.detections),
            'average_position': self.average_position,
            'average_size': sum(self.sizes) / len(self.sizes) if self.sizes else 0,
            'time_span': max(self.timestamps) - min(self.timestamps) if len(self.timestamps) > 1 else 0,
            'has_embedding': self.representative_embedding is not None,
            'importance_score': self.importance_score
        }


class AutoSpeakerDetector:
    """통계 기반 자동 화자 선정 시스템"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        # 얼굴 검출기 초기화
        self.face_cascade = None
        self._initialize_face_detector()
        
        # FaceNet 모델 초기화 (임베딩용)
        self.model_manager = None
        self.resnet = None
        self.face_transform = None
        self._initialize_facenet()
        
        # 스캔 설정 (전구간 분석)
        self.sample_rate = 0.5  # 50% 샘플링 (20% → 50%, 전구간 분석으로 개선)
        self.min_face_size = 30  # 최소 얼굴 크기
        self.clustering_threshold = 0.25  # 클러스터링 임계값 (0.35 → 0.25, 더 세밀한 클러스터링)
        self.min_cluster_size = 5  # 최소 클러스터 크기 (10 → 5, 더 유연한 클러스터링)
        
        # 중요도 점수 가중치 (화자 중심 개선)
        self.weights = {
            'frequency': 0.25,      # 등장 빈도 (0.30 → 0.25, 움직임 점수 추가로 조정)
            'size': 0.15,          # 총 얼굴 크기 (0.35 → 0.15, 크기 비중 대폭 감소)
            'center': 0.20,        # 화면 중앙 가중치 (0.25 → 0.20)
            'time_distribution': 0.15,  # 시간적 분포 (유지)
            'motion': 0.15,        # 움직임 점수 (새로 추가, 활발한 화자 감지)
            'consistency': 0.05,   # 크기 일관성 (0.10 → 0.05)
            'confidence': 0.05     # 평균 신뢰도 (유지)
        }
    
    def _initialize_face_detector(self):
        """얼굴 검출기 초기화"""
        cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        
        if Path(cascade_path).exists():
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.face_cascade.empty():
                if self.debug_mode:
                    logger.info("Haar Cascade 얼굴 검출기 로드 완료")
            else:
                raise RuntimeError("❌ Haar Cascade 로드 실패")
        else:
            raise RuntimeError(f"❌ Haar Cascade 파일 없음: {cascade_path}")
    
    def _initialize_facenet(self):
        """FaceNet 모델 초기화"""
        if not MODEL_MANAGER_AVAILABLE:
            if self.debug_mode:
                logger.warning("ModelManager 없음. 임베딩 기능 비활성화")
            return
            
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_manager = ModelManager(device)
            self.resnet = self.model_manager.get_resnet()
            
            # 얼굴 전처리 변환 (FaceNet용)
            self.face_transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            if self.debug_mode:
                logger.info(f"FaceNet 모델 로드 완료 (디바이스: {device})")
                
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as e:
            if self.debug_mode:
                logger.warning(f"FaceNet 초기화 실패: {e}")
            self.model_manager = None
            self.resnet = None
            self.face_transform = None
        except Exception as e:
            logger.error(f"예상치 못한 FaceNet 초기화 오류: {e}")
            self.model_manager = None
            self.resnet = None
            self.face_transform = None
    
    def extract_face_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[torch.Tensor]:
        """얼굴 크롭에서 임베딩 추출"""
        if self.resnet is None or self.face_transform is None:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # OpenCV BGR → PIL RGB 변환
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)
            
            # 전처리 및 배치 차원 추가
            face_tensor = self.face_transform(pil_image).unsqueeze(0)
            
            # GPU로 이동 (사용 가능한 경우)
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            
            # 임베딩 생성
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
                
            # L2 정규화
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.squeeze(0).cpu()  # CPU로 이동 후 배치 차원 제거
            
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            if self.debug_mode:
                logger.warning(f"임베딩 추출 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 임베딩 추출 오류: {e}")
            return None
    
    def scan_video(self, video_path: str) -> List[Dict[str, Any]]:
        """전체 영상 스캔하여 모든 얼굴 검출 데이터 수집"""
        logger.info(f"자동 화자 분석 시작: {video_path}")
        logger.debug(f"샘플링 비율: {self.sample_rate:.1%}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오 열기 실패: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = int(1 / self.sample_rate)
        
        print(f"   📹 총 {total_frames}프레임, {total_frames/fps:.1f}초")
        logger.debug(f"분석할 프레임: {total_frames//sample_interval}개")
        
        all_detections = []
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 샘플링: sample_interval마다만 처리
                if frame_idx % sample_interval == 0:
                    timestamp = frame_idx / fps
                    faces = self._detect_faces_in_frame(frame)
                    
                    for face_bbox, confidence in faces:
                        x1, y1, x2, y2 = face_bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        size = (x2 - x1) * (y2 - y1)
                        
                        # 임베딩 추출
                        embedding = self.extract_face_embedding(frame, face_bbox)
                        
                        detection_data = {
                            'frame_idx': frame_idx,
                            'timestamp': timestamp,
                            'bbox': face_bbox,
                            'center': (center_x, center_y),
                            'size': size,
                            'confidence': confidence,
                            'embedding': embedding
                        }
                        
                        all_detections.append(detection_data)
                
                frame_idx += 1
                
                # 진행률 표시
                if frame_idx % (sample_interval * 50) == 0:  # 50개 샘플마다
                    progress = (frame_idx / total_frames) * 100
                    elapsed = time.time() - start_time
                    logger.debug(f"진행률: {progress:.1f}% ({len(all_detections)}개 얼굴, {elapsed:.1f}초)")
        
        finally:
            cap.release()
        
        elapsed_time = time.time() - start_time
        logger.info(f"스캔 완료: {len(all_detections)}개 얼굴 발견 ({elapsed_time:.1f}초)")
        
        return all_detections
    
    def scan_video_left_right(self, video_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """좌우 영역별로 분리하여 전체 영상 스캔"""
        logger.info(f"좌우 분리 화자 분석 시작: {video_path}")
        logger.debug(f"샘플링 비율: {self.sample_rate:.1%}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오 열기 실패: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = int(1 / self.sample_rate)
        
        print(f"   📹 총 {total_frames}프레임, {total_frames/fps:.1f}초")
        logger.debug(f"분석할 프레임: {total_frames//sample_interval}개")
        print(f"   ⚖️ 좌우 분리 기준: x=960px")
        
        left_detections = []   # x < 960 (왼쪽 영역)
        right_detections = []  # x >= 960 (오른쪽 영역)
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 샘플링: sample_interval마다만 처리
                if frame_idx % sample_interval == 0:
                    timestamp = frame_idx / fps
                    faces = self._detect_faces_in_frame(frame)
                    
                    for face_bbox, confidence in faces:
                        x1, y1, x2, y2 = face_bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        size = (x2 - x1) * (y2 - y1)
                        
                        # 임베딩 추출
                        embedding = self.extract_face_embedding(frame, face_bbox)
                        
                        detection_data = {
                            'frame_idx': frame_idx,
                            'timestamp': timestamp,
                            'bbox': face_bbox,
                            'center': (center_x, center_y),
                            'size': size,
                            'confidence': confidence,
                            'embedding': embedding
                        }
                        
                        # 좌우 분리 (x=960 기준)
                        if center_x < 960:  # 왼쪽 영역
                            left_detections.append(detection_data)
                        else:  # 오른쪽 영역
                            right_detections.append(detection_data)
                
                frame_idx += 1
                
                # 진행률 표시
                if frame_idx % (sample_interval * 50) == 0:  # 50개 샘플마다
                    progress = (frame_idx / total_frames) * 100
                    elapsed = time.time() - start_time
                    logger.debug(f"진행률: {progress:.1f}% (좌:{len(left_detections)}개, 우:{len(right_detections)}개, {elapsed:.1f}초)")
        
        finally:
            cap.release()
        
        elapsed = time.time() - start_time
        logger.info(f"좌우 분리 스캔 완료: 좌측 {len(left_detections)}개, 우측 {len(right_detections)}개 ({elapsed:.1f}초)")
        
        return left_detections, right_detections
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """프레임에서 얼굴 검출 - MTCNN 우선, Haar Cascade 폴백"""
        faces = []
        
        # 1. MTCNN으로 얼굴 검출 시도 (ModelManager 사용 가능시)
        if self.model_manager and self.model_manager.mtcnn:
            try:
                mtcnn = self.model_manager.mtcnn
                # PIL Image로 변환 (MTCNN 요구사항)
                from PIL import Image
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    pil_frame = frame
                
                boxes, probs = mtcnn.detect(pil_frame)
                
                if boxes is not None and len(boxes) > 0:
                    for box, prob in zip(boxes, probs):
                        if prob > 0.5:  # 신뢰도 임계값
                            x1, y1, x2, y2 = box.astype(int)
                            # 바운딩 박스 검증
                            if x2 > x1 and y2 > y1 and (x2-x1) >= self.min_face_size and (y2-y1) >= self.min_face_size:
                                bbox = (x1, y1, x2, y2)
                                faces.append((bbox, float(prob)))
                    
                    if faces:  # MTCNN에서 얼굴을 찾으면 반환
                        return faces
            except Exception as e:
                if self.debug_mode:
                    logger.warning(f"MTCNN 얼굴 검출 실패, Haar Cascade 폴백: {e}")
        
        # 2. Haar Cascade 폴백
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            detected_faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            for (x, y, w, h) in detected_faces:
                # 얼굴 크기 및 종횡비 검증
                if w > 20 and h > 20 and w < 200 and h < 200:
                    aspect_ratio = w / h
                    if 0.7 < aspect_ratio < 1.3:
                        bbox = (x, y, x + w, y + h)
                        confidence = 0.9  # Haar는 신뢰도가 없으므로 고정값
                        faces.append((bbox, confidence))
        except Exception as e:
            if self.debug_mode:
                logger.warning(f"Haar Cascade 얼굴 검출 실패: {e}")
        
        return faces
    
    def cluster_faces(self, all_detections: List[Dict[str, Any]]) -> List[FaceCluster]:
        """얼굴 임베딩을 기반으로 클러스터링"""
        print(f"🔄 얼굴 클러스터링 시작...")
        
        # 임베딩이 있는 검출만 필터링
        valid_detections = [det for det in all_detections if det['embedding'] is not None]
        
        if len(valid_detections) < self.min_cluster_size:
            logger.warning(f"충분한 임베딩 데이터 없음 ({len(valid_detections)}개 < {self.min_cluster_size}개)")
            return self._fallback_clustering(all_detections)
        
        # 임베딩 스택
        embeddings = [det['embedding'] for det in valid_detections]
        embedding_matrix = torch.stack(embeddings).numpy()
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(embedding_matrix)
        # 거리 행렬 생성 (음수 방지를 위해 클리핑)
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, 2)  # 0~2 범위로 클리핑
        
        # DBSCAN 클러스터링
        if CLUSTERING_AVAILABLE:
            clustering = DBSCAN(
                eps=self.clustering_threshold,
                min_samples=self.min_cluster_size,
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
        else:
            # 기본 클러스터링 (첫 두 개만)
            cluster_labels = np.zeros(len(valid_detections))
            if len(valid_detections) > len(valid_detections) // 2:
                cluster_labels[len(valid_detections) // 2:] = 1
        
        # 클러스터 생성
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # 노이즈 제외
                continue
            if label not in clusters:
                clusters[label] = FaceCluster(label)
            clusters[label].add_detection(valid_detections[idx])
        
        # 클러스터 통계 계산
        cluster_list = list(clusters.values())
        for cluster in cluster_list:
            cluster.calculate_statistics()
        
        # 크기 필터링 (너무 작은 클러스터 제거)
        filtered_clusters = [c for c in cluster_list if len(c.detections) >= self.min_cluster_size]
        
        # 동일 인물 클러스터 병합 (중복 방지)
        merged_clusters = self.merge_similar_clusters(filtered_clusters)
        
        logger.info(f"클러스터링 완료: {len(merged_clusters)}개 클러스터 생성 (병합 후)")
        
        return merged_clusters
    
    def _split_single_cluster(self, cluster: FaceCluster, video_duration: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """단일 클러스터를 크기 기준으로 2개로 분할"""
        print(f"🔄 단일 클러스터 분할 시도 ({len(cluster.detections)}개 검출)")
        
        if len(cluster.detections) < 10:
            logger.error(f"검출 수가 너무 적음 ({len(cluster.detections)}개)")
            return None, None
        
        # 크기 기준으로 정렬 (큰 얼굴 vs 작은 얼굴)
        detections_with_size = [(det, det['size']) for det in cluster.detections]
        detections_with_size.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 60%와 하위 40%로 분할 (큰 얼굴이 더 중요)
        split_point = int(len(detections_with_size) * 0.6)
        
        # 분할된 클러스터 생성
        cluster1 = FaceCluster(0)  # 큰 얼굴들
        cluster2 = FaceCluster(1)  # 작은 얼굴들
        
        for i, (det, size) in enumerate(detections_with_size):
            if i < split_point:
                cluster1.add_detection(det)
            else:
                cluster2.add_detection(det)
        
        cluster1.calculate_statistics()
        cluster2.calculate_statistics()
        
        # 중요도 점수 계산
        cluster1.importance_score = self.calculate_importance_score(cluster1, video_duration)
        cluster2.importance_score = self.calculate_importance_score(cluster2, video_duration)
        
        logger.info(f"클러스터 분할 완료: {len(cluster1.detections)}개 + {len(cluster2.detections)}개")
        
        # 화자 정보 생성
        speaker1_info = {
            'cluster': cluster1,
            'representative_embedding': cluster1.representative_embedding,
            'average_position': cluster1.average_position,
            'importance_score': cluster1.importance_score,
            'appearance_count': len(cluster1.detections),
            'stats': cluster1.get_stats()
        }
        
        speaker2_info = {
            'cluster': cluster2,
            'representative_embedding': cluster2.representative_embedding,
            'average_position': cluster2.average_position,
            'importance_score': cluster2.importance_score,
            'appearance_count': len(cluster2.detections),
            'stats': cluster2.get_stats()
        }
        
        return speaker1_info, speaker2_info
    
    def calculate_motion_score(self, cluster: FaceCluster) -> float:
        """움직임 점수 계산: 위치 변화가 많은 얼굴 = 활발한 화자"""
        if len(cluster.positions) < 2:
            return 0.0
        
        total_movement = 0
        movement_count = 0
        
        for i in range(1, len(cluster.positions)):
            pos1 = cluster.positions[i-1]
            pos2 = cluster.positions[i]
            
            # 유클리드 거리 계산
            movement = ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5
            total_movement += movement
            movement_count += 1
        
        if movement_count == 0:
            return 0.0
        
        # 평균 움직임 계산
        avg_movement = total_movement / movement_count
        
        # 정규화 (0~1): 평균 픽셀 이동이 50px일 때 1.0
        motion_score = min(avg_movement / 50.0, 1.0)
        
        return motion_score
    
    def merge_similar_clusters(self, clusters: List[FaceCluster]) -> List[FaceCluster]:
        """유사한 클러스터 병합 (동일 인물 중복 방지)"""
        if len(clusters) <= 1:
            return clusters
            
        print(f"🔄 클러스터 병합 시작: {len(clusters)}개 클러스터")
        
        merged = []
        used = set()
        merge_threshold = 0.65  # 임베딩 유사도 임계값 (0.8→0.65로 더 엄격한 병합)
        
        import torch.nn.functional as F
        
        for i, cluster1 in enumerate(clusters):
            if i in used:
                continue
            
            # cluster1을 기준으로 병합할 클러스터들 찾기
            merge_candidates = [cluster1]
            
            # 다른 클러스터들과 비교
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                
                # 대표 임베딩이 있는 경우만 비교
                if (cluster1.representative_embedding is not None and 
                    cluster2.representative_embedding is not None):
                    
                    # 코사인 유사도 계산
                    similarity = F.cosine_similarity(
                        cluster1.representative_embedding.unsqueeze(0),
                        cluster2.representative_embedding.unsqueeze(0)
                    ).item()
                    
                    if similarity > merge_threshold:
                        merge_candidates.append(cluster2)
                        used.add(j)
                        
                        if self.debug_mode:
                            print(f"   🔗 클러스터 {i}와 {j} 병합 (유사도: {similarity:.3f})")
            
            # 병합 실행 (최소 2개 클러스터 유지 조건 추가)
            if len(merge_candidates) > 1:
                # 안전장치: 병합 후에도 최소 2개 클러스터가 남는지 확인
                remaining_clusters = len(clusters) - len([c for c in clusters if clusters.index(c) not in used]) - len(merge_candidates) + 1
                
                if remaining_clusters >= 2 or len(merged) == 0:  # 2개 이상 남거나 첫 번째 병합이면 진행
                    merged_cluster = self._merge_clusters(merge_candidates)
                    merged.append(merged_cluster)
                    
                    if self.debug_mode:
                        logger.info(f"{len(merge_candidates)}개 클러스터 병합 완료 (남은 클러스터: {remaining_clusters}개)")
                else:
                    # 병합하면 1개만 남으면 병합 취소
                    merged.append(cluster1)
                    if self.debug_mode:
                        logger.warning("병합 취소 (최소 2개 클러스터 유지를 위해)")
            else:
                merged.append(cluster1)
        
        logger.info(f"클러스터 병합 완료: {len(clusters)}개 → {len(merged)}개")
        return merged
    
    def _merge_clusters(self, clusters: List[FaceCluster]) -> FaceCluster:
        """여러 클러스터를 하나로 병합"""
        if len(clusters) == 1:
            return clusters[0]
        
        # 첫 번째 클러스터를 기준으로 나머지 병합
        base_cluster = clusters[0]
        
        for cluster in clusters[1:]:
            # 모든 검출 데이터 병합
            base_cluster.detections.extend(cluster.detections)
            base_cluster.embeddings.extend(cluster.embeddings)
            base_cluster.positions.extend(cluster.positions)
            base_cluster.sizes.extend(cluster.sizes)
            base_cluster.timestamps.extend(cluster.timestamps)
            base_cluster.confidences.extend(cluster.confidences)
        
        # 통계 재계산
        base_cluster.calculate_statistics()
        
        return base_cluster
    
    def _fallback_clustering(self, all_detections: List[Dict[str, Any]]) -> List[FaceCluster]:
        """임베딩 없을 때 위치 기반 폴백 클러스터링"""
        logger.warning("위치 기반 폴백 클러스터링 사용")
        
        if len(all_detections) < 10:
            return []
        
        # X 좌표로 정렬하여 좌우 구분
        sorted_detections = sorted(all_detections, key=lambda x: x['center'][0])
        mid_idx = len(sorted_detections) // 2
        
        # 좌우 클러스터 생성
        left_cluster = FaceCluster(0)
        right_cluster = FaceCluster(1)
        
        for i, detection in enumerate(sorted_detections):
            if i < mid_idx:
                left_cluster.add_detection(detection)
            else:
                right_cluster.add_detection(detection)
        
        left_cluster.calculate_statistics()
        right_cluster.calculate_statistics()
        
        return [left_cluster, right_cluster]
    
    def calculate_importance_score(self, cluster: FaceCluster, video_duration: float) -> float:
        """클러스터의 중요도 점수 계산"""
        if not cluster.detections:
            return 0.0
        
        # 1. 등장 빈도 점수 (정규화)
        frequency_score = min(len(cluster.detections) / 100, 1.0)
        
        # 2. 총 얼굴 크기 점수 (정규화)
        total_size = sum(cluster.sizes)
        size_score = min(total_size / 1000000, 1.0)
        
        # 3. 화면 중앙 가중치
        center_weights = []
        for pos in cluster.positions:
            x, y = pos
            # 1920x1080 기준 중앙(960, 540)에서의 거리
            distance_from_center = ((x - 960)**2 + (y - 540)**2)**0.5
            center_weight = max(0, 1 - distance_from_center / 800)
            center_weights.append(center_weight)
        center_score = sum(center_weights) / len(center_weights) if center_weights else 0
        
        # 4. 시간적 분포 점수
        if len(cluster.timestamps) > 1:
            time_span = max(cluster.timestamps) - min(cluster.timestamps)
            time_distribution_score = min(time_span / video_duration, 1.0)
        else:
            time_distribution_score = 0.0
        
        # 5. 크기 일관성 점수 (편차가 작을수록 높은 점수)
        if len(cluster.sizes) > 1:
            size_std = np.std(cluster.sizes)
            size_mean = np.mean(cluster.sizes)
            consistency_score = max(0, 1 - size_std / (size_mean + 1e-6))
        else:
            consistency_score = 1.0
        
        # 6. 평균 신뢰도
        confidence_score = sum(cluster.confidences) / len(cluster.confidences) if cluster.confidences else 0
        
        # 7. 움직임 점수 (활발한 화자 감지)
        motion_score = self.calculate_motion_score(cluster)
        
        # 최종 점수 계산 (가중 평균)
        importance_score = (
            frequency_score * self.weights['frequency'] +
            size_score * self.weights['size'] +
            center_score * self.weights['center'] +
            time_distribution_score * self.weights['time_distribution'] +
            motion_score * self.weights['motion'] +
            consistency_score * self.weights['consistency'] +
            confidence_score * self.weights['confidence']
        )
        
        return importance_score
    
    def select_main_speakers(self, clusters: List[FaceCluster], video_duration: float) -> Tuple[Optional[FaceCluster], Optional[FaceCluster]]:
        """주요 화자 2명 선정"""
        logger.debug("주요 화자 선정 중...")
        
        if len(clusters) < 2:
            logger.warning(f"충분한 클러스터 없음 ({len(clusters)}개)")
            return None, None
        
        # 각 클러스터의 중요도 점수 계산
        for cluster in clusters:
            cluster.importance_score = self.calculate_importance_score(cluster, video_duration)
        
        # 점수순으로 정렬
        sorted_clusters = sorted(clusters, key=lambda c: c.importance_score, reverse=True)
        
        speaker1 = sorted_clusters[0]
        speaker2 = sorted_clusters[1]
        
        # 결과 출력
        if self.debug_mode:
            logger.info("주요 화자 자동 선정 완료:")
            for i, speaker in enumerate([speaker1, speaker2], 1):
                stats = speaker.get_stats()
                print(f"   화자{i}: {stats['appearance_count']}회 등장, "
                      f"점수 {speaker.importance_score:.3f}, "
                      f"평균위치 {stats['average_position']}")
        
        return speaker1, speaker2
    
    def analyze_video(self, video_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """좌우 기반 전체 영상 분석하여 주요 화자 2명 반환"""
        start_time = time.time()
        
        print("\n" + "=" * 60)
        logger.debug("좌우 기반 화자 분석 시작")
        print("=" * 60)
        
        # 1단계: 좌우 분리 스캔
        left_detections, right_detections = self.scan_video_left_right(video_path)
        
        # 비디오 길이 계산
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps
        cap.release()
        
        logger.debug("좌우 분리 결과:")
        print(f"   왼쪽 영역: {len(left_detections)}개 얼굴")
        print(f"   오른쪽 영역: {len(right_detections)}개 얼굴")
        
        # 2단계: 각 영역에서 클러스터링
        speaker1_info = None
        speaker2_info = None
        
        # 왼쪽 영역에서 주요 화자 선정 (Person1)
        if len(left_detections) >= self.min_cluster_size:
            print(f"\n🔄 왼쪽 영역 클러스터링 시작...")
            left_clusters = self.cluster_faces(left_detections)
            
            if left_clusters:
                # 가장 많이 나온 클러스터 선택
                left_clusters.sort(key=lambda c: len(c.detections), reverse=True)
                main_left_cluster = left_clusters[0]
                main_left_cluster.importance_score = self.calculate_importance_score(main_left_cluster, video_duration)
                
                speaker1_info = self._create_speaker_info(main_left_cluster, "Person1 (Left)")
                logger.info(f"Person1 선정: {len(main_left_cluster.detections)}개 검출, 점수 {main_left_cluster.importance_score:.3f}")
            else:
                logger.error("왼쪽 영역 클러스터링 실패")
        else:
            logger.warning(f"왼쪽 영역 데이터 부족: {len(left_detections)}개 < {self.min_cluster_size}개")
        
        # 오른쪽 영역에서 주요 화자 선정 (Person2)
        if len(right_detections) >= self.min_cluster_size:
            print(f"\n🔄 오른쪽 영역 클러스터링 시작...")
            right_clusters = self.cluster_faces(right_detections)
            
            if right_clusters:
                # 가장 많이 나온 클러스터 선택
                right_clusters.sort(key=lambda c: len(c.detections), reverse=True)
                main_right_cluster = right_clusters[0]
                main_right_cluster.importance_score = self.calculate_importance_score(main_right_cluster, video_duration)
                
                speaker2_info = self._create_speaker_info(main_right_cluster, "Person2 (Right)")
                logger.info(f"Person2 선정: {len(main_right_cluster.detections)}개 검출, 점수 {main_right_cluster.importance_score:.3f}")
            else:
                logger.error("오른쪽 영역 클러스터링 실패")
        else:
            logger.warning(f"오른쪽 영역 데이터 부족: {len(right_detections)}개 < {self.min_cluster_size}개")
        
        # 3단계: 결과 검증 및 반환
        elapsed_time = time.time() - start_time
        print(f"\n🎉 좌우 기반 분석 완료 ({elapsed_time:.1f}초)")
        
        if speaker1_info and speaker2_info:
            logger.info("양쪽 화자 모두 선정 성공")
            return speaker1_info, speaker2_info
        elif speaker1_info or speaker2_info:
            logger.warning("한쪽 화자만 선정됨")
            return speaker1_info, speaker2_info
        else:
            logger.error("화자 선정 실패")
            return None, None
    
    def _create_speaker_info(self, cluster: FaceCluster, label: str = "") -> Dict[str, Any]:
        """클러스터로부터 화자 정보 딕셔너리 생성"""
        # 통계 정보 생성
        stats = cluster.get_stats()
        
        # 화자 정보 구성
        speaker_info = {
            'cluster': cluster,
            'representative_embedding': cluster.representative_embedding,
            'average_position': cluster.average_position,
            'importance_score': getattr(cluster, 'importance_score', 0.0),
            'appearance_count': len(cluster.detections),
            'stats': stats,
            'label': label
        }
        
        if self.debug_mode:
            print(f"📋 {label} 정보:")
            print(f"   검출 횟수: {speaker_info['appearance_count']}회")
            print(f"   중요도 점수: {speaker_info['importance_score']:.3f}")
            print(f"   평균 위치: {speaker_info['average_position']}")
            print(f"   평균 크기: {stats.get('average_size', 0):.1f}px")
        
        return speaker_info


class OneMinuteAnalyzer:
    """처음 1분(60초)을 100% 분석하여 확실한 화자 프로파일 생성 + IdentityBank 통합"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        # Phase 2: IdentityBank 통합
        from dual_face_tracker.core.identity_bank import IdentityBank
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.identity_bank = IdentityBank(max_samples=128, device=device)  # 1분 분석용 더 큰 뱅크
        
        # FaceNet 모델 초기화 (AutoSpeakerDetector와 동일)
        self.resnet = None
        self.face_transform = None
        self._initialize_facenet()
        
        # 얼굴 검출기 초기화
        self.face_cascade = None
        self.model_manager = None
        self._initialize_face_detector()
        
        # 클러스터링 설정
        self.clustering_eps = 0.3
        self.min_cluster_size = 20  # 1분 분석이므로 더 큰 클러스터 요구
        
    def _initialize_face_detector(self):
        """얼굴 검출기 초기화 (AutoSpeakerDetector와 동일)"""
        cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        
        if Path(cascade_path).exists():
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.face_cascade.empty():
                if self.debug_mode:
                    logger.info("Haar Cascade 얼굴 검출기 로드 완료")
            else:
                raise RuntimeError("❌ Haar Cascade 로드 실패")
        else:
            raise RuntimeError(f"❌ Haar Cascade 파일 없음: {cascade_path}")
            
        # ModelManager 자동 감지 시도
        try:
            from .model_manager import ModelManager
            self.model_manager = ModelManager()
            if self.debug_mode:
                logger.info("ModelManager 로드 완료 (MTCNN 사용 가능)")
        except ImportError:
            if self.debug_mode:
                logger.warning("ModelManager 없음 (Haar Cascade만 사용)")
    
    def _initialize_facenet(self):
        """FaceNet 모델 초기화 (AutoSpeakerDetector와 동일)"""
        try:
            from torchvision.models import inception_v3
            from facenet_pytorch import InceptionResnetV1
            
            # InceptionResnetV1 모델 로드
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            if torch.cuda.is_available():
                self.resnet = self.resnet.cuda()
            
            # 이미지 전처리 파이프라인
            self.face_transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            if self.debug_mode:
                logger.info("FaceNet 모델 로드 완료")
                
        except ImportError as e:
            if self.debug_mode:
                logger.warning(f"FaceNet 로드 실패: {e}")
            self.resnet = None
            self.face_transform = None

    def analyze_first_minute(self, video_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """처음 1분을 100% 분석하여 확실한 화자 프로파일 생성"""
        print("\n" + "=" * 70)
        logger.debug("1분 집중 분석 시작")
        print("=" * 70)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오 열기 실패: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyze_frames = min(int(fps * 60), total_frames)  # 60초 또는 전체 프레임 중 적은 것
        
        print(f"📹 비디오 정보: {total_frames}프레임, {total_frames/fps:.1f}초, {fps:.1f}fps")
        logger.debug(f"분석 범위: 처음 {analyze_frames}프레임 (60초)")
        print(f"⚖️ 좌우 분리 기준: x=960px")
        
        left_face_data = []   # 왼쪽 영역 모든 얼굴
        right_face_data = []  # 오른쪽 영역 모든 얼굴
        frame_idx = 0
        start_time = time.time()
        
        try:
            while frame_idx < analyze_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 모든 프레임 분석 (100% 샘플링)
                faces = self._detect_faces_in_frame(frame)
                
                for face_bbox, confidence in faces:
                    x1, y1, x2, y2 = face_bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    size = (x2 - x1) * (y2 - y1)
                    
                    # 임베딩 추출
                    embedding = self.extract_face_embedding(frame, face_bbox)
                    
                    face_info = {
                        'frame_idx': frame_idx,
                        'bbox': face_bbox,
                        'center': (center_x, center_y),
                        'size': size,
                        'confidence': confidence,
                        'embedding': embedding
                    }
                    
                    # 좌우 분리 (x=960 기준)
                    if center_x < 960:  # 왼쪽 영역
                        left_face_data.append(face_info)
                    else:  # 오른쪽 영역
                        right_face_data.append(face_info)
                
                frame_idx += 1
                
                # 진행률 표시
                if frame_idx % 300 == 0:  # 10초마다
                    progress = (frame_idx / analyze_frames) * 100
                    elapsed = time.time() - start_time
                    logger.debug(f"진행률: {progress:.1f}% (좌:{len(left_face_data)}, 우:{len(right_face_data)}, {elapsed:.1f}초)")
        
        finally:
            cap.release()
        
        elapsed = time.time() - start_time
        logger.info(f"1분 스캔 완료: 좌측 {len(left_face_data)}개, 우측 {len(right_face_data)}개 ({elapsed:.1f}초)")
        
        # 각 영역에서 화자 프로파일 생성
        person1_profile = self._create_speaker_profile(left_face_data, "Person1 (Left)")
        person2_profile = self._create_speaker_profile(right_face_data, "Person2 (Right)")
        
        return person1_profile, person2_profile

    def extract_face_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[torch.Tensor]:
        """얼굴 크롭에서 임베딩 추출 (AutoSpeakerDetector와 동일)"""
        if self.resnet is None or self.face_transform is None:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # OpenCV BGR → PIL RGB 변환
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)
            
            # 전처리 및 배치 차원 추가
            face_tensor = self.face_transform(pil_image).unsqueeze(0)
            
            # GPU로 이동 (사용 가능한 경우)
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            
            # 임베딩 생성
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
                
            # L2 정규화
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.squeeze(0).cpu()  # CPU로 이동 후 배치 차원 제거
            
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            if self.debug_mode:
                logger.warning(f"임베딩 추출 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 임베딩 추출 오류: {e}")
            return None

    def _detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """프레임에서 얼굴 검출 (AutoSpeakerDetector와 동일)"""
        faces = []
        
        # 1. MTCNN으로 얼굴 검출 시도 (ModelManager 사용 가능시)
        if self.model_manager and self.model_manager.mtcnn:
            try:
                mtcnn = self.model_manager.mtcnn
                # PIL Image로 변환 (MTCNN 요구사항)
                from PIL import Image
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    pil_frame = frame
                
                boxes, probs = mtcnn.detect(pil_frame)
                
                if boxes is not None and len(boxes) > 0:
                    for box, prob in zip(boxes, probs):
                        if prob > 0.5:  # 신뢰도 임계값
                            x1, y1, x2, y2 = box.astype(int)
                            # 바운딩 박스 검증
                            if x2 > x1 and y2 > y1 and (x2-x1) >= 30 and (y2-y1) >= 30:
                                bbox = (x1, y1, x2, y2)
                                faces.append((bbox, float(prob)))
                    
                    if faces:  # MTCNN에서 얼굴을 찾으면 반환
                        return faces
            except Exception as e:
                if self.debug_mode:
                    logger.warning(f"MTCNN 얼굴 검출 실패, Haar Cascade 폴백: {e}")
        
        # 2. Haar Cascade 폴백
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected_faces:
                bbox = (x, y, x + w, y + h)
                confidence = 0.8  # Haar Cascade는 확률을 제공하지 않으므로 고정값
                faces.append((bbox, confidence))
                
        except Exception as e:
            if self.debug_mode:
                logger.warning(f"Haar Cascade 얼굴 검출 실패: {e}")
        
        return faces

    def _create_speaker_profile(self, face_data: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
        """1분 데이터로 확실한 화자 프로파일 생성"""
        print(f"\n🔄 {label} 프로파일 생성 중...")
        
        if len(face_data) < self.min_cluster_size:
            logger.error(f"{label} 데이터 부족: {len(face_data)}개 < {self.min_cluster_size}개")
            return None
        
        # 임베딩이 있는 데이터만 필터링
        valid_data = [f for f in face_data if f['embedding'] is not None]
        
        if len(valid_data) < self.min_cluster_size:
            logger.error(f"{label} 유효 임베딩 부족: {len(valid_data)}개 < {self.min_cluster_size}개")
            return None
        
        # DBSCAN 클러스터링으로 같은 사람끼리 그룹화
        try:
            embeddings = [f['embedding'] for f in valid_data]
            embedding_matrix = torch.stack(embeddings).numpy()
            
            # 코사인 유사도 기반 클러스터링
            similarity_matrix = cosine_similarity(embedding_matrix)
            distance_matrix = 1 - similarity_matrix
            distance_matrix = np.clip(distance_matrix, 0, 2)
            
            if CLUSTERING_AVAILABLE:
                clustering = DBSCAN(
                    eps=self.clustering_eps,
                    min_samples=self.min_cluster_size,
                    metric='precomputed'
                )
                cluster_labels = clustering.fit_predict(distance_matrix)
            else:
                # 폴백: 모든 데이터를 하나의 클러스터로
                cluster_labels = np.zeros(len(valid_data))
            
            # 가장 큰 클러스터 = 주요 화자
            from collections import Counter
            label_counts = Counter(cluster_labels)
            # -1 (노이즈) 제외하고 가장 큰 클러스터
            valid_labels = [(label, count) for label, count in label_counts.items() if label != -1]
            
            if not valid_labels:
                logger.error(f"{label} 유효 클러스터 없음")
                return None
            
            main_cluster_label = max(valid_labels, key=lambda x: x[1])[0]
            main_faces = [valid_data[i] for i, l in enumerate(cluster_labels) if l == main_cluster_label]
            
            logger.info(f"{label} 클러스터링 완료: {len(main_faces)}개 검출 (전체 {len(cluster_labels)}개 중)")
            
        except Exception as e:
            logger.warning(f"{label} 클러스터링 실패, 전체 데이터 사용: {e}")
            main_faces = valid_data
        
        # Phase 2: IdentityBank를 사용한 강력한 프로파일 생성
        embeddings = [f['embedding'] for f in main_faces]
        centers = [f['center'] for f in main_faces]
        sizes = [f['size'] for f in main_faces]
        
        # IdentityBank에 임베딩 등록 (A/B 슬롯 결정은 label에 따라)
        slot = 'A' if 'Person1' in label or 'Left' in label else 'B'
        
        # 모든 임베딩을 IdentityBank에 업데이트
        for emb in embeddings:
            self.identity_bank.update(slot, emb)
        
        # IdentityBank에서 중앙값 프로토타입 생성 (노이즈 강건)
        prototype_embedding = self.identity_bank.proto(slot)
        
        profile = {
            'label': label,
            'slot': slot,  # A/B 슬롯 정보 추가
            'appearance_count': len(main_faces),
            
            # Phase 2: IdentityBank 프로토타입 (중앙값 기반)
            'reference_embedding': prototype_embedding,
            'identity_bank_size': len(self.identity_bank.bank[slot]),
            
            # 평균 위치 (좌우 기준점)
            'average_position': np.mean(centers, axis=0),
            
            # 평균 크기 (앞뒤 거리 추정)
            'average_size': np.mean(sizes),
            
            # 크기 범위 (최소/최대)
            'size_range': (min(sizes), max(sizes)),
            
            # 위치 범위 (움직임 범위)
            'position_range': {
                'x_min': min([c[0] for c in centers]),
                'x_max': max([c[0] for c in centers]),
                'y_min': min([c[1] for c in centers]),
                'y_max': max([c[1] for c in centers])
            }
        }
        
        if self.debug_mode:
            print(f"📋 {label} 프로파일 (IdentityBank 슬롯: {slot}):")
            print(f"   - 검출 횟수: {profile['appearance_count']}회")
            print(f"   - IdentityBank 크기: {profile['identity_bank_size']}개 임베딩")
            print(f"   - 평균 크기: {profile['average_size']:.0f}px")
            print(f"   - 평균 위치: ({profile['average_position'][0]:.0f}, {profile['average_position'][1]:.0f})")
            print(f"   - X 범위: {profile['position_range']['x_min']:.0f} ~ {profile['position_range']['x_max']:.0f}")
            print(f"   - Y 범위: {profile['position_range']['y_min']:.0f} ~ {profile['position_range']['y_max']:.0f}")
            logger.info(f"{label} 프로토타입: {'✅ 생성됨' if prototype_embedding is not None else '❌ 없음'}")
        
        return profile


if __name__ == "__main__":
    # 테스트 코드
    detector = AutoSpeakerDetector(debug_mode=True)
    
    # 샘플 비디오로 테스트
    video_path = "tests/videos/2people_sample1.mp4"
    if Path(video_path).exists():
        speaker1, speaker2 = detector.analyze_video(video_path)
        
        if speaker1 and speaker2:
            print("\n" + "="*50)
            logger.info("자동 화자 선정 결과")
            print("="*50)
            print(f"화자1: {speaker1['appearance_count']}회 등장, 점수 {speaker1['importance_score']:.3f}")
            print(f"화자2: {speaker2['appearance_count']}회 등장, 점수 {speaker2['importance_score']:.3f}")
        else:
            print("❌ 화자 선정 실패")
    else:
        logger.warning(f"테스트 비디오 없음: {video_path}")