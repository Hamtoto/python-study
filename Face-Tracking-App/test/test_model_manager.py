# -*- coding: utf-8 -*-
"""
ModelManager 단위 테스트 모듈
"""
import pytest
import torch
from core.model_manager import ModelManager

@pytest.fixture(scope="module")
def model_manager():
    """테스트용 ModelManager 인스턴스 제공"""
    # 테스트 환경에서 GPU가 없으면 CPU 사용
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return ModelManager(device)

def test_model_manager_initialization(model_manager):
    """ModelManager가 올바르게 초기화되는지 테스트"""
    assert model_manager is not None
    assert isinstance(model_manager, ModelManager)

def test_get_mtcnn_model(model_manager):
    """
    MTCNN 모델이 올바르게 로드되는지 테스트
    (실제 모델 로딩은 시간이 걸리므로, 이미 로드된 인스턴스를 반환하는지 확인)
    """
    mtcnn = model_manager.get_mtcnn()
    assert mtcnn is not None
    # MTCNN 모델의 특정 속성을 확인하여 로딩 여부 간접 확인
    assert hasattr(mtcnn, 'eval') # 모델 객체인지 확인

def test_get_resnet_model(model_manager):
    """
    ResNet 모델이 올바르게 로드되는지 테스트
    """
    resnet = model_manager.get_resnet()
    assert resnet is not None
    # ResNet 모델의 특정 속성을 확인하여 로딩 여부 간접 확인
    assert hasattr(resnet, 'eval') # 모델 객체인지 확인

def test_model_caching(model_manager):
    """
    모델이 캐싱되어 동일한 인스턴스를 반환하는지 테스트
    """
    mtcnn1 = model_manager.get_mtcnn()
    mtcnn2 = model_manager.get_mtcnn()
    assert mtcnn1 is mtcnn2

    resnet1 = model_manager.get_resnet()
    resnet2 = model_manager.get_resnet()
    assert resnet1 is resnet2

def test_face_tensor_pool(model_manager):
    """
    face_tensor_pool이 올바르게 동작하는지 테스트
    """
    # GPU 환경에서만 테스트 (CPU에서는 None 반환)
    if torch.cuda.is_available():
        tensor_pool = model_manager.get_face_tensor_pool()
        assert tensor_pool is not None
        assert isinstance(tensor_pool, torch.Tensor)
        assert tensor_pool.device.type == 'cuda'
        assert tensor_pool.shape == (1, 3, 160, 160)
    else:
        tensor_pool = model_manager.get_face_tensor_pool()
        assert tensor_pool is None
