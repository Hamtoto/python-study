#!/usr/bin/env python3
"""
MemoryPoolManager 문제 해결 확인 스크립트
"""

import asyncio

def test_memory_pool_creation():
    """MemoryPoolManager 생성 테스트"""
    try:
        from dual_face_tracker.core.memory_pool_manager import MemoryPoolManager, MemoryPoolConfig
        from dual_face_tracker.core.multi_stream_processor import MultiStreamConfig
        
        print("✅ 모듈 import 성공")
        
        # MemoryPoolConfig 직접 생성
        memory_config = MemoryPoolConfig(max_vram_usage=0.75)
        print("✅ MemoryPoolConfig 생성 성공")
        
        # MemoryPoolManager 생성  
        memory_pool = MemoryPoolManager(memory_config, gpu_id=0)
        print("✅ MemoryPoolManager 생성 성공")
        
        # MultiStreamConfig도 테스트
        multi_config = MultiStreamConfig(max_streams=4)
        print("✅ MultiStreamConfig 생성 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False

async def test_stream_manager_creation():
    """StreamManager 생성 테스트"""
    try:
        from dual_face_tracker.core.stream_manager import StreamManager
        
        # StreamManager 생성 (초기화 없이)
        stream_manager = StreamManager(max_streams=4, gpu_id=0)
        print("✅ StreamManager 생성 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ StreamManager 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔧 MemoryPoolManager 문제 해결 테스트")
    print("=" * 50)
    
    # 기본 생성 테스트
    success1 = test_memory_pool_creation()
    success2 = asyncio.run(test_stream_manager_creation())
    
    if success1 and success2:
        print("\n🎉 모든 기본 생성 테스트 성공!")
        print("이제 완전한 테스트를 실행해보세요:")
        print("./run_phase3_test.sh")
    else:
        print("\n❌ 기본 생성 테스트 실패")
        print("추가 수정이 필요합니다.")