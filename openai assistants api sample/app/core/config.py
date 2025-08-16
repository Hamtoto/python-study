import json
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    openai_api_key: str
    assistant_id: str
    host: str = "0.0.0.0"
    port: int = 8020
    debug: bool = True
    cors_origins: str = '["*"]'
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """CORS origins를 리스트로 변환"""
        try:
            return json.loads(self.cors_origins)
        except json.JSONDecodeError:
            return ["*"]


settings = Settings()