import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    APP_NAME: str = "이력서 PDF 분석 시스템"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 모델 설정
    MODEL_PATH: str = Field(
        default="./models",
        description="EasyOCR 모델 저장 경로"
    )
    USE_GPU: bool = Field(
        default=False,
        description="GPU 사용 여부"
    )
    MODEL_LANGUAGES: list = Field(
        default=["ko", "en"],
        description="인식할 언어 목록"
    )
    RECOG_NETWORK: str = Field(
        default="korean_g2",
        description="한국어 인식 네트워크"
    )
    
    # OCR 설정
    OCR_BATCH_SIZE: int = Field(
        default=8,
        description="OCR 배치 크기"
    )
    OCR_CONTRAST_THS: float = Field(
        default=0.1,
        description="대비 임계값"
    )
    OCR_WIDTH_THS: float = Field(
        default=0.7,
        description="너비 임계값"
    )
    OCR_HEIGHT_THS: float = Field(
        default=0.7,
        description="높이 임계값"
    )
    
    # 이미지 처리 설정
    MAX_IMAGE_DIMENSION: int = Field(
        default=2000,
        description="최대 이미지 크기"
    )
    DPI: int = Field(
        default=300,
        description="PDF 변환 DPI"
    )
    
    # API 설정
    API_PREFIX: str = "/api"
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "resume-analyzer"
    
    # 파일 업로드 설정
    UPLOAD_DIR: str = Field(
        default="./uploads",
        description="업로드 파일 저장 경로"
    )
    MAX_UPLOAD_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="최대 업로드 파일 크기"
    )
    ALLOWED_EXTENSIONS: set = Field(
        default={"pdf"},
        description="허용된 파일 확장자"
    )
    
    # 로깅 설정
    LOG_LEVEL: str = Field(
        default="INFO",
        description="로그 레벨"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷"
    )
    LOG_FILE: str = Field(
        default="app.log",
        description="로그 파일 경로"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 디렉토리 생성
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        
        # 절대 경로 변환
        self.MODEL_PATH = os.path.abspath(self.MODEL_PATH)
        self.UPLOAD_DIR = os.path.abspath(self.UPLOAD_DIR)
        self.LOG_FILE = os.path.abspath(self.LOG_FILE)

# 전역 설정 인스턴스
settings = Settings()

def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return settings 