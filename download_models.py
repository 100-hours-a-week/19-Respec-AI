import os
import sys
import logging
import easyocr
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_models(model_path: str = './models'):
    """
    EasyOCR 모델 다운로드
    Args:
        model_path: 모델 저장 경로
    """
    try:
        # 모델 경로 생성
        model_path = os.path.abspath(model_path)
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"모델 저장 경로: {model_path}")
        
        # EasyOCR 리더 초기화 (모델 자동 다운로드)
        logger.info("EasyOCR 모델 다운로드 시작...")
        reader = easyocr.Reader(
            ['ko', 'en'],
            model_storage_directory=model_path,
            download_enabled=True,
            recog_network='korean_g2',
            gpu=False,
            verbose=True
        )
        
        # 다운로드된 모델 파일 확인
        model_files = os.listdir(model_path)
        logger.info("다운로드된 모델 파일:")
        for file in model_files:
            logger.info(f"- {file}")
        
        logger.info("모델 다운로드 완료")
        return True
        
    except Exception as e:
        logger.error(f"모델 다운로드 실패: {str(e)}")
        return False

if __name__ == "__main__":
    # 커맨드라인 인자로 모델 경로 받기
    model_path = sys.argv[1] if len(sys.argv) > 1 else './models'
    
    # 모델 다운로드 실행
    success = download_models(model_path)
    sys.exit(0 if success else 1) 