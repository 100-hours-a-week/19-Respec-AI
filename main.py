from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tempfile
import os
import logging
import shutil
from typing import Optional
from model import ResumeTextExtractor, ResumeSpecExtractor, extract_specs_from_pdf
from app.api.endpoints import router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="이력서 PDF 분석 API",
    description="EasyOCR 기반 이력서 분석 시스템",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 설정
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# 모델 경로 설정
model_path = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_path, exist_ok=True)
os.environ["MODEL_PATH"] = model_path

# API 라우터 등록
app.include_router(router, prefix="/api")

# 임시 파일 정리 함수
def cleanup_temp_file(file_path: str):
    """임시 파일 정리"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.error(f"임시 파일 정리 실패: {str(e)}")

@app.get("/")
async def root():
    """
    루트 엔드포인트
    """
    return {
        "status": "online",
        "version": "2.0",
        "docs_url": "/docs",
        "api_prefix": "/api"
    }

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    PDF 이력서 분석 엔드포인트
    Args:
        file: PDF 파일
        background_tasks: 백그라운드 작업
    Returns:
        분석 결과
    """
    temp_path = None
    try:
        # 파일 확장자 검증
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="PDF 파일만 업로드 가능합니다."
            )

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # 백그라운드 작업으로 임시 파일 정리 등록
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)

        # PDF 분석 수행
        result = extract_specs_from_pdf(temp_path, model_path)

        if result['status'] == 'success':
            return JSONResponse(content=result)
        else:
            raise HTTPException(
                status_code=500,
                detail=result['message']
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"이력서 분석 중 오류 발생: {str(e)}", exc_info=True)
        # 임시 파일 정리
        if temp_path and os.path.exists(temp_path):
            cleanup_temp_file(temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"이력서 분석 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # EasyOCR 모델 초기화 테스트
        text_extractor = ResumeTextExtractor(model_path)
        return {
            "status": "healthy",
            "model_initialized": True
        }
    except Exception as e:
        logger.error(f"헬스 체크 실패: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/model-status")
async def model_status():
    """모델 상태 확인 엔드포인트"""
    try:
        model_files = os.listdir(model_path)
        return {
            "status": "success",
            "model_path": model_path,
            "model_files": model_files,
            "model_initialized": len(model_files) > 0
        }
    except Exception as e:
        logger.error(f"모델 상태 확인 실패: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)