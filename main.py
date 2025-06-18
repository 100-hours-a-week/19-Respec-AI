from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import os
from model import OCRModel

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 필요한 디렉토리 생성
REQUIRED_DIRS = ['static', 'templates']
for dir_name in REQUIRED_DIRS:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f"Created directory: {dir_name}")

app = FastAPI()

# 정적 파일 및 템플릿 폴더 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")   # ← 반드시 templates

# OCR 모델 초기화
ocr_model = OCRModel()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocrspec")
async def analyze_resume(file: UploadFile = File(...)):
    """이력서 PDF 분석"""
    try:
        # 파일 확장자 검사
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
        contents = await file.read()
        results = ocr_model.process_pdf(contents)
        
        if not results:
            raise HTTPException(status_code=400, detail="텍스트를 추출할 수 없습니다.")
        
        return {"full_text": results}
    except Exception as e:
        logger.error(f"이력서 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 