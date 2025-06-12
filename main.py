# main.py - 이력서 PDF 분석 웹 애플리케이션

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
from model import ResumeAnalyzer

app = FastAPI()

# 정적 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 업로드 디렉토리 설정
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 이력서 분석기 초기화
analyzer = ResumeAnalyzer()

@app.get("/")
async def home(request: Request):
    """홈페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    """이력서 분석 API"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
    # 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # 이력서 분석
        result = analyzer.analyze_pdf(file_path)
        
        # 분석 완료 후 파일 삭제
        os.remove(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 이력서 분석 서버 시작!")
    print("📋 http://localhost:8000 으로 접속하세요")
    uvicorn.run(app, host="0.0.0.0", port=8000) 