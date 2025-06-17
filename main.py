from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model import analyze_resume
import uvicorn
import os
import json
from typing import Optional

app = FastAPI(
    title="이력서 PDF 평가 API",
    description="EasyOCR 기반 이력서 평가 시스템",
    version="2.0"
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/ocrspec")
async def evaluate_resume(
    file: UploadFile = File(...),
    job_type: str = Form(...),
    nickname: Optional[str] = Form(None)
):
    try:
        result = await analyze_resume(
            file=file,
            job_type=job_type,
            nickname=nickname
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/submit-spec")
async def submit_spec(
    nickname: str = Form(...),
    job_type: str = Form(...),
    education: str = Form(...),
    experience: Optional[str] = Form(None)
):
    try:
        # JSON 문자열을 파이썬 객체로 변환
        education_data = json.loads(education)
        experience_data = json.loads(experience) if experience else []
        
        # 여기에서 데이터베이스 저장 로직을 구현할 수 있습니다
        
        return JSONResponse(content={
            "status": "success",
            "message": "스펙 정보가 성공적으로 저장되었습니다.",
            "data": {
                "nickname": nickname,
                "job_type": job_type,
                "education": education_data,
                "experience": experience_data
            }
        })
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)