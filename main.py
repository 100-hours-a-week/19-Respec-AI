from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import uvicorn
from resume_evaluation_system import ResumeEvaluationSystem

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="Spec Score API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 평가 시스템 초기화
evaluation_system = ResumeEvaluationSystem()

# ──────────────────────────
# 1) Pydantic 모델 정의
# ──────────────────────────
class University(BaseModel):
    school_name: str
    degree: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[float] = None
    gpa_max: Optional[float] = None

class Career(BaseModel):
    company: str
    role: Optional[str] = None
    work_month: Optional[int] = None

class Language(BaseModel):
    test: str
    score_or_grade: str

class Activity(BaseModel):
    name: str
    role: Optional[str] = None
    award: Optional[str] = None

class ResumeData(BaseModel):
    nickname: str
    final_edu: str
    final_status: str
    desired_job: str
    universities: Optional[List[Dict]] = []
    careers: Optional[List[Dict]] = []
    certificates: Optional[List[str]] = []
    languages: Optional[List[Dict]] = []
    activities: Optional[List[Dict]] = []

class ErrorResponse(BaseModel):
    message: str

# Route for the test page
@app.get("/", response_class=HTMLResponse)
async def get_test_page(request: Request):
    with open("/content/19-Respec-AI/templates/spec_test.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/spec/v1/post")
async def evaluate_resume(resume_data: ResumeData):
    """이력서 평가 엔드포인트"""
    try:
        # 입력 데이터 검증
        if not resume_data.nickname:
            raise HTTPException(status_code=400, detail="닉네임은 필수입니다.")
        if not resume_data.desired_job:
            raise HTTPException(status_code=400, detail="지원직종은 필수입니다.")
            
        print(f"🔍 평가 시작: {resume_data.nickname} ({resume_data.desired_job})")
        
        result = evaluation_system.evaluate_resume(resume_data.dict())
        
        print(f"✅ 평가 완료: {resume_data.nickname} -> {result['totalScore']:.2f}점")
        
        return result
    except Exception as e:
        error_msg = f"평가 중 오류 발생: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/status")
async def get_system_status():
    """시스템 상태 확인 엔드포인트"""
    try:
        return evaluation_system.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)