from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os, time
import uvicorn
from datetime import datetime
from model import SpecEvaluator

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="Spec Score Test API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 초기화
evaluator = SpecEvaluator()

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

class SpecV1(BaseModel):
    nickname: str
    final_edu: str
    final_status: str
    desired_job: str
    universities: Optional[List[University]] = []
    careers: Optional[List[Career]] = []
    certificates: Optional[List[str]] = []
    languages: Optional[List[Language]] = []
    activities: Optional[List[Activity]] = []

class SpecV1Response(BaseModel):
    nickname: str 
    totalScore: float

class ErrorResponse(BaseModel):
    message: str

# Route for the test page
@app.get("/", response_class=HTMLResponse)
async def get_test_page(request: Request):
    with open("templates/spec_test.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# API endpoint for spec evaluation
@app.post(
    "/spec/v1/post",
    response_model=SpecV1Response,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def evaluate_spec_v1(spec_data: SpecV1):
    """
    V1 API: 사용자의 학력, 경력, 자격증 등의 스펙 정보를 받아 평가합니다.
    """
    try:
        # 요청 시간 기록
        start_time = time.time()
        
        # SpecEvaluator를 사용하여 평가
        result = evaluator.predict(spec_data.dict())
        
        # 응답 시간 계산 및 로깅
        elapsed_time = time.time() - start_time
        print(f"[V1] {spec_data.nickname}의 평가 완료, 소요 시간: {elapsed_time:.2f}초")
        
        return result
    except Exception as e:
        # 오류 로깅
        print(f"[V1] 평가 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서버에서 예기치 못한 오류가 발생했습니다: {str(e)}"
        )

# 서버 실행
if __name__ == "__main__":
    # 개발 모드로 실행 (reload=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)