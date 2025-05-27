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
app = FastAPI(title="Spec Score API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 기반 모델 초기화
try:
    evaluator = SpecEvaluator()
    print("✅ RAG 기반 SpecEvaluator 초기화 성공")
except Exception as e:
    print(f"❌ SpecEvaluator 초기화 실패: {e}")
    evaluator = None

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
    with open("/content/19-Respec-AI/templates/spec_test.html", "r", encoding="utf-8") as f:
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
    V1 API: RAG 기반 사용자 스펙 평가
    
    - 벡터 유사도 검색으로 전공, 자격증, 활동의 직무 관련성 정확히 평가
    - LLM과 RAG 컨텍스트를 결합한 종합 평가
    - 실시간 동적 가중치 적용
    """
    
    # 평가기 상태 확인
    if evaluator is None:
        raise HTTPException(
            status_code=500,
            detail="평가 시스템이 초기화되지 않았습니다. 관리자에게 문의하세요."
        )
    
    try:
        # 요청 시간 기록
        start_time = time.time()
        
        # 입력 데이터 검증
        if not spec_data.nickname:
            raise HTTPException(status_code=400, detail="닉네임은 필수입니다.")
        if not spec_data.desired_job:
            raise HTTPException(status_code=400, detail="지원직종은 필수입니다.")
        
        print(f"🔍 RAG 평가 시작: {spec_data.nickname} ({spec_data.desired_job})")
        
        # RAG 기반 SpecEvaluator를 사용하여 평가
        result = evaluator.predict(spec_data.dict())
        
        # 응답 시간 계산 및 로깅
        elapsed_time = time.time() - start_time
        
        print(f"✅ RAG 평가 완료: {spec_data.nickname} -> {result.get('totalScore', 0):.2f}점 "
              f"(소요시간: {elapsed_time:.2f}초)")
        
        # 상세 정보 포함 여부 결정 (개발 모드에서만)
        include_details = os.getenv("INCLUDE_RAG_DETAILS", "false").lower() == "true"
        
        response = SpecV1Response(
            nickname=result["nickname"],
            totalScore=result["totalScore"],
            ragDetails=result.get("rag_details") if include_details else None
        )
        
        return response
        
    except HTTPException:
        # HTTP 예외는 그대로 재발생
        raise
    except Exception as e:
        # 기타 예외 처리
        error_msg = f"RAG 평가 중 예기치 못한 오류: {str(e)}"
        print(f"❌ {error_msg}")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )