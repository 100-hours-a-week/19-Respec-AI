# main.py - 이력서 PDF 분석 웹 애플리케이션
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os, time
import uvicorn
from datetime import datetime
import logging
from model import ResumeAnalysisModel

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Resume Analyzer V2 API",
    description="EasyOCR 기반 이력서 분석 및 스펙 평가 시스템",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 초기화
model = ResumeAnalysisModel()

# ──────────────────────────
# Pydantic 모델 정의
# ──────────────────────────
class University(BaseModel):
    school_name: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[float] = None
    gpa_max: Optional[float] = None

class Career(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    work_month: Optional[int] = None

class Language(BaseModel):
    test: Optional[str] = None
    score_or_grade: Optional[str] = None

class Activity(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    award: Optional[str] = None

class SpecDataV2(BaseModel):
    nickname: str
    final_edu: str
    final_status: str
    desired_job: str
    universities: Optional[List[University]] = []
    careers: Optional[List[Career]] = []
    certificates: Optional[List[str]] = []
    languages: Optional[List[Language]] = []
    activities: Optional[List[Activity]] = []
    filelink: Optional[str] = None

class SpecV2Response(BaseModel):
    nickname: str
    academicScore: float
    workExperienceScore: float
    certificationScore: float
    languageProficiencyScore: float
    extracurricularScore: float
    totalScore: float

class PortfolioRequest(BaseModel):
    nickname: str
    filelink: str

class PortfolioResponse(BaseModel):
    nickname: str
    detail: str

class ErrorResponse(BaseModel):
    message: str

class AnalysisResponse(BaseModel):
    extracted_text: str
    total_score: float
    grade: str
    education_score: float
    experience_score: float
    skills_score: float
    certifications_score: float
    languages_score: float
    education_level: str
    experience_years: int
    skills: List[str]
    certifications: List[str]
    languages: List[str]

# ──────────────────────────
# 유틸리티 함수들
# ──────────────────────────
def download_pdf_from_url(url: str) -> bytes:
    """URL에서 PDF 다운로드"""
    import requests
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"PDF 다운로드 실패: {e}")
        raise HTTPException(status_code=400, detail=f"PDF 다운로드 실패: {str(e)}")

def process_pdf_to_image(pdf_content: bytes):
    """PDF를 이미지로 변환"""
    import fitz  # PyMuPDF
    import numpy as np
    
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        page = doc[0]  # 첫 번째 페이지만
        
        # 고해상도로 렌더링
        mat = fitz.Matrix(3.0, 3.0)  # 3배 확대
        pix = page.get_pixmap(matrix=mat)
        
        # numpy 배열로 변환
        img_data = pix.tobytes("ppm")
        nparr = np.frombuffer(img_data, np.uint8)
        
        import cv2
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        return img
        
    except Exception as e:
        logger.error(f"PDF 처리 오류: {e}")
        raise HTTPException(status_code=400, detail=f"PDF 처리 실패: {str(e)}")

# ──────────────────────────
# API 엔드포인트들
# ──────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request):
    """메인 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume Analyzer V2</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #fff; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .post { background: #49cc90; }
            .get { background: #61affe; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Resume Analyzer V2 API</h1>
            <p>EasyOCR 기반 이력서 분석 및 스펙 평가 시스템</p>
            
            <h2>📋 Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/spec/v2/post</strong> - V2 스펙 분석 (포트폴리오 포함)
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/spec/portfolio</strong> - 포트폴리오 텍스트 추출
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/analyze</strong> - 이력서 파일 분석
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/clean_text</strong> - 개인정보 제거 + 텍스트 정리
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong> - 서버 상태 확인
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/performance</strong> - 성능 통계
            </div>
            
            <h2>🚀 Features</h2>
            <ul>
                <li>✅ 생년월일 및 개인정보 완전 제거</li>
                <li>✅ 짧고 핵심적인 텍스트 요약</li>
                <li>✅ 6개 카테고리 스펙 분석 (학력, 경력, 기술, 자격증, 어학, 활동)</li>
                <li>✅ PDF 포트폴리오 처리</li>
                <li>✅ GPU 가속 OCR</li>
                <li>✅ 성능 모니터링</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post(
    "/spec/v2/post",
    response_model=SpecV2Response,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def analyze_spec_v2(spec_data: SpecDataV2):
    """V2 API: 스펙 데이터 분석 (포트폴리오 포함)"""
    try:
        start_time = time.time()
        logger.info(f"V2 분석 시작: {spec_data.nickname}")
        
        # 포트폴리오 처리
        portfolio_text = ""
        if spec_data.filelink:
            try:
                pdf_content = download_pdf_from_url(spec_data.filelink)
                img = process_pdf_to_image(pdf_content)
                portfolio_text = model.extract_text(img)
                logger.info(f"포트폴리오 텍스트 추출 완료: {len(portfolio_text)} 문자")
            except Exception as e:
                logger.error(f"포트폴리오 처리 오류: {e}")
                # 포트폴리오 오류는 전체 분석을 중단하지 않음
        
        # 스펙 분석
        analysis_result = model.analyze_spec_data_v2(
            spec_data=spec_data.dict(),
            portfolio_text=portfolio_text
        )
        
        # V2 응답 형식
        response = SpecV2Response(
            nickname=spec_data.nickname,
            academicScore=round(analysis_result.get("education_score", 0), 2),
            workExperienceScore=round(analysis_result.get("experience_score", 0), 2),
            certificationScore=round(analysis_result.get("certification_score", 0), 2),
            languageProficiencyScore=round(analysis_result.get("language_score", 0), 2),
            extracurricularScore=round(analysis_result.get("activity_score", 0), 2),
            totalScore=round(analysis_result.get("total_score", 0), 2)
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"V2 분석 완료: {spec_data.nickname}, 총점: {response.totalScore}, 소요시간: {elapsed_time:.2f}초")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 분석 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서버에서 예기치 못한 오류가 발생했습니다: {str(e)}"
        )

@app.post(
    "/spec/portfolio",
    response_model=PortfolioResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def extract_portfolio_text(portfolio_request: PortfolioRequest):
    """포트폴리오 PDF에서 텍스트 추출"""
    try:
        start_time = time.time()
        logger.info(f"포트폴리오 텍스트 추출: {portfolio_request.nickname}")
        
        # PDF 처리
        pdf_content = download_pdf_from_url(portfolio_request.filelink)
        img = process_pdf_to_image(pdf_content)
        extracted_text = model.extract_text(img)
        
        elapsed_time = time.time() - start_time
        logger.info(f"포트폴리오 추출 완료: {len(extracted_text)} 문자, 소요시간: {elapsed_time:.2f}초")
        
        return PortfolioResponse(
            nickname=portfolio_request.nickname,
            detail=extracted_text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"포트폴리오 추출 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"포트폴리오 처리 실패: {str(e)}"
        )

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def analyze_resume(file: UploadFile = File(...)):
    """이력서 파일 분석"""
    try:
        start_time = time.time()
        logger.info(f"파일 분석 시작: {file.filename}")
        
        # 파일 읽기
        content = await file.read()
        
        # PDF 또는 이미지 처리
        if file.filename.lower().endswith('.pdf'):
            img = process_pdf_to_image(content)
        else:
            import cv2
            import numpy as np
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다.")
        
        # 분석 수행
        result = model.analyze_resume(img)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # 응답 형식 변환
        response = AnalysisResponse(
            extracted_text=result.get("extracted_text", ""),
            total_score=result.get("total_score", 0),
            grade=result.get("grade", "F"),
            education_score=result.get("education", {}).get("score", 0),
            experience_score=result.get("experience", {}).get("score", 0),
            skills_score=result.get("skills", {}).get("score", 0),
            certifications_score=result.get("certificates", {}).get("score", 0),
            languages_score=result.get("languages", {}).get("score", 0),
            education_level=result.get("education", {}).get("level", "Unknown"),
            experience_years=result.get("experience", {}).get("years", 0),
            skills=result.get("skills", {}).get("programming_languages", []) + 
                   result.get("skills", {}).get("frameworks", []),
            certifications=result.get("certificates", {}).get("certificates", []),
            languages=result.get("languages", {}).get("languages", [])
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"파일 분석 완료: 총점 {response.total_score}, 소요시간: {elapsed_time:.2f}초")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 분석 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 분석 실패: {str(e)}"
        )

@app.post("/clean_text")
async def clean_text_only(file: UploadFile = File(...)):
    """개인정보 제거 + 텍스트 정리"""
    try:
        start_time = time.time()
        logger.info(f"텍스트 정리: {file.filename}")
        
        # 파일 처리 (analyze와 동일)
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            img = process_pdf_to_image(content)
        else:
            import cv2
            import numpy as np
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 텍스트 추출 및 정리
        extracted_text = model.extract_text(img)
        
        elapsed_time = time.time() - start_time
        logger.info(f"텍스트 정리 완료: {len(extracted_text)} 문자, 소요시간: {elapsed_time:.2f}초")
        
        return {
            "success": True,
            "message": "텍스트 정리 완료",
            "result": {
                "cleaned_text": extracted_text,
                "length": len(extracted_text),
                "processing_time": f"{elapsed_time:.2f}초"
            }
        }
        
    except Exception as e:
        logger.error(f"텍스트 정리 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"텍스트 정리 실패: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "model": "EasyOCR Resume Analyzer V2",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/performance")
async def get_performance_stats():
    """성능 통계"""
    try:
        stats = model.get_performance_stats()
        return {
            "status": "success",
            "performance": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"성능 통계 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="성능 통계 조회 실패")

# 서버 실행
if __name__ == "__main__":
    # 개발 모드로 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 