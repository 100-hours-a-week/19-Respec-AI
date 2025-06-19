from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
import time
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import requests
from model import OCRModel
from pydantic import BaseModel, validator

# 로깅 설정 개선
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 필요한 디렉토리 생성
REQUIRED_DIRS = ['static', 'templates', 'temp']
for dir_name in REQUIRED_DIRS:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f"Created directory: {dir_name}")

app = FastAPI(
    title="이력서 분석 API",
    description="URL 기반 이력서 PDF 분석 및 구조화 서비스",
    version="2.0.0"
)

# 정적 파일 및 템플릿 폴더 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OCR 모델 초기화 (싱글톤 패턴)
ocr_model = None

def get_ocr_model():
    global ocr_model
    if ocr_model is None:
        ocr_model = OCRModel()
    return ocr_model

# Pydantic 모델 개선
class S3URLRequest(BaseModel):
    filelink: str = ""
    
    @validator('filelink')
    def validate_filelink(cls, v):
        if not v or not v.strip():
            raise ValueError("URL이 비어있습니다.")
        
        url = v.strip()
        
        # URL 형식 검증
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("유효한 URL 형식이 아닙니다.")
        except Exception:
            raise ValueError("URL 형식이 올바르지 않습니다.")
        
        # 파일 확장자 검증
        if not url.lower().endswith('.pdf'):
            raise ValueError("PDF 파일만 지원됩니다.")
        
        return url

# 응답 모델 정의
class ResumeAnalysisResponse(BaseModel):
    success: bool = True
    message: str = "분석이 완료되었습니다."
    processing_time: float = 0.0
    data: Dict[str, Any] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocrspec")
async def analyze_resume_from_url(request: S3URLRequest):
    """이력서 PDF 분석 (URL 기반) - 개선된 버전"""
    start_time = time.time()
    
    try:
        logger.info(f"=== 이력서 분석 시작 ===")
        logger.info(f"요청 URL: {request.filelink}")
        
        s3_url = request.filelink.strip()
        
        # URL 접근성 검증
        if not await validate_url_accessibility(s3_url):
            raise HTTPException(
                status_code=400, 
                detail="URL에 접근할 수 없습니다. 파일이 공개되어 있는지 확인해주세요."
            )
        
        # OCR 모델 가져오기
        ocr_model = get_ocr_model()
        
        # PDF 처리 및 텍스트 추출
        logger.info("PDF 텍스트 추출 시작...")
        results = ocr_model.process_pdf_from_url(s3_url)
        
        if not results:
            raise HTTPException(
                status_code=400, 
                detail="PDF에서 텍스트를 추출할 수 없습니다. 파일이 손상되었거나 이미지가 아닌 텍스트가 없을 수 있습니다."
            )
        
        logger.info(f"추출된 텍스트 라인 수: {len(results)}")
        
        # 구조화된 데이터로 파싱
        logger.info("구조화된 데이터 파싱 시작...")
        structured_data = ocr_model.parse_resume(results)
        
        # 응답 데이터 구성
        response_data = await build_response_data(structured_data)
        
        logger.info(f"분석 완료 (소요시간: {time.time() - start_time:.2f}초)")
        
        return response_data  # 래핑 없이 바로 반환
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"이력서 분석 중 오류 발생: {error_msg}")
        if "credentials" in error_msg.lower() or "access denied" in error_msg.lower():
            detail = "파일에 접근할 수 없습니다. 파일이 공개되어 있는지 확인해주세요."
        elif "404" in error_msg or "not found" in error_msg.lower():
            detail = "파일을 찾을 수 없습니다. URL을 확인해주세요."
        elif "timeout" in error_msg.lower():
            detail = "요청 시간이 초과되었습니다. 다시 시도해주세요."
        elif "connection" in error_msg.lower():
            detail = "네트워크 연결에 실패했습니다. 인터넷 연결을 확인해주세요."
        else:
            detail = f"처리 중 오류가 발생했습니다: {error_msg}"
        raise HTTPException(status_code=500, detail=detail)

async def validate_url_accessibility(url: str) -> bool:
    """URL 접근성 검증"""
    try:
        # HEAD 요청으로 파일 존재 여부 확인
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"URL 접근성 검증 실패: {e}")
        return True  # 검증 실패 시에도 진행 허용

async def build_response_data(structured_data: Dict) -> Dict[str, Any]:
    """응답 데이터 구성"""
    # 최종학력 정보를 universities에서 추출
    final_edu = "고등학교"  # 기본값
    final_status = "졸업"   # 기본값
    
    universities = structured_data.get("universities", [])
    if universities:
        latest_uni = universities[0]  # 가장 최근 학력
        degree = latest_uni.get("degree", "")
        if "대학교" in latest_uni.get("school_name", ""):
            final_edu = "대학교"
        elif "대학원" in latest_uni.get("school_name", ""):
            final_edu = "대학원"
        final_status = latest_uni.get("status", "졸업")
    
    response_data = {
        "final_edu": final_edu,
        "final_status": final_status,
        "desired_job": structured_data.get("desired_job", ""),
        "universities": [],
        "careers": [],
        "certificates": [],
        "languages": [],
        "activities": []
    }
    
    # 대학교 정보 변환 - status 필드 제거
    for university in structured_data.get("universities", []):
        university_info = {
            "school_name": university.get("school_name", ""),
            "degree": university.get("degree", ""),
            "major": university.get("major", ""),
            "gpa": university.get("gpa", 0.0),
            "gpa_max": university.get("gpa_max", 4.5)
        }
        response_data["universities"].append(university_info)
    
    # 경력 정보 변환 - company, role, work_month만 포함
    for career in structured_data.get("careers", []):
        career_info = {
            "company": career.get("company", ""),
            "role": career.get("role", ""),
            "work_month": career.get("work_month", 0)
        }
        response_data["careers"].append(career_info)
    
    # 자격증 정보 변환 - 제목만 추출
    for certificate in structured_data.get("certificates", []):
        cert_name = re.sub(r'\s*\([^)]*\)', '', certificate).strip()
        if cert_name:
            response_data["certificates"].append(cert_name)
    
    # 어학 정보 변환 - test, score_or_grade만 포함
    for language in structured_data.get("languages", []):
        language_info = {}
        
        # test가 빈 문자열이 아니면 추가
        test = language.get("test", "")
        if test:
            language_info["test"] = test
        
        # score_or_grade가 빈 문자열이 아니면 추가
        score_or_grade = language.get("score_or_grade", "")
        if score_or_grade:
            language_info["score_or_grade"] = score_or_grade
        
        # 최소한 하나의 필드라도 있으면 추가
        if language_info:
            response_data["languages"].append(language_info)
    
    # 활동 정보 변환 - name, role, award가 없으면 빈 문자열로 포함
    unique_activities = []  # 중복 제거를 위한 리스트
    for activity in structured_data.get("activities", []):
        activity_name = activity.get("name", "")
        activity_name = re.sub(r'\s*\([^)]*\)', '', activity_name).strip()
        # 중복 확인
        if not activity_name or activity_name in unique_activities:
            continue
        unique_activities.append(activity_name)
        activity_info = {
            "name": activity_name if activity_name else ""
        }
        # role, award가 없으면 빈 문자열로 포함
        role = activity.get("role", "")
        award = activity.get("award", "")
        activity_info["role"] = role.strip() if role else ""
        activity_info["award"] = award.strip() if award else ""
        response_data["activities"].append(activity_info)
    
    # 최종 응답
    return response_data

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 에러 핸들러 개선"""
    logger.error(f"Validation error: {exc}")
    
    # 요청 본문 로깅
    try:
        body = await request.body()
        logger.error(f"Request body: {body.decode()}")
    except:
        logger.error("Request body could not be read")
    
    # 더 친화적인 에러 메시지
    error_details = []
    for error in exc.errors():
        field = error.get("loc", ["unknown"])[-1]
        message = error.get("msg", "Validation error")
        error_details.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "입력 데이터가 올바르지 않습니다.",
            "details": error_details,
            "processing_time": 0.0,
            "data": {}
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "서버 내부 오류가 발생했습니다.",
            "processing_time": 0.0,
            "data": {}
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    ) 