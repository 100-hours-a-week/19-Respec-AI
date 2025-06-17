from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
import tempfile
import os
import logging
from typing import Optional, Dict, Any
from app.models.text_extractor import ResumeTextExtractor
from app.models.spec_extractor import ResumeSpecExtractor

# 로깅 설정
logger = logging.getLogger(__name__)

router = APIRouter()

class ResumeRequest(BaseModel):
    pdf_url: HttpUrl

class ResumeResponse(BaseModel):
    status: str
    result: Dict[str, Any]

@router.post("/ocrspec", response_model=ResumeResponse)
async def analyze_resume_from_url(request: ResumeRequest):
    """
    URL로부터 PDF 이력서를 분석하는 엔드포인트
    """
    temp_file = None
    try:
        # PDF 다운로드
        response = requests.get(str(request.pdf_url))
        response.raise_for_status()

        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()

        # 모델 초기화 (모델 경로 설정)
        model_path = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        os.makedirs(model_path, exist_ok=True)

        # 텍스트 추출
        text_extractor = ResumeTextExtractor(model_path)
        spec_extractor = ResumeSpecExtractor()

        # PDF 분석
        result = text_extractor.extract_text_from_pdf(temp_file.name)
        if not result:
            raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        # 스펙 추출
        specs = spec_extractor.extract_specs(result)
        
        # 응답 형식화
        formatted_result = {
            "학교": {
                "학교명": specs.get("education", [{}])[0].get("school"),
                "전공": specs.get("education", [{}])[0].get("major"),
                "학위": specs.get("education", [{}])[0].get("degree"),
                "학점": specs.get("education", [{}])[0].get("gpa")
            },
            "경력": {
                "회사명": specs.get("experience", [{}])[0].get("company"),
                "직무": specs.get("experience", [{}])[0].get("position"),
                "기간": specs.get("experience", [{}])[0].get("duration"),
                "부서": specs.get("experience", [{}])[0].get("department")
            },
            "활동": specs.get("certificates", []),
            "자격증": specs.get("languages", [])
        }

        return {
            "status": "success",
            "result": formatted_result
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"PDF 다운로드 실패: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF 다운로드 실패: {str(e)}")
    except Exception as e:
        logger.error(f"이력서 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이력서 분석 중 오류 발생: {str(e)}")
    finally:
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.get("/health")
async def health_check():
    """
    서비스 상태 확인 엔드포인트
    """
    try:
        model_path = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        return {
            "status": "healthy",
            "model_path": model_path,
            "model_exists": os.path.exists(model_path)
        }
    except Exception as e:
        logger.error(f"상태 확인 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 