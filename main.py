from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import os
from model import OCRModel
from pydantic import BaseModel

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
templates = Jinja2Templates(directory="templates")

# OCR 모델 초기화
ocr_model = OCRModel()

# Pydantic 모델
class S3URLRequest(BaseModel):
    filelink: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocrspec")
async def analyze_resume_from_url(request: S3URLRequest):
    """이력서 PDF 분석 (S3 URL)"""
    try:
        s3_url = request.filelink.strip()
        
        # URL 유효성 검사
        if not s3_url or not s3_url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="유효한 URL을 입력해주세요.")
        
        # PDF 파일인지 확인 (URL에서 파일 확장자 확인)
        if not s3_url.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일 URL만 처리 가능합니다.")
        
        # S3 URL에서 PDF 처리
        results = ocr_model.process_pdf_from_url(s3_url)
        
        if not results:
            raise HTTPException(status_code=400, detail="텍스트를 추출할 수 없습니다.")
        
        # 구조화된 데이터로 파싱
        structured_data = ocr_model.parse_resume(results)
        
        # 응답 형식에 맞게 변환
        response_data = {
            "nickname": "",  # 추후 구현
            "final_edu": "",
            "final_status": "",
            "desired_job": "",
            "universities": [],
            "careers": [],
            "certificates": [],
            "languages": [],
            "activities": []
        }
        
        # 대학교 정보 변환
        for uni in structured_data.get("universities", []):
            university_info = {
                "name": uni.get("name", ""),
                "degree": uni.get("degree", ""),
                "major": uni.get("major", ""),
                "status": uni.get("status", ""),
                "gpa": uni.get("gpa", 0.0),
                "gpa_max": uni.get("gpa_max", 4.5)
            }
            response_data["universities"].append(university_info)
        
        # 학위 우선순위 비교 개선
        degree_priority = ["박사", "석사", "학사", "전문학사", "고등학교"]
        highest_degree = None
        for uni in response_data["universities"]:
            deg = uni.get("degree", "")
            for d in degree_priority:
                if d in deg:
                    if (highest_degree is None) or (degree_priority.index(d) < degree_priority.index(highest_degree)):
                        highest_degree = d
        
        if highest_degree:
            response_data["final_edu"] = highest_degree
            response_data["final_status"] = "졸업"
        
        # 경력 정보 변환
        for career in structured_data.get("careers", []):
            career_info = {
                "company": career.get("company", ""),
                "role": career.get("role", ""),
                "work_month": career.get("work_month", 0)
            }
            response_data["careers"].append(career_info)
        
        # 자격증 정보 변환
        for cert in structured_data.get("certificates", []):
            if isinstance(cert, dict):
                response_data["certificates"].append(cert.get("name", ""))
            else:
                response_data["certificates"].append(str(cert))
        
        # 언어 정보 변환
        for lang in structured_data.get("languages", []):
            if isinstance(lang, dict):
                language_info = {
                    "test": lang.get("test", ""),
                    "score_or_grade": lang.get("score", "")
                }
                response_data["languages"].append(language_info)
            else:
                response_data["languages"].append({
                    "test": str(lang),
                    "score_or_grade": ""
                })
        
        # 활동 정보 변환
        for activity in structured_data.get("activities", []):
            if isinstance(activity, dict):
                activity_info = {
                    "name": activity.get("name", activity.get("description", "")),
                    "role": activity.get("role", ""),
                    "award": ""
                }
                response_data["activities"].append(activity_info)
            else:
                response_data["activities"].append({
                    "name": str(activity),
                    "role": "",
                    "award": ""
                })
        
        # 수상 정보를 활동에 추가
        for award in structured_data.get("awards", []):
            if isinstance(award, dict):
                award_info = {
                    "name": award.get("description", ""),
                    "role": "",
                    "award": "수상"
                }
                response_data["activities"].append(award_info)
            else:
                response_data["activities"].append({
                    "name": str(award),
                    "role": "",
                    "award": "수상"
                })
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"S3 URL에서 이력서 분석 중 오류 발생: {error_msg}")
        
        # 자격 증명 관련 에러인 경우 더 친화적인 메시지
        if "credentials" in error_msg.lower():
            raise HTTPException(
                status_code=500, 
                detail="S3 파일에 접근할 수 없습니다. 파일이 공개되어 있는지 확인하거나, AWS 자격 증명을 설정해주세요."
            )
        elif "404" in error_msg or "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다. URL을 확인해주세요.")
        else:
            raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {error_msg}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000) 