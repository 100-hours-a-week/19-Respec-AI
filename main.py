from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="Spec Score API")

# ──────────────────────────
# 1) Pydantic 모델 정의
# ──────────────────────────
class University(BaseModel):
    # 학교 이름
    school_name: str
    # 학위 (Optional)
    degree: Optional[str]
    # 전공 (Optional)
    major: Optional[str]
    # 평점 (Optional)
    gpa: Optional[float]
    # 평점 최대값 (Optional)
    gpa_max: Optional[float]

class Career(BaseModel):
    # 회사 이름
    company: str
    # 직책/역할 (Optional)
    role: Optional[str]

class Language(BaseModel):
    # 시험 종류 (예: TOEIC, TOEFL 등)
    test: str
    # 점수 또는 등급
    score_or_grade: str

class Activity(BaseModel):
    # 활동 이름
    name: str
    # 역할 (Optional)
    role: Optional[str]
    # 수상 내역 (Optional)
    award: Optional[str]

class SpecV1(BaseModel):
    # 지원자 닉네임
    nickname: str
    # 최종 학력
    final_edu: str
    # 학력 상태 (예: 졸업, 재학 등)
    final_status: str
    # 지원 직종
    desired_job: str
    # 대학 정보 리스트
    universities: Optional[List[University]]  = []
    # 경력 정보 리스트
    careers:     Optional[List[Career]]      = []
    # 자격증 리스트
    certificates: Optional[List[str]]        = []
    # 어학 정보 리스트
    languages:   Optional[List[Language]]    = []
    # 활동 정보 리스트
    activities:  Optional[List[Activity]]    = []

# ──────────────────────────
# 2) 점수 계산 함수
# ──────────────────────────
def score_education(final_edu: str) -> float:
    """
    최종 학력에 따라 고정 점수 반환
    - 고등학교: 20
    - 전문학사: 40
    - 학사: 60
    - 석사: 80
    - 박사: 100
    - 기타: 0
    """
    mapping = {"고등학교":20, "전문학사":40, "학사":60, "석사":80, "박사":100}
    return mapping.get(final_edu, 0.0)

def score_universities(unis: List[University]) -> float:
    """
    대학 목록의 GPA를 백분율로 평균낸 점수 반환
    유효한 GPA/gpa_max만 계산에 포함
    """
    valid = [u for u in unis if u.gpa and u.gpa_max and u.gpa_max > 0]
    if not valid:
        return 0.0
    return sum((u.gpa / u.gpa_max) * 100 for u in valid) / len(valid)

def score_careers(careers: List[Career]) -> float:
    """
    경력 개수당 20점씩 부여, 최대 100점
    """
    return min(len(careers) * 20.0, 100.0)

def score_certificates(certs: List[str]) -> float:
    """
    자격증 개수당 10점씩 부여, 최대 100점
    """
    return min(len(certs) * 10.0, 100.0)

def score_languages(langs: List[Language]) -> float:
    """
    각 언어 시험별 최고점으로 정규화하여 평균 점수 계산
    - TOEIC (0~990), TOEFL (0~120), IELTS (0~9), OPIC 등급 매핑
    """
    total, cnt = 0.0, 0
    for lang in langs:
        t = lang.test.upper()
        try:
            s = float(lang.score_or_grade)
        except:
            continue
        if t == "TOEIC" and 0 <= s <= 990:
            total += (s / 990.0) * 100.0; cnt += 1
        elif t == "TOEFL" and 0 <= s <= 120:
            total += (s / 120.0) * 100.0; cnt += 1
        elif t == "IELTS" and 0 <= s <= 9:
            total += (s / 9.0) * 100.0; cnt += 1
        elif t == "OPIC":
            grades = {
                "NL":10, "NM":20, "NH":30,
                "IL":40, "IM":50, "IH":60,
                "AL":80, "AM":90, "AH":100
            }
            sc = grades.get(lang.score_or_grade.upper(), 0)
            if sc > 0:
                total += sc; cnt += 1
    return (total / cnt) if cnt else 0.0

def score_activities(acts: List[Activity]) -> float:
    """
    활동 당 기본 10점, 수상 시 추가 10점, 최대 100점
    """
    total = 0.0
    for a in acts:
        total += 10.0 + (10.0 if a.award else 0.0)
    return min(total, 100.0)

# ──────────────────────────
# 3) 카테고리별 가중치
# ──────────────────────────
WEIGHTS = {
    "education":    0.20,  # 학력
    "university":   0.20,  # 대학 GPA
    "careers":      0.15,  # 경력
    "certificates": 0.10,  # 자격증
    "languages":    0.20,  # 어학
    "activities":   0.15,  # 활동
}

# ──────────────────────────
# 4) 루트 및 헬스체크 엔드포인트
# ──────────────────────────
@app.get("/")
def root():
    # 서비스 가동 확인용
    return {"message": "Spec Score API is up and running"}

@app.get("/health")
def health():
    # 헬스체크(상태 확인)
    return {"status": "ok"}

# ──────────────────────────
# 5) 스펙 점수 계산 엔드포인트
# ──────────────────────────
@app.post("/spec/v1/post")
def calculate_spec(spec: SpecV1):
    """
    클라이언트로부터 받은 SpecV1 객체를 기반으로
    각 카테고리 점수를 계산하여 가중합 후, 20~100 사이로 클램프하고 반환
    """
    try:
        # 카테고리별 점수 산출
        edu  = score_education(spec.final_edu)
        uni  = score_universities(spec.universities)
        car  = score_careers(spec.careers)
        cert = score_certificates(spec.certificates)
        lang = score_languages(spec.languages)
        act  = score_activities(spec.activities)

        # 가중치 합산
        total = (
            edu  * WEIGHTS["education"] +
            uni  * WEIGHTS["university"] +
            car  * WEIGHTS["careers"] +
            cert * WEIGHTS["certificates"] +
            lang * WEIGHTS["languages"] +
            act  * WEIGHTS["activities"]
        )
        # 최소 20점, 최대 100점 클램프 후 소수점 둘째 자리까지
        total = round(min(max(total, 20.0), 100.0), 2)

        # 결과 반환
        return {"nickname": spec.nickname, "totalScore": total}

    except Exception as e:
        # 내부 오류 발생 시 500 리턴
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────
# 6) 서버 실행
# ──────────────────────────
if __name__ == "__main__":
    import uvicorn
    # 개발 모드로 실행 (reload=True)
    uvicorn.run("spec_api:app", host="0.0.0.0", port=8000, reload=True)
