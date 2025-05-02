from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os, time, redis    
import uvicorn
from datetime import datetime
from kv_cache_implementation import SpecEvaluator
from batch_processing import BatchProcessor
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

# Redis 환경 변수 설정
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"

# 캐시 TTL 설정 (24시간)
CACHE_TTL = 86400

# 모델 초기화
evaluator = SpecEvaluator(
    use_redis=REDIS_ENABLED,
    redis_host=REDIS_HOST,
    redis_port=REDIS_PORT,
    redis_db=REDIS_DB,
    cache_ttl=CACHE_TTL
)

batch_processor = BatchProcessor(evaluator, batch_size=10, max_workers=4)

# Redis 클라이언트 초기화 (포트폴리오 텍스트 캐싱용)
redis_client = None
if REDIS_ENABLED:
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB + 1,  # 모델 캐시와 다른 DB 사용
            decode_responses=True
        )
        print("Redis 포트폴리오 캐시 연결 성공")
    except Exception as e:
        print(f"Redis 포트폴리오 캐시 연결 실패: {e}")
        redis_client = None

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

class SpecV1Respone(BaseModel):
    # 지원자 닉네임
    nickname: str 
    # 총점
    totalScore: int

class ErrorResponse(BaseModel):
    # 오류 메시지
    message: str = Field
# ──────────────────────────
# 2) 점수 계산 함수
# ──────────────────────────
@app.post(
        "/spec/v1/post", 
    response_model=SpecV1Respone,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["스펙 평가"]
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

# ──────────────────────────
# 3) 서버 실행
# ──────────────────────────
if __name__ == "__main__":
    # 개발 모드로 실행 (reload=True)
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)