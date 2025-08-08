# AI 기반 이력서 평가 시스템 (Respec-AI)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)

> **RAG(Retrieval-Augmented Generation) 기술과 파인튜닝된 LLM을 활용한 지능형 이력서 평가 API**
Respec-AI는 기업의 채용 담당자를 위한 AI 기반 이력서 자동 평가 시스템입니다. 
- **12개 직종별** 맞춤형 평가 기준 적용
- **벡터 데이터베이스** 기반 유사도 검색으로 정확한 스펙 매칭
- **파인튜닝된 Qwen2 모델**을 통한 전문적인 평가 제공

## 주요 기능

### 지능형 평가 시스템
- **RAG 기반 평가**: 전공, 대학, 자격증, 경력의 직무 연관성 벡터 검색
- **다단계 점수 계산**: 직무별 가중치 적용한 정밀 평가
- **실시간 언어 검증**: 15개 어학시험 점수 자동 검증 및 표준화
### 지원 직종 (12개)
경영·사무, 마케팅·광고·홍보, 무역·유통, 인터넷·IT, 생산·제조, 영업·고객상담, 건설, 금융, 연구개발·설계, 디자인, 미디어, 전문·특수직

## 기술 스택

### AI/ML
- **LLM**: Qwen2-1.5B (파인튜닝 버전)
- **임베딩**: ko-sroberta-multitask (한국어 특화)
- **벡터 DB**: PostgreSQL + pgvector

### Backend
- **API**: FastAPI + Uvicorn
- **DB**: PostgreSQL (가중치, 평가기준 저장)

## 설치 및 실행

### 1. 시스템 요구사항
```bash
# OS: Ubuntu 20.04+ 권장
# Python: 3.8+
# CUDA: 12.2 (GPU 필수)
# RAM: 8GB+ 권장
# GPU: VRAM 4GB+ 권장
# PostgreSQL: 12+ (pgvector 확장 필요)
```

### 2. 환경 설정
```bash
# 시스템 패키지 설치
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

### 3. 프로젝트 설치
```bash
# 프로젝트 클론
git clone https://github.com/100-hours-a-week/19-Respec-AI.git
cd 19-Respec-AI

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip

# PyTorch CUDA 12.1 버전 설치 (CUDA 12.2 호환)
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 나머지 패키지 설치
pip install -r requirements.txt --no-cache-dir
```

### 4. 환경변수 설정
```bash
# .env 파일 생성 및 편집
cp .env.example .env
nano .env
```

**.env 설정 예시:**
```env
# PostgreSQL 데이터베이스 설정
HOST=localhost
DATABASE=respec_ai_db
USER=your_username
PASSWORD=your_password

# AI 모델 설정
MODEL_NAME=Qwen/Qwen2-1.5B-Instruct
PEFT_MODEL_PATH=syahaeun/qwen2-resume-evaluator
```

### 5. 데이터베이스 설정
```bash
# PostgreSQL 설치 및 설정
sudo apt install -y postgresql postgresql-contrib

# pgvector 확장 설치
sudo apt install -y postgresql-14-pgvector

# 데이터베이스 생성 (별도 스크립트 필요)
# psql -U postgres -c "CREATE DATABASE respec_ai_db;"
```
## 실행 방법

### 개발 환경 실행
```bash
# FastAPI 서버 실행
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 프로덕션 환경 실행
```bash
# FastAPI 서버 실행 (워커 프로세스 4개)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API 사용 방법

### 기본 접근
- **Web UI**: http://localhost:8000 (테스트 페이지)
- **API 문서**: http://localhost:8000/docs (Swagger UI)
- **상세 API 문서**: [API 문서](https://github.com/100-hours-a-week/19-Respec-WIKI/wiki/AI-API) 참조
