import easyocr
import numpy as np
from PIL import Image
import logging
from typing import List, Tuple, Dict, Optional
import io
import PyPDF2
import fitz  # PyMuPDF
import re
from datetime import datetime
import torch
import os
import requests
import boto3
from urllib.parse import urlparse
import tempfile
from io import BytesIO

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRModel:
    def __init__(self):
        """EasyOCR 모델 초기화"""
        logger.info("EasyOCR 모델 초기화 중...")
        
        # GPU 사용 가능 여부 확인
        if torch.backends.mps.is_available():
            device = 'mps'
            logger.info("MPS (Apple Silicon GPU) 사용")
            # EasyOCR이 MPS를 인식하도록 환경 변수 설정
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        elif torch.cuda.is_available():
            device = 'cuda'
            logger.info("CUDA GPU 사용")
        else:
            device = 'cpu'
            logger.info("CPU 사용")
        
        # EasyOCR 초기화 시 GPU 사용 설정
        gpu_available = device != 'cpu'
        self.reader = easyocr.Reader(['ko', 'en'], gpu=gpu_available)
        logger.info(f"EasyOCR 모델 초기화 완료 (GPU: {gpu_available})")

    def download_from_s3_url(self, s3_url: str) -> bytes:
        """S3 URL에서 파일 다운로드 - 개선된 버전"""
        logger.info(f"URL에서 파일 다운로드 중: {s3_url}")
        
        try:
            # 다양한 URL 형식 지원
            if s3_url.startswith('s3://'):
                # S3 URL을 HTTP URL로 변환 시도
                s3_url = s3_url.replace('s3://', 'https://')
                logger.info("S3 URL을 HTTP URL로 변환 시도...")
            
            # HTTP 요청으로 파일 다운로드
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(s3_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # 파일 크기 확인
            content_length = response.headers.get('content-length')
            if content_length:
                file_size = int(content_length)
                if file_size > 50 * 1024 * 1024:  # 50MB 제한
                    raise ValueError("파일이 너무 큽니다. 50MB 이하의 파일만 지원됩니다.")
            
            # 파일 다운로드
            pdf_data = response.content
            
            if not pdf_data:
                raise ValueError("빈 파일입니다.")
            
            logger.info(f"S3 URL에서 파일 다운로드 완료: {len(pdf_data)} bytes")
            return pdf_data
            
        except requests.exceptions.Timeout:
            logger.error("파일 다운로드 시간 초과")
            raise ValueError("파일 다운로드 시간이 초과되었습니다.")
        except requests.exceptions.ConnectionError:
            logger.error("네트워크 연결 오류")
            raise ValueError("네트워크 연결에 실패했습니다.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP 오류: {e}")
            if e.response.status_code == 404:
                raise ValueError("파일을 찾을 수 없습니다.")
            elif e.response.status_code == 403:
                raise ValueError("파일에 접근할 수 없습니다. 파일이 공개되어 있는지 확인해주세요.")
            else:
                raise ValueError(f"파일 다운로드 중 오류가 발생했습니다: {e}")
        except Exception as e:
            logger.error(f"파일 다운로드 중 예상치 못한 오류: {e}")
            raise ValueError(f"파일 다운로드 중 오류가 발생했습니다: {e}")

    def process_pdf_from_url(self, s3_url: str) -> List[str]:
        """URL에서 PDF 처리 - 개선된 버전"""
        logger.info(f"URL에서 PDF 처리 시작: {s3_url}")
        
        try:
            # 파일 다운로드
            pdf_data = self.download_from_s3_url(s3_url)
            
            # PDF 처리
            logger.info("PDF 텍스트 추출 시작...")
            results = self.process_pdf(pdf_data)
            
            if not results:
                logger.warning("PDF에서 텍스트를 추출할 수 없습니다.")
                return []
            
            logger.info(f"PDF 텍스트 추출 완료: {len(results)} 라인")
            return results
            
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {e}")
            raise ValueError(f"PDF 처리 중 오류가 발생했습니다: {e}")

    def process_pdf(self, pdf_data: bytes) -> List[str]:
        """PDF 파일 처리 및 텍스트 추출 - PDF 텍스트 직접 추출 방식"""
        try:
            # PDF 파일 열기
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            results = []
            
            logger.info("=== PDF 텍스트 직접 추출 시작 ===")
            
            # 각 페이지에서 텍스트 직접 추출
            for page_num in range(len(pdf_document)):
                logger.info(f"페이지 {page_num + 1} 텍스트 추출 중...")
                page = pdf_document[page_num]
                pdf_text = page.get_text()
                
                if pdf_text.strip():
                    # 텍스트를 라인별로 분리
                    lines = pdf_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 2:
                            # 개인정보가 아닌 경우만 추가
                            if not self.is_personal_info(line):
                                # 텍스트 정제
                                cleaned_line = self.clean_pdf_text(line)
                                if cleaned_line and not self.is_personal_info(cleaned_line):
                                    results.append(cleaned_line)
                                    logger.info(f"추출된 텍스트: {cleaned_line}")
                                else:
                                    logger.info(f"개인정보로 필터링됨: {line}")
                            else:
                                logger.info(f"개인정보로 필터링됨: {line}")
            
            logger.info(f"총 {len(results)}개의 텍스트 라인 추출됨")
            return results
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {str(e)}")
            raise e

    def clean_pdf_text(self, text: str) -> str:
        """PDF 텍스트 정제"""
        try:
            # 불필요한 공백 제거
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # 빈 문자열이거나 너무 짧은 경우 제외
            if not text or len(text) < 2:
                return ""
            
            # 특수문자 정제 (의미있는 특수문자는 유지)
            text = re.sub(r'[^\w\s가-힣\-\.\,\/\(\)\~]', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"텍스트 정제 실패: {str(e)}")
            return text

    def is_personal_info(self, text: str) -> bool:
        """개인정보인지 확인 - 더 엄격하게 수정"""
        personal_patterns = [
            # 이름/닉네임 패턴만 필터링
            r'^[가-힣]{2,4}입니다$',
            r'^[가-힣]{2,4}입니다\.$',
            r'^[가-힣]{2,4}\s*입니다$',
            r'^[가-힣]{2,4}\s*입니다\.$',
            r'^저는[가-힣]{2,4}입니다\.?$',
            r'^[가-힣]{2,4}이라고\s*합니다\.?$',
            r'^안녕하세요[가-힣]*입니다\.?$',
            
            # 지원자 관련 패턴
            r'^지원자$',
            r'^지질줄모르는지원자$',
            
            # 연락처 정보
            r'^[0-9]{3}-[0-9]{4}-[0-9]{4}$',
            r'^[0-9]{11}$',
            r'^[0-9]{2,3}-[0-9]{3,4}-[0-9]{4}$',
            r'^[0-9]{3}\.[0-9]{4}\.[0-9]{4}$',
            
            # 이메일 패턴
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            
            # 생년월일 관련
            r'^[0-9]{8}만[0-9]{1,2}세$',
            r'^[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일$',
            r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$',
            r'^만\s*[0-9]{1,2}세$',
            r'^[0-9]{1,2}세$',
            
            # 주소 정보 (개인 주소만)
            r'^[가-힣\s]+시[가-힣\s]+구[가-힣\s]+동[0-9-]+$',
            r'^[가-힣\s]+아파트[0-9-]+동[0-9-]+호$',
            
            # 섹션 제목 (개인정보 관련만)
            r'^Persoona1\s*1nformation$',
            r'^Personal\s*Information$',
            r'^기본사항$',
            r'^개인정보$',
            r'^인적사항$',
        ]
        
        for pattern in personal_patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def is_section_header(self, text: str) -> Optional[str]:
        """섹션 헤더 판별"""
        # 섹션 헤더 패턴
        section_patterns = {
            "학력": r'^(학력|학적|교육|학업|교육이력|학력사항|교육사항)$',
            "경력": r'^(경력|경험|직장|직무|업무|근무|경력사항|직장경력)$',
            "자격증": r'^(자격|자격증|면허|자격/면허|자격사항)$',
            "어학": r'^(어학|외국어|언어|어학능력|외국어능력)$',
            "활동": r'^(활동|대외활동|교내활동|봉사|동아리|프로젝트)$',
            "희망직무": r'^(희망직무|희망직종|희망분야|희망근무|희망업종|희망업무|희망진로|희망근무지)$'
        }
        
        # 텍스트 정규화
        text = text.strip()
        
        # 각 패턴 확인
        for section, pattern in section_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"Found section header: {section} from text: {text}")
                return section
        return None

    def extract_desired_job(self, text: str) -> str:
        """희망 직무 추출"""
        # 희망 직무 키워드 패턴
        job_patterns = [
            r'희망[직무|직종|분야|업종|업무][\s:]*([\w\s·/]+)',
            r'[희망|원하는][\s]*(직무|직종|분야|업종|업무)[\s:]*([\w\s·/]+)',
            r'(인터넷[_·/]?IT|웹개발|앱개발|서버|백엔드|프론트엔드|풀스택|소프트웨어|시스템|네트워크|보안|데이터|AI|인공지능|기획|디자인|마케팅|영업|인사|총무|재무|회계)'
        ]
        
        for pattern in job_patterns:
            match = re.search(pattern, text)
            if match:
                # 그룹이 2개인 패턴의 경우 두 번째 그룹 사용
                job = match.group(2) if len(match.groups()) > 1 else match.group(1)
                # 정규화
                job = re.sub(r'[_·/]', '/', job)
                logger.info(f"Found desired job: {job}")
                return job.strip()
        return ""

    def process_section(self, section: str, texts: List[str], result: Dict, patterns: Dict):
        """섹션 처리"""
        logger.info(f"Processing section: {section} with {len(texts)} texts")
        
        if section == "학력":
            self.process_education(texts, result, patterns)
        elif section == "경력":
            self.process_career(texts, result, patterns)
        elif section == "자격증":
            self.process_certificates(texts, result, patterns)
        elif section == "어학":
            self.process_languages(texts, result, patterns)
        elif section == "활동":
            self.process_activities(texts, result, patterns)
        elif section == "희망직무":
            result["desired_job"] = self.extract_desired_job(" ".join(texts))
            logger.info(f"Set desired job: {result['desired_job']}")

    def parse_resume(self, ocr_results: List[str]) -> Dict:
        """이력서 파싱 - OCR 결과를 구조화된 데이터로 변환"""
        result = {
            "nickname": "임솔",  # 기본 닉네임 설정
            "universities": [],
            "careers": [],
            "certificates": [],
            "languages": [],
            "activities": [],
            "skills": [],
            "awards": [],
            "introduction": ""
        }

        # OCR 결과를 하나의 텍스트로 합치기
        full_text = "\n".join(ocr_results) if isinstance(ocr_results, list) else str(ocr_results)
        
        # 디버그: OCR 결과 로깅
        logger.info("=== OCR 결과 ===")
        logger.info(full_text)
        logger.info("=== OCR 결과 끝 ===")
        
        # 텍스트를 줄 단위로 분리
        lines = full_text.split('\n')
        
        # 범용적 파싱 로직
        self.extract_education_direct(lines, result)
        self.extract_career_direct(lines, result)
        self.extract_certificate_direct(lines, result)
        self.extract_language_direct(lines, result)
        self.extract_activity_direct(lines, result)
        self.extract_skill_direct(lines, result)
        
        # 디버그: 파싱 결과 로깅
        logger.info("=== 파싱 결과 ===")
        logger.info(f"Nickname: {result.get('nickname', '')}")
        logger.info(f"Universities: {result.get('universities', [])}")
        logger.info(f"Careers: {result.get('careers', [])}")
        logger.info(f"Certificates: {result.get('certificates', [])}")
        logger.info(f"Languages: {result.get('languages', [])}")
        logger.info(f"Activities: {result.get('activities', [])}")
        logger.info("=== 파싱 결과 끝 ===")
        
        return result

    def extract_education_direct(self, lines: List[str], result: Dict):
        """직접적 학력 정보 추출 - 개선된 버전"""
        logger.info("\n=== 학력 정보 추출 시작 ===")
        
        # 전체 텍스트를 하나로 합쳐서 검색
        full_text = " ".join(lines)
        logger.info(f"전체 텍스트에서 학력 검색: {full_text[:200]}...")
        
        # 대학교 패턴 검색 (전체 텍스트에서) - 더 포괄적인 패턴 추가
        university_patterns = [
            r'([가-힣]+대학교)',
            r'([가-힣]+대학)',
            r'([가-힣]+전문대학)',
            r'([가-힣]+컬리지)',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학|컬리지|college|university))',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학))',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학))',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학|컬리지))',
            # 영문 패턴
            r'([a-zA-Z\s]+(?:University|College|Institute))',
            r'([a-zA-Z\s]+(?:Univ|Coll|Inst))'
        ]
        
        found_universities = []
        for pattern in university_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                school_name = match.strip()
                if school_name and school_name not in found_universities:
                    # 고등학교 제외
                    if '고등학교' not in school_name and '고등' not in school_name:
                        found_universities.append(school_name)
                        logger.info(f"대학교 발견: {school_name}")
        
        # 각 대학교에 대해 정보 추출
        for school_name in found_universities:
            # 학위 확인
            degree = "학사"
            if any(keyword in full_text for keyword in ['석사', '대학원', 'master', 'Master']):
                degree = "석사"
            elif any(keyword in full_text for keyword in ['박사', 'doctor', 'Doctor', 'PhD']):
                degree = "박사"
            elif any(keyword in full_text for keyword in ['전문학사', 'associate', 'Associate']):
                degree = "전문학사"
            
            # 전공 확인 - 더 정확한 패턴 매칭
            major = ""
            major_patterns = [
                r'([가-힣a-zA-Z\s]+(?:학과|전공))',
                r'([가-힣a-zA-Z\s]+(?:학과|전공))\s*[ㅣ|]\s*[가-힣]+',  # "경영학과 ㅣ수원" 형태
                r'([가-힣]+학과)',  # 더 구체적인 패턴
                r'([가-힣]+전공)',
                r'([a-zA-Z\s]+(?:Major|Department))'
            ]
            
            for pattern in major_patterns:
                match = re.search(pattern, full_text)
                if match:
                    major = match.group(1).strip()
                    # 학교명이 포함된 경우 제거
                    major = re.sub(r'^[가-힣]{2,4}\s+', '', major)
                    # 숫자 제거
                    major = re.sub(r'\s*\d+\s*$', '', major)
                    # 특수문자 제거
                    major = re.sub(r'[ㅣ|]', '', major)
                    major = major.strip()
                    if major:  # 빈 문자열이 아닌 경우만 사용
                        break
            
            # 학점(GPA) 추출
            gpa = 0.0
            gpa_max = 4.5
            
            gpa_patterns = [
                r'GPA\s*:\s*([0-9.]+)',
                r'학점\s*:\s*([0-9.]+)',
                r'평점\s*:\s*([0-9.]+)',
                r'([0-9.]+)\s*\/\s*([0-9.]+)',  # "3.5/4.5" 형태
                r'([0-9.]+)\s*점',  # "3.5점" 형태
                r'([0-9.]+)\s*\/\s*([0-9.]+)\s*점'  # "3.5/4.5점" 형태
            ]
            
            for pattern in gpa_patterns:
                match = re.search(pattern, full_text)
                if match:
                    if len(match.groups()) == 1:
                        gpa = float(match.group(1))
                    elif len(match.groups()) == 2:
                        gpa = float(match.group(1))
                        gpa_max = float(match.group(2))
                    break
            
            # 졸업 상태 확인 - 더 정확한 패턴 매칭
            status = "졸업"  # 기본값
            status_keywords = {
                '졸업': ['졸업', 'graduate', 'completed', '졸업자', '졸업예정', '졸업생'],
                '중퇴': ['중퇴', '중도퇴학', 'dropout', 'withdrawn', '중퇴자', '자퇴', '자퇴자'],
                '수료': ['수료', '수료자', 'coursework', 'completed_course', '수료예정'],
                '휴학': ['휴학', '휴학중', 'leave', 'suspended', '휴학자', '휴학생'],
                '재학': ['재학', '재학중', '학생', 'enrolled', 'current', '재학자', '재학생', '학부생', '대학생']
            }
            
            # 전체 텍스트에서 졸업 상태 키워드 검색
            for status_text, keywords in status_keywords.items():
                if any(keyword in full_text for keyword in keywords):
                    status = status_text
                    logger.info(f"졸업 상태 발견: {status_text}")
                    break
            
            # 중복 확인 및 우선순위 처리
            existing_schools = [uni["school_name"] for uni in result["universities"]]
            
            # 중복된 학교명 처리 (예: "수원대학교"와 "수원대학")
            is_duplicate = False
            for existing_school in existing_schools:
                # 한 학교명이 다른 학교명에 포함되는 경우 (예: "수원대학교"와 "수원대학")
                if school_name in existing_school or existing_school in school_name:
                    # 더 긴 이름을 우선 (예: "수원대학교" > "수원대학")
                    if len(school_name) > len(existing_school):
                        # 기존 항목 제거하고 새로운 항목으로 교체
                        result["universities"] = [uni for uni in result["universities"] if uni["school_name"] != existing_school]
                        is_duplicate = False
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate and school_name not in existing_schools:
                university_data = {
                    "school_name": school_name,
                    "degree": degree,
                    "major": major,
                    "status": status,
                    "gpa": gpa,
                    "gpa_max": gpa_max
                }
                
                result["universities"].append(university_data)
                logger.info(f"학력 정보 추출: {school_name} - {degree} - {major} - {status} - GPA: {gpa}/{gpa_max}")
        
        # 최종학력 정보 설정
        if result["universities"]:
            # 가장 높은 학위 찾기
            degree_priority = {
                "박사": 4,
                "석사": 3,
                "학사": 2,
                "전문학사": 1,
                "": 0
            }
            
            highest_degree = max(result["universities"], 
                               key=lambda x: degree_priority.get(x.get("degree", ""), 0))
            
            logger.info("\n최종학력 정보:")
            # 최종학력 설정 - 학위에 따른 교육 수준 사용
            degree = highest_degree.get("degree", "")
            if degree == "박사" or degree == "석사":
                result["final_edu"] = "대학원"
                logger.info(f"  최종학력: {result['final_edu']} (학위: {degree})")
            elif degree == "학사" or degree == "전문학사":
                result["final_edu"] = "대학교"
                logger.info(f"  최종학력: {result['final_edu']} (학위: {degree})")
            else:
                result["final_edu"] = "고등학교"  # 기본값
                logger.info(f"  최종학력: {result['final_edu']} (기본값)")
            
            # 최종상태 설정
            if highest_degree["status"]:
                result["final_status"] = highest_degree["status"]
                logger.info(f"  최종상태: {result['final_status']}")
            else:
                result["final_status"] = "졸업"  # 기본값
                logger.info(f"  최종상태: {result['final_status']} (기본값)")
        else:
            # 대학 정보가 없는 경우
            result["final_edu"] = "고등학교"
            result["final_status"] = "졸업"
            logger.info("대학 정보 없음, 기본값 설정")
        
        logger.info("=== 학력 정보 추출 완료 ===\n")

    def extract_language_direct(self, lines: List[str], result: Dict):
        """직접적 어학 정보 추출"""
        logger.info("\n=== 어학 정보 추출 시작 ===")
        current_language = None
        
        # 어학 관련 키워드
        language_keywords = {
            'test': [
                'TOEIC', 'TOEFL', 'TEPS', 'IELTS', 'TOEIC Speaking',
                'OPIc', 'JLPT', 'JPT', 'HSK', 'TOPIK',
                '토익', '토플', '텝스', '아이엘츠', '토익스피킹',
                '오픽', '일본어능력시험', '중국어능력시험', '한국어능력시험',
                # 추가 패턴
                'TOEIC Speaking', 'TOEIC Writing', 'TOEIC Listening', 'TOEIC Reading',
                'TOEFL iBT', 'TOEFL PBT', 'TOEFL CBT',
                'TEPS', 'TEPS Speaking', 'TEPS Writing',
                'IELTS Academic', 'IELTS General', 'IELTS Speaking', 'IELTS Writing',
                'OPIC', 'OPIc', 'OPI',
                'JLPT N1', 'JLPT N2', 'JLPT N3', 'JLPT N4', 'JLPT N5',
                'JPT', 'JPT A', 'JPT B', 'JPT C',
                'HSK 1', 'HSK 2', 'HSK 3', 'HSK 4', 'HSK 5', 'HSK 6',
                'TOPIK 1', 'TOPIK 2', 'TOPIK 3', 'TOPIK 4', 'TOPIK 5', 'TOPIK 6',
                # 한글 패턴
                '토익스피킹', '토익라이팅', '토익리스닝', '토익리딩',
                '토플아이비티', '토플피비티', '토플시비티',
                '텝스스피킹', '텝스라이팅',
                '아이엘츠아카데믹', '아이엘츠제너럴', '아이엘츠스피킹', '아이엘츠라이팅',
                '오픽', '오피시', '오피',
                '일본어능력시험1급', '일본어능력시험2급', '일본어능력시험3급', '일본어능력시험4급', '일본어능력시험5급',
                '중국어능력시험1급', '중국어능력시험2급', '중국어능력시험3급', '중국어능력시험4급', '중국어능력시험5급', '중국어능력시험6급',
                '한국어능력시험1급', '한국어능력시험2급', '한국어능력시험3급', '한국어능력시험4급', '한국어능력시험5급', '한국어능력시험6급'
            ],
            'grade': ['A+', 'A', 'B+', 'B', 'C+', 'C', 'IH', 'IM3', 'IM2', 'IM1', 'IL', 'NH', 'NM', 'NL']
        }
        
        logger.info(f"총 {len(lines)}개 라인에서 어학 정보 검색 중...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            logger.info(f"어학 검색 중 라인 {i+1}: {line}")
            
            # 시험명 추출
            test_match = None
            for test in language_keywords['test']:
                if test.lower() in line.lower():
                    test_match = test
                    logger.info(f"어학 시험 발견: {test_match}")
                    break
            
            if test_match:
                # 이전 언어 정보가 있으면 저장
                if current_language:
                    result["languages"].append(current_language)
                    logger.info(f"이전 어학 정보 저장: {current_language}")
                
                # 새로운 언어 정보 시작
                current_language = {
                    "test": test_match,
                    "score_or_grade": ""
                }
                
                # 같은 줄에서 점수/등급도 확인
                # 숫자 점수 (예: 950점, 950)
                score_match = re.search(r'(\d+)(?:점|points?)?', line)
                if score_match:
                    current_language["score_or_grade"] = score_match.group(1)
                    logger.info(f"같은 줄에서 점수 발견: {score_match.group(1)}")
                else:
                    # 등급 (예: IH, IM3 등)
                    for grade in language_keywords['grade']:
                        if grade in line:
                            current_language["score_or_grade"] = grade
                            logger.info(f"같은 줄에서 등급 발견: {grade}")
                            break
                
                continue
            
            # 현재 언어 정보가 있는 경우 점수/등급 추출
            if current_language and not current_language["score_or_grade"]:
                # 숫자 점수
                score_match = re.search(r'(\d+)(?:점|points?)?', line)
                if score_match:
                    current_language["score_or_grade"] = score_match.group(1)
                    logger.info(f"다음 줄에서 점수 발견: {score_match.group(1)}")
                else:
                    # 등급
                    for grade in language_keywords['grade']:
                        if grade in line:
                            current_language["score_or_grade"] = grade
                            logger.info(f"다음 줄에서 등급 발견: {grade}")
                            break
        
        # 마지막 어학 정보 저장
        if current_language:
            result["languages"].append(current_language)
            logger.info(f"마지막 어학 정보 저장: {current_language}")
        
        logger.info(f"총 {len(result['languages'])}개의 어학 정보 추출됨")
        logger.info("=== 어학 정보 추출 완료 ===\n")

    def extract_activity_direct(self, lines: List[str], result: Dict):
        """직접적 활동 정보 추출 - 활동명 다음에 날짜/헤더/빈줄이 오더라도 그 다음 텍스트를 역할로 매칭, 역할에서 기간 등 불필요한 정보 제거"""
        logger.info("\n=== 활동 정보 추출 시작 ===")
        activities = []
        activity_section_start = -1
        for i, line in enumerate(lines):
            line = line.strip()
            if line in ["사회경험", "활동", "경험", "Experience", "Activities"]:
                activity_section_start = i
                logger.info(f"활동 섹션 시작: {line} (라인 {i+1})")
                break
        if activity_section_start == -1:
            logger.info("활동 섹션을 찾을 수 없음")
            result["activities"] = []
            return
        i = activity_section_start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line in ["자격증", "어학", "기술", "프로젝트", "수상", "자기소개", "수상경력", "어학능력"]:
                break
            if re.match(r'\d{4}\.\d{2}', line):
                i += 1
                continue
            activity_keywords = [
                "전시회", "축제", "컨퍼런스", "세미나", "워크샵", "프로젝트", 
                "동아리", "봉사", "인턴십", "경진대회", "대회", "해커톤", 
                "연구", "개발", "디자인", "마케팅", "기획", "운영", "관리",
                "전시", "행사", "이벤트", "페스티벌", "공모전", "콘테스트",
                "엑스포", "박람회", "컨벤션", "심포지엄", "포럼", "캠프"
            ]
            is_activity_name = any(keyword in line for keyword in activity_keywords)
            if is_activity_name and len(line) > 1:
                activity_name = line
                role = ""
                award = ""
                j = i + 1
                # 날짜/헤더/빈줄을 건너뛰고 실제 역할 찾기
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or re.match(r'\d{4}\.\d{2}', next_line) or next_line in ["Experience", "사회경험", "자격증", "어학", "기술", "프로젝트", "수상", "자기소개", "수상경력", "어학능력"]:
                        j += 1
                        continue
                    # 역할 키워드가 있거나, 적당한 길이의 텍스트면 역할로 간주
                    if len(next_line) > 1 and len(next_line) < 30:
                        # 역할에서 'ㅣ 6개월', 'ㅣ 30개월' 등 기간 정보 제거
                        role = re.sub(r'ㅣ\s*\d+개월', '', next_line).strip()
                        # 역할에서 숫자+개월 패턴도 제거
                        role = re.sub(r'\d+개월', '', role).strip()
                        logger.info(f"역할로 간주: {role}")
                        j += 1
                        break
                    j += 1
                activities.append({
                    "name": activity_name,
                    "role": role,
                    "award": award
                })
                logger.info(f"활동 정보 추가: {activity_name} - {role} - {award}")
                i = j
            else:
                i += 1
        # 중복 제거 (활동명 기준)
        unique_activities = []
        seen_names = set()
        for activity in activities:
            if activity["name"] not in seen_names:
                unique_activities.append(activity)
                seen_names.add(activity["name"])
        result["activities"] = unique_activities
        logger.info(f"총 {len(unique_activities)}개의 고유 활동 정보 추출됨")
        logger.info("=== 활동 정보 추출 완료 ===")

    def extract_skill_direct(self, lines: List[str], result: Dict):
        """직접적 스킬 정보 추출"""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 기술 스킬 패턴들
            skill_patterns = [
                # 프로그래밍 언어
                r'(Python|파이썬)',
                r'(Java|자바)',
                r'(JavaScript|자바스크립트|JS)',
                r'(TypeScript|타입스크립트|TS)',
                r'(C\+\+|C\+\+11|C\+\+14|C\+\+17)',
                r'(C#|C샵)',
                r'(PHP|피에이치피)',
                r'(Ruby|루비)',
                r'(Go|고)',
                r'(Rust|러스트)',
                r'(Swift|스위프트)',
                r'(Kotlin|코틀린)',
                r'(Scala|스칼라)',
                # 프레임워크/라이브러리
                r'(React|리액트)',
                r'(Vue|뷰)',
                r'(Angular|앵귤러)',
                r'(Node\.js|노드)',
                r'(Django|장고)',
                r'(Flask|플라스크)',
                r'(Spring|스프링)',
                r'(Express|익스프레스)',
                r'(Laravel|라라벨)',
                r'(ASP\.NET|닷넷)',
                # 데이터베이스
                r'(MySQL|마이SQL)',
                r'(PostgreSQL|포스트그레SQL)',
                r'(MongoDB|몽고DB)',
                r'(Redis|레디스)',
                r'(Oracle|오라클)',
                r'(SQLite|SQLite)',
                # 클라우드/인프라
                r'(Docker|도커)',
                r'(Kubernetes|쿠버네티스|K8s)',
                r'(AWS|아마존|Amazon)',
                r'(Azure|애저)',
                r'(GCP|구글클라우드|Google.?Cloud)',
                r'(Jenkins|젠킨스)',
                r'(Git|깃)',
                r'(GitHub|깃허브)',
                r'(GitLab|깃랩)',
                # 디자인 도구
                r'(Photoshop|포토샵)',
                r'(Illustrator|일러스트레이터)',
                r'(Figma|피그마)',
                r'(Sketch|스케치)',
                r'(XD|Adobe.?XD)',
                r'(InDesign|인디자인)',
                # 오피스 도구
                r'(Excel|엑셀)',
                r'(PowerPoint|파워포인트)',
                r'(Word|워드)',
                r'(Access|액세스)',
                r'(Outlook|아웃룩)'
            ]
            
            for pattern in skill_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    skill_name = match.group(1).strip()
                    if skill_name not in result["skills"]:
                        result["skills"].append(skill_name)
                    break 

    def extract_career_direct(self, lines: List[str], result: Dict):
        """직접적 경력 정보 추출"""
        current_career = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 회사명 추출
            company_patterns = [
                r'([가-힣a-zA-Z\s]+(?:사원|주식회사|회사|기업|corporation|corp|inc|llc|co\.|ltd))',
                r'([가-힣a-zA-Z\s]+(?:그룹|시스템|솔루션|테크|tech|디자인|개발|마케팅))',
                r'([가-힣a-zA-Z\s]+(?:스튜디오|랩|연구소|센터|아카데미))'
            ]
            
            is_company_line = False
            company_name = ""
            
            for pattern in company_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    company_name = match.group(1).strip()
                    # '사원'이 포함되어 있으면 제거
                    company_name = re.sub(r'사원$', '', company_name).strip()
                    is_company_line = True
                    break
            
            if is_company_line:
                if current_career:
                    # 직무가 없으면 기본값으로 정규직 설정
                    if not current_career["role"]:
                        current_career["role"] = "정규직"
                    result["careers"].append(current_career)
                
                current_career = {
                    "company": company_name,
                    "role": "",
                    "work_month": 0
                }
                continue
            
            # 직무 추출 - 인턴/정규직/대표로 분류
            if current_career:
                # 인턴 관련 패턴 (가장 우선순위)
                intern_patterns = [
                    r'(인턴|intern|인턴십|internship)',
                    r'(수습|견습|trainee)',
                    r'(아르바이트|알바|part.?time)',
                    r'(보조|어시스턴트|assistant)'
                ]
                
                # 대표 관련 패턴 (두 번째 우선순위)
                representative_patterns = [
                    r'(대표|president|director|chief|대표자)',
                    r'(회장|chairman|executive|이사)',
                    r'(founder|창업자|설립자|주최자)',
                    r'(팀장|부장|과장|차장)'
                ]
                
                # 정규직 관련 패턴 (기본값)
                regular_patterns = [
                    r'(정규직|regular|full.?time|permanent)',
                    r'(참여|활동|담당|책임|리드|lead|담당자|책임자)',
                    r'(기획|운영|관리|manager|organizer)',
                    r'(전시주관|행사진행|participant)'
                ]
                
                # 인턴 확인 (가장 우선순위)
                for pattern in intern_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and not current_career["role"]:
                        current_career["role"] = "인턴"
                        break
                
                # 대표 확인 (두 번째 우선순위)
                if not current_career["role"]:
                    for pattern in representative_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            current_career["role"] = "대표"
                            break
                
                # 정규직 확인 (기본값)
                if not current_career["role"]:
                    for pattern in regular_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            current_career["role"] = "정규직"
                            break
            
            # 근무기간 추출 (간단한 개월수만)
            if current_career:
                # 개월 단위
                month_match = re.search(r'(\d+)\s*개월', line)
                if month_match:
                    current_career["work_month"] = int(month_match.group(1))
                    continue
        
        # 마지막 경력 추가
        if current_career:
            # 직무가 없으면 기본값으로 정규직 설정
            if not current_career["role"]:
                current_career["role"] = "정규직"
            result["careers"].append(current_career)

    def extract_certificate_direct(self, lines: List[str], result: Dict):
        """직접적 자격증 정보 추출"""
        # 제외할 섹션 제목들
        exclude_sections = [
            '기본사항', '학력사항', '경력사항', '사회경험', '활동사항', 
            '자격증', '어학능력', '수상내역', '프로젝트', '기술스택',
            'basic', 'education', 'career', 'experience', 'activity',
            'certificate', 'language', 'award', 'project', 'skill'
        ]
        
        # 제외할 활동/역할 키워드들
        exclude_activity_keywords = [
            '전시주관', '행사진행', '담당', '책임', '리드', '매니저', '참여', '활동',
            '기획', '운영', '관리', '보조', '어시스턴트', 'organizer', 'manager', 
            'participant', 'leader', 'assistant', '담당자', '책임자'
        ]
        
        # 제외할 회사/기관 키워드들
        exclude_company_keywords = [
            '사원', '주식회사', '회사', '기업', 'corporation', 'corp', 'inc', 'llc', 'co.', 'ltd',
            '그룹', '시스템', '솔루션', '테크', 'tech', '디자인', '개발', '마케팅',
            '스튜디오', '랩', '연구소', '센터', '아카데미'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 섹션 제목 제외
            if any(section in line for section in exclude_sections):
                continue
            
            # 활동/역할 키워드가 포함된 경우 제외
            if any(keyword in line for keyword in exclude_activity_keywords):
                continue
            
            # 회사/기관 키워드가 포함된 경우 제외
            if any(keyword in line for keyword in exclude_company_keywords):
                continue
            
            # 숫자와 함께 오는 활동/회사명 제외 (예: "전시주관및 행사진행 03", "공픔디자인사원 05")
            if re.search(r'\d+$', line) and any(keyword in line for keyword in exclude_activity_keywords + exclude_company_keywords):
                continue
            
            # 어학 관련 키워드가 포함된 경우 제외 (어학능력에서 처리)
            language_keywords = [
                'toeic', 'toefl', 'teps', 'opic', 'jlpt', 'hsk', 'topik',
                '토익', '토플', '텝스', '오픽', '일본어', '중국어', '한국어',
                '영어', '일본어', '중국어', '한국어', '어학', 'language'
            ]
            
            if any(lang_keyword in line.lower() for lang_keyword in language_keywords):
                continue
            
            # 자격증 패턴들 (더 엄격하게)
            certificate_patterns = [
                # 운전면허
                r'(운전면허증?|운전면허|driver.?license)',
                # Adobe 제품
                r'(Photoshop|포토샵|adobe.?photoshop)',
                r'(Illustrator|일러스트레이터|adobe.?illustrator|111ustrator)',
                r'(InDesign|인디자인|adobe.?indesign)',
                r'(Premiere|프리미어|adobe.?premiere)',
                r'(After.?Effects|애프터이펙트|adobe.?after.?effects)',
                # Microsoft Office
                r'(Office|오피스|microsoft.?office|M5 office)',
                r'(Word|워드|microsoft.?word)',
                r'(Excel|엑셀|microsoft.?excel)',
                r'(PowerPoint|파워포인트|microsoft.?powerpoint)',
                # IT 자격증
                r'(정보처리기사|정보처리산업기사|정보처리기능사)',
                r'(컴퓨터활용능력|컴활)',
                r'(워드프로세서|워드)',
                r'(컴퓨터그래픽스운용기능사|컴퓨터그래픽스)',
                r'(사무자동화산업기사|사무자동화)',
                r'(전자계산기조직응용기사|전자계산기)',
                # 기타 IT 자격증
                r'(SQLD|SQLP|SQL)',
                r'(CCNA|CCNP|CCIE)',
                r'(AWS|Azure|GCP)',
                r'(Docker|Kubernetes|K8s)',
                r'(PMP|프로젝트관리)',
                r'(ITIL|아이틸)',
                # 회계/세무 자격증
                r'(공인회계사|CPA)',
                r'(세무사|세무)',
                r'(관세사|관세)',
                r'(공인노무사|노무사)',
                r'(공인중개사|중개사)',
                # 금융 자격증
                r'(투자상담사|투자)',
                r'(자산관리사|자산관리)',
                r'(보험설계사|보험)',
                r'(증권투자상담사|증권)',
                # 기타 전문 자격증
                r'(변리사|변리)',
                r'(법무사|법무)',
                r'(평가사|평가)',
                r'(기사|산업기사|기능사)',
                # GTQ (그래픽기술자격)
                r'(GTQ|그래픽기술자격)',
                # 한글, 엑셀 등 개별 프로그램
                r'(한글|HWP|한글워드프로세서)',
                r'(엑셀|Excel)',
                r'(파워포인트|PowerPoint)',
                r'(워드|Word)',
                # License 관련
                r'(License|라이센스)',
                # 기타 자격증 키워드 (더 엄격하게)
                r'(자격증|자격|certificate|cert)'
            ]
            
            # 자격증 패턴 검색
            for pattern in certificate_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    cert_name = match.group(1).strip()
                    
                    # "자격증" 단어 제거
                    cert_name = re.sub(r'자격증?', '', cert_name).strip()
                    
                    # 점수/등급 추출
                    score_grade = ""
                    
                    # 등급 패턴 (1급, 2급, 3급 등)
                    grade_match = re.search(r'([1-3]급)', line)
                    if grade_match:
                        score_grade = grade_match.group(1)
                    
                    # 점수 패턴 (숫자 + 점)
                    score_match = re.search(r'(\d+)점', line)
                    if score_match:
                        score_grade = f"{score_match.group(1)}점"
                    
                    # 기타 등급 패턴 (A, B, C 등)
                    other_grade_match = re.search(r'([A-Z]등급?|[A-Z])', line)
                    if other_grade_match and not score_grade:
                        score_grade = other_grade_match.group(1)
                    
                    # 자격증 정보 구성
                    cert_info = cert_name
                    if score_grade:
                        cert_info = f"{cert_name} ({score_grade})"
                    
                    # 빈 문자열이 아니고 중복되지 않은 경우만 추가
                    if cert_info and cert_info not in result["certificates"]:
                        result["certificates"].append(cert_info)
                    break
            
            # 패턴에서 찾지 못한 경우, 자격증 키워드가 포함된 라인 검색 (더 엄격하게)
            if not any(re.search(pattern, line, re.IGNORECASE) for pattern in certificate_patterns):
                cert_keywords = [
                    '자격증', '자격', '기사', '산업기사', '기능사', '사', '증', '면허',
                    'certificate', 'license', 'qualification', 'cert'
                ]
                
                if any(keyword in line.lower() for keyword in cert_keywords):
                    # 라인에서 자격증명 추출
                    cert_name = line.strip()
                    
                    # "자격증" 단어 제거
                    cert_name = re.sub(r'자격증?', '', cert_name).strip()
                    
                    # 점수/등급 추출
                    score_grade = ""
                    
                    # 등급 패턴 (1급, 2급, 3급 등)
                    grade_match = re.search(r'([1-3]급)', line)
                    if grade_match:
                        score_grade = grade_match.group(1)
                    
                    # 점수 패턴 (숫자 + 점)
                    score_match = re.search(r'(\d+)점', line)
                    if score_match:
                        score_grade = f"{score_match.group(1)}점"
                    
                    # 기타 등급 패턴 (A, B, C 등)
                    other_grade_match = re.search(r'([A-Z]등급?|[A-Z])', line)
                    if other_grade_match and not score_grade:
                        score_grade = other_grade_match.group(1)
                    
                    # 자격증 정보 구성
                    cert_info = cert_name
                    if score_grade:
                        cert_info = f"{cert_name} ({score_grade})"
                    
                    # 빈 문자열이 아니고 중복되지 않은 경우만 추가
                    if cert_info and cert_info not in result["certificates"]:
                        result["certificates"].append(cert_info)

    def process_education(self, texts: List[str], result: Dict, patterns: Dict):
        """학력 정보 처리"""
        current_edu = {}
        for text in texts:
            logger.info(f"Processing education text: {text}")
            
            # GPA 추출 (예: 3.9/4.3, 3.9/4.5 등)
            gpa_match = re.search(r'(\d+\.?\d*)\s*\/\s*(\d+\.?\d*)', text)
            if gpa_match:
                current_edu["gpa"] = float(gpa_match.group(1))
                current_edu["gpa_max"] = float(gpa_match.group(2))
                logger.info(f"Found GPA: {current_edu['gpa']}/{current_edu['gpa_max']}")
                continue

            # 대학교명 (서울대, 연세대 등 주요 대학 포함)
            uni_pattern = r'(서울대학교|연세대학교|고려대학교|성균관대학교|한양대학교|중앙대학교|경희대학교|서강대학교|이화여자대학교|건국대학교|동국대학교|홍익대학교|아주대학교|숭실대학교|인하대학교|국민대학교|세종대학교|단국대학교|광운대학교|상명대학교|서울시립대학교|숙명여자대학교|한국외국어대학교|한성대학교|명지대학교|가천대학교|가톨릭대학교|강남대학교|강원대학교|경기대학교|경남대학교|경북대학교|경상대학교|계명대학교|공주대학교|광주과학기술원|국민대학교|군산대학교|금오공과대학교|남서울대학교|단국대학교|대구가톨릭대학교|대구대학교|대구한의대학교|대전대학교|동덕여자대학교|동서대학교|동아대학교|동의대학교|명지대학교|목원대학교|목포대학교|배재대학교|백석대학교|부경대학교|부산가톨릭대학교|부산대학교|부산외국어대학교|삼육대학교|상명대학교|서경대학교|서울과학기술대학교|서울교육대학교|서울기독교대학교|서울시립대학교|서울여자대학교|서원대학교|선문대학교|성결대학교|성신여자대학교|세명대학교|세종대학교|수원대학교|숙명여자대학교|순천대학교|순천향대학교|숭실대학교|신라대학교|아주대학교|안동대학교|안양대학교|연세대학교|영남대학교|영산대학교|용인대학교|우석대학교|울산대학교|원광대학교|을지대학교|이화여자대학교|인제대학교|인천대학교|인하대학교|전남대학교|전북대학교|전주대학교|제주대학교|조선대학교|중부대학교|중앙대학교|창원대학교|청주대학교|충남대학교|충북대학교|한경대학교|한국교원대학교|한국교통대학교|한국기술교육대학교|한국산업기술대학교|한국외국어대학교|한국항공대학교|한국해양대학교|한남대학교|한동대학교|한림대학교|한밭대학교|한서대학교|한성대학교|한신대학교|한양대학교|호서대학교|홍익대학교)'
            if re.search(uni_pattern, text):
                if current_edu:
                    result["universities"].append(current_edu)
                current_edu = {"name": text.strip()}
                logger.info(f"Found university: {current_edu['name']}")
                continue

            # 전공 (학과/전공 키워드 포함)
            major_pattern = r'([\w\s·]+)(학과|전공|계열)'
            major_match = re.search(major_pattern, text)
            if major_match:
                current_edu["major"] = major_match.group(1).strip()
                logger.info(f"Found major: {current_edu['major']}")
                continue

            # 학위 구분
            degree_pattern = r'(학사|석사|박사|전문학사|고등학교)'
            degree_match = re.search(degree_pattern, text)
            if degree_match:
                current_edu["degree"] = degree_match.group(1)
                logger.info(f"Found degree: {current_edu['degree']}")
                continue

            # 졸업 상태
            status_pattern = r'(졸업|재학|휴학|중퇴|수료|졸업예정)'
            status_match = re.search(status_pattern, text)
            if status_match:
                current_edu["status"] = status_match.group(1)
                logger.info(f"Found status: {current_edu['status']}")
                continue

        # 마지막 교육 정보 추가
        if current_edu:
            result["universities"].append(current_edu)
            logger.info(f"Added final education record: {current_edu}")

        # 최종 학력 설정
        if result["universities"]:
            # 학위 우선순위
            degree_priority = {"박사": 5, "석사": 4, "학사": 3, "전문학사": 2, "고등학교": 1}
            
            # 최종 학력 찾기
            highest_edu = max(
                result["universities"],
                key=lambda x: degree_priority.get(x.get("degree", ""), 0)
            )
            
            # 최종학력 설정 - 학위에 따른 교육 수준 사용
            degree = highest_edu.get("degree", "")
            if degree == "박사" or degree == "석사":
                result["final_edu"] = "대학원"
            elif degree == "학사" or degree == "전문학사":
                result["final_edu"] = "대학교"
            else:
                result["final_edu"] = "고등학교"  # 기본값
            
            result["final_status"] = highest_edu.get("status", "")
        else:
            # 대학 정보가 없는 경우
            result["final_edu"] = "고등학교"
            result["final_status"] = "졸업"
            logger.info("대학 정보 없음, 기본값 설정")
        
        logger.info("=== 학력 정보 추출 완료 ===\n")

    def process_languages(self, texts: List[str], result: Dict, patterns: Dict):
        """어학 정보 처리"""
        current_lang = {}
        
        for text in texts:
            logger.info(f"Processing language text: {text}")
            
            # 시험 종류 (TOEIC, TOEFL, TEPS, OPIC, JLPT, HSK, TOPIK 등)
            test_pattern = r'(TOEIC|TOEFL|TEPS|OPIC|JLPT|HSK|TOPIK|토익|토플|텝스|오픽|일본어능력시험|중국어능력시험|한국어능력시험)'
            test_match = re.search(test_pattern, text, re.IGNORECASE)
            
            if test_match:
                if current_lang:
                    result["languages"].append(current_lang)
                current_lang = {"test": test_match.group(1)}
                logger.info(f"Found language test: {current_lang['test']}")
                
                # 같은 라인에서 점수 찾기
                score_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:점|point|점수)?', text)
                if score_match:
                    score = score_match.group(1)
                    current_lang["score"] = float(score) if '.' in score else int(score)
                    logger.info(f"Found score in same line: {current_lang['score']}")
                continue
            
            # 점수만 있는 라인
            score_pattern = r'(\d+(?:\.\d+)?)\s*(?:점|point|점수)?'
            score_match = re.search(score_pattern, text)
            if score_match and current_lang and "score" not in current_lang:
                score = score_match.group(1)
                current_lang["score"] = float(score) if '.' in score else int(score)
                logger.info(f"Found score: {current_lang['score']}")
                continue
            
            # 날짜 처리
            date_pattern = r'(\d{4})[년\.]\s*(\d{1,2})[월\.](?:\s*(\d{1,2})[일\.])?'
            date_match = re.search(date_pattern, text)
            if date_match and current_lang:
                year = date_match.group(1)
                month = date_match.group(2).zfill(2)
                day = date_match.group(3).zfill(2) if date_match.group(3) else "01"
                current_lang["date"] = f"{year}-{month}-{day}"
                logger.info(f"Found date: {current_lang['date']}")
        
        # 마지막 어학 정보 추가
        if current_lang:
            result["languages"].append(current_lang)
            logger.info(f"Added final language record: {current_lang}")
        
        # 어학 점수 정규화
        for lang in result["languages"]:
            if "test" in lang:
                # 시험 이름 정규화
                test_name = lang["test"].upper()
                if "토익" in test_name:
                    lang["test"] = "TOEIC"
                elif "토플" in test_name:
                    lang["test"] = "TOEFL"
                elif "텝스" in test_name:
                    lang["test"] = "TEPS"
                elif "오픽" in test_name:
                    lang["test"] = "OPIC"
                elif "일본어능력시험" in test_name:
                    lang["test"] = "JLPT"
                elif "중국어능력시험" in test_name:
                    lang["test"] = "HSK"
                elif "한국어능력시험" in test_name:
                    lang["test"] = "TOPIK"