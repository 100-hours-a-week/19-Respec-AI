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
        """S3 URL에서 PDF 파일 다운로드"""
        try:
            logger.info(f"URL에서 파일 다운로드 중: {s3_url}")
            
            # URL 파싱
            parsed_url = urlparse(s3_url)
            
            # AWS S3 URL인지 확인
            if parsed_url.scheme == 'https' and 's3' in parsed_url.netloc and 'amazonaws.com' in parsed_url.netloc:
                # AWS S3 URL인 경우 - 먼저 일반 HTTP 요청으로 시도
                try:
                    logger.info("AWS S3 URL을 일반 HTTP 요청으로 처리 시도...")
                    response = requests.get(s3_url, timeout=30)
                    response.raise_for_status()
                    logger.info(f"S3 URL에서 파일 다운로드 완료: {len(response.content)} bytes")
                    return response.content
                except Exception as s3_error:
                    logger.warning(f"S3 URL을 일반 HTTP로 처리 실패: {str(s3_error)}")
                    logger.info("boto3를 사용한 S3 접근 시도...")
                    
                    # boto3를 사용한 S3 접근 시도
                    try:
                        bucket_name = parsed_url.netloc.split('.')[0]
                        object_key = parsed_url.path.lstrip('/')
                        
                        # boto3 클라이언트 생성
                        s3_client = boto3.client('s3')
                        
                        # 파일 다운로드
                        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                        file_content = response['Body'].read()
                        
                        logger.info(f"S3에서 파일 다운로드 완료: {len(file_content)} bytes")
                        return file_content
                    except Exception as boto_error:
                        logger.error(f"boto3 S3 접근 실패: {str(boto_error)}")
                        raise Exception(f"S3 URL 접근 실패. 파일이 공개되어 있는지 확인하세요: {str(boto_error)}")
            else:
                # 일반 HTTP/HTTPS URL
                response = requests.get(s3_url, timeout=30)
                response.raise_for_status()
                logger.info(f"일반 URL에서 파일 다운로드 완료: {len(response.content)} bytes")
                return response.content
                
        except Exception as e:
            logger.error(f"URL에서 파일 다운로드 실패: {str(e)}")
            raise

    def process_pdf_from_url(self, s3_url: str) -> List[str]:
        """S3 URL에서 PDF 파일을 다운로드하고 처리"""
        try:
            # S3 URL에서 파일 다운로드
            pdf_data = self.download_from_s3_url(s3_url)
            
            # PDF 처리
            results = self.process_pdf(pdf_data)
            
            # 결과를 텍스트 리스트로 변환
            if isinstance(results, list):
                return results
            else:
                return [str(results)]
                
        except Exception as e:
            logger.error(f"URL에서 PDF 처리 실패: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 및 정제
        1. 기본 정규화
        2. 특수문자 제거
        3. 불필요한 공백 제거
        4. 의미 있는 텍스트만 필터링
        """
        try:
            # 기본 정규화
            text = text.strip()
            
            # 연속된 공백을 하나로
            text = ' '.join(text.split())
            
            # 한글, 영문, 숫자만 남기고 나머지는 제거
            text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
            
            # 연속된 특수문자 제거
            text = re.sub(r'[^\w\s]+', '', text)
            
            # 앞뒤 특수문자 제거
            text = text.strip('!@#$%^&*()_+-=[]{}|;:,.<>?')
            
            # 빈 문자열이거나 너무 짧은 경우 제외
            if not text or len(text) < 2:
                return ""
            
            # 불필요한 공백 제거
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"텍스트 전처리 실패: {str(e)}")
            return ""

    def correct_ocr_errors(self, text: str) -> str:
        """OCR 오류 교정
        1. 자주 발생하는 OCR 오류 패턴 수정
        2. 숫자-문자 혼동 수정
        3. 한글 자모 분리/결합 오류 수정
        """
        try:
            # 숫자-문자 혼동 수정
            corrections = {
                'O': '0',  # 영문 O를 숫자 0으로
                'l': '1',  # 영문 l을 숫자 1로
                'I': '1',  # 영문 I를 숫자 1로
                'Z': '2',  # 영문 Z를 숫자 2로
                'S': '5',  # 영문 S를 숫자 5로
                'G': '6',  # 영문 G를 숫자 6로
                'B': '8',  # 영문 B를 숫자 8로
            }
            
            for wrong, correct in corrections.items():
                text = text.replace(wrong, correct)
            
            # 한글 자모 분리/결합 오류 수정
            text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', text)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR 오류 교정 실패: {str(e)}")
            return text

    def process_pdf(self, pdf_data: bytes) -> List[Dict]:
        """PDF 파일 처리 및 텍스트 추출
        1. PDF를 이미지로 변환
        2. EasyOCR로 텍스트 추출
        3. 텍스트 전처리 및 정제
        4. 결과 반환
        """
        try:
            # PDF 파일 열기
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            results = []
            
            # 각 페이지 처리
            for page_num in range(len(pdf_document)):
                logger.info(f"페이지 {page_num + 1} 처리 중...")
                page = pdf_document[page_num]
                
                # 페이지를 이미지로 변환
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # EasyOCR로 텍스트 추출
                ocr_results = self.reader.readtext(np.array(img))
                
                # 결과 처리
                page_texts = []
                for bbox, text, conf in ocr_results:
                    if conf > 0.5:  # 신뢰도가 50% 이상인 텍스트만 처리
                        # 텍스트 전처리
                        cleaned_text = self.preprocess_text(text)
                        if cleaned_text:
                            # OCR 오류 교정
                            corrected_text = self.correct_ocr_errors(cleaned_text)
                            page_texts.append({
                                'text': corrected_text,
                                'confidence': conf,
                                'position': bbox[0][1]  # y 좌표로 정렬
                            })
                
                # y 좌표로 정렬
                page_texts.sort(key=lambda x: x['position'])
                
                # 연속된 텍스트 병합
                merged_texts = []
                current_text = ""
                current_y = None
                
                for item in page_texts:
                    if current_y is None:
                        current_text = item['text']
                        current_y = item['position']
                    elif abs(item['position'] - current_y) < 10:  # 같은 줄로 간주
                        current_text += " " + item['text']
                    else:
                        if current_text:
                            merged_texts.append(current_text)
                        current_text = item['text']
                        current_y = item['position']
                
                if current_text:
                    merged_texts.append(current_text)
                
                results.extend(merged_texts)
            
            return results
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {str(e)}")
            return []

    def parse_resume(self, ocr_results: List[str]) -> Dict:
        """이력서 파싱 - OCR 결과를 구조화된 데이터로 변환"""
        result = {
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
        self.parse_universal_resume(lines, result)
        
        # 디버그: 파싱 결과 로깅
        logger.info("=== 파싱 결과 ===")
        logger.info(f"Universities: {result.get('universities', [])}")
        logger.info(f"Careers: {result.get('careers', [])}")
        logger.info(f"Certificates: {result.get('certificates', [])}")
        logger.info(f"Languages: {result.get('languages', [])}")
        logger.info(f"Activities: {result.get('activities', [])}")
        logger.info("=== 파싱 결과 끝 ===")
        
        return result

    def parse_universal_resume(self, lines: List[str], result: Dict):
        """범용적 이력서 파싱 로직"""
        # 전체 텍스트를 하나로 합치기
        full_text = "\n".join(lines)
        
        # 1. 학력 정보 추출
        self.extract_education_direct(lines, result)
        
        # 2. 경력 정보 추출
        self.extract_career_direct(lines, result)
        
        # 3. 자격증 정보 추출
        self.extract_certificate_direct(lines, result)
        
        # 4. 어학 정보 추출
        self.extract_language_direct(lines, result)
        
        # 5. 활동 정보 추출
        self.extract_activity_direct(lines, result)
        
        # 6. 스킬 정보 추출
        self.extract_skill_direct(lines, result)

    def extract_education_direct(self, lines: List[str], result: Dict):
        """직접적 학력 정보 추출"""
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 학교명 추출
            school_patterns = [
                r'([가-힣a-zA-Z\s]+(?:고등학교|대학교|대학|전문대학|컬리지|college|university))',
                r'([가-힣a-zA-Z\s]+(?:학교|학교))'
            ]
            
            for pattern in school_patterns:
                match = re.search(pattern, line)
                if match:
                    school_name = match.group(1).strip()
                    
                    # 학위 확인
                    degree = "학사"
                    if "고등학교" in school_name:
                        degree = "고등학교"
                    elif any(keyword in line for keyword in ['석사', '대학원', 'master']):
                        degree = "석사"
                    elif any(keyword in line for keyword in ['박사', 'doctor']):
                        degree = "박사"
                    elif any(keyword in line for keyword in ['전문학사', 'associate']):
                        degree = "전문학사"
                    
                    # 전공 확인
                    major = ""
                    if "학과" in line or "전공" in line:
                        major_match = re.search(r'([가-힣a-zA-Z\s]+(?:학과|전공))', line)
                        if major_match:
                            major = major_match.group(1).strip()
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if "학과" in next_line or "전공" in next_line:
                            major = next_line
                    
                    # 중복 확인
                    existing_schools = [uni["name"] for uni in result["universities"]]
                    if school_name not in existing_schools:
                        result["universities"].append({
                            "name": school_name,
                            "degree": degree,
                            "major": major,
                            "gpa": 0.0,
                            "gpa_max": 4.5
                        })
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
                    is_company_line = True
                    break
            
            if is_company_line:
                if current_career:
                    result["careers"].append(current_career)
                
                current_career = {
                    "company": company_name,
                    "role": "",
                    "work_month": 0
                }
                continue
            
            # 직무 추출
            if current_career:
                role_patterns = [
                    r'(디자이너|개발자|프로그래머|엔지니어|매니저|팀장|사원|주임|대리|과장|차장|부장|이사|CEO|CTO|CFO|COO)',
                    r'([가-힣a-zA-Z\s]+(?:담당|책임|리드|매니저|어시스턴트|인턴|보조))',
                    r'(designer|developer|engineer|manager|assistant|intern)'
                ]
                
                for pattern in role_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and not current_career["role"]:
                        current_career["role"] = match.group(1).strip()
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
            result["careers"].append(current_career)

    def extract_certificate_direct(self, lines: List[str], result: Dict):
        """직접적 자격증 정보 추출"""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 자격증 패턴들
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
                # 어학 자격증
                r'(토익|TOEIC|toeic)',
                r'(토플|TOEFL|toefl)',
                r'(오픽|OPIc|opic)',
                r'(텝스|TEPS|teps)',
                r'(일본어능력시험|JLPT|jlpt)',
                r'(중국어능력시험|HSK|hsk)',
                r'(한국어능력시험|TOPIK|topik)',
                # 기타 자격증
                r'(기사|산업기사|기능사)',
                r'(자격증|자격|certificate|cert)'
            ]
            
            for pattern in certificate_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    cert_name = match.group(1).strip()
                    if cert_name not in result["certificates"]:
                        result["certificates"].append(cert_name)
                    break

    def extract_language_direct(self, lines: List[str], result: Dict):
        """직접적 어학 정보 추출"""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 어학 시험 패턴들
            language_patterns = [
                r'(토익|TOEIC|toeic)\s*(\d+)',
                r'(토플|TOEFL|toefl)\s*(\d+)',
                r'(오픽|OPIc|opic)\s*([가-힣a-zA-Z]+)',
                r'(텝스|TEPS|teps)\s*(\d+)',
                r'(일본어능력시험|JLPT|jlpt)\s*([가-힣a-zA-Z]+)',
                r'(중국어능력시험|HSK|hsk)\s*([가-힣a-zA-Z]+)',
                r'(한국어능력시험|TOPIK|topik)\s*([가-힣a-zA-Z]+)',
                r'(영어|English)\s*([가-힣a-zA-Z]+)',
                r'(일본어|Japanese)\s*([가-힣a-zA-Z]+)',
                r'(중국어|Chinese)\s*([가-힣a-zA-Z]+)'
            ]
            
            for pattern in language_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    test_name = match.group(1).strip()
                    score = match.group(2).strip()
                    
                    # 중복 확인
                    existing = [lang["test"] for lang in result["languages"]]
                    if test_name not in existing:
                        result["languages"].append({
                            "test": test_name,
                            "score_or_grade": score
                        })
                    break

    def extract_activity_direct(self, lines: List[str], result: Dict):
        """직접적 활동 정보 추출"""
        current_activity = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 활동 패턴들
            activity_patterns = [
                r'([가-힣a-zA-Z\s]+(?:전시회|축제|컨퍼런스|세미나|워크샵|프로젝트|동아리|봉사|인턴십|경진대회|대회|해커톤|해커톤))',
                r'([가-힣a-zA-Z\s]+(?:연구|개발|디자인|마케팅|기획|운영|관리))',
                r'([가-힣a-zA-Z\s]+(?:클럽|모임|단체|조직|협회))'
            ]
            
            is_activity_line = False
            activity_name = ""
            
            for pattern in activity_patterns:
                match = re.search(pattern, line)
                if match:
                    activity_name = match.group(1).strip()
                    is_activity_line = True
                    break
            
            if is_activity_line:
                if current_activity:
                    result["activities"].append(current_activity)
                
                current_activity = {
                    "name": activity_name,
                    "role": "",
                    "award": ""
                }
                continue
            
            # 역할 추출
            if current_activity:
                role_patterns = [
                    r'(전시주관|행사진행|기획|운영|참여|활동|담당|책임|리드|매니저|어시스턴트|인턴)',
                    r'([가-힣a-zA-Z\s]+(?:담당|책임|리드|매니저|어시스턴트|인턴))',
                    r'(organizer|manager|assistant|intern|participant|leader)'
                ]
                
                for pattern in role_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and not current_activity["role"]:
                        current_activity["role"] = match.group(1).strip()
                        break
            
            # 수상 내역 추출
            if current_activity:
                award_patterns = [
                    r'(수상|우수상|최우수상|대상|금상|은상|동상|특별상|장려상)',
                    r'(award|prize|winner|champion|finalist)'
                ]
                
                for pattern in award_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and not current_activity["award"]:
                        current_activity["award"] = match.group(1).strip()
                        break
        
        # 마지막 활동 추가
        if current_activity:
            result["activities"].append(current_activity)

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