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
        5. 닉네임/이름 삭제
        """
        try:
            # 기본 정규화
            text = text.strip()
            
            # 연속된 공백을 하나로
            text = ' '.join(text.split())
            
            # 개인정보 제거 (이름, 닉네임 등) - 학력 정보는 유지
            personal_info_patterns = [
                # 기본 이름 패턴
                r'^[가-힣]{2,4}입니다$',  # "김다슴입니다" 같은 패턴
                r'^[가-힣]{2,4}입니다\.$',  # "김다슴입니다." 같은 패턴
                r'^[가-힣]{2,4}\s*입니다$',  # 공백 포함된 패턴
                r'^[가-힣]{2,4}\s*입니다\.$',  # 공백과 마침표 포함된 패턴
                
                # 지원자 관련 패턴
                r'^지원자$',  # "지원자" 단독
                r'^지질줄모르는지원자$',  # 오타 포함된 지원자
                r'^[가-힣]*지원자[가-힣]*$',  # 지원자 포함된 모든 패턴
                
                # 자기소개 패턴
                r'^[가-힣]{2,4}입니다\.?$',  # "홍길동입니다" 형태
                r'^안녕하세요[가-힣]*입니다\.?$',  # "안녕하세요 홍길동입니다"
                r'^저는[가-힣]{2,4}입니다\.?$',  # "저는 홍길동입니다"
                r'^[가-힣]{2,4}이라고\s*합니다\.?$',  # "홍길동이라고 합니다"
                r'^[가-힣]{2,4}이라고\s*합니다$',  # "홍길동이라고 합니다"
                
                # 닉네임/별명 패턴
                r'^[가-힣a-zA-Z0-9]{2,10}$',  # 2-10자 한글/영문/숫자 조합 (단독 라인)
                r'^[가-힣]{2,4}\s*\([가-힣a-zA-Z0-9]{2,10}\)$',  # "홍길동(닉네임)" 형태
                r'^[가-힣a-zA-Z0-9]{2,10}\s*\([가-힣]{2,4}\)$',  # "닉네임(홍길동)" 형태
                
                # 연락처 관련 (이름 포함)
                r'^[가-힣]{2,4}\s*:\s*[0-9-]+$',  # "홍길동: 010-1234-5678"
                r'^[가-힣]{2,4}\s*[0-9-]+$',  # "홍길동 010-1234-5678"
                
                # 이메일 패턴 (이름 포함)
                r'^[가-힣]{2,4}\s*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # "홍길동@email.com"
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # 일반 이메일
                
                # 생년월일 관련 (이름 포함)
                r'^[가-힣]{2,4}\s*[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일$',  # "홍길동 1995년 3월 15일"
                r'^[가-힣]{2,4}\s*[0-9]{4}-[0-9]{2}-[0-9]{2}$',  # "홍길동 1995-03-15"
                
                # 주소 관련 (이름 포함)
                r'^[가-힣]{2,4}\s*[가-힣\s]+시[가-힣\s]+구[가-힣\s]+동',  # "홍길동 서울시 강남구 역삼동"
                
                # 기타 개인정보 패턴
                r'^이름\s*:\s*[가-힣]{2,4}$',  # "이름: 홍길동"
                r'^성명\s*:\s*[가-힣]{2,4}$',  # "성명: 홍길동"
                r'^닉네임\s*:\s*[가-힣a-zA-Z0-9]{2,10}$',  # "닉네임: nickname"
                r'^별명\s*:\s*[가-힣a-zA-Z0-9]{2,10}$',  # "별명: nickname"
                
                # 개인 식별 정보 (학력 정보는 유지)
                r'^[0-9]{8}만[0-9]{1,2}세$',  # "19961115만26세"
                r'^[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일$',  # "1996년 11월 15일"
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$',  # "1996-11-15"
                r'^만\s*[0-9]{1,2}세$',  # "만 26세"
                r'^[0-9]{1,2}세$',  # "26세"
                
                # 연락처 정보
                r'^[0-9]{3}-[0-9]{4}-[0-9]{4}$',  # "010-1234-5678"
                r'^[0-9]{11}$',  # "01012345678"
                r'^[0-9]{2,3}-[0-9]{3,4}-[0-9]{4}$',  # 전화번호 패턴
                
                # 주소 정보 (개인 주소)
                r'^[가-힣\s]+시[가-힣\s]+구[가-힣\s]+동[0-9-]+$',  # "수원시영통구영통로100한빛아파트101동1000호"
                r'^[가-힣\s]+아파트[0-9-]+동[0-9-]+호$',  # "한빛아파트101동1000호"
                
                # 섹션 제목 (개인정보 관련)
                r'^기본사항$',  # "기본사항"
                r'^개인정보$',  # "개인정보"
                r'^인적사항$',  # "인적사항"
                r'^Personal\s*Information$',  # "Personal Information"
                r'^Persoona1\s*1nformation$',  # OCR 오타 포함
            ]
            
            for pattern in personal_info_patterns:
                if re.match(pattern, text.strip()):
                    return ""  # 빈 문자열로 반환하여 제거
            
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
        4. 특정 단어 오타 수정
        5. 일반적인 OCR 오류 패턴 수정
        6. 개인정보 제거 (이름, 닉네임 등)
        """
        try:
            # 개인정보 제거 (이름, 닉네임 등)
            personal_info_patterns = [
                # 기본 이름 패턴
                r'^[가-힣]{2,4}입니다$',  # "김다슴입니다" 같은 패턴
                r'^[가-힣]{2,4}입니다\.$',  # "김다슴입니다." 같은 패턴
                r'^[가-힣]{2,4}\s*입니다$',  # 공백 포함된 패턴
                r'^[가-힣]{2,4}\s*입니다\.$',  # 공백과 마침표 포함된 패턴
                
                # 지원자 관련 패턴
                r'^지원자$',  # "지원자" 단독
                r'^지질줄모르는지원자$',  # 오타 포함된 지원자
                r'^[가-힣]*지원자[가-힣]*$',  # 지원자 포함된 모든 패턴
                
                # 자기소개 패턴
                r'^[가-힣]{2,4}입니다\.?$',  # "홍길동입니다" 형태
                r'^안녕하세요[가-힣]*입니다\.?$',  # "안녕하세요 홍길동입니다"
                r'^저는[가-힣]{2,4}입니다\.?$',  # "저는 홍길동입니다"
                r'^[가-힣]{2,4}이라고\s*합니다\.?$',  # "홍길동이라고 합니다"
                r'^[가-힣]{2,4}이라고\s*합니다$',  # "홍길동이라고 합니다"
                
                # 닉네임/별명 패턴
                r'^[가-힣a-zA-Z0-9]{2,10}$',  # 2-10자 한글/영문/숫자 조합 (단독 라인)
                r'^[가-힣]{2,4}\s*\([가-힣a-zA-Z0-9]{2,10}\)$',  # "홍길동(닉네임)" 형태
                r'^[가-힣a-zA-Z0-9]{2,10}\s*\([가-힣]{2,4}\)$',  # "닉네임(홍길동)" 형태
                
                # 연락처 관련 (이름 포함)
                r'^[가-힣]{2,4}\s*:\s*[0-9-]+$',  # "홍길동: 010-1234-5678"
                r'^[가-힣]{2,4}\s*[0-9-]+$',  # "홍길동 010-1234-5678"
                
                # 이메일 패턴 (이름 포함)
                r'^[가-힣]{2,4}\s*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # "홍길동@email.com"
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # 일반 이메일
                
                # 생년월일 관련 (이름 포함)
                r'^[가-힣]{2,4}\s*[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일$',  # "홍길동 1995년 3월 15일"
                r'^[가-힣]{2,4}\s*[0-9]{4}-[0-9]{2}-[0-9]{2}$',  # "홍길동 1995-03-15"
                
                # 주소 관련 (이름 포함)
                r'^[가-힣]{2,4}\s*[가-힣\s]+시[가-힣\s]+구[가-힣\s]+동',  # "홍길동 서울시 강남구 역삼동"
                
                # 기타 개인정보 패턴
                r'^이름\s*:\s*[가-힣]{2,4}$',  # "이름: 홍길동"
                r'^성명\s*:\s*[가-힣]{2,4}$',  # "성명: 홍길동"
                r'^닉네임\s*:\s*[가-힣a-zA-Z0-9]{2,10}$',  # "닉네임: nickname"
                r'^별명\s*:\s*[가-힣a-zA-Z0-9]{2,10}$',  # "별명: nickname"
            ]
            
            for pattern in personal_info_patterns:
                if re.match(pattern, text.strip()):
                    return ""  # 빈 문자열로 반환하여 제거
            
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
            
            # 특정 단어 오타 수정
            word_corrections = {
                '요 청': '요청',
                '공픔디': '공공디',
                '공픔디 자인사원': '공공디자인사원',
                '공픔디자인사원': '공공디자인사원',
                '액 스포전시회': '액스포전시회',
                '지질줄모르는지원자': '지원자',
                'M5 Ai': 'M5 AI',
                '111ustrator': 'Illustrator'
            }
            
            for wrong, correct in word_corrections.items():
                text = text.replace(wrong, correct)
            
            # 일반적인 OCR 오류 패턴 수정
            # 1. 공백 오류 (단어 사이 불필요한 공백)
            text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로
            text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', text)  # 한글 사이 공백 제거
            
            # 2. 숫자와 문자 사이 공백 제거
            text = re.sub(r'(\d)\s+([가-힣a-zA-Z])', r'\1\2', text)
            text = re.sub(r'([가-힣a-zA-Z])\s+(\d)', r'\1\2', text)
            
            # 3. 자주 혼동되는 문자 수정
            char_corrections = {
                '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
                '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
                'ｌ': 'l', 'ｌ': 'I', 'Ｏ': 'O', 'Ｓ': 'S', 'Ｚ': 'Z',
                'Ｇ': 'G', 'Ｂ': 'B'
            }
            
            for wrong, correct in char_corrections.items():
                text = text.replace(wrong, correct)
            
            # 4. 한글 자모 오류 수정
            korean_corrections = {
                'ㅇㅏ': '아', 'ㅇㅓ': '어', 'ㅇㅗ': '오', 'ㅇㅜ': '우',
                'ㄱㅏ': '가', 'ㄴㅏ': '나', 'ㄷㅏ': '다', 'ㄹㅏ': '라',
                'ㅁㅏ': '마', 'ㅂㅏ': '바', 'ㅅㅏ': '사', 'ㅈㅏ': '자',
                'ㅊㅏ': '차', 'ㅋㅏ': '카', 'ㅌㅏ': '타', 'ㅍㅏ': '파',
                'ㅎㅏ': '하'
            }
            
            for wrong, correct in korean_corrections.items():
                text = text.replace(wrong, correct)
            
            # 5. 일반적인 단어 오타 수정
            common_corrections = {
                '이력서': '이력서',
                '자기소개서': '자기소개서',
                '경력사항': '경력사항',
                '학력사항': '학력사항',
                '자격사항': '자격사항',
                '어학능력': '어학능력',
                '활동사항': '활동사항',
                '수상내역': '수상내역',
                '프로젝트': '프로젝트',
                '기술스택': '기술스택',
                '대학교': '대학교',
                '전문대학': '전문대학',
                '고등학교': '고등학교',
                '운전면허': '운전면허',
                '운전면허증': '운전면허증'
            }
            
            for wrong, correct in common_corrections.items():
                if wrong in text:
                    text = text.replace(wrong, correct)
            
            # 6. 불필요한 특수문자 제거 (단, 의미있는 특수문자는 유지)
            text = re.sub(r'[^\w\s가-힣\-\.\,\/\(\)]', '', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR 오류 교정 실패: {str(e)}")
            return text

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
                                if cleaned_line:
                                    results.append(cleaned_line)
                                    logger.info(f"추출된 텍스트: {cleaned_line}")
            
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
        """개인정보인지 확인"""
        personal_patterns = [
            r'^[가-힣]{2,4}입니다$',
            r'^지원자$',
            r'^[0-9]{8}만[0-9]{1,2}세$',
            r'^[0-9]{4}\.[0-9]{2}\.[0-9]{2}\s*\(만\s*[0-9]{1,2}세\)$',
            r'^[0-9]{3}-[0-9]{4}-[0-9]{4}$',
            r'^[0-9]{11}$',
            r'^[가-힣\s]+시[가-힣\s]+구[가-힣\s]+동[0-9-]+$',
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
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
        """범용적 이력서 파싱 - 모든 섹션 추출"""
        logger.info("=== 범용 이력서 파싱 시작 ===")
        
        # 학력 관련 키워드 디버깅
        education_keywords = ['대학교', '대학', '전문대학', '학과', '전공', '학사', '석사', '박사', '졸업', '재학', '휴학', '수료', '중퇴', 'GPA', '학점', '평점']
        found_education_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 학력 관련 키워드가 포함된 라인 찾기
            for keyword in education_keywords:
                if keyword in line:
                    found_education_lines.append(f"라인 {i+1}: {line}")
                    break
        
        if found_education_lines:
            logger.info("=== 발견된 학력 관련 라인들 ===")
            for line in found_education_lines:
                logger.info(line)
        else:
            logger.info("학력 관련 키워드를 찾을 수 없습니다.")
        
        # 기존 추출 로직 실행
        self.extract_education_direct(lines, result)
        self.extract_career_direct(lines, result)
        self.extract_certificate_direct(lines, result)
        self.extract_language_direct(lines, result)
        self.extract_activity_direct(lines, result)
        self.extract_skill_direct(lines, result)

    def extract_education_direct(self, lines: List[str], result: Dict):
        """직접적 학력 정보 추출 - 최종학력만 추출"""
        # 전체 텍스트를 하나로 합쳐서 검색
        full_text = " ".join(lines)
        logger.info(f"전체 텍스트에서 학력 검색: {full_text[:200]}...")
        
        # 대학교 패턴 검색 (전체 텍스트에서)
        university_patterns = [
            r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학|컬리지|college|university))',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학))',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학))',
            r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학|컬리지))'
        ]
        
        found_universities = []
        for pattern in university_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                school_name = match.strip()
                if school_name and school_name not in found_universities:
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
            
            # 전공 확인
            major = ""
            major_patterns = [
                r'([가-힣a-zA-Z\s]+(?:학과|전공))',
                r'([가-힣a-zA-Z\s]+(?:학과|전공))\s*[ㅣ|]\s*[가-힣]+'  # "경영학과 ㅣ수원" 형태
            ]
            
            for pattern in major_patterns:
                match = re.search(pattern, full_text)
                if match:
                    major = match.group(1).strip()
                    # 학교명이 포함된 경우 제거
                    major = re.sub(r'^[가-힣]{2,4}\s+', '', major)
                    # 숫자 제거
                    major = re.sub(r'\s*\d+\s*$', '', major)
                    major = major.strip()
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
            
            # 졸업 상태 확인
            status = "졸업"  # 기본값
            status_keywords = {
                '졸업': ['졸업', 'graduate', 'completed', '졸업자'],
                '중퇴': ['중퇴', '중도퇴학', 'dropout', 'withdrawn', '중퇴자'],
                '수료': ['수료', '수료자', 'coursework', 'completed_course'],
                '휴학': ['휴학', '휴학중', 'leave', 'suspended', '휴학자'],
                '재학': ['재학', '재학중', '학생', 'enrolled', 'current', '재학자']
            }
            
            for status_text, keywords in status_keywords.items():
                if any(keyword in full_text for keyword in keywords):
                    status = status_text
                    break
            
            # 중복 확인
            existing_schools = [uni["name"] for uni in result["universities"]]
            if school_name not in existing_schools:
                university_data = {
                    "name": school_name,
                    "degree": degree,
                    "major": major,
                    "status": status,
                    "gpa": gpa,
                    "gpa_max": gpa_max
                }
                
                result["universities"].append(university_data)
                logger.info(f"학력 정보 추출: {school_name} - {degree} - {major} - {status} - GPA: {gpa}/{gpa_max}")
        
        # 기존 라인별 추출 로직도 유지
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 학교명 추출 (고등학교 제외, 대학교/대학/전문대학만)
            school_patterns = [
                r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학|컬리지|college|university))',
                r'([가-힣a-zA-Z\s]+(?:대학교|대학))',
                r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학))',
                r'([가-힣a-zA-Z\s]+(?:대학교|대학|전문대학|컬리지))'
            ]
            
            for pattern in school_patterns:
                match = re.search(pattern, line)
                if match:
                    school_name = match.group(1).strip()
                    
                    # 학위 확인
                    degree = "학사"
                    if any(keyword in line for keyword in ['석사', '대학원', 'master', 'Master']):
                        degree = "석사"
                    elif any(keyword in line for keyword in ['박사', 'doctor', 'Doctor', 'PhD']):
                        degree = "박사"
                    elif any(keyword in line for keyword in ['전문학사', 'associate', 'Associate']):
                        degree = "전문학사"
                    
                    # 전공 확인
                    major = ""
                    if "학과" in line or "전공" in line:
                        # 학과/전공 패턴 찾기
                        major_match = re.search(r'([가-힣a-zA-Z\s]+(?:학과|전공))', line)
                        if major_match:
                            major = major_match.group(1).strip()
                            # 학교명이 포함된 경우 제거 (예: "수원 경영학과" -> "경영학과")
                            # 학교명 패턴 제거 (2-4글자 학교명 + 공백)
                            major = re.sub(r'^[가-힣]{2,4}\s+', '', major)
                            # 숫자 제거 (예: "경영학과 02" -> "경영학과")
                            major = re.sub(r'\s*\d+\s*$', '', major)
                            major = major.strip()
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if "학과" in next_line or "전공" in next_line:
                            major = next_line
                            # 학교명 제거
                            major = re.sub(r'^[가-힣]{2,4}\s+', '', major)
                            # 숫자 제거 (예: "경영학과 02" -> "경영학과")
                            major = re.sub(r'\s*\d+\s*$', '', major)
                            major = major.strip()
                    
                    # 학점(GPA) 추출
                    gpa = 0.0
                    gpa_max = 4.5
                    
                    # 현재 라인에서 GPA 찾기
                    gpa_patterns = [
                        r'GPA\s*:\s*([0-9.]+)',
                        r'학점\s*:\s*([0-9.]+)',
                        r'평점\s*:\s*([0-9.]+)',
                        r'([0-9.]+)\s*\/\s*([0-9.]+)',  # "3.5/4.5" 형태
                        r'([0-9.]+)\s*점',  # "3.5점" 형태
                        r'([0-9.]+)\s*\/\s*([0-9.]+)\s*점'  # "3.5/4.5점" 형태
                    ]
                    
                    for pattern in gpa_patterns:
                        match = re.search(pattern, line)
                        if match:
                            if len(match.groups()) == 1:
                                gpa = float(match.group(1))
                            elif len(match.groups()) == 2:
                                gpa = float(match.group(1))
                                gpa_max = float(match.group(2))
                            break
                    
                    # 다음 라인에서 GPA 찾기
                    if gpa == 0.0 and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        for pattern in gpa_patterns:
                            match = re.search(pattern, next_line)
                            if match:
                                if len(match.groups()) == 1:
                                    gpa = float(match.group(1))
                                elif len(match.groups()) == 2:
                                    gpa = float(match.group(1))
                                    gpa_max = float(match.group(2))
                                break
                    
                    # 졸업 상태 확인 (졸업, 중퇴, 수료, 휴학, 재학)
                    status = "졸업"  # 기본값
                    status_keywords = {
                        '졸업': ['졸업', 'graduate', 'completed', '졸업자'],
                        '중퇴': ['중퇴', '중도퇴학', 'dropout', 'withdrawn', '중퇴자'],
                        '수료': ['수료', '수료자', 'coursework', 'completed_course'],
                        '휴학': ['휴학', '휴학중', 'leave', 'suspended', '휴학자'],
                        '재학': ['재학', '재학중', '학생', 'enrolled', 'current', '재학자']
                    }
                    
                    for status_text, keywords in status_keywords.items():
                        if any(keyword in line for keyword in keywords):
                            status = status_text
                            break
                    
                    # 중복 확인
                    existing_schools = [uni["name"] for uni in result["universities"]]
                    if school_name not in existing_schools:
                        university_data = {
                            "name": school_name,
                            "degree": degree,
                            "major": major,
                            "status": status,  # 졸업 상태 추가
                            "gpa": gpa,
                            "gpa_max": gpa_max
                        }
                        
                        result["universities"].append(university_data)
                        logger.info(f"학력 정보 추출: {school_name} - {degree} - {major} - {status} - GPA: {gpa}/{gpa_max}")
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

    def extract_language_direct(self, lines: List[str], result: Dict):
        """직접적 어학 정보 추출 - 지정된 외국어 시험만 추출"""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 지정된 외국어 시험 패턴들 (업데이트된 목록)
            language_patterns = [
                # 영어 시험
                r'(TOEIC|토익)\s*(\d+)',
                r'(TOEFL|토플)\s*(\d+)',
                r'(TEPS|텝스)\s*(\d+)',
                r'(G-TELP|지텔프)\s*(\d+)',
                r'(FLEX|플렉스)\s*(\d+)',
                r'(OPIc|오픽)\s*([가-힣a-zA-Z]+)',
                r'(TOEIC\s*Speaking|토익\s*스피킹)\s*(\d+)',
                r'(TEPS\s*Speaking|텝스\s*스피킹)\s*(\d+)',
                r'(G-TELP\s*Speaking|지텔프\s*스피킹)\s*(\d+)',
                r'(IELTS|아이엘츠)\s*([0-9.]+)',
                
                # SNULT (독어, 불어, 러시아어, 중국어, 일본어, 스페인어)
                r'(SNULT)\s*([가-힣a-zA-Z]+)',
                r'(SNULT)\s*(\d+)',
                
                # HSK (중국어)
                r'(HSK|중국어능력시험)\s*([가-힣a-zA-Z]+)',
                r'(HSK|중국어능력시험)\s*(\d+)',
                
                # JPT (일본어)
                r'(JPT|일본어능력시험)\s*(\d+)',
                r'(JPT|일본어능력시험)\s*([가-힣a-zA-Z]+)',
                
                # FLEX (독어, 불어, 스페인어, 러시아어, 일본어, 중국어)
                r'(FLEX)\s*([가-힣a-zA-Z]+)',
                r'(FLEX)\s*(\d+)',
                
                # OPIc (중국어, 러시아어, 스페인어, 일본어, 베트남어)
                r'(OPIc)\s*([가-힣a-zA-Z]+)',
                r'(OPIc)\s*(\d+)',
                
                # JLPT (일본어능력시험)
                r'(JLPT|일본어능력시험)\s*([가-힣a-zA-Z]+)',
                r'(JLPT|일본어능력시험)\s*(\d+)',
                
                # TOPIK (한국어능력시험)
                r'(TOPIK|한국어능력시험)\s*([가-힣a-zA-Z]+)',
                r'(TOPIK|한국어능력시험)\s*(\d+)',
                
                # 기타 언어 시험
                r'(DELF|달프)\s*([가-힣a-zA-Z]+)',  # 프랑스어
                r'(DALF|달프)\s*([가-힣a-zA-Z]+)',  # 프랑스어
                r'(TestDaF|테스트다프)\s*([가-힣a-zA-Z]+)',  # 독일어
                r'(Goethe|괴테)\s*([가-힣a-zA-Z]+)',  # 독일어
                r'(DELE|델레)\s*([가-힣a-zA-Z]+)',  # 스페인어
                r'(TORFL|토르플)\s*([가-힣a-zA-Z]+)',  # 러시아어
                r'(CILS|칠스)\s*([가-힣a-zA-Z]+)',  # 이탈리아어
                r'(CELI|첼리)\s*([가-힣a-zA-Z]+)',  # 이탈리아어
                r'(CELPE-Bras|셀프브라스)\s*([가-힣a-zA-Z]+)',  # 포르투갈어
                r'(TOCFL|톡플)\s*([가-힣a-zA-Z]+)',  # 중국어(대만)
                r'(VSTEP|브스텝)\s*([가-힣a-zA-Z]+)',  # 베트남어
            ]
            
            for pattern in language_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    test_name = match.group(1).strip()
                    score = match.group(2).strip()
                    
                    # 점수가 숫자인 경우 점수 형식으로 변환
                    if score.isdigit():
                        if int(score) > 1000:  # TOEIC, TOEFL 등
                            score = f"{score}점"
                        else:  # 기타 점수
                            score = f"{score}점"
                    elif score.replace('.', '').isdigit():  # IELTS 등 소수점 포함
                        score = f"{score}점"
                    
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
                r'([가-힣a-zA-Z\s]+(?:클럽|모임|단체|조직|협회))',
                r'([가-힣a-zA-Z\s]+(?:스튜디오|랩|연구소|센터|아카데미))'
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
            
            # 역할 추출 - 원본 텍스트 그대로 추출
            if current_activity and not current_activity["role"]:
                # 역할 관련 키워드가 포함된 라인 찾기
                role_keywords = [
                    '담당', '책임', '리드', '매니저', '인턴', '정규직', '대표', '참여', '활동',
                    '기획', '운영', '관리', '전시주관', '행사진행', '보조', '어시스턴트',
                    'organizer', 'manager', 'intern', 'regular', 'representative', 'participant', 'leader',
                    'assistant', '담당자', '책임자'
                ]
                
                # 역할 키워드가 포함된 라인을 역할로 설정
                if any(keyword in line for keyword in role_keywords):
                    # 역할만 추출 (숫자, 날짜, 단위 등 제거)
                    role_text = line.strip()
                    
                    # 불필요한 패턴 제거
                    role_text = re.sub(r'\d+', '', role_text)  # 숫자 제거
                    role_text = re.sub(r'개월|월|년|일|시간|분', '', role_text)  # 시간 단위 제거
                    role_text = re.sub(r'[0-9]+', '', role_text)  # 추가 숫자 제거
                    role_text = re.sub(r'\s+', ' ', role_text)  # 연속된 공백을 하나로
                    role_text = role_text.strip()
                    
                    # 빈 문자열이 아닌 경우에만 설정
                    if role_text:
                        current_activity["role"] = role_text
            
            # 수상 내역 추출 (텍스트로)
            if current_activity:
                award_patterns = [
                    r'(수상|우수상|최우수상|대상|금상|은상|동상|특별상|장려상)',
                    r'(award|prize|winner|champion|finalist)',
                    r'(1등|2등|3등|4등|5등)',
                    r'(금|은|동|특별|장려)'
                ]
                
                for pattern in award_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and not current_activity["award"]:
                        award_text = match.group(1).strip()
                        # 수상 내역을 더 구체적으로 표시
                        if award_text in ['1등', '금']:
                            current_activity["award"] = "금상"
                        elif award_text in ['2등', '은']:
                            current_activity["award"] = "은상"
                        elif award_text in ['3등', '동']:
                            current_activity["award"] = "동상"
                        elif award_text in ['특별']:
                            current_activity["award"] = "특별상"
                        elif award_text in ['장려']:
                            current_activity["award"] = "장려상"
                        else:
                            current_activity["award"] = award_text
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