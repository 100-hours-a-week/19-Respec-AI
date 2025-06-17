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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRModel:
    def __init__(self):
        """EasyOCR 모델 초기화"""
        logger.info("EasyOCR 모델 초기화 중...")
        self.reader = easyocr.Reader(['ko', 'en'])
        logger.info("EasyOCR 모델 초기화 완료")

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

    def parse_resume(self, ocr_results: List[Dict]) -> Dict:
        """이력서 파싱"""
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

        current_section = None
        section_text = []
        
        # 정규표현식 패턴
        patterns = {
            "university": {
                "name": r'([가-힣a-zA-Z\s]+대학교|대학)',
                "degree": r'(수료|전문학사|학사|석사|박사)',
                "major": r'([가-힣a-zA-Z\s]+(학과|전공))',
                "status": r'(졸업|재학|휴학|수료)',
                "date": r'(\d{4})[년\.](\d{1,2})[월\.](\d{1,2})[일]?'
            },
            "career": {
                "company": r'([가-힣a-zA-Z\s]+(주식회사|기업|회사|기관|㈜))',
                "role": r'([가-힣a-zA-Z\s]+(직원|사원|인턴|대리|과장|차장|부장|이사|팀장|매니저))',
                "duration": r'(\d{4})[년\.](\d{1,2})[월\.](\d{1,2})[일]?\s*[-~]\s*(\d{4})[년\.](\d{1,2})[월\.](\d{1,2})[일]?'
            },
            "language": {
                "test": r'(TOEIC|TOEFL|TEPS|G-TELP|FLEX|OPIc|TOEIC Speaking|TEPS Speaking|G-TELP Speaking|IELTS|SNULT|HSK|JPT)',
                "score": r'(\d+|[A-Z]+[0-9]*)',
                "date": r'(\d{4})[년\.](\d{1,2})[월\.](\d{1,2})[일]?'
            }
        }

        # Process each text item
        for item in ocr_results:
            text = item['text'].strip()
            if not text:
                continue

            # Check if this is a section header
            section = self.is_section_header(text)
            if section:
                # Process previous section if exists
                if current_section and section_text:
                    self.process_section(current_section, section_text, result, patterns)
                current_section = section
                section_text = []
            elif current_section:
                section_text.append(text)

        # Process the last section
        if current_section and section_text:
            self.process_section(current_section, section_text, result, patterns)

        return result

    def is_section_header(self, text: str) -> str:
        """섹션 헤더 식별"""
        sections = {
            "학력": ["학력", "교육", "학교", "전공", "학위", "education", "academic"],
            "경력": ["경력", "직무경험", "업무경험", "인턴십", "경험", "근무", "career", "experience", "work"],
            "자격증": ["자격증", "자격", "면허", "수료증", "certificate", "license"],
            "어학": ["어학", "외국어", "language", "toeic", "토익", "토플", "toefl", "ielts"],
            "활동": ["활동", "대외활동", "프로젝트", "동아리", "봉사활동", "activity", "project"],
            "기술": ["기술", "스킬", "skill", "기술스택", "tech stack"],
            "수상": ["수상", "수상경력", "award", "prize"],
            "자기소개": ["자기소개", "소개", "introduction", "about"]
        }
        
        text = text.lower()
        for section, keywords in sections.items():
            if any(keyword.lower() in text for keyword in keywords):
                return section
        return ""

    def process_section(self, section: str, texts: List[str], result: Dict, patterns: Dict):
        """섹션 처리"""
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
        elif section == "기술":
            self.process_skills(texts, result, patterns)
        elif section == "수상":
            self.process_awards(texts, result, patterns)
        elif section == "자기소개":
            result["introduction"] = " ".join(texts)

    def process_education(self, texts: List[str], result: Dict, patterns: Dict):
        """학력 정보 처리"""
        current_edu = {}
        for text in texts:
            # 대학교명
            if re.search(patterns["university"]["name"], text):
                if current_edu:
                    result["universities"].append(current_edu)
                current_edu = {"name": text.strip()}
            # 학위
            elif re.search(patterns["university"]["degree"], text):
                current_edu["degree"] = text.strip()
            # 전공
            elif re.search(patterns["university"]["major"], text):
                current_edu["major"] = text.strip()
            # 상태
            elif re.search(patterns["university"]["status"], text):
                current_edu["status"] = text.strip()
            # 날짜
            elif re.search(patterns["university"]["date"], text):
                date = self.normalize_date(text)
                if date:
                    current_edu["date"] = date

        if current_edu:
            result["universities"].append(current_edu)

    def process_career(self, texts: List[str], result: Dict, patterns: Dict):
        """경력 정보 처리"""
        current_career = {}
        for text in texts:
            # 회사명
            if re.search(patterns["career"]["company"], text):
                if current_career:
                    result["careers"].append(current_career)
                current_career = {"company": text.strip()}
            # 직무
            elif re.search(patterns["career"]["role"], text):
                current_career["role"] = text.strip()
            # 기간
            elif re.search(patterns["career"]["duration"], text):
                dates = re.findall(r'(\d{4})[년\.](\d{1,2})[월\.](\d{1,2})[일]?', text)
                if len(dates) >= 2:
                    start_date = f"{dates[0][0]}-{dates[0][1].zfill(2)}-{dates[0][2].zfill(2)}"
                    end_date = f"{dates[1][0]}-{dates[1][1].zfill(2)}-{dates[1][2].zfill(2)}"
                    current_career["start_date"] = start_date
                    current_career["end_date"] = end_date

        if current_career:
            result["careers"].append(current_career)

    def process_certificates(self, texts: List[str], result: Dict, patterns: Dict):
        """자격증 정보 처리"""
        for text in texts:
            cert = {"name": text.strip()}
            date = self.normalize_date(text)
            if date:
                cert["date"] = date
            result["certificates"].append(cert)

    def process_languages(self, texts: List[str], result: Dict, patterns: Dict):
        """어학 정보 처리"""
        current_lang = {}
        for text in texts:
            # 시험 종류
            if re.search(patterns["language"]["test"], text):
                if current_lang:
                    result["languages"].append(current_lang)
                current_lang = {"test": text.strip()}
            # 점수
            elif re.search(patterns["language"]["score"], text):
                current_lang["score"] = text.strip()
            # 날짜
            elif re.search(patterns["language"]["date"], text):
                date = self.normalize_date(text)
                if date:
                    current_lang["date"] = date

        if current_lang:
            result["languages"].append(current_lang)

    def process_activities(self, texts: List[str], result: Dict, patterns: Dict):
        """활동 정보 처리"""
        current_activity = {"description": " ".join(texts)}
        date = self.normalize_date(" ".join(texts))
        if date:
            current_activity["date"] = date
        result["activities"].append(current_activity)

    def process_skills(self, texts: List[str], result: Dict, patterns: Dict):
        """기술 스택 처리"""
        for text in texts:
            skills = [skill.strip() for skill in text.split(',')]
            result["skills"].extend(skills)

    def process_awards(self, texts: List[str], result: Dict, patterns: Dict):
        """수상 정보 처리"""
        current_award = {"description": " ".join(texts)}
        date = self.normalize_date(" ".join(texts))
        if date:
            current_award["date"] = date
        result["awards"].append(current_award)

    def normalize_date(self, date_str: str) -> Optional[str]:
        """날짜 형식 정규화"""
        try:
            # 다양한 날짜 형식 처리
            date_patterns = [
                r'(\d{4})[년\.](\d{1,2})[월\.](\d{1,2})[일]?',
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
                r'(\d{4})(\d{2})(\d{2})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    year, month, day = match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # 상대적 날짜 처리 (예: "현재", "~")
            if date_str in ["현재", "~", "present", "now"]:
                return datetime.now().strftime("%Y-%m-%d")
                
            return None
        except Exception as e:
            logger.warning(f"날짜 변환 실패: {str(e)}")
            return None 