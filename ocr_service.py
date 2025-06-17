import easyocr
import requests
import tempfile
import os
from typing import Dict, List, Optional, Tuple
import re
import logging
import traceback
from PIL import Image
import io
import numpy as np
from pdf2image import convert_from_path
import mimetypes
from datetime import datetime
from dateutil import parser
from hanspell import spell_checker
from koreanize import koreanize
import unicodedata

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        """텍스트 정규화"""
        # 유니코드 정규화
        text = unicodedata.normalize('NFKC', text)
        # 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 처리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        return text.strip()

    @staticmethod
    def fix_korean_spelling(text: str) -> str:
        """한글 맞춤법 검사 및 수정"""
        try:
            result = spell_checker.check(text)
            return result.checked
        except Exception as e:
            logger.warning(f"맞춤법 검사 실패: {str(e)}")
            return text

    @staticmethod
    def normalize_date(date_str: str) -> Optional[str]:
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

class OCRService:
    def __init__(self):
        try:
            logger.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
            self.text_processor = TextProcessor()
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OCR: {str(e)}")
            raise Exception(f"Failed to initialize OCR: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 기본 정규화
        text = self.text_processor.normalize_text(text)
        # 한글 맞춤법 검사
        text = self.text_processor.fix_korean_spelling(text)
        return text

    def process_ocr_results(self, results: List[Tuple]) -> List[Dict]:
        """OCR 결과 처리 및 전처리"""
        processed_results = []
        for (bbox, text, prob) in results:
            if prob < 0.5:  # 신뢰도가 낮은 텍스트는 건너뛰기
                continue
                
            # 텍스트 전처리
            processed_text = self.preprocess_text(text)
            if not processed_text:
                continue
                
            # 위치 정보 계산
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            
            processed_results.append({
                'text': processed_text,
                'bbox': bbox,
                'confidence': prob,
                'center_y': center_y,
                'center_x': center_x
            })
        
        # 수직 위치로 정렬
        processed_results.sort(key=lambda x: x['center_y'])
        return processed_results

    def is_section_header(self, text: str) -> str:
        """섹션 헤더 식별 개선"""
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

    def parse_resume(self, ocr_results: List[Dict]) -> Dict:
        """이력서 파싱 개선"""
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
        
        # 정규표현식 패턴 개선
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

    def process_section(self, section: str, texts: List[str], result: Dict, patterns: Dict):
        """섹션 처리 개선"""
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
                date = self.text_processor.normalize_date(text)
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
            date = self.text_processor.normalize_date(text)
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
                date = self.text_processor.normalize_date(text)
                if date:
                    current_lang["date"] = date

        if current_lang:
            result["languages"].append(current_lang)

    def process_activities(self, texts: List[str], result: Dict, patterns: Dict):
        """활동 정보 처리"""
        current_activity = {"description": " ".join(texts)}
        date = self.text_processor.normalize_date(" ".join(texts))
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
        date = self.text_processor.normalize_date(" ".join(texts))
        if date:
            current_award["date"] = date
        result["awards"].append(current_award)

    def process_resume(self, file_url: str) -> Dict:
        """이력서 처리 메인 함수"""
        try:
            # 파일 다운로드
            file_path, ext = self.download_file(file_url)
            
            # 파일 처리
            if ext.lower() == '.pdf':
                results = self.process_pdf(file_path)
            else:
                results = self.process_image(file_path)
            
            # OCR 결과 처리
            processed_results = self.process_ocr_results(results)
            
            # 이력서 파싱
            resume_data = self.parse_resume(processed_results)
            
            # 임시 파일 삭제
            os.unlink(file_path)
            
            return resume_data
            
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to process resume: {str(e)}")

    def download_file(self, url: str) -> tuple:
        """Download file from URL and save to temporary file"""
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception("Failed to download file")
            
            # Get content type
            content_type = response.headers.get('content-type', '')
            
            # Determine file extension
            if 'pdf' in content_type:
                ext = '.pdf'
            elif 'image' in content_type:
                ext = mimetypes.guess_extension(content_type) or '.png'
            else:
                raise Exception("Unsupported file type")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name, ext
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise Exception(f"Failed to download file: {str(e)}")

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process PDF file and extract text"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            all_results = []
            for img in images:
                # Convert PIL Image to numpy array
                img_np = np.array(img)
                
                # Extract text using EasyOCR
                results = self.reader.readtext(img_np)
                
                # Process results
                for (bbox, text, prob) in results:
                    # Calculate center y-coordinate for vertical position
                    center_y = (bbox[0][1] + bbox[2][1]) / 2
                    all_results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': prob,
                        'center_y': center_y
                    })
            
            # Sort by vertical position
            all_results.sort(key=lambda x: x['center_y'])
            
            return all_results
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")

    def process_image(self, image_path: str) -> List[Dict]:
        """Process image file and extract text"""
        try:
            # Read the image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Extract text using EasyOCR
            results = self.reader.readtext(image_np)
            
            # Process results
            processed_results = []
            for (bbox, text, prob) in results:
                # Calculate center y-coordinate for vertical position
                center_y = (bbox[0][1] + bbox[2][1]) / 2
                processed_results.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': prob,
                    'center_y': center_y
                })
            
            # Sort by vertical position
            processed_results.sort(key=lambda x: x['center_y'])
            
            return processed_results
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"Failed to process image: {str(e)}") 