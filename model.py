import os
import easyocr
import numpy as np
from PIL import Image
import cv2
import re
import logging
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeTextExtractor:
    def __init__(self, model_path: str = './models'):
        """
        이력서 텍스트 추출기 초기화
        Args:
            model_path: EasyOCR 모델 저장 경로
        """
        try:
            # 모델 디렉토리 생성
            os.makedirs(model_path, exist_ok=True)
            
            logger.info("EasyOCR 모델 초기화 중...")
            self.reader = easyocr.Reader(
                ['ko', 'en'],
                gpu=False,
                model_storage_directory=model_path,
                download_enabled=True,
                recog_network='korean_g2',  # 한국어 모델 변경
                detector=True,
                recognizer=True,
                verbose=True
            )
            logger.info("EasyOCR 모델 초기화 완료")
        except Exception as e:
            logger.error(f"EasyOCR 초기화 실패: {str(e)}")
            raise RuntimeError(f"EasyOCR 모델 초기화 실패: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리
        Args:
            image: 원본 이미지
        Returns:
            전처리된 이미지
        """
        try:
            # 이미지 유효성 검사
            if image is None or image.size == 0:
                raise ValueError("유효하지 않은 이미지입니다.")

            # 이미지 크기 조정 (너무 큰 이미지 처리)
            max_dimension = 2000
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                image = cv2.resize(image, None, fx=scale, fy=scale)

            # RGB to 그레이스케일
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # 이진화
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 모폴로지 연산으로 텍스트 선명도 향상
            kernel = np.ones((1,1), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            return morph
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {str(e)}")
            raise RuntimeError(f"이미지 전처리 실패: {str(e)}")

    def extract_text_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 추출
        Args:
            image: 이미지 배열
        Returns:
            추출된 텍스트 정보 리스트
        """
        try:
            # 이미지 유효성 검사
            if image is None or image.size == 0:
                raise ValueError("유효하지 않은 이미지입니다.")

            # 이미지 전처리
            processed_image = self._preprocess_image(image)
            
            # 텍스트 추출 (최적화된 파라미터)
            results = self.reader.readtext(
                processed_image,
                paragraph=True,
                batch_size=8,  # 배치 크기 증가
                contrast_ths=0.1,  # 대비 임계값 낮춤
                adjust_contrast=0.5,
                width_ths=0.7,  # 너비 임계값 증가
                height_ths=0.7,  # 높이 임계값 증가
                slope_ths=0.2,  # 기울기 임계값 추가
                ycenter_ths=0.5,  # y축 중심 임계값 추가
                add_margin=0.1,  # 여백 추가
                text_threshold=0.6,  # 텍스트 감지 임계값
                link_threshold=0.4,  # 텍스트 연결 임계값
                low_text=0.4,  # 낮은 텍스트 임계값
                canvas_size=1280,  # 캔버스 크기 설정
                mag_ratio=1.5  # 확대 비율
            )
            
            extracted_texts = []
            for result in results:
                if len(result) == 3:
                    bbox, text, conf = result
                else:
                    bbox, text = result
                    conf = 0.5
                
                # 텍스트 정제
                text = self._clean_text(text)
                
                if conf > 0.3 and text:  # 신뢰도 필터링 및 빈 텍스트 제외
                    extracted_texts.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': bbox
                    })
            
            if not extracted_texts:
                logger.warning("추출된 텍스트가 없습니다.")
            
            return extracted_texts
        except Exception as e:
            logger.error(f"텍스트 추출 실패: {str(e)}")
            raise RuntimeError(f"텍스트 추출 실패: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """
        추출된 텍스트 정제
        Args:
            text: 원본 텍스트
        Returns:
            정제된 텍스트
        """
        if not isinstance(text, str):
            return ""
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정리 (한글, 영어, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?()\-]', '', text)
        
        # 연속된 점(.) 제거
        text = re.sub(r'\.+', '.', text)
        
        # 불필요한 공백 제거
        text = text.strip()
        
        return text

class ResumeSpecExtractor:
    def __init__(self):
        """스펙 추출기 초기화"""
        self.text_patterns = {
            'education': [
                r'([가-힣]+대학[교]?).*?([가-힣]+(?:학과|전공|학부))',
                r'학점\s*?[:]?\s*(\d+\.?\d*)\s*?[/]?\s*(\d+\.?\d*)',
                r'GPA\s*?[:]?\s*(\d+\.?\d*)\s*?[/]?\s*(\d+\.?\d*)',
                r'평점\s*?[:]?\s*(\d+\.?\d*)\s*?[/]?\s*(\d+\.?\d*)',
                r'(학사|석사|박사|학위)'
            ],
            'experience': [
                r'([가-힣a-zA-Z0-9]+(?:도서관|주식회사|회사|기업|㈜)).*?(\d+년\s*\d*개월|\d+개월)',
                r'(인턴|대리|과장|차장|부장|팀장|매니저|주임|사원)',
                r'근무[기간]?\s*:?\s*(\d+)\s*(?:개월|년)',
                r'([가-힣a-zA-Z0-9]+(?:팀|부서|센터|연구소|실|본부))'
            ],
            'job_field': [
                r'(인터넷_IT|마케팅_광고_홍보|무역_유통|생산_제조|영업_고객상담|디자인|미디어)',
                r'지원\s*분야\s*:?\s*([가-힣a-zA-Z0-9_]+)'
            ],
            'certificates': [
                r'([가-힣a-zA-Z0-9]+(?:기사|기술사|자격증|1급|2급|산업기사))',
                r'(JLPT|TOEIC|TOEFL|OPIC|HSK)(?:\s*[\d]+급|\s*[\d]+점)?'
            ],
            'languages': [
                r'(TOEIC|TOEFL|OPIC|JLPT|HSK)\s*(?:점수|:)?\s*(\d{3,4})',
                r'([가-힣]+어)\s*(?:능력|실력|수준)\s*:\s*(?:상|중|하)'
            ],
            'nickname': [
                r'닉네임\s*:?\s*([a-zA-Z0-9가-힣]+)',
                r'이름\s*:?\s*([a-zA-Z0-9가-힣]+)'
            ]
        }

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?()\-]', '', text)
        return text.strip()

    def _extract_pattern_matches(self, text: str, patterns: List[str]) -> List[Dict[str, Any]]:
        """패턴 매칭 결과 추출"""
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'text': match.group(0),
                    'groups': match.groups(),
                    'span': match.span()
                })
        return matches

    def _find_nearby_text(self, text: str, start: int, end: int, window: int = 50) -> str:
        """주변 텍스트 찾기"""
        start = max(0, start - window)
        end = min(len(text), end + window)
        return text[start:end]

    def extract_specs(self, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        텍스트에서 스펙 정보 추출
        Args:
            texts: 추출된 텍스트 리스트
        Returns:
            스펙 정보
        """
        try:
            # 전체 텍스트 결합
            full_text = ' '.join([item['text'] for item in texts])
            full_text = self._clean_text(full_text)
            
            specs = {
                'nickname': None,
                'education': [],
                'experience': [],
                'job_field': None,
                'certificates': [],
                'languages': [],
                'confidence_scores': {}  # 신뢰도 점수 추가
            }
            
            # 닉네임 추출
            for pattern in self.text_patterns['nickname']:
                match = re.search(pattern, full_text)
                if match:
                    specs['nickname'] = match.group(1)
                    break
            
            # 학력 정보 추출
            edu_info = {
                'school': None,
                'major': None,
                'degree': None,
                'gpa': None
            }
            
            for pattern in self.text_patterns['education']:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    if '대학' in match.group(0):
                        edu_info['school'] = match.group(1)
                        if len(match.groups()) > 1:
                            edu_info['major'] = match.group(2)
                    elif '학점' in match.group(0) or 'GPA' in match.group(0) or '평점' in match.group(0):
                        if len(match.groups()) >= 2:
                            edu_info['gpa'] = f"{match.group(1)}/{match.group(2)}"
                    elif '학위' in match.group(0):
                        edu_info['degree'] = match.group(1)
            
            if any(edu_info.values()):
                specs['education'].append(edu_info)
            
            # 경력 정보 추출
            exp_info = {
                'company': None,
                'position': None,
                'duration': None,
                'department': None
            }
            
            for pattern in self.text_patterns['experience']:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    if '도서관' in match.group(0) or '회사' in match.group(0):
                        exp_info['company'] = match.group(1)
                    elif '인턴' in match.group(0) or '사원' in match.group(0):
                        exp_info['position'] = match.group(1)
                    elif '개월' in match.group(0):
                        exp_info['duration'] = match.group(1) + '개월'
                    elif '팀' in match.group(0) or '부서' in match.group(0):
                        exp_info['department'] = match.group(1)
            
            if any(exp_info.values()):
                specs['experience'].append(exp_info)
            
            # 지원 분야 추출
            for pattern in self.text_patterns['job_field']:
                match = re.search(pattern, full_text)
                if match:
                    specs['job_field'] = match.group(1)
                    break
            
            # 자격증 추출
            for pattern in self.text_patterns['certificates']:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    specs['certificates'].append(match.group(0))
            
            # 어학 능력 추출
            for pattern in self.text_patterns['languages']:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    specs['languages'].append(match.group(0))
            
            # 신뢰도 점수 계산
            for field, patterns in self.text_patterns.items():
                matches = self._extract_pattern_matches(full_text, patterns)
                if matches:
                    specs['confidence_scores'][field] = len(matches) / len(patterns)
            
            return specs
            
        except Exception as e:
            logger.error(f"스펙 추출 실패: {str(e)}")
            raise RuntimeError(f"스펙 추출 실패: {str(e)}")

def extract_specs_from_pdf(pdf_path: str, model_path: str = './models') -> Dict[str, Any]:
    """
    PDF 파일에서 스펙 추출
    Args:
        pdf_path: PDF 파일 경로
        model_path: EasyOCR 모델 경로
    Returns:
        추출된 스펙 정보
    """
    try:
        # PDF 파일 존재 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # PDF 파일 열기
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"PDF 파일을 열 수 없습니다: {str(e)}")

        # PDF 페이지 수 확인
        if doc.page_count == 0:
            raise ValueError("PDF 파일에 페이지가 없습니다.")

        # PDF를 이미지로 변환
        images = []
        for page in doc:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(np.array(img))
            except Exception as e:
                logger.warning(f"페이지 {page.number + 1} 변환 실패: {str(e)}")
                continue

        if not images:
            raise ValueError("PDF를 이미지로 변환할 수 없습니다.")

        # 텍스트 추출기 초기화
        text_extractor = ResumeTextExtractor(model_path)
        
        # 각 이미지에서 텍스트 추출
        all_texts = []
        for i, img in enumerate(images):
            try:
                texts = text_extractor.extract_text_from_image(img)
                all_texts.extend(texts)
            except Exception as e:
                logger.warning(f"페이지 {i + 1} 텍스트 추출 실패: {str(e)}")
                continue

        if not all_texts:
            raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")

        # 스펙 추출
        spec_extractor = ResumeSpecExtractor()
        specs = spec_extractor.extract_specs(all_texts)
        
        return {
            'status': 'success',
            'specs': specs,
            'page_count': len(images),
            'extracted_text_count': len(all_texts)
        }
        
    except Exception as e:
        logger.error(f"PDF 처리 실패: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        # 리소스 정리
        if 'doc' in locals():
            doc.close()