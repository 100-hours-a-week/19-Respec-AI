import os
import easyocr
import numpy as np
import cv2
import fitz
from PIL import Image
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResumeTextExtractor:
    def __init__(self, model_path: str = './models'):
        """
        이력서 텍스트 추출기 초기화
        Args:
            model_path: EasyOCR 모델 저장 경로
        """
        try:
            # 모델 경로 설정
            self.model_path = os.path.abspath(model_path)
            os.makedirs(self.model_path, exist_ok=True)
            
            logger.info(f"EasyOCR 모델 초기화 중... (경로: {self.model_path})")
            
            # EasyOCR 리더 초기화
            self.reader = easyocr.Reader(
                ['ko', 'en'],  # 한국어, 영어 지원
                gpu=False,     # CPU 모드 사용
                model_storage_directory=self.model_path,
                download_enabled=True,  # 필요한 모델 자동 다운로드
                recog_network='korean_g2',  # 한국어 인식 모델
                verbose=False
            )
            
            # 모델 파일 확인
            model_files = os.listdir(self.model_path)
            logger.info(f"모델 파일 목록: {model_files}")
            
            logger.info("EasyOCR 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"EasyOCR 초기화 실패: {str(e)}")
            raise RuntimeError(f"EasyOCR 모델 초기화 실패: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDF에서 텍스트 추출
        Args:
            pdf_path: PDF 파일 경로
        Returns:
            추출된 텍스트 리스트
        """
        try:
            # PDF 파일 열기
            doc = fitz.open(pdf_path)
            all_texts = []

            # 각 페이지 처리
            for page_num in range(len(doc)):
                page = doc[page_num]
                # 고해상도로 이미지 추출 (300 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # 이미지로 변환
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                # 이미지 전처리
                processed_img = self._preprocess_image(img_np)
                
                # 텍스트 추출
                try:
                    texts = self.reader.readtext(
                        processed_img,
                        paragraph=True,  # 문단 단위로 추출
                        batch_size=8,    # 배치 크기
                        detail=0,        # 텍스트만 반환
                        contrast_ths=0.3,  # 대비 임계값
                        adjust_contrast=0.5,  # 대비 조정
                        width_ths=0.7,   # 너비 임계값
                        height_ths=0.7,  # 높이 임계값
                    )
                    
                    # 결과 저장
                    for text in texts:
                        if text.strip():  # 빈 텍스트 제외
                            all_texts.append({
                                'text': text,
                                'page': page_num + 1
                            })
                            
                except Exception as e:
                    logger.warning(f"페이지 {page_num + 1} 텍스트 추출 실패: {str(e)}")
                    continue

            return all_texts
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {str(e)}")
            raise RuntimeError(f"PDF 처리 실패: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리
        Args:
            image: 원본 이미지
        Returns:
            전처리된 이미지
        """
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 이진화 (적응형 임계값)
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # 블록 크기
                2    # 상수
            )
            
            return binary
            
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {str(e)}")
            return image  # 전처리 실패 시 원본 반환

    def extract_text_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 추출
        Args:
            image: 이미지 배열
        Returns:
            추출된 텍스트 정보 리스트
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("유효하지 않은 이미지입니다.")

            processed_image = self._preprocess_image(image)
            
            results = self.reader.readtext(
                processed_image,
                paragraph=True,
                batch_size=8,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                width_ths=0.7,
                height_ths=0.7,
                slope_ths=0.2,
                ycenter_ths=0.5,
                add_margin=0.1,
                text_threshold=0.6,
                link_threshold=0.4,
                low_text=0.4,
                canvas_size=1280,
                mag_ratio=1.5
            )
            
            extracted_texts = []
            for result in results:
                bbox, text, conf = result
                text = self._clean_text(text)
                
                if conf > 0.3 and text:
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
        text = text.strip()
        text = ' '.join(text.split())
        
        return text 