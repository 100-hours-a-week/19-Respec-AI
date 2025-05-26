from typing import List, Tuple, Optional
import easyocr
import cv2
import numpy as np
from PIL import Image

class OCRModel:
    def __init__(self, languages: List[str] = ['ko', 'en']):
        """
        OCR 모델 초기화
        
        Args:
            languages: 인식할 언어 리스트 (기본값: ['ko', 'en'])
        """
        self.reader = easyocr.Reader(languages)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        OCR 성능 향상을 위한 이미지 전처리
        
        Args:
            image: 입력 이미지 (numpy array)
            
        Returns:
            전처리된 이미지
        """
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Otsu's 이진화로 텍스트 강조
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary

    def detect_text(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        이미지에서 텍스트 검출 및 인식
        
        Args:
            image: 입력 이미지 (numpy array)
            
        Returns:
            검출된 텍스트 정보 리스트 [(bbox, text, confidence), ...]
        """
        # 이미지 전처리
        processed_image = self.preprocess_image(image)
        
        # OCR 수행
        results = self.reader.readtext(processed_image)
        
        return results

    def detect_text_from_pil(self, pil_image: Image.Image) -> List[Tuple[List[List[int]], str, float]]:
        """
        PIL Image에서 텍스트 검출 및 인식
        
        Args:
            pil_image: PIL Image 객체
            
        Returns:
            검출된 텍스트 정보 리스트 [(bbox, text, confidence), ...]
        """
        # PIL Image를 numpy array로 변환
        image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return self.detect_text(image_np)