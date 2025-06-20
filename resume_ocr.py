import easyocr
import os
from pdf2image import convert_from_path
import numpy as np
from hanspell import spell_checker
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeOCR:
    def __init__(self):
        """EasyOCR 초기화"""
        try:
            logger.info("EasyOCR 초기화 중...")
            self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
            logger.info("EasyOCR 초기화 완료")
        except Exception as e:
            logger.error(f"EasyOCR 초기화 실패: {str(e)}")
            raise

    def fix_spelling(self, text):
        """한글 맞춤법 검사 및 수정"""
        try:
            result = spell_checker.check(text)
            return result.checked
        except Exception as e:
            logger.warning(f"맞춤법 검사 실패: {str(e)}")
            return text

    def process_pdf(self, pdf_path):
        """PDF 파일에서 텍스트 추출"""
        try:
            # PDF를 이미지로 변환
            logger.info(f"PDF 변환 중: {pdf_path}")
            images = convert_from_path(pdf_path)
            
            all_text = []
            # 각 페이지 처리
            for i, image in enumerate(images):
                logger.info(f"페이지 {i+1} 처리 중...")
                
                # PIL Image를 numpy 배열로 변환
                image_np = np.array(image)
                
                # EasyOCR로 텍스트 추출
                results = self.reader.readtext(image_np)
                
                # 추출된 텍스트 처리
                page_text = []
                for (bbox, text, prob) in results:
                    if prob > 0.5:  # 신뢰도가 50% 이상인 텍스트만 사용
                        # 맞춤법 검사 및 수정
                        corrected_text = self.fix_spelling(text)
                        page_text.append(corrected_text)
                
                all_text.extend(page_text)
            
            return all_text
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {str(e)}")
            raise

def main():
    # ResumeOCR 인스턴스 생성
    ocr = ResumeOCR()
    
    # PDF 파일 경로 입력 받기
    pdf_path = input("이력서 PDF 파일 경로를 입력하세요: ")
    
    if not os.path.exists(pdf_path):
        print("파일이 존재하지 않습니다.")
        return
    
    try:
        # PDF 처리
        print("\n이력서 처리 중...")
        extracted_text = ocr.process_pdf(pdf_path)
        
        # 결과 출력
        print("\n=== 추출된 텍스트 ===")
        for i, text in enumerate(extracted_text, 1):
            print(f"{i}. {text}")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 