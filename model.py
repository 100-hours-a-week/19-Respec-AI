import os
import fitz
from PIL import Image
import easyocr
import re

class ResumeOCRModel:
    """
    이력서 PDF → 텍스트 → 영역별 점수 계산 (EasyOCR 기반)
    """
    def __init__(self, lang=['ko', 'en'], min_confidence=0.7, dpi=300):
        self.reader = easyocr.Reader(lang)
        self.min_confidence = min_confidence
        self.dpi = dpi

    def pdf_to_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        image_paths = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_path = f"{os.path.splitext(pdf_path)[0]}_page_{i+1}.png"
            img.save(img_path)
            image_paths.append(img_path)
        return image_paths

    def ocr_images(self, image_paths):
        all_text = []
        for img_path in image_paths:
            results = self.reader.readtext(img_path)
            for _, text, confidence in results:
                if confidence >= self.min_confidence:
                    all_text.append(text)
        return all_text

    def analyze(self, pdf_path):
        image_paths = self.pdf_to_images(pdf_path)
        text_lines = self.ocr_images(image_paths)
        analysis = self.score_resume(text_lines)
        return analysis

    def score_resume(self, text_lines):
        nickname = None
        for line in text_lines:
            if "이름" in line or "name" in line.lower():
                nickname = line.split(":")[-1].strip()
                break
        if not nickname:
            for line in text_lines:
                match = re.search(r"[가-힣]{2,}|[a-zA-Z]{2,}", line)
                if match:
                    nickname = match.group()
                    break

        academic_keywords = ["대학교", "University", "학과", "전공", "학점"]
        academicScore = sum(any(k in line for k in academic_keywords) for line in text_lines) * 10

        work_keywords = ["근무", "회사", "경력", "인턴", "직장"]
        workExperienceScore = sum(any(k in line for k in work_keywords) for line in text_lines) * 10

        cert_keywords = ["자격증", "기사", "Certificate", "Engineer", "취득"]
        certificationScore = sum(any(k in line for k in cert_keywords) for line in text_lines) * 10

        lang_keywords = ["TOEIC", "TOEFL", "OPIc", "JLPT", "HSK", "영어", "일본어", "중국어"]
        languageProficiencyScore = sum(any(k in line for k in lang_keywords) for line in text_lines) * 10

        extra_keywords = ["동아리", "활동", "봉사", "공모전", "수상", "프로젝트"]
        extracurricularScore = sum(any(k in line for k in extra_keywords) for line in text_lines) * 10

        # 점수 상한선(100점 만점 등)
        academicScore = min(academicScore, 100)
        workExperienceScore = min(workExperienceScore, 100)
        certificationScore = min(certificationScore, 100)
        languageProficiencyScore = min(languageProficiencyScore, 100)
        extracurricularScore = min(extracurricularScore, 100)

        totalScore = round((academicScore + workExperienceScore + certificationScore +
                            languageProficiencyScore + extracurricularScore) / 5, 2)

        return {
            "nickname": nickname or "",
            "academicScore": academicScore,
            "workExperienceScore": workExperienceScore,
            "certificationScore": certificationScore,
            "languageProficiencyScore": languageProficiencyScore,
            "extracurricularScore": extracurricularScore,
            "totalScore": totalScore
        }