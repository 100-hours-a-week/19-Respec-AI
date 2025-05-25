from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, time
import uvicorn
import easyocr
import pdf2image
import cv2
import numpy as np
from io import BytesIO
import tempfile

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="Spec Score OCR API V2")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR 초기화
reader = easyocr.Reader(['ko', 'en'])

# ──────────────────────────
# Pydantic 모델 정의
# ──────────────────────────
class OCRResult(BaseModel):
    page: int
    text: str
    confidence: float
    position: List[List[int]]

class OCRResponse(BaseModel):
    success: bool
    message: str
    elapsed_time: float
    results: List[List[OCRResult]]

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_type: Optional[str] = None

# Route for the test page
@app.get("/", response_class=HTMLResponse)
async def get_test_page(request: Request):
    """
    OCR 테스트를 위한 웹 인터페이스를 제공합니다.
    """
    with open("templates/ocr_test.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/v2/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(file: UploadFile = File(...)):
    """
    PDF 문서에서 텍스트를 추출하여 반환합니다.
    
    - **file**: PDF 파일 (multipart/form-data)
    - **returns**: 페이지별 OCR 결과와 텍스트 위치 정보
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail={"success": False, "message": "PDF 파일만 업로드 가능합니다."}
        )
    
    try:
        start_time = time.time()
        
        # 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        # PDF를 이미지로 변환
        images = pdf2image.convert_from_path(temp_pdf_path)
        results = []

        for i, image in enumerate(images):
            # 이미지 전처리
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
            image_np = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # OCR 수행
            detections = reader.readtext(image_np)
            
            # 결과 변환
            page_results = []
            for bbox, text, confidence in detections:
                result = OCRResult(
                    page=i+1,
                    text=text,
                    confidence=confidence,
                    position=[[int(x) for x in point] for point in bbox]
                )
                page_results.append(result)
            
            results.append(page_results)

        # 임시 파일 삭제
        os.unlink(temp_pdf_path)
        
        elapsed_time = time.time() - start_time
        print(f"[OCR] PDF 처리 완료: {file.filename}, 소요 시간: {elapsed_time:.2f}초")
        
        return OCRResponse(
            success=True,
            message="PDF 처리가 완료되었습니다.",
            elapsed_time=elapsed_time,
            results=results
        )

    except Exception as e:
        print(f"[OCR] 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "PDF 처리 중 오류가 발생했습니다.",
                "error_type": str(type(e).__name__)
            }
        )

@app.post("/api/v2/ocr/image", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    """
    이미지에서 텍스트를 추출하여 반환합니다.
    
    - **file**: 이미지 파일 (PNG, JPG, JPEG, TIFF, BMP)
    - **returns**: OCR 결과와 텍스트 위치 정보
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": "지원되는 이미지 형식만 업로드 가능합니다."}
        )
    
    try:
        start_time = time.time()
        
        # 이미지 읽기
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 이미지 전처리
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # OCR 수행
        detections = reader.readtext(image)
        
        # 결과 변환
        results = [[]]  # 단일 페이지를 2D 리스트로 변환
        for bbox, text, confidence in detections:
            result = OCRResult(
                page=1,
                text=text,
                confidence=confidence,
                position=[[int(x) for x in point] for point in bbox]
            )
            results[0].append(result)
        
        elapsed_time = time.time() - start_time
        print(f"[OCR] 이미지 처리 완료: {file.filename}, 소요 시간: {elapsed_time:.2f}초")
        
        return OCRResponse(
            success=True,
            message="이미지 처리가 완료되었습니다.",
            elapsed_time=elapsed_time,
            results=results
        )

    except Exception as e:
        print(f"[OCR] 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "이미지 처리 중 오류가 발생했습니다.",
                "error_type": str(type(e).__name__)
            }
        )

if __name__ == "__main__":
    print("OCR 서버가 시작됩니다...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 