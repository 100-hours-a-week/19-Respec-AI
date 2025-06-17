import os
import tempfile
from typing import Any, Dict, List, Tuple
import easyocr
import numpy as np
from PIL import Image
import re
import logging
import fitz  # PyMuPDF
from datetime import datetime
import cv2

# 로깅 설정 개선
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# EasyOCR reader 초기화 (한국어와 영어 지원)
try:
    reader = easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory='./models')
    logger.info("EasyOCR 초기화 완료")
except Exception as e:
    logger.error(f"EasyOCR 초기화 실패: {str(e)}")
    raise

def convert_pdf_to_images(pdf_bytes: bytes) -> List[np.ndarray]:
    """PDF를 이미지로 변환"""
    images = []
    try:
        # 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        # PyMuPDF로 PDF 열기
        doc = fitz.open(temp_pdf_path)
        
        for page in doc:
            # 페이지를 이미지로 변환 (해상도 향상)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2배 확대
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))

        # 임시 파일 삭제
        os.unlink(temp_pdf_path)
        
        return images
    except Exception as e:
        logger.error(f"PDF 이미지 변환 중 오류 발생: {str(e)}")
        raise

def extract_text_from_images(images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """이미지에서 텍스트 추출"""
    all_results = []
    try:
        for img in images:
            # 이미지 전처리
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 이미지 향상
            img = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            # EasyOCR로 텍스트 추출 (다양한 옵션 적용)
            results = reader.readtext(
                img,
                paragraph=True,  # 문단 단위로 추출
                batch_size=4,    # 배치 크기 조정
                contrast_ths=0.3,  # 대비 임계값
                adjust_contrast=0.5,  # 대비 조정
                width_ths=0.5,   # 너비 임계값
                height_ths=0.5   # 높이 임계값
            )

            for result in results:
                if len(result) == 3:
                    bbox, text, conf = result
                else:
                    bbox, text = result
                    conf = 0.5

                if conf > 0.3:  # 신뢰도 기준
                    text = preprocess_text(text)
                    if text:  # 빈 문자열이 아닌 경우만 추가
                        # 주변 텍스트도 함께 저장
                        nearby = []
                        for other in results:
                            if other != result:
                                other_bbox = other[0]
                                if is_nearby(bbox, other_bbox, threshold=150):
                                    nearby.append(preprocess_text(other[1]))

                        all_results.append({
                            'text': text,
                            'confidence': conf,
                            'bbox': bbox,
                            'nearby_text': nearby
                        })

    except Exception as e:
        logger.error(f"텍스트 추출 중 오류 발생: {str(e)}")
        raise

    return all_results

def is_nearby(bbox1, bbox2, threshold=150):
    """두 bbox가 가까이 있는지 확인"""
    center1 = ((bbox1[0][0] + bbox1[2][0])/2, (bbox1[0][1] + bbox1[2][1])/2)
    center2 = ((bbox2[0][0] + bbox2[2][0])/2, (bbox2[0][1] + bbox2[2][1])/2)
    
    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    return distance < threshold

def preprocess_text(text: str) -> str:
    """텍스트 전처리 개선"""
    if not isinstance(text, str):
        return ""
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 특수문자 정리 (일부 특수문자 허용)
    text = re.sub(r'[^\w\s\.\-\/\(\):,~&+%·∙•※@가-힣]', '', text)
    
    # 연속된 점(.) 제거
    text = re.sub(r'\.+', '.', text)
    
    # 불필요한 공백 제거
    text = text.strip()
    
    # 최소 길이 확인
    if len(text) < 2:
        return ""
        
    return text

def find_nearby_text(target_result: Dict, all_results: List[Dict], max_distance: int = 150) -> List[str]:
    """특정 텍스트 주변의 다른 텍스트 찾기"""
    nearby_texts = []
    target_bbox = target_result['bbox']
    target_center = ((target_bbox[0][0] + target_bbox[2][0])/2, (target_bbox[0][1] + target_bbox[2][1])/2)
    
    for result in all_results:
        if result == target_result:
            continue
        
        bbox = result['bbox']
        center = ((bbox[0][0] + bbox[2][0])/2, (bbox[0][1] + bbox[2][1])/2)
        
        # 유클리드 거리 계산
        distance = ((target_center[0] - center[0])**2 + (target_center[1] - center[1])**2)**0.5
        if distance < max_distance:
            nearby_texts.append(result['text'])
    
    return nearby_texts

def extract_specs(text_results: List[Dict], nickname: str = None) -> Dict:
    """텍스트에서 스펙 정보 추출"""
    specs = {
        'education': [],      # 학력 정보
        'experience': [],     # 경력 정보
        'certificates': [],   # 자격증
        'languages': [],      # 어학능력
        'activities': [],     # 활동/네트워킹
        'job_type': None,    # 지원분야
        'education_level': None,  # 최종학력
        'graduation_status': None,  # 졸업상태
        'nickname': nickname  # 닉네임 (사용자 입력)
    }
    
    # 전체 텍스트를 하나의 문자열로 결합
    full_text = ' '.join([r['text'] for r in text_results])
    
    # 학력 정보 추출 (필수)
    education_keywords = ['대학교', '학과', '전공', '학사', '석사', '박사', '학점']
    current_edu = None
    
    for result in text_results:
        text = result['text']
        if any(keyword in text for keyword in education_keywords):
            if current_edu is None:
                current_edu = {
                    'school': None,
                    'major': None,
                    'degree': None,
                    'gpa': None,
                    'gpa_max': 4.5
                }
            
            # 학교명 추출
            school_match = re.search(r'([가-힣]+대학교)', text)
            if school_match:
                current_edu['school'] = school_match.group(1)
            
            # 전공 추출
            major_match = re.search(r'([가-힣]+(?:학과|전공))', text)
            if major_match:
                current_edu['major'] = major_match.group(1)
            
            # 학위 추출
            if '학사' in text:
                current_edu['degree'] = '학사'
                specs['education_level'] = '대학교'
            elif '석사' in text:
                current_edu['degree'] = '석사'
                specs['education_level'] = '대학원'
            elif '박사' in text:
                current_edu['degree'] = '박사'
                specs['education_level'] = '대학원'
            
            # 학점 추출 (필수)
            # 다양한 형식의 학점 패턴 매칭
            gpa_patterns = [
                r'(\d+\.?\d*)\s*(?:/|만점|\(만점\))\s*(\d+\.?\d*)',  # 3.5/4.5 또는 3.5만점4.5
                r'학점\s*:?\s*(\d+\.?\d*)\s*(?:/|만점|\(만점\))\s*(\d+\.?\d*)',  # 학점: 3.5/4.5
                r'(\d+\.?\d*)\s*학점',  # 3.5학점
                r'GPA\s*:?\s*(\d+\.?\d*)',  # GPA: 3.5
                r'평점\s*:?\s*(\d+\.?\d*)'  # 평점: 3.5
            ]
            
            for pattern in gpa_patterns:
                gpa_match = re.search(pattern, text)
                if gpa_match:
                    if len(gpa_match.groups()) == 2:
                        current_edu['gpa'] = float(gpa_match.group(1))
                        current_edu['gpa_max'] = float(gpa_match.group(2))
                    else:
                        current_edu['gpa'] = float(gpa_match.group(1))
                    break
            
            # 졸업상태 추출
            if '졸업' in text:
                specs['graduation_status'] = '졸업'
            elif '재학' in text:
                specs['graduation_status'] = '재학중'
            elif '휴학' in text:
                specs['graduation_status'] = '휴학중'
            
            if current_edu['school'] or current_edu['major']:
                specs['education'].append(current_edu)
                current_edu = None
    
    # 경력 정보 추출 (필수)
    experience_keywords = ['회사', '기업', '근무', '인턴', '직무', '업무', '주임', '대리', '과장', '차장', '부장', '이사']
    current_exp = None
    
    for result in text_results:
        text = result['text']
        if any(keyword in text for keyword in experience_keywords):
            if current_exp is None:
                current_exp = {
                    'company': None,
                    'position': None,
                    'position_type': None,
                    'duration': None
                }
            
            # 회사명 추출
            company_match = re.search(r'([가-힣a-zA-Z0-9]+(?:회사|기업|주식회사|㈜))', text)
            if company_match:
                current_exp['company'] = company_match.group(1)
            
            # 직책 추출
            position_match = re.search(r'([가-힣]+(?:주임|대리|과장|차장|부장|이사|팀장|매니저|담당|사원))', text)
            if position_match:
                current_exp['position'] = position_match.group(1)
            
            # 근무형태 추출
            if '인턴' in text:
                current_exp['position_type'] = '인턴'
            elif '계약' in text:
                current_exp['position_type'] = '계약직'
            elif current_exp['position']:
                current_exp['position_type'] = '정규직'
            
            # 근무기간 추출
            duration_match = re.search(r'(\d+)\s*(?:개월|년)', text)
            if duration_match:
                current_exp['duration'] = int(duration_match.group(1))
            
            if current_exp['company'] or current_exp['position']:
                specs['experience'].append(current_exp)
                current_exp = None
    
    # 자격증 정보 추출 (필수)
    cert_keywords = [
        '자격증', '기사', '산업기사', '기술사', '자격', '면허', '수료증',
        'certification', 'certificate', '1급', '2급', '사', '士', '검정',
        '한국사', 'TOEIC', 'TOEFL', 'OPIC', 'JLPT', 'HSK', '토익', '토플',
        '운전', '보안', '안전', '평가사', '관리사', '지도사', '상담사', '강사'
    ]
    
    current_cert = None
    cert_found = set()  # 중복 방지를 위한 집합
    
    for result in text_results:
        text = result['text']
        
        # 이미 찾은 자격증은 건너뛰기
        if text in cert_found:
            continue
            
        if any(keyword in text for keyword in cert_keywords):
            # 자격증명 추출을 위한 다양한 패턴
            cert_patterns = [
                r'([가-힣a-zA-Z0-9]+(?:자격증|기사|산업기사|기술사|면허증?))',  # 기본 패턴
                r'([가-힣a-zA-Z0-9]+(?:급|사|士|검정|평가사|관리사|지도사|상담사|강사))',  # 직무 관련
                r'(TOEIC|TOEFL|OPIC|JLPT|HSK|토익|토플)(?:\s*\d*)?',  # 어학 자격증
                r'([가-힣a-zA-Z0-9]+운전면허)',  # 운전면허
                r'([가-힣a-zA-Z0-9]+(?:수료증|인증서|자격))'  # 기타 자격
            ]
            
            cert_name = None
            for pattern in cert_patterns:
                match = re.search(pattern, text)
                if match:
                    cert_name = match.group(1)
                    break
            
            if cert_name and cert_name not in cert_found:
                current_cert = {
                    'name': cert_name,
                    'date': None
                }
                
                # 취득일자 추출 (다양한 형식 지원)
                date_patterns = [
                    r'(\d{4})[./년]\s*(\d{1,2})[./월]?',  # YYYY.MM 또는 YYYY년 MM월
                    r'(\d{4})\s*년?\s*(\d{1,2})\s*월?',   # YYYY MM
                    r'(\d{2})[./]\s*(\d{1,2})',          # YY.MM
                ]
                
                # 주변 텍스트에서 날짜 찾기
                nearby_texts = find_nearby_text(result, text_results, max_distance=200)
                all_texts = [text] + nearby_texts
                
                for text_to_check in all_texts:
                    for pattern in date_patterns:
                        date_match = re.search(pattern, text_to_check)
                        if date_match:
                            year = date_match.group(1)
                            month = date_match.group(2)
                            
                            # YY 형식을 YYYY로 변환
                            if len(year) == 2:
                                year = '20' + year if int(year) < 50 else '19' + year
                                
                            current_cert['date'] = f"{year}.{month.zfill(2)}"
                            break
                    if current_cert['date']:
                        break
                
                if current_cert['name']:
                    cert_found.add(cert_name)
                    specs['certificates'].append(current_cert)
                    current_cert = None
    
    # 어학능력 정보 추출 (필수)
    lang_keywords = ['TOEIC', 'TOEFL', 'OPIC', 'JLPT', 'HSK', '토익', '토플', '영어', '일본어', '중국어']
    current_lang = None
    
    for result in text_results:
        text = result['text']
        if any(keyword in text.upper() for keyword in lang_keywords):
            if current_lang is None:
                current_lang = {
                    'test': None,
                    'score': None
                }
            
            # 시험명 추출
            test_match = re.search(r'(TOEIC|TOEFL|OPIC|JLPT|HSK|토익|토플)', text, re.IGNORECASE)
            if test_match:
                current_lang['test'] = test_match.group(1)
            
            # 점수/등급 추출
            score_match = re.search(r'(\d+)\s*(?:점|급)|Level\s*(\d+)', text)
            if score_match:
                current_lang['score'] = score_match.group(0)
            
            if current_lang['test']:
                specs['languages'].append(current_lang)
                current_lang = None
    
    # 활동 정보 추출 (필수)
    activity_keywords = ['동아리', '봉사', '대외활동', '프로젝트', '학생회', '활동']
    current_act = None
    
    for result in text_results:
        text = result['text']
        if any(keyword in text for keyword in activity_keywords):
            if current_act is None:
                current_act = {
                    'name': None,
                    'role': None,
                    'duration': None
                }
            
            # 활동명 추출
            act_match = re.search(r'([가-힣a-zA-Z0-9]+(?:동아리|봉사|대외활동|프로젝트|학생회))', text)
            if act_match:
                current_act['name'] = act_match.group(1)
            
            # 역할/직책 추출
            role_match = re.search(r'([가-힣]+(?:장|위원|담당))', text)
            if role_match:
                current_act['role'] = role_match.group(1)
            
            # 활동기간 추출
            duration_match = re.search(r'(\d+)\s*(?:개월|년)', text)
            if duration_match:
                current_act['duration'] = int(duration_match.group(1))
            
            if current_act['name']:
                specs['activities'].append(current_act)
                current_act = None
    
    # 지원분야 추출
    job_keywords = {
        '인터넷_IT': ['IT', '개발', '프로그래밍', '소프트웨어', '웹', '앱'],
        '경영_사무': ['경영', '사무', '총무', '인사', '회계', '재무'],
        '마케팅_광고_홍보': ['마케팅', '광고', '홍보', '기획', 'PR', '브랜드'],
        '디자인': ['디자인', '그래픽', 'UI', 'UX', '시각', '산업디자인'],
        '영업_고객상담': ['영업', '세일즈', '고객', '상담', 'CS'],
        '서비스': ['서비스', '호텔', '요식', '관광', '레저'],
        '연구_RND': ['연구', 'R&D', '개발', '설계', '엔지니어'],
        '생산_제조': ['생산', '제조', '품질', '공정', '설비'],
        '교육': ['교육', '강사', '교사', '강의', '학습'],
        '의료_복지': ['의료', '복지', '간호', '요양', '보건'],
        '미디어_콘텐츠': ['미디어', '콘텐츠', '방송', '출판', '엔터테인먼트']
    }
    
    for job_type, keywords in job_keywords.items():
        if any(keyword in full_text for keyword in keywords):
            specs['job_type'] = job_type
            break
    
    if not specs['job_type']:
        specs['job_type'] = '인터넷_IT'  # 기본값 설정
    
    return specs

async def analyze_resume(file, job_type: str = None, nickname: str = None):
    """PDF 파일에서 스펙 정보 추출"""
    try:
        logger.info("PDF 분석 시작")
        contents = await file.read()
        
        # PDF를 이미지로 변환
        logger.debug("PDF를 이미지로 변환 중...")
        images = convert_pdf_to_images(contents)
        if not images:
            raise ValueError("PDF를 이미지로 변환할 수 없습니다.")
        logger.debug(f"{len(images)}개의 이미지 변환 완료")
        
        # EasyOCR로 텍스트 추출
        logger.debug("텍스트 추출 중...")
        text_results = extract_text_from_images(images)
        if not text_results:
            raise ValueError("텍스트를 추출할 수 없습니다.")
        logger.debug(f"{len(text_results)}개의 텍스트 블록 추출 완료")
        
        # 스펙 정보 추출
        logger.debug("스펙 정보 추출 중...")
        specs = extract_specs(text_results, nickname)
        
        # 지원분야 설정
        if job_type:
            specs['job_type'] = job_type
        
        logger.info("PDF 분석 완료")
        return {
            "status": "success",
            "spec_info": specs
        }
        
    except ValueError as e:
        logger.error(f"데이터 처리 오류: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"PDF 분석 중 예외 발생: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"PDF 분석 중 오류 발생: {str(e)}"
        } 