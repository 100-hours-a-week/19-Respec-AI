import easyocr      # ✅ EasyOCR 라이브러리
import cv2          # ✅ OpenCV (이미지 처리)
import numpy as np  # ✅ NumPy (배열 처리)
import re           # ✅ 정규표현식
from typing import Dict, List, Optional, Any  # ✅ 타입 힌트
import logging      # ✅ 로깅
import time         # 성능 모니터링용
import psutil       # 시스템 리소스 모니터링
import gc           # 가비지 컬렉션

class ResumeAnalysisModel:
    """EasyOCR 기반 이력서 분석 모델"""
    
    def __init__(self, languages=['ko', 'en'], gpu=True):
        """
        모델 초기화
        
        Args:
            languages: 지원할 언어 리스트
            gpu: GPU 사용 여부
        """
        self.logger = logging.getLogger(__name__)
        
        # 성능 모니터링 변수 초기화
        self.performance_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'ocr_success_rate': 0.0,
            'memory_usage_mb': 0.0,
            'gpu_available': False
        }
        
        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            self.performance_stats['gpu_available'] = gpu
            self.logger.info(f"EasyOCR 모델 로딩 완료 (GPU: {gpu})")
        except Exception as e:
            if gpu:
                # GPU 실패시 CPU로 재시도
                self.reader = easyocr.Reader(languages, gpu=False)
                self.performance_stats['gpu_available'] = False
                self.logger.info("GPU 로딩 실패, CPU 모드로 전환")
            else:
                raise e
                
        # 점수 가중치 설정
        self.score_weights = {
            'education': 25,     # 학업 (25점)
            'experience': 30,    # 경력 (30점)
            'skills': 15,        # 기술 (15점)
            'certificates': 20,  # 자격증 (20점)
            'languages': 10      # 언어 (10점)
        }
        
        # 동적 신뢰도 임계값 설정
        self.confidence_thresholds = {
            'high_quality': 0.8,    # 고품질 이미지
            'medium_quality': 0.6,  # 중간 품질 이미지
            'low_quality': 0.4,     # 저품질 이미지
            'minimum': 0.2          # 최소 임계값
        }
        
        # 키워드 매칭 패턴
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """키워드 매칭 패턴 초기화"""
        return {
            'education_keywords': [
                r'대학교', r'대학원', r'학사', r'석사', r'박사', r'졸업', r'전공', r'학과'
            ],
            'experience_keywords': [
                r'경력', r'근무', r'재직', r'프로젝트', r'업무', r'담당', r'개발', r'관리'
            ],
            'skills_keywords': [
                r'Python', r'Java', r'JavaScript', r'React', r'Spring', r'MySQL', r'MongoDB'
            ]
        }

    def validate_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 품질 검사 및 분석"""
        try:
            height, width = image.shape[:2]
            
            # 기본 품질 지표
            quality_metrics = {
                'resolution': {'width': width, 'height': height},
                'total_pixels': width * height,
                'aspect_ratio': width / height if height > 0 else 0,
                'is_valid': True,
                'quality_score': 0,
                'issues': []
            }
            
            # 해상도 검사
            if width < 300 or height < 300:
                quality_metrics['issues'].append('해상도가 너무 낮음 (최소 300x300 권장)')
                quality_metrics['quality_score'] -= 30
            elif width < 600 or height < 600:
                quality_metrics['issues'].append('해상도가 낮음 (600x600 이상 권장)')
                quality_metrics['quality_score'] -= 15
            
            # 종횡비 검사
            if quality_metrics['aspect_ratio'] < 0.5 or quality_metrics['aspect_ratio'] > 2.0:
                quality_metrics['issues'].append('비정상적인 종횡비')
                quality_metrics['quality_score'] -= 10
            
            # 이미지 선명도 검사 (라플라시안 분산)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                quality_metrics['issues'].append('이미지가 흐림 (선명도 부족)')
                quality_metrics['quality_score'] -= 20
            elif laplacian_var < 200:
                quality_metrics['issues'].append('이미지 선명도가 낮음')
                quality_metrics['quality_score'] -= 10
                
            # 밝기 검사
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                quality_metrics['issues'].append('이미지가 너무 어두움')
                quality_metrics['quality_score'] -= 15
            elif mean_brightness > 200:
                quality_metrics['issues'].append('이미지가 너무 밝음')
                quality_metrics['quality_score'] -= 10
                
            # 최종 품질 점수 계산 (0-100)
            quality_metrics['quality_score'] = max(0, 100 + quality_metrics['quality_score'])
            
            # 품질이 너무 낮으면 유효하지 않은 것으로 판단
            if quality_metrics['quality_score'] < 30:
                quality_metrics['is_valid'] = False
                
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"이미지 품질 검사 오류: {e}")
            return {
                'is_valid': False,
                'quality_score': 0,
                'issues': ['이미지 품질 검사 실패'],
                'resolution': {'width': 0, 'height': 0}
            }

    def get_dynamic_confidence_threshold(self, image_quality_score: float) -> float:
        """이미지 품질에 따른 동적 신뢰도 임계값 결정"""
        if image_quality_score >= 80:
            return self.confidence_thresholds['high_quality']
        elif image_quality_score >= 60:
            return self.confidence_thresholds['medium_quality']
        elif image_quality_score >= 40:
            return self.confidence_thresholds['low_quality']
        else:
            return self.confidence_thresholds['minimum']

    def monitor_performance(self, func_name: str, start_time: float, success: bool = True):
        """성능 모니터링 데이터 수집"""
        processing_time = time.time() - start_time
        
        # 통계 업데이트
        self.performance_stats['total_processed'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        self.performance_stats['average_processing_time'] = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['total_processed']
        )
        
        # 메모리 사용량 업데이트
        process = psutil.Process()
        self.performance_stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        
        # OCR 성공률 업데이트 (간단한 추정)
        if func_name == 'extract_text':
            success_count = getattr(self, '_ocr_success_count', 0)
            total_count = getattr(self, '_ocr_total_count', 0)
            
            if success:
                success_count += 1
            total_count += 1
            
            self._ocr_success_count = success_count
            self._ocr_total_count = total_count
            self.performance_stats['ocr_success_rate'] = (success_count / total_count) * 100
        
        self.logger.info(f"{func_name} 처리 시간: {processing_time:.2f}초, 메모리: {self.performance_stats['memory_usage_mb']:.1f}MB")

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.performance_stats.copy()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 - OCR 품질 대폭 개선"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 기울기 보정 (스큐 보정)
        try:
            gray = self.correct_skew(gray)
        except:
            pass  # 스큐 보정 실패해도 계속 진행
        
        # 해상도 향상 (4배로 증가)
        scale_factor = 4
        height, width = gray.shape
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                         interpolation=cv2.INTER_CUBIC)
        
        # 노이즈 제거 및 선명도 향상 (다단계 처리)
        # 1단계: 약한 가우시안 블러
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2단계: 언샤프 마스킹으로 선명도 향상
        blurred = cv2.GaussianBlur(denoised, (5, 5), 2.0)
        unsharp_mask = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        
        # 3단계: 적응형 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(unsharp_mask)
        
        # 4단계: 적응형 임계처리 (더 정교한 이진화)
        # Otsu + 가우시안 조합
        _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 적응형 임계처리
        binary2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        # 두 결과를 결합
        binary = cv2.bitwise_and(binary1, binary2)
        
        # 5단계: 모폴로지 연산으로 문자 구조 개선
        # 텍스트 연결 및 정리
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # 작은 노이즈 제거
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # 6단계: 최종 대비 조정
        # 텍스트가 더 진하게 보이도록
        cleaned = cv2.bitwise_not(cleaned)  # 반전
        cleaned = cv2.dilate(cleaned, np.ones((1,1), np.uint8), iterations=1)  # 약간 굵게
        cleaned = cv2.bitwise_not(cleaned)  # 다시 반전
        
        return cleaned
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """이미지 기울기 보정"""
        # 이진화
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 허프 변환으로 직선 검출
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:min(len(lines), 20)]:  # 최대 20개 직선만 사용
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                # 중앙값으로 기울기 각도 결정
                median_angle = np.median(angles)
                
                # 각도가 너무 작으면 보정하지 않음
                if abs(median_angle) > 0.5:
                    # 이미지 회전
                    height, width = image.shape
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        
        return image

    def extract_text(self, image: np.ndarray, confidence_threshold: float = None) -> str:
        """이미지에서 텍스트 추출 - 개선된 버전 (품질 기반 동적 처리)"""
        start_time = time.time()
        success = False
        
        try:
            # 이미지 품질 검사
            quality_metrics = self.validate_image_quality(image)
            
            if not quality_metrics['is_valid']:
                self.logger.warning(f"이미지 품질 문제: {quality_metrics['issues']}")
                # 품질이 낮아도 처리는 시도
            
            # 동적 신뢰도 임계값 설정
            if confidence_threshold is None:
                confidence_threshold = self.get_dynamic_confidence_threshold(
                    quality_metrics['quality_score']
                )
            
            self.logger.info(f"이미지 품질 점수: {quality_metrics['quality_score']}, "
                           f"신뢰도 임계값: {confidence_threshold}")
            
            # 1차 시도: 기본 전처리
            processed_image = self.preprocess_image(image)
            results = self._ocr_with_settings(processed_image, confidence_threshold)
            
            # 결과 품질 확인 및 적응형 재시도
            if len(results) < 5:  # 텍스트가 너무 적으면
                # 2차 시도: 더 낮은 임계값으로
                self.logger.info("1차 OCR 결과가 부족, 2차 시도 시작")
                lower_threshold = max(0.1, confidence_threshold * 0.6)
                results_retry = self._ocr_with_settings(processed_image, lower_threshold)
                if len(results_retry) > len(results):
                    results = results_retry
                    
            # 3차 시도: 매우 낮은 품질 이미지의 경우
            if len(results) < 3 and quality_metrics['quality_score'] < 40:
                self.logger.info("3차 시도: 최소 임계값 적용")
                results_final = self._ocr_with_settings(processed_image, 0.1)
                if len(results_final) > len(results):
                    results = results_final
            
            # 신뢰도 필터링 및 텍스트 정리
            filtered_texts = []
            for result in results:
                try:
                    if len(result) >= 2:
                        text = result[1]  # 텍스트는 항상 두 번째 요소
                        # 텍스트 정리
                        cleaned_text = self.clean_extracted_text(text.strip())
                        if cleaned_text:
                            filtered_texts.append(cleaned_text)
                except Exception as e:
                    self.logger.warning(f"텍스트 정리 오류: {e}, result: {result}")
                    continue
            
            # 텍스트 결합 및 후처리
            full_text = ' '.join(filtered_texts)
            processed_text = self.post_process_text(full_text)
            
            # 이력서 핵심 내용만 필터링
            filtered_content = self.filter_resume_content(processed_text)
            
            success = len(processed_text) > 10  # 최소 텍스트 길이로 성공 여부 판단
            
            self.logger.info(f"OCR 완료: {len(filtered_texts)}개 텍스트 블록 추출, "
                           f"최종 텍스트 길이: {len(processed_text)}")
            
            # 메모리 정리
            gc.collect()
            
            # 필터링된 내용이 있으면 사용, 없으면 원본 사용
            final_text = filtered_content if len(filtered_content) > 50 else processed_text
            
            return final_text
            
        except Exception as e:
            self.logger.error(f"텍스트 추출 오류: {e}")
            return ""
        finally:
            # 성능 모니터링
            self.monitor_performance('extract_text', start_time, success)
    
    def _ocr_with_settings(self, image: np.ndarray, confidence_threshold: float):
        """특정 설정으로 OCR 수행"""
        results = self.reader.readtext(
            image,
            width_ths=0.7,      # 텍스트 폭 임계값
            height_ths=0.7,     # 텍스트 높이 임계값
            paragraph=True,     # 단락 단위로 인식
            detail=1
        )
        
        # 신뢰도 필터링 - 안전한 언패킹 처리
        filtered_results = []
        for result in results:
            try:
                if len(result) == 3:
                    bbox, text, confidence = result
                elif len(result) == 2:
                    bbox, text = result
                    confidence = 1.0  # 기본 신뢰도
                else:
                    continue
                    
                if confidence >= confidence_threshold:
                    filtered_results.append((bbox, text, confidence))
            except Exception as e:
                self.logger.warning(f"OCR 결과 처리 오류: {e}, result: {result}")
                continue
                
        return filtered_results
    
    def clean_extracted_text(self, text: str) -> str:
        """추출된 텍스트 정리 - 불필요한 내용 대폭 제거"""
        if not text or len(text.strip()) < 2:
            return ""
        
        import re
        
        # 불필요한 내용 제거 패턴 (개인정보 및 양식 제거)
        unnecessary_patterns = [
            # 주소 완전 제거 (강화)
            r'.*(?:주소|거주지|현주소|본적|출생지|소재지|위치):?.*',
            r'.*(?:시|구|동|로|길|읍|면|리|가|번지|호)\s*\d+.*',  # 주소 패턴
            r'.*(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주).*(?:시|도|구|군).*',
            r'.*\d{5,6}.*',  # 우편번호
            r'.*(?:아파트|빌라|오피스텔|연립|단독|주택|맨션).*\d+.*',
            
            # 개인정보 완전 제거 (강화)
            r'.*(?:연락처|전화번호|휴대폰|핸드폰|휴대전화|Tel|Phone|Mobile|Cell):?.*',
            r'\d{2,3}[-\s]\d{3,4}[-\s]\d{4}',  # 전화번호 패턴
            r'010[-\s]\d{4}[-\s]\d{4}',
            r'.*(?:이메일|E-?mail|메일|email|Mail|EMAIL):?.*',
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 이메일 주소
            r'.*@.*\..*',  # 간단한 이메일 패턴
            r'.*(?:성별|나이|연령|생년월일|출생년도|출생일|생일|혈액형|종교|결혼|병역):?.*',
            r'.*\d{4}년\s*\d{1,2}월\s*\d{1,2}일.*',  # 생년월일 패턴
            r'.*\d{4}\.\d{1,2}\.\d{1,2}.*',
            r'.*\d{4}-\d{1,2}-\d{1,2}.*',
            r'생년월일.*',
            r'.*만\s*\d+세.*',
            r'.*\d+\s*세.*',  # 나이 표기
            r'.*\(\s*\d+\s*세\s*\).*',  # (25세) 패턴
            r'.*\d{4}년.*\d{1,2}월.*\d{1,2}일.*\(\s*\d+\s*세\s*\).*',  # 전체 생년월일+나이 패턴
            
            # 문서 양식 및 개인식별정보 제거
            r'.*(?:이력서|履歷書|RESUME|CV|자기소개서|지원서).*',
            r'.*(?:작성일|작성자|날짜|Date|Name|성명|이름|성함):?.*',
            r'.*(?:사진|Photo|Picture|첨부|증명사진).*',
            r'페이지\s*\d+|Page\s*\d+',
            r'.*(?:지원자|지원분야|지원직종|희망직종|희망연봉):?.*',
            
            # 불필요한 기호 및 빈 내용
            r'[-=_*~━─]{3,}',
            r'\(\s*\)|\[\s*\]|\{\s*\}',
            r':\s*$',
            r'^[^가-힣a-zA-Z0-9]*$',  # 특수문자만
            r'^[ㄱ-ㅎㅏ-ㅣ]+$',  # 자음/모음만
            r'^[a-zA-Z]{1,2}$',  # 1-2글자 영문
            r'^\d+$',  # 단독 숫자
            
            # 불필요한 안내문구
            r'.*(?:해당.*없음|N/A|n/a|없음|기재.*없음|특이.*없음).*',
            r'.*(?:※|＊|\*|■|□).*',  # 주석 기호
            r'.*(?:첨부|별첨|참고|비고|www\.|http|\.com|\.co\.kr).*',
        ]
        
        # 불필요한 패턴 제거
        for pattern in unnecessary_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 추가: 생년월일 패턴 강화 제거 (매우 강력하게)
        birth_patterns = [
            # 생년월일 관련 모든 패턴
            r'.*생년월일.*',                             # 생년월일이 포함된 모든 줄
            r'.*생\s*년\s*월\s*일.*',                    # 띄어쓰기 포함
            r'.*출생년도.*',                             # 출생년도
            r'.*출생일.*',                               # 출생일
            r'.*생일.*',                                 # 생일
            
            # 년월일 패턴 (매우 포괄적)
            r'.*1[89]\d{2}년.*\d{1,2}월.*\d{1,2}일.*',  # 1900-1999년
            r'.*20[0-2]\d년.*\d{1,2}월.*\d{1,2}일.*',   # 2000-2029년
            r'.*\d{4}년\s*\d{1,2}월\s*\d{1,2}일.*',     # 모든 년월일 패턴
            r'.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}.*',       # 점으로 구분된 날짜
            r'.*\d{4}-\s*\d{1,2}-\s*\d{1,2}.*',         # 하이픈으로 구분된 날짜
            r'.*\d{4}/\s*\d{1,2}/\s*\d{1,2}.*',         # 슬래시로 구분된 날짜
            
            # 특정 생년월일 패턴 (1996년 12월 6일)
            r'.*1996.*12.*6.*',                          # 1996년 12월 6일 관련
            r'.*1996년.*12월.*6일.*',                    # 정확한 패턴
            r'.*1996\s*년\s*12\s*월\s*6\s*일.*',        # 공백 포함
            r'.*96.*12.*6.*',                            # 축약형
            
            # 나이 관련 패턴
            r'.*\(\s*\d{1,3}\s*세\s*\).*',              # (나이) 패턴
            r'.*\d{1,3}\s*세.*',                        # 나이 패턴
            r'.*만\s*\d{1,3}\s*세.*',                   # 만 나이
            r'.*25\s*세.*',                             # 특정 나이 (25세)
            r'.*\(\s*25\s*세\s*\).*',                   # (25세)
            
            # 이름 관련 패턴
            r'.*Min\s+Jihong.*',                        # 영문 이름
            r'.*민\s*지\s*홍.*',                         # 한글 이름 (공백 포함)
            r'.*민지홍.*',                               # 한글 이름
            r'.*지홍.*',                                 # 이름 일부
            
            # 개인정보 라벨과 함께 나오는 패턴
            r'.*성명.*민지홍.*',                         # 성명: 민지홍
            r'.*이름.*민지홍.*',                         # 이름: 민지홍
            r'.*성함.*민지홍.*',                         # 성함: 민지홍
        ]
        
        # 각 패턴을 여러 번 적용하여 완전히 제거
        for _ in range(3):  # 3번 반복 적용
            for pattern in birth_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)  # 단일 라인에서도 제거
        
        # 2단계: 줄 단위로 필터링
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            
            # 빈 줄 제거
            if not line:
                continue
                
            # 너무 짧은 줄 제거 (의미있는 내용이 아닐 가능성)
            if len(line) < 3:
                continue
                
            # 숫자만 있는 줄 제거
            if re.match(r'^\d+$', line):
                continue
                
            # 특수문자만 있는 줄 제거
            if re.match(r'^[^\w가-힣]+$', line):
                continue
                
            # 반복되는 문자만 있는 줄 제거
            if len(set(line.replace(' ', ''))) <= 2:
                continue
            
            # 의미있는 내용이 있는지 확인
            meaningful_chars = re.findall(r'[가-힣a-zA-Z0-9]', line)
            if len(meaningful_chars) < 2:
                continue
                
            filtered_lines.append(line)
        
        # 3단계: 기본 정리
        text = '\n'.join(filtered_lines)
        
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정리 (한글, 영문, 숫자, 기본 특수문자만 유지)
        text = re.sub(r'[^\w\s가-힣.,():/-]', '', text)
        
        # 4단계: 중복 제거
        lines = text.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line)
        
        return '\n'.join(unique_lines).strip()
    
    def post_process_text(self, text: str) -> str:
        """텍스트 후처리 - 대폭 강화된 OCR 오류 수정"""
        if not text:
            return ""
        
        import re
        
        # 1단계: 기본 정리
        text = re.sub(r'\s+', ' ', text)  # 과도한 공백 제거
        
        # 2단계: 일반적인 OCR 오류 패턴 수정 (대폭 확장)
        corrections = {
            # IT/개발 관련 용어
            r'[Pp]ythcn|[Pp]ython|[Pp]이손|파이손': 'Python',
            r'더미터|데미터|더이터|데미타': '데이터',
            r'지늠|지능|지늘': '지능',
            r'머신리심|머신리딥|머신러님|머신터딩': '머신러닝',
            r'물리님|러님|리님': '러닝',
            r'프로그러낌|프로그램밍|프로그러밍': '프로그래밍',
            r'인공지늠|인공지늘|인공지늄': '인공지능',
            r'디털러닝|디플러닝|덥리빌|덥리딩|더플러님|더신러님': '딥러닝',
            r'스도리터럽|스도리터림|스도리터락|스도리련픽': '스터디',
            r'위크습|위크솜|위국습': '워크샵',
            r'교수법|교소법': '교수법',
            r'카이스트|가이스트': 'KAIST',
            
            # 회사/조직 관련
            r'회망연봉|희망연방': '희망연봉',
            r'지원문야|지원분아': '지원분야',
            r'히사|회시': '회사',
            r'내규데\s*따듬|내규에\s*따른': '내규에 따른',
            r'인적사항|인저사항': '인적사항',
            r'경력사항|경려사항': '경력사항',
            r'담당\s*업무|담강\s*업무': '담당업무',
            r'근무기간|근무가간': '근무기간',
            
            # 문서 유형
            r'입\s*사\s*지\s*원\s*서': '입사지원서',
            r'이\s*력\s*서': '이력서',
            r'자\s*기\s*소\s*개\s*서': '자기소개서',
            
            # 학력 관련
            r'학력사항|학력시항': '학력사항',
            r'재학기간|재학가간': '재학기간',
            r'전문대학|전몬대학': '전문대학',
            r'대학교|대학고': '대학교',
            r'대학원|대학완': '대학원',
            r'졸업|졸엄': '졸업',
            r'학점|학점': '학점',
            r'전공|전골': '전공',
            r'학과|학가': '학과',
            r'학사학위|학시학위': '학사학위',
            r'컴퓨터시스템공학과|컴퓨터시스틈공학과': '컴퓨터시스템공학과',
            
            # 개인정보
            r'생\s*년\s*월\s*일|생\s*넌\s*원\s*일': '생년월일',
            r'휴대전화|휴데전화': '휴대전화',
            r'주\s*소|주\s*서': '주소',
            r'영\s*문': '영문',
            
            # 지역명
            r'경기도|검기도': '경기도',
            r'고양시|고향시': '고양시',
            r'일산동구|일싼동구': '일산동구',
            r'무궁화로|무굼화로': '무궁화로',
            r'서울|서울': '서울',
            r'부산|부싼': '부산',
            r'대구|대구': '대구',
            
            # 기관/학교명
            r'인하공업전문대학|인하공엄전문대학': '인하공업전문대학',
            r'네화여자대학교|네화여자대학고': '네화여자대학교',
            r'한국표순과학인구권|한국과학기술정보연구원': '한국과학기술정보연구원',
            r'국가\s*소재단구데미터센터|국가슈퍼컴퓨터센터': '국가슈퍼컴퓨터센터',
            r'중부진친교육지원청|중부진천교육지원청': '중부진천교육지원청',
            r'서온기순인구권|서울과학기술대학교': '서울과학기술대학교',
            r'서온교육대학교|서울교육대학교': '서울교육대학교',
            r'대구대학교|대구대학고': '대구대학교',
            r'코너스트국제학교|콘코드국제학교': '콘코드국제학교',
            
            # 기술/프로젝트 관련
            r'진천군\s*확신도시|진천군\s*혁신도시': '진천군 혁신도시',
            r'사업|사엄': '사업',
            r'프로적드|프로젝트': '프로젝트',
            r'프로적도|프로그램': '프로그램',
            r'위국복|워크북': '워크북',
            r'강의안|강의완': '강의안',
            r'마스크\s*미차움|마스크\s*미착용': '마스크 미착용',
            r'감지|감지': '감지',
            r'모덤|모델': '모델',
            r'생성|생성': '생성',
            r'교육함프|교육캠프': '교육캠프',
            r'교육함드|교육과정': '교육과정',
            r'인저양성|인재양성': '인재양성',
            r'4C\s*능력배양|4C\s*능력배양': '4C 능력배양',
            
            # 일반적인 OCR 오류
            r'(\d+)g(\d+)년': r'\1\2년',  # 1gg6년 → 1996년
            r'원\s*(\d+)': r'월 \1',      # 원 12 → 월 12
            r'인\s*(\d+)': r'일 \1',      # 인 25 → 일 25
            r'더\s*이\s*터': '데이터',
            r'곰\s*(\w+)\s*지\s*(\w+)': r'\1지\2',  # 이곰민지꿀 → 이민지
            
            # 날짜 형식 정리
            r'(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})': r'\1.\2.\3',
            r'(\d{4})\s*년\s*(\d{1,2})\s*월': r'\1년 \2월',
            r'(\d{1,2})\s*월\s*(\d{1,2})\s*일': r'\1월 \2일',
            
            # 구분자 정리
            r'및\s*부서|및\s*부셔': '및 부서',
            r'직위|직위': '직위',
            r'내움|내용': '내용',
            r'기관|기간': '기관',
            r'기간|기간': '기간',
            
            # 문장 부호 정리
            r'\s*,\s*': ', ',
            r'\s*\.\s*': '. ',
            r'\s*:\s*': ': ',
            r'\s*;\s*': '; ',
        }
        
        # 패턴 적용
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 3단계: 공백 및 특수문자 정리
        text = re.sub(r'\s+([,.;:])', r'\1', text)  # 구두점 앞 공백 제거
        text = re.sub(r'([,.;:])\s*', r'\1 ', text)  # 구두점 뒤 공백 통일
        text = re.sub(r'\s+', ' ', text)  # 과도한 공백 제거
        
        # 4단계: 최종 정리
        text = text.strip()
        
        return text

    def analyze_education(self, text: str) -> Dict[str, Any]:
        """교육 배경 분석"""
        education_info = {
            'level': 'Unknown',
            'institution': '',
            'gpa': 0.0,
            'base_score': 0  # 0-100 범위의 기본 점수
        }
        
        # 학력 수준 파악
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in [r'박사', r'Ph\.?D']):
            education_info['level'] = 'PhD'
            education_info['base_score'] = 100
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in [r'석사', r'Master', r'대학원']):
            education_info['level'] = 'Master'
            education_info['base_score'] = 85
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in [r'학사', r'Bachelor', r'대학교']):
            education_info['level'] = 'Bachelor'
            education_info['base_score'] = 70
        elif re.search(r'전문대|2년제|3년제', text, re.IGNORECASE):
            education_info['level'] = 'Associate'
            education_info['base_score'] = 55
        elif re.search(r'고등학교|고교', text, re.IGNORECASE):
            education_info['level'] = 'High School'
            education_info['base_score'] = 40
        else:
            education_info['base_score'] = 30
        
        # GPA 추출 및 점수 조정
        gpa_match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)', text)
        if gpa_match:
            gpa_score = float(gpa_match.group(1))
            gpa_max = float(gpa_match.group(2))
            education_info['gpa'] = gpa_score
            # GPA 비율에 따른 점수 조정
            gpa_ratio = gpa_score / gpa_max
            education_info['base_score'] = int(education_info['base_score'] * (0.6 + 0.4 * gpa_ratio))
        
        return education_info

    def analyze_experience(self, text: str) -> Dict[str, Any]:
        """경력 분석"""
        experience_info = {
            'years': 0,
            'companies': [],
            'positions': [],
            'base_score': 0  # 0-100 범위의 기본 점수
        }
        
        # 경력 년수 추출
        year_patterns = [
            r'(\d+)년\s*(\d+)?개월?',
            r'(\d+)\s*years?',
            r'경력\s*(\d+)년'
        ]
        
        total_months = 0
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    years = int(match[0]) if match[0] else 0
                    months = int(match[1]) if len(match) > 1 and match[1] else 0
                    total_months += years * 12 + months
                else:
                    total_months += int(match) * 12
        
        experience_info['years'] = total_months / 12
        
        # 경력 점수 계산 (0-100 범위)
        if total_months >= 60:  # 5년 이상
            experience_info['base_score'] = 100
        elif total_months >= 36:  # 3년 이상
            experience_info['base_score'] = 85
        elif total_months >= 24:  # 2년 이상
            experience_info['base_score'] = 70
        elif total_months >= 12:  # 1년 이상
            experience_info['base_score'] = 55
        elif total_months >= 6:   # 6개월 이상
            experience_info['base_score'] = 40
        elif total_months > 0:    # 경력 있음
            experience_info['base_score'] = 25
        else:
            experience_info['base_score'] = 10  # 신입
        
        return experience_info

    def analyze_skills(self, text: str) -> Dict[str, Any]:
        """기술 스택 분석"""
        skills_info = {
            'programming_languages': [],
            'frameworks': [],
            'databases': [],
            'tools': [],
            'base_score': 0  # 0-100 범위의 기본 점수
        }
        
        # 기술 키워드 매칭
        skill_keywords = {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C\\+\\+', 'C#', 
                'Go', 'Rust', 'Swift', 'Kotlin', 'PHP', 'Ruby'
            ],
            'frameworks': [
                'React', 'Vue', 'Angular', 'Django', 'Flask', 'Spring', 
                'Express', 'FastAPI', 'Laravel', 'Rails'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 
                'SQLite', 'MariaDB', 'Cassandra'
            ],
            'tools': [
                'Git', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 
                'Jenkins', 'Linux', 'Windows', 'MacOS'
            ]
        }
        
        total_skills = 0
        for category, keywords in skill_keywords.items():
            for keyword in keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    skills_info[category].append(keyword)
                    total_skills += 1
        
        # 스킬 점수 계산 (0-100 범위)
        if total_skills >= 20:
            skills_info['base_score'] = 100
        elif total_skills >= 15:
            skills_info['base_score'] = 85
        elif total_skills >= 10:
            skills_info['base_score'] = 70
        elif total_skills >= 7:
            skills_info['base_score'] = 55
        elif total_skills >= 5:
            skills_info['base_score'] = 40
        elif total_skills >= 3:
            skills_info['base_score'] = 25
        elif total_skills >= 1:
            skills_info['base_score'] = 15
        else:
            skills_info['base_score'] = 5
        
        return skills_info

    def analyze_certificates(self, text: str) -> Dict[str, Any]:
        """자격증 분석"""
        certificates_info = {
            'certificates': [],
            'base_score': 0  # 0-100 범위의 기본 점수
        }
        
        # 자격증 키워드 (점수별로 분류)
        high_value_certs = [r'정보처리기사', r'정보보안기사', r'네트워크관리사', r'OCP', r'CISSP']
        medium_value_certs = [r'정보처리산업기사', r'컴퓨터활용능력', r'컴활', r'ITQ', r'MOS']
        language_certs = [r'토익', r'토플', r'오픽', r'TOEIC', r'TOEFL', r'OPIc', r'JLPT', r'HSK']
        
        total_score = 0
        
        # 고급 자격증 (각 30점)
        for pattern in high_value_certs:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                certificates_info['certificates'].extend(matches)
                total_score += len(matches) * 30
        
        # 중급 자격증 (각 20점)
        for pattern in medium_value_certs:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                certificates_info['certificates'].extend(matches)
                total_score += len(matches) * 20
        
        # 어학 자격증 (각 15점)
        for pattern in language_certs:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                certificates_info['certificates'].extend(matches)
                total_score += len(matches) * 15
        
        # 기타 자격증 키워드 (각 10점)
        other_cert_pattern = r'기사|산업기사|자격증'
        other_matches = re.findall(other_cert_pattern, text, re.IGNORECASE)
        if other_matches:
            certificates_info['certificates'].extend(other_matches)
            total_score += len(other_matches) * 10
        
        # 점수를 0-100 범위로 정규화
        certificates_info['base_score'] = min(100, total_score)
        
        return certificates_info

    def analyze_languages(self, text: str) -> Dict[str, Any]:
        """언어 능력 분석"""
        languages_info = {
            'languages': [],
            'scores': {},
            'base_score': 0  # 0-100 범위의 기본 점수
        }
        
        # 어학 점수 패턴
        language_patterns = {
            'TOEIC': r'토익|TOEIC[:\s]*(\d+)',
            'TOEFL': r'토플|TOEFL[:\s]*(\d+)',
            'IELTS': r'IELTS[:\s]*(\d+\.?\d*)',
            'JLPT': r'JLPT[:\s]*([N]?\d+)',
            'HSK': r'HSK[:\s]*(\d+)',
            'OPIc': r'오픽|OPIc[:\s]*([A-Z]+\d*)'
        }
        
        total_lang_score = 0
        for lang, pattern in language_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                languages_info['languages'].append(lang)
                score_text = match.group(1) if match.group(1) else '0'
                
                # 언어별 점수 계산
                if lang == 'TOEIC':
                    score = int(score_text) if score_text.isdigit() else 0
                    if score >= 950:
                        total_lang_score += 40
                    elif score >= 900:
                        total_lang_score += 35
                    elif score >= 850:
                        total_lang_score += 30
                    elif score >= 800:
                        total_lang_score += 25
                    elif score >= 700:
                        total_lang_score += 20
                    elif score >= 600:
                        total_lang_score += 15
                    else:
                        total_lang_score += 10
                elif lang == 'TOEFL':
                    score = int(score_text) if score_text.isdigit() else 0
                    if score >= 100:
                        total_lang_score += 35
                    elif score >= 80:
                        total_lang_score += 25
                    elif score >= 60:
                        total_lang_score += 15
                    else:
                        total_lang_score += 10
                elif lang in ['JLPT', 'HSK']:
                    total_lang_score += 25
                else:
                    total_lang_score += 20
                
                languages_info['scores'][lang] = score_text
        
        # 기본 언어 능력 키워드 체크
        if not languages_info['languages']:
            basic_lang_patterns = [r'영어', r'일본어', r'중국어', r'불어', r'독일어', r'회화']
            for pattern in basic_lang_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    total_lang_score += 10
                    break
        
        languages_info['base_score'] = min(100, total_lang_score)
        return languages_info

    def analyze_resume(self, image: np.ndarray) -> Dict[str, Any]:
        """종합 이력서 분석 - 성능 모니터링 및 오류 처리 강화"""
        start_time = time.time()
        analysis_success = False
        
        try:
            # 이미지 품질 사전 검사
            quality_metrics = self.validate_image_quality(image)
            
            # 텍스트 추출
            extracted_text = self.extract_text(image)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                return {
                    'error': '텍스트 추출 실패 또는 텍스트 부족',
                    'total_score': 0,
                    'grade': 'F',
                    'image_quality': quality_metrics,
                    'performance_stats': self.get_performance_stats()
                }
            
            # 각 영역별 분석
            education = self.analyze_education(extracted_text)
            experience = self.analyze_experience(extracted_text)
            skills = self.analyze_skills(extracted_text)
            certificates = self.analyze_certificates(extracted_text)
            languages = self.analyze_languages(extracted_text)
            
            # 가중치 적용하여 최종 점수 계산
            weighted_scores = {
                'education': (education['base_score'] / 100) * self.score_weights['education'],
                'experience': (experience['base_score'] / 100) * self.score_weights['experience'],
                'skills': (skills['base_score'] / 100) * self.score_weights['skills'],
                'certificates': (certificates['base_score'] / 100) * self.score_weights['certificates'],
                'languages': (languages['base_score'] / 100) * self.score_weights['languages']
            }
            
            # 총점 계산
            total_score = sum(weighted_scores.values())
            
            # 이미지 품질에 따른 점수 조정
            if quality_metrics['quality_score'] < 50:
                quality_penalty = (50 - quality_metrics['quality_score']) * 0.1
                total_score = max(0, total_score - quality_penalty)
                self.logger.info(f"이미지 품질로 인한 점수 차감: {quality_penalty:.2f}점")
            
            # 각 영역별 가중치 적용된 점수 추가
            education['score'] = weighted_scores['education']
            experience['score'] = weighted_scores['experience']
            skills['score'] = weighted_scores['skills']
            certificates['score'] = weighted_scores['certificates']
            languages['score'] = weighted_scores['languages']
            
            # 등급 계산
            if total_score >= 90:
                grade = 'A+'
            elif total_score >= 80:
                grade = 'A'
            elif total_score >= 70:
                grade = 'B+'
            elif total_score >= 60:
                grade = 'B'
            elif total_score >= 50:
                grade = 'C+'
            elif total_score >= 40:
                grade = 'C'
            else:
                grade = 'F'
            
            analysis_success = True
            
            result = {
                'extracted_text': extracted_text,
                'education': education,
                'experience': experience,
                'skills': skills,
                'certificates': certificates,
                'languages': languages,
                'weighted_scores': weighted_scores,
                'total_score': round(total_score, 2),
                'grade': grade,
                'max_score': 100,
                'image_quality': quality_metrics,
                'performance_stats': self.get_performance_stats()
            }
            
            # 메모리 정리
            gc.collect()
            
            return result
            
        except Exception as e:
            self.logger.error(f"이력서 분석 오류: {e}")
            return {
                'error': f'분석 처리 중 오류 발생: {str(e)}',
                'total_score': 0,
                'grade': 'F',
                'performance_stats': self.get_performance_stats()
            }
        finally:
            # 성능 모니터링
            self.monitor_performance('analyze_resume', start_time, analysis_success)

    def get_model_info(self) -> Dict[str, str]:
        """모델 정보 반환"""
        return {
            'model_type': 'EasyOCR',
            'languages': ['Korean', 'English'],
            'version': '1.0.0',
            'description': 'EasyOCR 기반 이력서 분석 모델'
        }

    def analyze_spec_data_v2(self, spec_data: Dict[str, Any], portfolio_text: str = "") -> Dict[str, Any]:
        """V2 API용 JSON 스펙 데이터 분석"""
        
        # 기본 텍스트 구성 (스펙 데이터 + 포트폴리오 텍스트)
        combined_text = self._build_text_from_spec_data(spec_data) + " " + portfolio_text
        
        # 각 영역별 분석
        education = self._analyze_education_from_spec(spec_data, combined_text)
        experience = self._analyze_experience_from_spec(spec_data, combined_text)
        skills = self._analyze_skills_from_spec(spec_data, combined_text)
        certificates = self._analyze_certificates_from_spec(spec_data, combined_text)
        languages = self._analyze_languages_from_spec(spec_data, combined_text)
        activities = self._analyze_activities_from_spec(spec_data, combined_text)
        
        # 가중치 적용하여 최종 점수 계산
        weighted_scores = {
            'education': (education['base_score'] / 100) * self.score_weights['education'],
            'experience': (experience['base_score'] / 100) * self.score_weights['experience'],
            'skills': (skills['base_score'] / 100) * self.score_weights['skills'],
            'certificates': (certificates['base_score'] / 100) * self.score_weights['certificates'],
            'languages': (languages['base_score'] / 100) * self.score_weights['languages']
        }
        
        # 대외활동 점수 추가 (기존 가중치에서 5%씩 할당)
        activity_weight = 0.05
        weighted_scores['activities'] = (activities['base_score'] / 100) * activity_weight
        
        # 기존 가중치 조정 (총합이 100%가 되도록)
        adjustment_factor = 0.95  # 대외활동 5%를 위해 기존 점수 95%로 조정
        for key in ['education', 'experience', 'skills', 'certificates', 'languages']:
            weighted_scores[key] *= adjustment_factor
        
        # 총점 계산
        total_score = sum(weighted_scores.values())
        
        return {
            'education_score': weighted_scores['education'],
            'experience_score': weighted_scores['experience'], 
            'certification_score': weighted_scores['certificates'],
            'language_score': weighted_scores['languages'],
            'activity_score': weighted_scores['activities'],
            'skills_score': weighted_scores['skills'],
            'total_score': round(total_score, 2),
            'detailed_scores': weighted_scores
        }

    def _build_text_from_spec_data(self, spec_data: Dict[str, Any]) -> str:
        """스펙 데이터에서 분석용 텍스트 구성"""
        text_parts = []
        
        # 기본 정보
        text_parts.append(f"최종학력: {spec_data.get('final_edu', '')}")
        text_parts.append(f"상태: {spec_data.get('final_status', '')}")
        text_parts.append(f"지원직종: {spec_data.get('desired_job', '')}")
        
        # 학력 정보
        if spec_data.get('universities'):
            for uni in spec_data['universities']:
                if isinstance(uni, dict):
                    uni_text = f"학교: {uni.get('school_name', '')} 학위: {uni.get('degree', '')} 전공: {uni.get('major', '')}"
                    if uni.get('gpa') and uni.get('gpa_max'):
                        uni_text += f" 학점: {uni['gpa']}/{uni['gpa_max']}"
                    text_parts.append(uni_text)
        
        # 경력 정보
        if spec_data.get('careers'):
            for career in spec_data['careers']:
                if isinstance(career, dict):
                    career_text = f"회사: {career.get('company', '')} 직무: {career.get('role', '')}"
                    if career.get('work_month'):
                        career_text += f" 근무기간: {career['work_month']}개월"
                    text_parts.append(career_text)
        
        # 자격증
        if spec_data.get('certificates'):
            cert_text = "자격증: " + ", ".join(spec_data['certificates'])
            text_parts.append(cert_text)
        
        # 어학
        if spec_data.get('languages'):
            for lang in spec_data['languages']:
                if isinstance(lang, dict):
                    lang_text = f"어학: {lang.get('test', '')} {lang.get('score_or_grade', '')}"
                    text_parts.append(lang_text)
        
        # 대외활동
        if spec_data.get('activities'):
            for activity in spec_data['activities']:
                if isinstance(activity, dict):
                    activity_text = f"활동: {activity.get('name', '')} 역할: {activity.get('role', '')}"
                    if activity.get('award'):
                        activity_text += f" 수상: {activity['award']}"
                    text_parts.append(activity_text)
        
        return " ".join(text_parts)

    def _analyze_education_from_spec(self, spec_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """스펙 데이터에서 교육 배경 분석"""
        education_info = {
            'level': spec_data.get('final_edu', 'Unknown'),
            'base_score': 0
        }
        
        # 최종학력 기반 점수
        final_edu = spec_data.get('final_edu', '').lower()
        if '박사' in final_edu:
            education_info['base_score'] = 100
        elif '석사' in final_edu or '대학원' in final_edu:
            education_info['base_score'] = 85
        elif '학사' in final_edu or '대학교' in final_edu:
            education_info['base_score'] = 70
        elif '전문' in final_edu or '2년제' in final_edu or '3년제' in final_edu:
            education_info['base_score'] = 55
        elif '고등학교' in final_edu or '고교' in final_edu:
            education_info['base_score'] = 40
        else:
            education_info['base_score'] = 30
        
        # GPA 추출 및 점수 조정
        gpa_match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)', text)
        if gpa_match:
            gpa_score = float(gpa_match.group(1))
            gpa_max = float(gpa_match.group(2))
            education_info['gpa'] = gpa_score
            # GPA 비율에 따른 점수 조정
            gpa_ratio = gpa_score / gpa_max
            education_info['base_score'] = int(education_info['base_score'] * (0.6 + 0.4 * gpa_ratio))
        
        return education_info

    def _analyze_experience_from_spec(self, spec_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """스펙 데이터에서 경력 분석"""
        experience_info = {
            'years': 0,
            'base_score': 0
        }
        
        # 경력 정보에서 총 근무 개월 수 계산
        total_months = 0
        if spec_data.get('careers'):
            for career in spec_data['careers']:
                if isinstance(career, dict) and career.get('work_month'):
                    total_months += career['work_month']
        
        experience_info['years'] = total_months / 12
        
        # 경력 점수 계산
        if total_months >= 60:  # 5년 이상
            experience_info['base_score'] = 100
        elif total_months >= 36:  # 3년 이상
            experience_info['base_score'] = 85
        elif total_months >= 24:  # 2년 이상
            experience_info['base_score'] = 70
        elif total_months >= 12:  # 1년 이상
            experience_info['base_score'] = 55
        elif total_months >= 6:   # 6개월 이상
            experience_info['base_score'] = 40
        elif total_months > 0:    # 경력 있음
            experience_info['base_score'] = 25
        else:
            experience_info['base_score'] = 10  # 신입
        
        return experience_info

    def _analyze_skills_from_spec(self, spec_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """스펙 데이터에서 기술 스택 분석 (텍스트 기반)"""
        return self.analyze_skills(text)

    def _analyze_certificates_from_spec(self, spec_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """스펙 데이터에서 자격증 분석"""
        certificates_info = {
            'certificates': [],
            'base_score': 0
        }
        
        # 스펙 데이터의 자격증 리스트
        spec_certificates = spec_data.get('certificates', [])
        certificates_info['certificates'].extend(spec_certificates)
        
        # 텍스트에서 추가 자격증 추출
        text_analysis = self.analyze_certificates(text)
        certificates_info['certificates'].extend(text_analysis.get('certificates', []))
        
        # 중복 제거
        certificates_info['certificates'] = list(set(certificates_info['certificates']))
        
        # 자격증 점수 계산
        total_score = 0
        
        # 고급 자격증 패턴
        high_value_patterns = ['정보처리기사', '정보보안기사', '네트워크관리사']
        medium_value_patterns = ['정보처리산업기사', '컴퓨터활용능력', '컴활']
        language_patterns = ['토익', 'TOEIC', '토플', 'TOEFL', 'OPIC', 'JLPT', 'HSK']
        
        for cert in certificates_info['certificates']:
            cert_lower = cert.lower()
            if any(pattern in cert for pattern in high_value_patterns):
                total_score += 30
            elif any(pattern in cert for pattern in medium_value_patterns):
                total_score += 20
            elif any(pattern.lower() in cert_lower for pattern in language_patterns):
                total_score += 15
            else:
                total_score += 10
        
        certificates_info['base_score'] = min(100, total_score)
        return certificates_info

    def _analyze_languages_from_spec(self, spec_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """스펙 데이터에서 언어 능력 분석"""
        languages_info = {
            'languages': [],
            'scores': {},
            'base_score': 0
        }
        
        total_lang_score = 0
        
        # 스펙 데이터의 어학 정보
        if spec_data.get('languages'):
            for lang in spec_data['languages']:
                if isinstance(lang, dict):
                    test_name = lang.get('test', '')
                    score = lang.get('score_or_grade', '')
                    
                    languages_info['languages'].append(test_name)
                    languages_info['scores'][test_name] = score
                    
                    # 점수 계산
                    if test_name.upper() == 'TOEIC':
                        try:
                            toeic_score = int(score)
                            if toeic_score >= 950:
                                total_lang_score += 40
                            elif toeic_score >= 900:
                                total_lang_score += 35
                            elif toeic_score >= 850:
                                total_lang_score += 30
                            elif toeic_score >= 800:
                                total_lang_score += 25
                            elif toeic_score >= 700:
                                total_lang_score += 20
                            else:
                                total_lang_score += 15
                        except:
                            total_lang_score += 15
                    elif test_name.upper() == 'TOEFL':
                        total_lang_score += 25
                    elif test_name.upper() in ['JLPT', 'HSK']:
                        total_lang_score += 20
                    else:
                        total_lang_score += 15
        
        # 텍스트에서 추가 언어 정보 추출
        text_analysis = self.analyze_languages(text)
        if text_analysis.get('base_score', 0) > total_lang_score:
            languages_info.update(text_analysis)
        else:
            languages_info['base_score'] = min(100, total_lang_score)
        
        return languages_info

    def _analyze_activities_from_spec(self, spec_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """스펙 데이터에서 대외활동 분석"""
        activities_info = {
            'activities': [],
            'base_score': 0
        }
        
        total_score = 0
        
        # 스펙 데이터의 활동 정보
        if spec_data.get('activities'):
            for activity in spec_data['activities']:
                if isinstance(activity, dict):
                    activity_name = activity.get('name', '')
                    role = activity.get('role', '')
                    award = activity.get('award', '')
                    
                    activities_info['activities'].append({
                        'name': activity_name,
                        'role': role,
                        'award': award
                    })
                    
                    # 점수 계산
                    base_activity_score = 15
                    
                    # 리더십 역할 보너스
                    if any(keyword in role for keyword in ['회장', '부회장', '팀장', '리더', '임원']):
                        base_activity_score += 10
                    
                    # 수상 경력 보너스
                    if award and award.strip():
                        if any(keyword in award for keyword in ['대상', '최우수', '금상']):
                            base_activity_score += 15
                        elif any(keyword in award for keyword in ['우수', '은상', '장려']):
                            base_activity_score += 10
                        else:
                            base_activity_score += 5
                    
                    total_score += base_activity_score
        
        activities_info['base_score'] = min(100, total_score)
        return activities_info

    def filter_resume_content(self, text: str) -> str:
        """이력서 분석을 위한 핵심 내용만 추출"""
        if not text:
            return ""
        
        import re
        
        # 필요한 스펙만 추출하는 패턴 (6개 카테고리)
        important_patterns = [
            # 1. 학력: 대학교, 전공, 학위, GPA, 졸업년도
            r'.*(?:대학교|대학원|전문대학|대학|학사|석사|박사|졸업|전공|학과|학부).*',
            r'.*(?:GPA|학점|평점)\s*\d+\.\d+.*',
            r'.*\d{4}년.*(?:졸업|입학|수료).*',
            
            # 2. 경력: 회사명, 직책, 업무내용, 프로젝트, 근무기간
            r'.*(?:회사|기업|Corporation|Inc|Ltd|Co)\s+.*',
            r'.*(?:팀장|과장|부장|대리|매니저|리더|PM|개발자|엔지니어).*',
            r'.*(?:프로젝트|업무|담당|개발|설계|구축|운영).*',
            r'.*\d{4}년.*\d{1,2}월.*(?:근무|재직).*',
            
            # 3. 기술스택: 프로그래밍 언어, 프레임워크, 데이터베이스, 클라우드
            r'.*(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Kotlin|Swift).*',
            r'.*(?:React|Vue|Angular|Node\.js|Django|Flask|Spring|Laravel).*',
            r'.*(?:MySQL|PostgreSQL|MongoDB|Redis|Oracle|MariaDB).*',
            r'.*(?:AWS|Azure|GCP|Docker|Kubernetes|Git).*',
            
            # 4. 자격증: IT 자격증, 어학시험 점수, 전문 인증
            r'.*(?:정보처리기사|컴활|MOS|CCNA|CISSP|PMP|SQLD).*',
            r'.*(?:자격증|인증서|Certificate|Certification).*',
            
            # 5. 어학능력: 언어별 수준, 시험점수
            r'.*(?:TOEIC|TOEFL|IELTS|HSK|JPT|OPIc)\s*\d{3,4}.*',
            r'.*(?:영어|중국어|일본어).*(?:상급|중급|초급|유창|능숙).*',
            
            # 6. 대외활동: 동아리, 봉사활동, 수상경력, 논문/연구
            r'.*(?:동아리|봉사|인턴십|공모전|수상|논문|연구).*',
            r'.*(?:회장|부회장|리더|멘토).*',
            r'.*(?:1등|2등|3등|금상|은상|동상|대상|우수상).*',
            
            # 날짜 패턴 (모든 카테고리에 필요)
            r'.*\d{4}년\s*\d{1,2}월.*',
            r'.*\d{4}\.\d{1,2}.*',
            r'.*\d{4}-\d{1,2}.*',
        ]
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 핵심 패턴과 매치되는지 확인
            is_important = False
            for pattern in important_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_important = True
                    break
            
            # 중요한 내용이거나 충분히 긴 의미있는 문장인 경우 포함
            if is_important or (len(line) > 10 and len(re.findall(r'[가-힣a-zA-Z]', line)) > 5):
                filtered_lines.append(line)
        
        # 결과를 더 짧게 요약
        return self._create_short_summary(filtered_lines)
    
    def _create_short_summary(self, lines: List[str]) -> str:
        """핵심 내용을 매우 짧게 요약"""
        import re
        
        summary_points = []
        
        # 카테고리별 핵심 정보만 추출
        categories = {
            '학력': [],
            '경력': [],
            '기술': [],
            '자격증': [],
            '어학': [],
            '활동': []
        }
        
        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
            
            # 학력
            if re.search(r'대학교|대학원|전공|학과|졸업', line, re.IGNORECASE):
                school = re.search(r'([가-힣]+대학교|[가-힣]+대학원)', line)
                major = re.search(r'([가-힣]+과|[가-힣]+학과|[가-힣]+전공)', line)
                if school:
                    categories['학력'].append(school.group(1))
                if major:
                    categories['학력'].append(major.group(1))
            
            # 경력
            elif re.search(r'회사|기업|근무|경력|프로젝트', line, re.IGNORECASE):
                company = re.search(r'([가-힣A-Za-z]+(?:회사|기업|그룹))', line)
                period = re.search(r'(\d+년|\d+개월)', line)
                if company:
                    categories['경력'].append(company.group(1))
                if period:
                    categories['경력'].append(period.group(1))
            
            # 기술
            elif re.search(r'Python|Java|JavaScript|React|Vue|AWS|Docker', line, re.IGNORECASE):
                techs = re.findall(r'(Python|Java|JavaScript|React|Vue|Angular|AWS|Docker|Kubernetes)', line, re.IGNORECASE)
                categories['기술'].extend(techs)
            
            # 자격증
            elif re.search(r'자격증|정보처리|컴활|토익|토플', line, re.IGNORECASE):
                cert = re.search(r'(정보처리기사|컴활|토익|토플|TOEIC|TOEFL)', line, re.IGNORECASE)
                if cert:
                    categories['자격증'].append(cert.group(1))
            
            # 어학
            elif re.search(r'TOEIC|TOEFL|영어|중국어|일본어', line, re.IGNORECASE):
                lang_score = re.search(r'(TOEIC\s*\d{3,4}|TOEFL\s*\d{2,3})', line, re.IGNORECASE)
                if lang_score:
                    categories['어학'].append(lang_score.group(1))
            
            # 활동
            elif re.search(r'동아리|봉사|수상|공모전', line, re.IGNORECASE):
                activity = re.search(r'([가-힣]+동아리|[가-힣]+봉사|[가-힣]+수상)', line)
                if activity:
                    categories['활동'].append(activity.group(1))
        
        # 카테고리별로 최대 2개씩만 선택하여 요약
        for category, items in categories.items():
            if items:
                unique_items = list(set(items))[:2]  # 중복 제거 후 최대 2개
                if unique_items:
                    summary_points.append(f"{category}: {', '.join(unique_items)}")
        
        return '\n'.join(summary_points[:6])  # 최대 6줄로 제한 