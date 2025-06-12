import easyocr
import cv2
import numpy as np
import json
import hashlib
import time
from functools import lru_cache

class SpecEvaluator:
    def __init__(self):
        print("EasyOCR 모델 로딩 중...")
        # EasyOCR 모델 로드 (GPU 사용 시도, 실패시 CPU)
        try:
            self.reader = easyocr.Reader(['ko', 'en'], gpu=True)
            print("GPU 모드로 EasyOCR 로딩 완료!")
        except:
            self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
            print("CPU 모드로 EasyOCR 로딩 완료!")
        
        # 메모리 캐시 초기화
        self.memory_cache = {}
        
        # 기본 평가 가중치 설정
        self.weights = {
            "universities": 0.3,  # 학력
            "careers": 0.35,      # 경력
            "certificates": 0.15, # 자격증
            "languages": 0.15,    # 어학
            "activities": 0.05    # 활동
        }
        
        # 점수 정규화 파라미터
        self.min_score = 40  # 최소 점수
        self.max_score = 95  # 최대 점수
        
        # 캐시 통계
        self.cache_hits = 0
        self.cache_misses = 0
        
        print("모델 초기화 완료!")

    def preprocess_image(self, image):
        """이미지 전처리 함수"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase DPI (resize)
        scale_percent = 200
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Text sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened

    def extract_text(self, image):
        """이미지에서 텍스트 추출"""
        # 이미지 전처리
        processed_img = self.preprocess_image(image)
        
        # EasyOCR로 텍스트 추출
        results = self.reader.readtext(processed_img)
        
        # 신뢰도가 높은 텍스트만 필터링 (50% 이상)
        filtered_text = []
        for (bbox, text, prob) in results:
            if prob > 0.5:
                filtered_text.append(text)
        
        return " ".join(filtered_text)

    def generate_cache_key(self, spec_data):
        """스펙 데이터로부터 캐시 키 생성"""
        spec_json = json.dumps(spec_data, sort_keys=True, default=str)
        return hashlib.md5(spec_json.encode()).hexdigest()

    def get_from_cache(self, cache_key):
        """캐시에서 결과 조회"""
        return self.memory_cache.get(cache_key)

    def save_to_cache(self, cache_key, result):
        """결과를 캐시에 저장"""
        self.memory_cache[cache_key] = result
        # 메모리 캐시 크기 제한 (1000개 항목)
        if len(self.memory_cache) > 1000:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

    def rule_based_evaluate(self, spec_data):
        """규칙 기반 평가 방식"""
        scores = {}
        job = spec_data.get("desired_job", "일반")
        
        # 최종학력 평가
        final_edu = spec_data.get("final_edu", "")
        final_status = spec_data.get("final_status", "")
        final_edu_score = 0

        # 최종학력 기본 점수 설정
        if "대학원" in final_edu:
            final_edu_score = 95
        elif "대학교" in final_edu and not "2_3년제" in final_edu:
            final_edu_score = 85
        elif "2_3년제" in final_edu:
            final_edu_score = 80
        elif "고등학교" in final_edu:
            final_edu_score = 75
        else:
            final_edu_score = 65

        # 학력 상태에 따른 조정
        if "졸업" in final_status:
            final_edu_score *= 1.0
        elif "수료" in final_status:
            final_edu_score *= 0.95
        elif "재학" in final_status:
            final_edu_score *= 0.9
        elif "휴학" in final_status:
            final_edu_score *= 0.85
        elif "중퇴" in final_status:
            final_edu_score *= 0.8

        scores["final_edu"] = min(100, final_edu_score)

        # 학력 평가
        uni_score = 0
        if "universities" in spec_data and spec_data["universities"]:
            uni_count = len(spec_data["universities"])
            for uni in spec_data["universities"]:
                school_score = 80  # 기본 점수
                
                # GPA 점수
                gpa = uni.get("gpa", 0)
                gpa_max = uni.get("gpa_max", 4.5)
                
                if gpa and gpa_max:
                    gpa_score = (gpa / gpa_max) * 100
                else:
                    gpa_score = 75
                
                uni_item_score = (school_score * 0.6 + gpa_score * 0.4)
                uni_score += uni_item_score
            
            uni_score = uni_score / uni_count
        
        scores["universities"] = uni_score

        # 경력 평가
        career_score = 0
        if "careers" in spec_data and spec_data["careers"]:
            career_count = len(spec_data["careers"])
            for career in spec_data["careers"]:
                company_score = 75  # 기본 점수
                role_score = 75     # 기본 점수
                
                work_month = career.get("work_month", 0)
                experience_weight = 1.0
                
                if work_month:
                    if work_month < 3:
                        experience_weight = 0.5
                    elif work_month < 6:
                        experience_weight = 0.8
                    elif work_month < 12:
                        experience_weight = 1.0
                    elif work_month < 24:
                        experience_weight = 1.2
                    else:
                        experience_weight = 1.5
                
                career_item_score = (company_score * 0.5 + role_score * 0.5) * experience_weight
                career_score += career_item_score
            
            career_score = career_score / career_count
            career_score = max(0, min(100, career_score))
        
        scores["careers"] = career_score

        # 자격증 평가
        cert_score = 0
        if "certificates" in spec_data and spec_data["certificates"]:
            cert_count = len(spec_data["certificates"])
            for cert in spec_data["certificates"]:
                cert_score += 80  # 기본 점수
            cert_score = cert_score / cert_count
        
        scores["certificates"] = cert_score

        # 어학 능력 평가
        lang_score = 0
        if "languages" in spec_data and spec_data["languages"]:
            lang_count = len(spec_data["languages"])
            for lang in spec_data["languages"]:
                test = lang.get("test", "").upper()
                score_or_grade = lang.get("score_or_grade", "")
                
                test_score = 0
                if test == "TOEIC":
                    try:
                        toeic_score = float(score_or_grade)
                        if toeic_score >= 900:
                            test_score = 95
                        elif toeic_score >= 800:
                            test_score = 90
                        elif toeic_score >= 700:
                            test_score = 85
                        elif toeic_score >= 600:
                            test_score = 80
                        else:
                            test_score = 75
                    except:
                        test_score = 75
                else:
                    test_score = 80
                
                lang_score += test_score
            
            lang_score = lang_score / lang_count
        
        scores["languages"] = lang_score

        # 활동 평가
        activity_score = 0
        if "activities" in spec_data and spec_data["activities"]:
            activity_count = len(spec_data["activities"])
            for activity in spec_data["activities"]:
                activity_score += 80  # 기본 점수
            activity_score = activity_score / activity_count
        
        scores["activities"] = activity_score

        # 종합 점수 계산
        total_score = 0
        weights = self.weights.copy()
        weights["final_edu"] = 0.15
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        for key, score in scores.items():
            if score > 0:
                total_score += score * weights.get(key, 0.1)
        
        if total_score == 0:
            total_score = 40
        
        # 점수 정규화
        total_score = self.min_score + (total_score / 100) * (self.max_score - self.min_score)
        
        # 포트폴리오 보너스
        if "filelink" in spec_data and spec_data["filelink"]:
            total_score = min(self.max_score, total_score + 3)
        
        return round(total_score, 2)

    def predict(self, spec_data):
        """스펙 정보를 받아 평가 결과 반환"""
        try:
            # 캐시 키 생성
            cache_key = self.generate_cache_key(spec_data)
            
            # 캐시에서 결과 조회
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                self.cache_hits += 1
                print(f"캐시 적중! (총 {self.cache_hits}번 적중)")
                return cached_result
            
            self.cache_misses += 1
            print(f"캐시 미스 (총 {self.cache_misses}번 미스)")
            
            # 시작 시간 기록
            start_time = time.time()
            
            # 이미지가 있는 경우 텍스트 추출
            if "image" in spec_data:
                extracted_text = self.extract_text(spec_data["image"])
                spec_data["extracted_text"] = extracted_text
                print(f"추출된 텍스트: {extracted_text[:100]}...")
            
            # 규칙 기반 평가 수행
            score = self.rule_based_evaluate(spec_data)
            
            # 소요 시간 계산
            elapsed_time = time.time() - start_time
            print(f"평가 소요 시간: {elapsed_time:.2f}초")
            
            # 결과 생성
            result = {
                "nickname": spec_data.get("nickname", "이름 없음"),
                "totalScore": score
            }
            
            # 결과 캐싱
            self.save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return {
                "nickname": spec_data.get("nickname", "이름 없음"),
                "totalScore": self.min_score
            }
    
    def get_cache_stats(self):
        """캐시 통계 정보 반환"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2)
        } 