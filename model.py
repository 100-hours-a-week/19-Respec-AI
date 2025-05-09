import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from tqdm import tqdm
import hashlib
import time
import redis
from functools import lru_cache

class SpecEvaluator:
    def __init__(self, use_redis=True, redis_host="localhost", redis_port=6379, redis_db=0, cache_ttl=86400):
        # 기존 초기화 코드는 동일하게 유지
        print("XGLM-564M 모델 로딩 중...")
        # XGLM-564M 모델 로드
        self.model_name = "facebook/xglm-564M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # 기존 코드 생략...
        
        # 기본 평가 가중치 설정 (각 항목별 중요도)
        self.weights = {
            "universities": 0.3,  # 학력
            "careers": 0.35,      # 경력
            "certificates": 0.15, # 자격증
            "languages": 0.15,    # 어학
            "activities": 0.05    # 활동
        }
        
        # 직무별 가중치 조정값 (기본 가중치에 가감)
        self.job_weights = {
            "경영·사무": {
                "universities": 0.05,   # 학력 가중치 증가
                "careers": 0.05,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": 0.05,      # 어학 가중치 증가
                "activities": -0.20     # 활동 가중치 감소
            },
            "마케팅·광고·홍보": {
                "universities": -0.05,  # 학력 가중치 감소
                "careers": 0.05,        # 경력 가중치 증가
                "certificates": 0.00,   # 자격증 가중치 유지
                "languages": 0.05,      # 어학 가중치 증가
                "activities": -0.05     # 활동 가중치 감소
            },
            "무역·유통": {
                "universities": 0.00,   # 학력 가중치 유지
                "careers": 0.10,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": 0.05,      # 어학 가중치 증가
                "activities": -0.20     # 활동 가중치 감소
            },
            "인터넷·IT": {
                "universities": 0.00,   # 학력 가중치 유지
                "careers": 0.10,        # 경력 가중치 증가
                "certificates": 0.10,   # 자격증 가중치 증가
                "languages": -0.05,     # 어학 가중치 감소
                "activities": -0.15     # 활동 가중치 감소
            },
            "생산·제조": {
                "universities": -0.05,  # 학력 가중치 감소
                "careers": 0.15,        # 경력 가중치 크게 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": -0.10,     # 어학 가중치 감소
                "activities": -0.05     # 활동 가중치 감소
            },
            "영업·고객상담": {
                "universities": -0.10,  # 학력 가중치 크게 감소
                "careers": 0.10,        # 경력 가중치 증가
                "certificates": 0.00,   # 자격증 가중치 유지
                "languages": 0.10,      # 어학 가중치 증가
                "activities": -0.10     # 활동 가중치 감소
            },
            "건설": {
                "universities": 0.00,   # 학력 가중치 유지
                "careers": 0.15,        # 경력 가중치 크게 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": -0.15,     # 어학 가중치 크게 감소
                "activities": -0.05     # 활동 가중치 감소
            },
            "금융": {
                "universities": 0.10,   # 학력 가중치 증가
                "careers": 0.05,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": 0.00,      # 어학 가중치 유지
                "activities": -0.20     # 활동 가중치 감소
            },
            "연구개발·설계": {
                "universities": 0.15,   # 학력 가중치 크게 증가
                "careers": 0.05,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": -0.05,     # 어학 가중치 감소
                "activities": -0.20     # 활동 가중치 감소
            },
            "디자인": {
                "universities": -0.05,  # 학력 가중치 감소
                "careers": 0.10,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": -0.10,     # 어학 가중치 감소
                "activities": 0.00      # 활동 가중치 유지
            },
            "미디어": {
                "universities": -0.05,  # 학력 가중치 감소
                "careers": 0.05,        # 경력 가중치 증가
                "certificates": 0.00,   # 자격증 가중치 유지
                "languages": 0.00,      # 어학 가중치 유지
                "activities": 0.00      # 활동 가중치 유지
            },
            "전문·특수직": {
                "universities": 0.10,   # 학력 가중치 증가
                "careers": 0.05,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": 0.00,      # 어학 가중치 유지
                "activities": -0.20     # 활동 가중치 감소
            }
        }
        
        # 점수 정규화 파라미터 - 기존과 동일
        self.min_score = 50  # 최소 점수
        self.max_score = 95  # 최대 점수
        
        # 캐시 통계 - 기존과 동일
        self.cache_hits = 0
        self.cache_misses = 0
        
        print("모델 초기화 완료!")

    def _prepare_prompt_wrapper(self, spec_data_json):
        """JSON 문자열을 딕셔너리로 변환 후 실제 프롬프트 준비 함수 호출"""
        spec_data = json.loads(spec_data_json)
        return self._prepare_prompt_uncached(spec_data)
    
    def _prepare_prompt_uncached(self, spec_data):
        """스펙 데이터를 평가하기 위한 프롬프트 준비"""
        # 지원직종에 따른 맞춤형 프롬프트 생성
        job = spec_data.get("desired_job", "일반")
        
        # 지원직종에 따른 맞춤형 프롬프트 생성
        job_specific_description = ""
        if job == "경영·사무":
            job_specific_description = "경영 및 사무 직종은 조직 관리, 행정 업무, 문서작성 등의 능력이 중요합니다."
        elif job == "마케팅·광고·홍보":
            job_specific_description = "마케팅, 광고, 홍보 분야는 창의성, 커뮤니케이션 능력, 트렌드 분석 능력이 중요합니다."
        elif job == "무역·유통":
            job_specific_description = "무역 및 유통 분야는 글로벌 비즈니스 역량, 어학 능력, 물류 이해도가 중요합니다."
        elif job == "인터넷·IT":
            job_specific_description = "IT 분야는 기술적 전문성, 프로그래밍 능력, 문제 해결 능력이 중요합니다."
        elif job == "생산·제조":
            job_specific_description = "생산 및 제조 분야는 공정 이해도, 품질 관리 능력, 현장 경험이 중요합니다."
        elif job == "영업·고객상담":
            job_specific_description = "영업 및 고객상담 분야는 설득력, 대인관계 능력, 고객 니즈 파악 능력이 중요합니다."
        elif job == "건설":
            job_specific_description = "건설 분야는 전문 기술, 현장 경험, 안전 관리 능력이 중요합니다."
        elif job == "금융":
            job_specific_description = "금융 분야는 수리 능력, 경제 이해도, 분석력이 중요합니다."
        elif job == "연구개발·설계":
            job_specific_description = "연구개발 및 설계 분야는 전문 지식, 분석력, 창의적 문제 해결 능력이 중요합니다."
        elif job == "디자인":
            job_specific_description = "디자인 분야는 창의성, 트렌드 감각, 시각적 표현 능력이 중요합니다."
        elif job == "미디어":
            job_specific_description = "미디어 분야는 콘텐츠 제작 능력, 스토리텔링, 커뮤니케이션 능력이 중요합니다."
        elif job == "전문·특수직":
            job_specific_description = "전문 및 특수직은 해당 분야의 깊은 전문성, 자격증, 경험이 중요합니다."
        
        prompt = f"""
        다음은 '{job}' 직종에 지원한 지원자의 스펙입니다. 
        {job_specific_description}
        이 스펙을 100점 만점으로 평가해주세요.

        [지원자 정보]
        닉네임: {spec_data.get('nickname', '이름 없음')}
        최종학력: {spec_data.get('final_edu', '정보 없음')} ({spec_data.get('final_status', '정보 없음')})
        지원 직종: {job}

        """
        
        # 대학 정보 추가
        if "universities" in spec_data and spec_data["universities"]:
            prompt += "[학력 사항]\n"
            for uni in spec_data["universities"]:
                gpa_info = f"{uni.get('gpa', '정보 없음')}/{uni.get('gpa_max', '정보 없음')}" if 'gpa' in uni else "정보 없음"
                prompt += f"- {uni.get('school_name', '정보 없음')}, {uni.get('degree', '정보 없음')}, {uni.get('major', '정보 없음')}, GPA: {gpa_info}\n"
        
        # 경력 정보 추가 (work_month 포함)
        if "careers" in spec_data and spec_data["careers"]:
            prompt += "\n[경력 사항]\n"
            for career in spec_data["careers"]:
                work_month_info = f", 근무기간: {career.get('work_month', '정보 없음')}개월" if 'work_month' in career else ""
                prompt += f"- {career.get('company', '정보 없음')}, {career.get('role', '정보 없음')}{work_month_info}\n"
        
        # 자격증 정보 추가
        if "certificates" in spec_data and spec_data["certificates"]:
            prompt += "\n[자격증]\n"
            for cert in spec_data["certificates"]:
                prompt += f"- {cert}\n"
        
        # 어학 정보 추가
        if "languages" in spec_data and spec_data["languages"]:
            prompt += "\n[어학 능력]\n"
            for lang in spec_data["languages"]:
                prompt += f"- {lang.get('test', '정보 없음')}: {lang.get('score_or_grade', '정보 없음')}\n"
        
        # 활동 정보 추가
        if "activities" in spec_data and spec_data["activities"]:
            prompt += "\n[대외 활동]\n"
            for activity in spec_data["activities"]:
                award_info = f", 수상: {activity.get('award')}" if activity.get('award') else ""
                prompt += f"- {activity.get('name', '정보 없음')}, 역할: {activity.get('role', '정보 없음')}{award_info}\n"
        
        # 평가 지시 추가
        prompt += f"""
        위 지원자의 스펙을 '{job}' 직종에 적합한지 평가하여 점수를 매겨주세요.
        점수는 0-100 사이의 숫자로만 표현해 주세요.

        종합 평가 점수:"""
        
        return prompt
    
    def generate_cache_key(self, spec_data):
        """스펙 데이터로부터 캐시 키 생성"""
        # JSON 직렬화 및 정렬하여 일관된 해시 생성
        spec_json = json.dumps(spec_data, sort_keys=True)
        return hashlib.md5(spec_json.encode()).hexdigest()

    def get_from_cache(self, cache_key):
        """캐시에서 결과 조회"""
        if self.use_redis:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                print(f"Redis 캐시 조회 실패: {e}")
        
        # Redis 실패 시 메모리 캐시 확인
        return self.memory_cache.get(cache_key)

    def save_to_cache(self, cache_key, result):
        """결과를 캐시에 저장"""
        if self.use_redis:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(result)
                )
            except Exception as e:
                print(f"Redis 캐시 저장 실패: {e}")
                # Redis 실패 시 메모리 캐시 사용
                self.memory_cache[cache_key] = result
        else:
            # Redis 미사용 시 메모리 캐시 사용
            self.memory_cache[cache_key] = result
            
            # 메모리 캐시 크기 제한 (1000개 항목)
            if len(self.memory_cache) > 1000:
                # 가장 오래된 키 하나 제거
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]

    def evaluate_with_llm(self, prompt, spec_data):  # spec_data 파라미터 추가
        """XGLM 모델을 사용하여 스펙을 평가 (KV 캐시 활용)"""
        # 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 모델 추론 (KV 캐시 활용)
        with torch.no_grad():
            # KV 캐시 사용 설정
            if self.use_kv_cache:
                # use_cache=True를 명시적으로 설정하여 KV 캐시 활성화
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=5,  # 점수만 생성하므로 짧게 설정
                    temperature=0.7,   # 약간의 다양성 허용
                    top_p=0.9,         # 핵심 토큰에 집중
                    do_sample=True,    # 샘플링 사용
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,    # KV 캐시 활성화
                )
            else:
                # KV 캐시를 사용하지 않는 경우
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=5,  # 점수만 생성하므로 짧게 설정
                    temperature=0.7,   # 약간의 다양성 허용
                    top_p=0.9,         # 핵심 토큰에 집중
                    do_sample=True,    # 샘플링 사용
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,   # KV 캐시 비활성화
                )
        
        # 결과 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 이후 생성된 텍스트만 추출
        generated_score = generated_text[len(prompt):]
        
        # 생성된 텍스트에서 숫자만 추출 시도
        try:
            # 숫자 형태의 문자열 추출
            import re
            score_match = re.search(r'\d+(\.\d+)?', generated_score)  # 소수점도 인식하도록 수정
            if score_match:
                score = float(score_match.group())  # 정수가 아닌 실수로 변환
                # 점수 범위 제한
                score = max(0.0, min(100.0, score))
                return score
            else:
                # 숫자를 찾지 못한 경우 규칙 기반 평가로 대체
                return self.rule_based_evaluate(spec_data)
        except:
            # 오류 발생 시 규칙 기반 평가로 대체
            return self.rule_based_evaluate(spec_data)

    def rule_based_evaluate(self, spec_data):
        """규칙 기반 평가 방식 (LLM 평가가 실패할 경우 백업)"""
        scores = {}
        
        # 직무에 따른 가중치 조정
        job = spec_data.get("desired_job", "일반")
        weights = self.weights.copy()
        
        if job in self.job_weights:
            for key, adjustment in self.job_weights[job].items():
                weights[key] = max(0.01, min(0.95, weights[key] + adjustment))  # 가중치가 너무 크거나 작아지지 않도록
        
        # 정규화를 위해 가중치 합을 1로 조정
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        # 학력 평가
        uni_score = 0
        if "universities" in spec_data and spec_data["universities"]:
            uni_count = len(spec_data["universities"])
            uni_score = 0
            for uni in spec_data["universities"]:
                # 대학 점수 계산
                school_score = 0
                school_name = uni.get("school_name", "").lower()
                
                # 학교 이름 기반 점수 (단순화된 예시)
                if "서울대" in school_name or "연세대" in school_name or "고려대" in school_name:
                    school_score = 95
                elif "성균관대" in school_name or "한양대" in school_name:
                    school_score = 90
                elif "중앙대" in school_name or "경희대" in school_name or "이화여대" in school_name:
                    school_score = 85
                elif "대학교" in school_name:
                    school_score = 80
                else:
                    school_score = 75
                
                # 학위 가중치
                degree = uni.get("degree", "")
                degree_weight = {
                    "박사": 1.2,
                    "석사": 1.1,
                    "학사": 1.0,
                    "전문학사": 0.9,
                    "수료": 0.8
                }.get(degree, 0.8)
                
                # GPA 점수 (정규화)
                gpa = uni.get("gpa", 0)
                gpa_max = uni.get("gpa_max", 4.5)
                
                if gpa and gpa_max:
                    gpa_score = (gpa / gpa_max) * 100
                else:
                    gpa_score = 75  # 기본값
                
                # 전공 관련성 (직무별 관련 전공 매핑)
                major = uni.get("major", "").lower()
                major_relevance = 1.0  # 기본값
                
                # 지원 직무와 전공 관련성
                if job == "경영·사무":
                    if "경영" in major or "경제" in major or "회계" in major or "통계" in major:
                        major_relevance = 1.2
                    elif "행정" in major or "사무" in major:
                        major_relevance = 1.1
                    elif "인문" in major or "사회" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "마케팅·광고·홍보":
                    if "마케팅" in major or "광고" in major or "홍보" in major or "미디어" in major:
                        major_relevance = 1.2
                    elif "경영" in major or "커뮤니케이션" in major or "심리" in major:
                        major_relevance = 1.1
                    elif "디자인" in major or "사회" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "무역·유통":
                    if "무역" in major or "유통" in major or "물류" in major:
                        major_relevance = 1.2
                    elif "경영" in major or "경제" in major or "국제" in major:
                        major_relevance = 1.1
                    elif "사회" in major or "외국어" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "인터넷·IT":
                    if "컴퓨터" in major or "소프트웨어" in major or "전산" in major or "정보" in major:
                        major_relevance = 1.2
                    elif "전자" in major or "통신" in major:
                        major_relevance = 1.1
                    elif "수학" in major or "자연과학" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "생산·제조":
                    if "기계" in major or "산업" in major or "제조" in major or "생산" in major:
                        major_relevance = 1.2
                    elif "공학" in major or "재료" in major:
                        major_relevance = 1.1
                    elif "화학" in major or "물리" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "영업·고객상담":
                    if "영업" in major or "마케팅" in major or "경영" in major:
                        major_relevance = 1.2
                    elif "심리" in major or "커뮤니케이션" in major:
                        major_relevance = 1.1
                    elif "사회" in major or "인문" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "건설":
                    if "건축" in major or "토목" in major or "건설" in major:
                        major_relevance = 1.2
                    elif "도시" in major or "공학" in major:
                        major_relevance = 1.1
                    elif "자연과학" in major or "지리" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "금융":
                    if "금융" in major or "경제" in major or "경영" in major or "통계" in major:
                        major_relevance = 1.2
                    elif "회계" in major or "수학" in major:
                        major_relevance = 1.1
                    elif "컴퓨터" in major or "사회" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "연구개발·설계":
                    if "공학" in major or "자연과학" in major or "연구" in major:
                        major_relevance = 1.2
                    elif "수학" in major or "통계" in major:
                        major_relevance = 1.1
                    elif "컴퓨터" in major or "기술" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "디자인":
                    if "디자인" in major or "시각" in major or "산업디자인" in major:
                        major_relevance = 1.2
                    elif "미술" in major or "예술" in major:
                        major_relevance = 1.1
                    elif "건축" in major or "컴퓨터" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "미디어":
                    if "미디어" in major or "방송" in major or "언론" in major or "영상" in major:
                        major_relevance = 1.2
                    elif "커뮤니케이션" in major or "광고" in major:
                        major_relevance = 1.1
                    elif "문학" in major or "사회" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "전문·특수직":
                    if "법학" in major or "의학" in major or "약학" in major or "특수" in major:
                        major_relevance = 1.2
                    elif "교육" in major or "심리" in major:
                        major_relevance = 1.1
                    elif "사회" in major or "복지" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                # 학력 총합 점수
                uni_item_score = (school_score * 0.4 + gpa_score * 0.3) * degree_weight * major_relevance
                uni_score += uni_item_score
            
            # 여러 대학이 있을 경우 평균 내기
            uni_score = uni_score / uni_count
        
        scores["universities"] = uni_score
        
        # 경력 평가 - work_month 반영
        career_score = 0
        if "careers" in spec_data and spec_data["careers"]:
            career_count = len(spec_data["careers"])
            for career in spec_data["careers"]:
                company = career.get("company", "").lower()
                role = career.get("role", "").lower()
                
                # 회사 규모/인지도 점수 (단순화)
                company_score = 0
                if "삼성" in company or "네이버" in company or "카카오" in company or "lg" in company:
                    company_score = 95
                elif "기업" in company or "그룹" in company or "주식회사" in company:
                    company_score = 85
                else:
                    company_score = 75
                
                # 역할 관련성 점수
                role_score = 0
                
                if job == "경영·사무":
                    if "사무" in role or "총무" in role or "인사" in role or "행정" in role:
                        role_score = 95
                    elif "경영" in role or "관리" in role:
                        role_score = 90
                    elif "비서" in role or "사원" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "마케팅·광고·홍보":
                    if "마케팅" in role or "광고" in role or "홍보" in role or "브랜드" in role:
                        role_score = 95
                    elif "기획" in role or "카피" in role:
                        role_score = 90
                    elif "디지털" in role or "sns" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "무역·유통":
                    if "무역" in role or "유통" in role or "물류" in role or "구매" in role:
                        role_score = 95
                    elif "수출" in role or "수입" in role:
                        role_score = 90
                    elif "통관" in role or "운송" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "인터넷·IT":
                    if "개발" in role or "프로그래머" in role or "엔지니어" in role:
                        role_score = 95
                    elif "웹" in role or "앱" in role or "서버" in role:
                        role_score = 90
                    elif "디자인" in role or "기획" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "생산·제조":
                    if "생산" in role or "제조" in role or "공정" in role:
                        role_score = 95
                    elif "품질" in role or "관리" in role:
                        role_score = 90
                    elif "조립" in role or "검사" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "영업·고객상담":
                    if "영업" in role or "세일즈" in role or "판매" in role:
                        role_score = 95
                    elif "상담" in role or "고객" in role or "cs" in role:
                        role_score = 90
                    elif "매장" in role or "관리" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "건설":
                    if "건설" in role or "현장" in role or "공사" in role:
                        role_score = 95
                    elif "감리" in role or "설계" in role:
                        role_score = 90
                    elif "안전" in role or "관리" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "금융":
                    if "금융" in role or "은행" in role or "투자" in role:
                        role_score = 95
                    elif "회계" in role or "경리" in role:
                        role_score = 90
                    elif "자산" in role or "관리" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "연구개발·설계":
                    if "연구" in role or "개발" in role or "r&d" in role:
                        role_score = 95
                    elif "설계" in role or "디자인" in role:
                        role_score = 90
                    elif "분석" in role or "기술" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "디자인":
                    if "디자인" in role or "그래픽" in role or "ui" in role or "ux" in role:
                        role_score = 95
                    elif "일러스트" in role or "시각" in role:
                        role_score = 90
                    elif "편집" in role or "웹" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "미디어":
                    if "미디어" in role or "방송" in role or "pd" in role or "기자" in role:
                        role_score = 95
                    elif "영상" in role or "촬영" in role:
                        role_score = 90
                    elif "편집" in role or "작가" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                elif job == "전문·특수직":
                    if "의사" in role or "변호사" in role or "회계사" in role or "교수" in role:
                        role_score = 95
                    elif "교사" in role or "전문가" in role:
                        role_score = 90
                    elif "약사" in role or "상담사" in role:
                        role_score = 85
                    else:
                        role_score = 75
                
                # 근무 기간 가중치 (work_month 반영)
                work_month = career.get("work_month", 0)
                experience_weight = 1.0  # 기본값
                
                # 근무 기간에 따른 가중치 (로그 스케일로 점진적 증가)
                if work_month:
                    # 6개월 미만: 0.8, 6-12개월: 1.0, 12-24개월: 1.2, 24-36개월: 1.3, 36개월 이상: 1.5
                    if work_month < 6:
                        experience_weight = 0.8
                    elif work_month < 12:
                        experience_weight = 1.0
                    elif work_month < 24:
                        experience_weight = 1.2
                    elif work_month < 36:
                        experience_weight = 1.3
                    else:
                        experience_weight = 1.5
                
                # 경력 점수 계산 (회사 30%, 역할 50%, 경력 기간 20%)
                career_item_score = (company_score * 0.3 + role_score * 0.5) * experience_weight
                career_score += career_item_score
            
            # 여러 경력이 있을 경우 평균
            career_score = career_score / career_count
        
        scores["careers"] = career_score
        # 자격증 평가
        cert_score = 0
        if "certificates" in spec_data and spec_data["certificates"]:
            cert_count = len(spec_data["certificates"])
            for cert in spec_data["certificates"]:
                cert = cert.lower()
                
                # 직무 관련 자격증 점수
                if job == "경영·사무":
                    if "경영지도사" in cert or "사무" in cert or "행정사" in cert:
                        cert_score += 95
                    elif "세무" in cert or "회계" in cert or "컴퓨터활용능력" in cert:
                        cert_score += 90
                    elif "엑셀" in cert or "워드" in cert or "정보처리" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "마케팅·광고·홍보":
                    if "마케팅" in cert or "광고" in cert or "홍보" in cert:
                        cert_score += 95
                    elif "사회조사" in cert or "빅데이터" in cert:
                        cert_score += 90
                    elif "인터넷" in cert or "통계" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "무역·유통":
                    if "무역" in cert or "유통" in cert or "물류" in cert or "관세사" in cert:
                        cert_score += 95
                    elif "국제" in cert or "수출" in cert:
                        cert_score += 90
                    elif "유통" in cert or "무역영어" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "인터넷·IT":
                    if "정보처리" in cert or "aws" in cert or "데이터베이스" in cert or "cloud" in cert:
                        cert_score += 95
                    elif "네트워크" in cert or "보안" in cert:
                        cert_score += 90
                    elif "코딩" in cert or "프로그래밍" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "생산·제조":
                    if "품질" in cert or "생산" in cert or "제조" in cert:
                        cert_score += 95
                    elif "안전" in cert or "기계" in cert:
                        cert_score += 90
                    elif "전기" in cert or "관리" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "영업·고객상담":
                    if "영업" in cert or "판매" in cert or "고객" in cert:
                        cert_score += 95
                    elif "마케팅" in cert or "cs" in cert:
                        cert_score += 90
                    elif "세일즈" in cert or "상담" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "건설":
                    if "건축" in cert or "토목" in cert or "건설" in cert:
                        cert_score += 95
                    elif "안전" in cert or "감리" in cert:
                        cert_score += 90
                    elif "설비" in cert or "cad" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "금융":
                    if "금융" in cert or "증권" in cert or "회계사" in cert or "투자" in cert:
                        cert_score += 95
                    elif "은행" in cert or "보험" in cert:
                        cert_score += 90
                    elif "재무" in cert or "펀드" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "연구개발·설계":
                    if "기술사" in cert or "연구" in cert or "설계" in cert:
                        cert_score += 95
                    elif "기사" in cert or "r&d" in cert:
                        cert_score += 90
                    elif "cad" in cert or "분석" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "디자인":
                    if "디자인" in cert or "그래픽" in cert or "ui" in cert:
                        cert_score += 95
                    elif "포토샵" in cert or "일러스트" in cert:
                        cert_score += 90
                    elif "멀티미디어" in cert or "웹" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                # 미디어 분야 자격증 평가 계속
                elif job == "미디어":
                    if "방송" in cert or "미디어" in cert or "언론" in cert:
                        cert_score += 95
                    elif "영상" in cert or "촬영" in cert:
                        cert_score += 90
                    elif "편집" in cert or "작가" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                elif job == "전문·특수직":
                    if "변호사" in cert or "의사" in cert or "약사" in cert or "회계사" in cert:
                        cert_score += 95
                    elif "세무사" in cert or "교사" in cert:
                        cert_score += 90
                    elif "상담사" in cert or "공인" in cert:
                        cert_score += 85
                    else:
                        cert_score += 75
                
                else:  # 기타 직종
                    cert_score += 80  # 기본 점수
            
            # 자격증 평균 점수
            cert_score = cert_score / cert_count
        
        scores["certificates"] = cert_score
        
        # 어학 능력 평가
        lang_score = 0
        if "languages" in spec_data and spec_data["languages"]:
            lang_count = len(spec_data["languages"])
            for lang in spec_data["languages"]:
                test = lang.get("test", "").upper()
                score_or_grade = lang.get("score_or_grade", "")
                
                # 각 시험별 점수 평가
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
                
                elif test == "TOEFL":
                    try:
                        toefl_score = float(score_or_grade)
                        if toefl_score >= 100:
                            test_score = 95
                        elif toefl_score >= 90:
                            test_score = 90
                        elif toefl_score >= 80:
                            test_score = 85
                        elif toefl_score >= 70:
                            test_score = 80
                        else:
                            test_score = 75
                    except:
                        test_score = 75
                
                elif test == "OPIC":
                    grade_map = {
                        "AL": 95, "IH": 90, "IM3": 85, "IM2": 80, 
                        "IM1": 75, "IL": 70, "NH": 65, "NM": 60, "NL": 55
                    }
                    test_score = grade_map.get(score_or_grade, 75)
                
                else:  # 기타 어학 시험
                    test_score = 80  # 기본 점수
                
                # 직무별 어학 중요도 가중치 부여
                job_lang_weight = 1.0
                
                if job == "무역·유통":
                    job_lang_weight = 1.3  # 무역은 외국어 매우 중요
                elif job == "영업·고객상담":
                    job_lang_weight = 1.2  # 글로벌 고객 응대 가능성
                elif job == "마케팅·광고·홍보" or job == "금융" or job == "미디어":
                    job_lang_weight = 1.1  # 약간 중요
                elif job == "생산·제조" or job == "건설":
                    job_lang_weight = 0.9  # 약간 덜 중요
                
                lang_score += test_score * job_lang_weight
            
            # 어학 평균 점수
            lang_score = lang_score / lang_count
        
        scores["languages"] = lang_score
        
        # 활동 평가
        activity_score = 0
        if "activities" in spec_data and spec_data["activities"]:
            activity_count = len(spec_data["activities"])
            for activity in spec_data["activities"]:
                name = activity.get("name", "").lower()
                role = activity.get("role", "").lower()
                award = activity.get("award", "").lower()
                
                # 활동 점수 계산
                activity_item_score = 0
                
                # 역할 가중치
                role_weight = 1.0
                if "회장" in role or "대표" in role:
                    role_weight = 1.2
                elif "임원" in role or "팀장" in role:
                    role_weight = 1.1
                elif "부원" in role or "사원" in role:
                    role_weight = 0.9
                
                # 수상 가중치
                award_weight = 1.0
                if award:
                    if "대상" in award or "금상" in award:
                        award_weight = 1.3
                    elif "은상" in award or "우수상" in award:
                        award_weight = 1.2
                    elif "동상" in award or "장려상" in award:
                        award_weight = 1.1
                
                # 직무 관련성 점수
                relevance_score = 0
                
                if job == "경영·사무":
                    if "경영" in name or "기획" in name or "창업" in name:
                        relevance_score = 95
                    elif "행정" in name or "사무" in name:
                        relevance_score = 90
                    elif "경제" in name or "회계" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "마케팅·광고·홍보":
                    if "마케팅" in name or "광고" in name or "홍보" in name:
                        relevance_score = 95
                    elif "브랜드" in name or "기획" in name:
                        relevance_score = 90
                    elif "디자인" in name or "sns" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "무역·유통":
                    if "무역" in name or "유통" in name or "국제" in name:
                        relevance_score = 95
                    elif "비즈니스" in name or "교류" in name:
                        relevance_score = 90
                    elif "외국어" in name or "경제" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "인터넷·IT":
                    if "개발" in name or "코딩" in name or "프로그래밍" in name or "해커톤" in name:
                        relevance_score = 95
                    elif "전산" in name or "컴퓨터" in name or "it" in name:
                        relevance_score = 90
                    elif "소프트웨어" in name or "스타트업" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "생산·제조":
                    if "생산" in name or "제조" in name or "공정" in name:
                        relevance_score = 95
                    elif "품질" in name or "관리" in name:
                        relevance_score = 90
                    elif "공학" in name or "설계" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "영업·고객상담":
                    if "영업" in name or "세일즈" in name or "판매" in name:
                        relevance_score = 95
                    elif "고객" in name or "상담" in name:
                        relevance_score = 90
                    elif "서비스" in name or "마케팅" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "건설":
                    if "건설" in name or "건축" in name or "토목" in name:
                        relevance_score = 95
                    elif "설계" in name or "감리" in name:
                        relevance_score = 90
                    elif "공학" in name or "안전" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "금융":
                    if "금융" in name or "투자" in name or "경제" in name:
                        relevance_score = 95
                    elif "회계" in name or "재무" in name:
                        relevance_score = 90
                    elif "증권" in name or "은행" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "연구개발·설계":
                    if "연구" in name or "개발" in name or "r&d" in name:
                        relevance_score = 95
                    elif "설계" in name or "분석" in name:
                        relevance_score = 90
                    elif "과학" in name or "공학" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "디자인":
                    if "디자인" in name or "창작" in name or "미술" in name:
                        relevance_score = 95
                    elif "시각" in name or "ui" in name:
                        relevance_score = 90
                    elif "그래픽" in name or "편집" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "미디어":
                    if "미디어" in name or "방송" in name or "언론" in name:
                        relevance_score = 95
                    elif "영상" in name or "기자" in name:
                        relevance_score = 90
                    elif "촬영" in name or "편집" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "전문·특수직":
                    if "법률" in name or "의료" in name or "전문" in name:
                        relevance_score = 95
                    elif "교육" in name or "연구" in name:
                        relevance_score = 90
                    elif "상담" in name or "봉사" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                else:  # 기타 직종
                    relevance_score = 80  # 기본 점수
                
                # 활동 총 점수
                activity_item_score = relevance_score * role_weight * award_weight
                activity_score += activity_item_score
            
            # 활동 평균 점수
            activity_score = activity_score / activity_count
        
        scores["activities"] = activity_score
        
        # 종합 점수 계산
        total_score = 0
        for key, score in scores.items():
            if score > 0:  # 점수가 있는 항목만 계산
                total_score += score * weights.get(key, 0.1)
        
        # 만약 모든 항목이 0점이면 기본 점수 반환
        if total_score == 0:
            total_score = 65
        
        # 점수 정규화 (좀 더 넓은 구간으로 분산)
        total_score = self.min_score + (total_score / 100) * (self.max_score - self.min_score)
        
        # 포트폴리오 보너스 점수 (V2 API의 filelink가 있는 경우)
        if "filelink" in spec_data and spec_data["filelink"]:
            portfolio_bonus = 3  # 포트폴리오 제출 시 3점 추가
            total_score = min(self.max_score, total_score + portfolio_bonus)
        
        # 소수점 2자리까지 반환 (반올림하지 않고 소수점 유지)
        return round(total_score, 2)

    def predict(self, spec_data):
        """스펙 정보를 받아 평가 결과 반환 (캐싱 적용)"""
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
            
            # JSON 문자열로 변환하여 캐시 가능한 형태로 만듦
            spec_data_json = json.dumps(spec_data, sort_keys=True)
            # 프롬프트 생성
            prompt = self.prepare_prompt_cached(spec_data_json)
            
            # 시작 시간 기록
            start_time = time.time()
            
            # LLM으로 평가 - spec_data도 함께 전달
            score = self.evaluate_with_llm(prompt, spec_data)
            
            # 소요 시간 계산
            elapsed_time = time.time() - start_time
            print(f"모델 추론 시간: {elapsed_time:.2f}초")
            
            # 결과 생성 - 소수점 2자리 유지
            result = {
                "nickname": spec_data.get("nickname", "이름 없음"),
                "totalScore": score
            }
            
            # 결과 캐싱
            self.save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            # 오류 발생 시 규칙 기반 평가로 대체
            score = self.rule_based_evaluate(spec_data)
            return {
                "nickname": spec_data.get("nickname", "이름 없음"),
                "totalScore": score
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