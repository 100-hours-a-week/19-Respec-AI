import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from tqdm import tqdm

class SpecEvaluator:
    def __init__(self):
        print("XGLM-564M 모델 로딩 중...")
        # XGLM-564M 모델 로드
        self.model_name = "facebook/xglm-564M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # GPU 사용 가능하면 GPU로 이동
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 중인 디바이스: {self.device}")
        self.model.to(self.device)
        
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
            "백엔드 개발자": {
                "universities": 0.05,   # 학력 가중치 증가
                "careers": 0.10,        # 경력 가중치 증가
                "certificates": 0.05,   # 자격증 가중치 증가
                "languages": -0.10,     # 어학 가중치 감소
                "activities": -0.10     # 활동 가중치 감소
            },
            "프론트엔드 개발자": {
                "universities": -0.05,
                "careers": 0.10,
                "certificates": 0.05,
                "languages": -0.05,
                "activities": -0.05
            },
            "데이터 사이언티스트": {
                "universities": 0.10,
                "careers": 0.05,
                "certificates": 0.05,
                "languages": -0.10,
                "activities": -0.10
            },
            "마케팅": {
                "universities": -0.10,
                "careers": 0.10,
                "languages": 0.10,
                "certificates": -0.05,
                "activities": -0.05
            }
        }
        
        # 점수 정규화 파라미터
        self.min_score = 50  # 최소 점수
        self.max_score = 95  # 최대 점수
        
        print("모델 초기화 완료!")

    def prepare_prompt(self, spec_data):
        """스펙 데이터를 평가하기 위한 프롬프트 준비"""
        # 지원직종에 따른 맞춤형 프롬프트 생성
        job = spec_data.get("desired_job", "일반")
        
        prompt = f"""
        다음은 '{job}' 직종에 지원한 지원자의 스펙입니다. 
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
        
        # 경력 정보 추가
        if "careers" in spec_data and spec_data["careers"]:
            prompt += "\n[경력 사항]\n"
            for career in spec_data["careers"]:
                prompt += f"- {career.get('company', '정보 없음')}, {career.get('role', '정보 없음')}\n"
        
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

    def evaluate_with_llm(self, prompt):
        """XGLM 모델을 사용하여 스펙을 평가"""
        # 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 모델 추론
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=5,  # 점수만 생성하므로 짧게 설정
                temperature=0.7,   # 약간의 다양성 허용
                top_p=0.9,         # 핵심 토큰에 집중
                do_sample=True,    # 샘플링 사용
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 이후 생성된 텍스트만 추출
        generated_score = generated_text[len(prompt):]
        
        # 생성된 텍스트에서 숫자만 추출 시도
        try:
            # 숫자 형태의 문자열 추출
            import re
            score_match = re.search(r'\d+', generated_score)
            if score_match:
                score = int(score_match.group())
                # 점수 범위 제한
                score = max(0, min(100, score))
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
                
                # 전공 관련성 (단순화)
                major = uni.get("major", "").lower()
                major_relevance = 1.0  # 기본값
                
                # 지원 직무와 전공 관련성
                if job == "백엔드 개발자" or job == "프론트엔드 개발자":
                    if "컴퓨터" in major or "소프트웨어" in major or "전산" in major or "정보" in major:
                        major_relevance = 1.2
                    elif "전자" in major or "공학" in major:
                        major_relevance = 1.1
                    elif "자연과학" in major or "수학" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "데이터 사이언티스트":
                    if "통계" in major or "데이터" in major or "수학" in major:
                        major_relevance = 1.2
                    elif "컴퓨터" in major or "전산" in major:
                        major_relevance = 1.1
                    elif "공학" in major or "자연과학" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                elif job == "마케팅":
                    if "마케팅" in major or "경영" in major or "광고" in major:
                        major_relevance = 1.2
                    elif "커뮤니케이션" in major or "미디어" in major or "심리" in major:
                        major_relevance = 1.1
                    elif "사회과학" in major or "인문" in major:
                        major_relevance = 1.0
                    else:
                        major_relevance = 0.9
                
                # 학력 총합 점수
                uni_item_score = (school_score * 0.4 + gpa_score * 0.3) * degree_weight * major_relevance
                uni_score += uni_item_score
            
            # 여러 대학이 있을 경우 평균 내기
            uni_score = uni_score / uni_count
        
        scores["universities"] = uni_score
        
        # 경력 평가
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
                elif "기업" in company or "그룹" in company:
                    company_score = 85
                else:
                    company_score = 75
                
                # 역할 관련성 점수
                role_score = 0
                if job == "백엔드 개발자":
                    if "백엔드" in role or "서버" in role or "개발자" in role or "엔지니어" in role:
                        role_score = 95
                    elif "개발" in role or "프로그래머" in role:
                        role_score = 85
                    else:
                        role_score = 70
                
                elif job == "프론트엔드 개발자":
                    if "프론트엔드" in role or "ui" in role or "ux" in role or "웹" in role:
                        role_score = 95
                    elif "개발" in role or "디자이너" in role:
                        role_score = 85
                    else:
                        role_score = 70
                
                elif job == "데이터 사이언티스트":
                    if "데이터" in role or "분석" in role or "사이언티스트" in role:
                        role_score = 95
                    elif "연구" in role or "통계" in role:
                        role_score = 85
                    else:
                        role_score = 70
                
                elif job == "마케팅":
                    if "마케팅" in role or "광고" in role or "브랜드" in role:
                        role_score = 95
                    elif "홍보" in role or "기획" in role:
                        role_score = 85
                    else:
                        role_score = 70
                
                # 경력 점수 계산 (회사 40%, 역할 60%)
                career_item_score = company_score * 0.4 + role_score * 0.6
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
                
                # 직무 관련 자격증 점수 (단순화)
                if job == "백엔드 개발자" or job == "프론트엔드 개발자":
                    if "정보처리" in cert or "aws" in cert or "데이터베이스" in cert or "cloud" in cert:
                        cert_score += 95
                    elif "코딩" in cert or "프로그래밍" in cert or "개발" in cert:
                        cert_score += 85
                    else:
                        cert_score += 70
                
                elif job == "데이터 사이언티스트":
                    if "데이터" in cert or "분석" in cert or "통계" in cert or "빅데이터" in cert:
                        cert_score += 95
                    elif "python" in cert or "sql" in cert or "머신러닝" in cert:
                        cert_score += 85
                    else:
                        cert_score += 70
                
                elif job == "마케팅":
                    if "마케팅" in cert or "광고" in cert or "디지털마케팅" in cert:
                        cert_score += 95
                    elif "사회조사" in cert or "포토샵" in cert:
                        cert_score += 85
                    else:
                        cert_score += 70
                
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
                
                lang_score += test_score
            
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
                if job == "백엔드 개발자" or job == "프론트엔드 개발자":
                    if "개발" in name or "코딩" in name or "프로그래밍" in name or "해커톤" in name:
                        relevance_score = 95
                    elif "전산" in name or "컴퓨터" in name or "it" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "데이터 사이언티스트":
                    if "데이터" in name or "분석" in name or "통계" in name:
                        relevance_score = 95
                    elif "ai" in name or "머신러닝" in name:
                        relevance_score = 85
                    else:
                        relevance_score = 75
                
                elif job == "마케팅":
                    if "마케팅" in name or "광고" in name or "홍보" in name:
                        relevance_score = 95
                    elif "기획" in name or "브랜드" in name:
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
        
        # 점수 정규화
        total_score = self.min_score + (total_score / 100) * (self.max_score - self.min_score)
        total_score = round(total_score)
        
        return total_score

    def predict(self, spec_data):
        """스펙 정보를 받아 평가 결과 반환"""
        try:
            # 프롬프트 생성
            prompt = self.prepare_prompt(spec_data)
            
            # LLM으로 평가
            score = self.evaluate_with_llm(prompt)
            
            # 결과 반환
            return {
                "nickname": spec_data.get("nickname", "이름 없음"),
                "totalScore": score
            }
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            # 오류 발생 시 규칙 기반 평가로 대체
            score = self.rule_based_evaluate(spec_data)
            return {
                "nickname": spec_data.get("nickname", "이름 없음"),
                "totalScore": score
            }
