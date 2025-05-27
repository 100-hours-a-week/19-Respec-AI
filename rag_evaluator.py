import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple
import json
from vector_database import VectorDatabase
from embedding_utils import EmbeddingUtils

class ResumeEvaluationRAG:
    """RAG 기반 이력서 평가 시스템"""
    
    def __init__(self, db_config: Dict, model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"):
        # LLM 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 패딩 토큰 설정 - 핵심 수정사항
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 벡터 데이터베이스 및 임베딩 유틸리티 초기화
        self.vector_db = VectorDatabase(db_config)
        self.embedding_utils = EmbeddingUtils()
        
        # RAG 검색 설정
        self.similarity_threshold = 0.7  # 유사도 0.7 이상만 유효한 매칭
        self.default_score = 0.3  # 매칭되지 않는 경우 기본 점수
        
        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_model.to(self.device)
    
    def evaluate_education(self, universities: List[Dict], job_field: str) -> Dict:
        """학력 평가 - RAG 기반 전공 유사도 검색"""
        if not universities:
            return {"score": 0, "details": "학력 정보 없음"}
        
        total_score = 0
        details = []
        
        for univ in universities:
            major = univ.get('major', '')
            gpa = univ.get('gpa', 0)
            gpa_max = univ.get('gpa_max', 4.5)
            
            if not major:
                # 전공 정보가 없는 경우
                univ_score = 0.2
                details.append({
                    "major": "전공 정보 없음",
                    "matched_major": "매칭 불가",
                    "similarity": 0,
                    "total": univ_score
                })
            else:
                # RAG 기반 전공 유사도 검색
                similar_majors = self.vector_db.search_similar_majors(major, job_field, top_k=1)
                
                if similar_majors and len(similar_majors) > 0:
                    best_match = similar_majors[0]
                    base_relevance = best_match['relevance_score']
                    similarity = best_match['similarity']
                    
                    # 유사도 기반 동적 점수 계산
                    if similarity >= self.similarity_threshold:
                        dynamic_relevance = base_relevance * similarity
                    else:
                        dynamic_relevance = self.default_score
                    
                    # GPA 점수 계산
                    gpa_score = (gpa / gpa_max) if gpa_max > 0 else 0
                    
                    # 최종 점수 (전공 70% + GPA 30%)
                    univ_score = (dynamic_relevance * 0.7) + (gpa_score * 0.3)
                    
                    details.append({
                        "major": major,
                        "matched_major": best_match['major_name'],
                        "base_relevance": base_relevance,
                        "similarity": similarity,
                        "dynamic_relevance": dynamic_relevance,
                        "gpa_score": gpa_score,
                        "total": univ_score
                    })
                else:
                    # 매칭되는 전공이 없는 경우
                    univ_score = self.default_score * 0.7
                    details.append({
                        "major": major,
                        "matched_major": "매칭 없음",
                        "dynamic_relevance": self.default_score,
                        "gpa_score": (gpa / gpa_max) if gpa_max > 0 else 0,
                        "total": univ_score
                    })
            
            total_score += univ_score
        
        return {
            "score": min(total_score, 1.0),
            "details": details
        }
    
    def evaluate_certificates(self, certificates: List[str], job_field: str) -> Dict:
        """자격증 평가 - RAG 기반 자격증 유사도 검색"""
        if not certificates:
            return {"score": 0, "details": "자격증 없음"}
        
        total_weight = 0
        details = []
        processed_certs = set()
        
        for cert in certificates:
            # 각 자격증별로 RAG 검색 수행
            similar_certs = self.vector_db.search_similar_certificates(cert, job_field, top_k=1)
            
            if similar_certs and len(similar_certs) > 0:
                best_match = similar_certs[0]
                cert_name = best_match['certificate_name']
                base_weight = best_match['weight_score']
                similarity = best_match['similarity']
                
                # 중복 처리 방지
                cert_key = f"{cert}_{cert_name}"
                if cert_key not in processed_certs:
                    if similarity >= self.similarity_threshold:
                        dynamic_weight = base_weight * similarity
                    else:
                        dynamic_weight = self.default_score * 0.5
                    
                    total_weight += dynamic_weight
                    processed_certs.add(cert_key)
                    
                    details.append({
                        "input_certificate": cert,
                        "matched_certificate": cert_name,
                        "base_weight": base_weight,
                        "similarity": similarity,
                        "dynamic_weight": dynamic_weight
                    })
            else:
                # 매칭되지 않는 자격증
                default_weight = self.default_score * 0.3
                total_weight += default_weight
                details.append({
                    "input_certificate": cert,
                    "matched_certificate": "매칭 없음",
                    "dynamic_weight": default_weight
                })
        
        # 정규화 (최대 3개 주요 자격증 기준)
        normalized_score = min(total_weight / 2.5, 1.0)
        
        return {
            "score": normalized_score,
            "details": details
        }
    
    def evaluate_activities(self, activities: List[Dict], job_field: str) -> Dict:
        """활동 평가 - RAG 기반 활동 유사도 검색"""
        if not activities:
            return {"score": 0, "details": "활동 정보 없음"}
        
        total_score = 0
        details = []
        
        for activity in activities:
            activity_name = activity.get('name', '')
            activity_role = activity.get('role', '')
            
            # 활동명과 역할을 결합하여 더 정확한 평가
            combined_text = f"{activity_name} {activity_role}".strip()
            
            if not combined_text:
                continue
            
            # RAG 검색으로 유사한 활동 찾기
            similar_activities = self.vector_db.search_similar_activities(combined_text, job_field, top_k=1)
            
            if similar_activities and len(similar_activities) > 0:
                best_match = similar_activities[0]
                base_relevance = best_match['relevance_score']
                similarity = best_match['similarity']
                
                if similarity >= self.similarity_threshold:
                    dynamic_score = base_relevance * similarity
                    # 역할에 따른 추가 가중치
                    role_bonus = self.get_role_bonus(activity_role)
                    final_score = dynamic_score * role_bonus
                else:
                    final_score = self.default_score * 0.4
                
                details.append({
                    "activity_name": activity_name,
                    "role": activity_role,
                    "matched_activity": best_match['activity_keyword'],
                    "base_relevance": base_relevance,
                    "similarity": similarity,
                    "role_bonus": role_bonus if similarity >= self.similarity_threshold else 1.0,
                    "final_score": final_score
                })
            else:
                # 매칭되지 않는 활동
                final_score = self.default_score * 0.2
                details.append({
                    "activity_name": activity_name,
                    "role": activity_role,
                    "matched_activity": "매칭 없음",
                    "final_score": final_score
                })
            
            total_score += final_score
        
        # 정규화 (활동 수에 따른 평균)
        normalized_score = min(total_score / len(activities), 1.0) if activities else 0
        
        return {
            "score": normalized_score,
            "details": details
        }
    
    def get_role_bonus(self, role: str) -> float:
        """역할에 따른 보너스 점수"""
        if not role:
            return 1.0
            
        role_lower = role.lower()
        
        if any(keyword in role_lower for keyword in ['회장', '대표', '팀장', '리더']):
            return 1.2
        elif any(keyword in role_lower for keyword in ['부회장', '부팀장', '부대표']):
            return 1.1
        elif any(keyword in role_lower for keyword in ['운영진', '임원', '기획']):
            return 1.05
        else:
            return 1.0
    
    def evaluate_experience(self, careers: List[Dict], job_field: str) -> Dict:
        """경력 평가 - 기존 로직 유지하되 RAG 확장 가능"""
        if not careers:
            return {"score": 0, "details": "경력 없음"}
        
        total_months = 0
        details = []
        
        for career in careers:
            work_month = career.get('work_month', 0)
            company = career.get('company', '')
            role = career.get('role', '')
            
            total_months += work_month
            details.append({
                "company": company,
                "role": role,
                "months": work_month
            })
        
        # 경력 점수 계산 (2년 경력을 만점 기준)
        experience_score = min(total_months / 24.0, 1.0)
        
        return {
            "score": experience_score,
            "details": details
        }
    
    def generate_llm_evaluation(self, resume_data: Dict, component_scores: Dict) -> str:
        """LLM을 활용한 종합 평가 생성"""
        
        # 평가 컨텍스트 구성
        context = f"""
지원자 정보:
- 최종학력: {resume_data.get('final_edu', '')}
- 지원직종: {resume_data.get('desired_job', '')}

세부 평가 점수:
- 학력 점수: {component_scores['education']['score']:.2f}/1.0
- 자격증 점수: {component_scores['certificates']['score']:.2f}/1.0
- 활동 점수: {component_scores['activities']['score']:.2f}/1.0
- 경력 점수: {component_scores['experience']['score']:.2f}/1.0
- 종합 점수: {component_scores['total_score']:.2f}/100

위 정보를 바탕으로 {resume_data.get('desired_job', '')} 직종 지원자로서의 적합성을 100점만점으로 평가해주세요.
점수만 출력하세요. 예시: 72.84

평가: """
        
        try:
            # LLM 입력 토큰화
            inputs = self.tokenizer(
                context, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            ).to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # 결과 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            evaluation_text = response[len(context):].strip()
            
            if not evaluation_text:
                evaluation_text = f"{component_scores['total_score']:.2f}"
            
            return evaluation_text
            
        except Exception as e:
            print(f"LLM 평가 생성 오류: {e}")
            return f"{component_scores['total_score']:.2f}"
    
    def evaluate_resume(self, resume_data: Dict) -> Dict:
        """이력서 종합 평가"""
        
        job_field = resume_data.get('desired_job', '인터넷·IT')
        
        # 각 영역별 평가
        education_eval = self.evaluate_education(resume_data.get('universities', []), job_field)
        certificates_eval = self.evaluate_certificates(resume_data.get('certificates', []), job_field)
        activities_eval = self.evaluate_activities(resume_data.get('activities', []), job_field)
        experience_eval = self.evaluate_experience(resume_data.get('careers', []), job_field)
        
        # 가중치 적용하여 종합 점수 계산
        weights = {
            'education': 0.3,    # 학력 30%
            'certificates': 0.25, # 자격증 25%
            'experience': 0.25,   # 경력 25%
            'activities': 0.2,    # 활동 20%
        }
        
        total_score = (
            education_eval['score'] * weights['education'] +
            certificates_eval['score'] * weights['certificates'] +
            experience_eval['score'] * weights['experience'] +
            activities_eval['score'] * weights['activities']
        ) * 100  # 100점 만점으로 변환
        
        component_scores = {
            'education': education_eval,
            'certificates': certificates_eval,
            'experience': experience_eval,
            'activities': activities_eval,
            'total_score': total_score
        }
        
        # LLM 평가 생성
        llm_evaluation = self.generate_llm_evaluation(resume_data, component_scores)
        
        return {
            'total_score': total_score,
            'component_scores': component_scores,
            'llm_evaluation': llm_evaluation
        }