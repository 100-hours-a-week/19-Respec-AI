from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum


class ScoreCategory(Enum):
    """점수 카테고리 정의"""
    BASIC = "기본 점수"
    MAJOR = "전공"
    UNIVERSITY = "학교"
    CERTIFICATION = "자격증"
    EXPERIENCE = "경력"
    LANGUAGE = "어학"
    ACTIVITY = "활동"


@dataclass
class WeightConfig:
    """가중치 설정을 위한 데이터 클래스"""
    education_max: float  # 학력 전체 만점
    major_ratio: float    # 전공 비율
    university_ratio: float  # 학교 비율
    cert_max: float       # 자격증 전체 만점
    cert_relevance_ratio: float  # 자격증 관련성 비율
    cert_count_ratio: float      # 자격증 개수 비율
    experience_max: float        # 경력 전체 만점
    exp_relevance_ratio: float   # 경력 관련성 비율
    exp_duration_ratio: float    # 경력 기간 비율
    language_max: float          # 어학 전체 만점
    activity_max: float          # 활동 전체 만점
    activity_relevance_ratio: float  # 활동 관련성 비율
    activity_role_ratio: float       # 활동 역할 비율

    @classmethod
    def from_tuple(cls, weights: Tuple) -> 'WeightConfig':
        """기존 튜플 형태의 가중치를 WeightConfig로 변환"""
        return cls(
            education_max=float(weights[0]),
            major_ratio=float(weights[1]),
            university_ratio=float(weights[2]),
            cert_max=float(weights[3]),
            cert_relevance_ratio=float(weights[4]),
            cert_count_ratio=float(weights[5]),
            experience_max=float(weights[6]),
            exp_relevance_ratio=float(weights[7]),
            exp_duration_ratio=float(weights[8]),
            language_max=float(weights[9]),
            activity_max=float(weights[10]),
            activity_relevance_ratio=float(weights[11]),
            activity_role_ratio=float(weights[12])
        )


class ScoreCalculator:
    """점수 계산 전담 클래스"""
    
    BASE_SCORE = 40.0
    
    def __init__(self, weight_config: WeightConfig):
        self.weights = weight_config
        self.scores = {category.value: 0.0 for category in ScoreCategory}
        self.scores[ScoreCategory.BASIC.value] = self.BASE_SCORE
    
    def calculate_major_score(self, similarity: float) -> float:
        """전공 점수 계산"""
        max_score = self.weights.education_max * self.weights.major_ratio
        
        if similarity >= 0.9:
            return max_score
        elif similarity >= 0.7:
            return max_score * 0.8
        elif similarity >= 0.5:
            return max_score * 0.5
        else:
            return max_score * 0.2
    
    def calculate_university_score(self, rank: int) -> float:
        """대학교 점수 계산"""
        max_score = self.weights.education_max * self.weights.university_ratio
        
        if rank <= 3:
            return max_score
        elif rank <= 10:
            return max_score * 0.9
        elif rank <= 20:
            return max_score * 0.8
        elif rank <= 30:
            return max_score * 0.75
        elif rank <= 50:
            return max_score * 0.7
        elif rank <= 100:
            return max_score * 0.6
        elif rank <= 200:
            return max_score * 0.65
        elif rank <= 300:
            return max_score * 0.5
        else:
            return max_score * 0.4
    
    def calculate_certification_score(self, matches: List[Dict]) -> float:
        """자격증 점수 계산"""
        total_score = 0.0
        relevance_score = 0.0
        
        # 관련성 점수 계산
        for match in matches[:3]:
            similarity = match.get('similarity', 0)
            if similarity >= 0.8:
                relevance_score += self.weights.cert_max * self.weights.cert_relevance_ratio * 0.8
            elif similarity >= 0.6:
                relevance_score += self.weights.cert_max * self.weights.cert_relevance_ratio * 0.5
            elif similarity >= 0.4:
                relevance_score += self.weights.cert_max * self.weights.cert_relevance_ratio * 0.3
        
        # 개수 보너스
        count_bonus = min(len(matches) * 1.0, 
                         self.weights.cert_max * self.weights.cert_count_ratio)
        
        total_score = relevance_score + count_bonus
        return min(total_score, self.weights.cert_max)
    
    def calculate_experience_score(self, matches: List[Dict]) -> float:
        """경력 점수 계산"""
        total_score = 0.0
        
        for match in matches[:3]:
            similarity = match.get('similarity', 0)
            duration_months = match.get('work_month', 0)
            
            # 직무 관련성 점수
            if similarity >= 0.8:
                relevance_score = self.weights.experience_max * self.weights.exp_relevance_ratio
            elif similarity >= 0.6:
                relevance_score = self.weights.experience_max * self.weights.exp_relevance_ratio * 0.7
            else:
                relevance_score = self.weights.experience_max * self.weights.exp_relevance_ratio * 0.3
            
            # 경력 기간 점수
            duration_score = min(duration_months * (self.weights.experience_max * self.weights.exp_duration_ratio),
                               self.weights.experience_max * self.weights.exp_duration_ratio)
            
            total_score += (relevance_score + duration_score)
        
        return min(total_score, self.weights.experience_max)
    
    def calculate_activity_score(self, matches: List[Dict]) -> float:
        """활동 점수 계산"""
        total_score = 0.0
        
        for match in matches[:3]:
            similarity = match.get('similarity', 0)
            
            if similarity >= 0.8:
                activity_score = self.weights.activity_max * self.weights.activity_relevance_ratio * similarity
            elif similarity >= 0.6:
                activity_score = self.weights.activity_max * self.weights.activity_relevance_ratio * similarity * 0.8
            else:
                continue
                
            total_score += activity_score
        
        return min(total_score, self.weights.activity_max * 0.7)
    
    def get_total_score(self) -> float:
        """총점 계산"""
        return sum(self.scores.values())
    
    def normalize_to_100(self) -> Dict[str, float]:
        """각 영역별 점수를 100점 만점으로 정규화"""
        max_scores = {
            "academic": self.weights.education_max,
            "certification": self.weights.cert_max,
            "workExperience": self.weights.experience_max,
            "languageProficiency": self.weights.language_max,
            "extracurricular": self.weights.activity_max
        }
        
        # 학력은 전공+학교 점수 합계
        education_score = self.scores[ScoreCategory.MAJOR.value] + self.scores[ScoreCategory.UNIVERSITY.value]
        
        categories = {
            "academic": education_score,
            "certification": self.scores[ScoreCategory.CERTIFICATION.value],
            "workExperience": self.scores[ScoreCategory.EXPERIENCE.value],
            "languageProficiency": self.scores[ScoreCategory.LANGUAGE.value],
            "extracurricular": self.scores[ScoreCategory.ACTIVITY.value]
        }
        
        normalized = {}
        for category, current_score in categories.items():
            if max_scores[category] > 0:
                normalized[category] = (current_score / max_scores[category]) * 100
            else:
                normalized[category] = 0.0
                
        return normalized


class ScoreReporter:
    """점수 출력 전담 클래스"""
    
    @staticmethod
    def print_score_breakdown(scores: Dict[str, float]) -> str:
        """점수 분석 결과 출력 및 프롬프트용 텍스트 반환"""
        # 콘솔 출력
        print("\n=== 🎯 이력서 점수 분석 결과 ===")
        total = 0.0
        prompt_lines = []
        
        for category, score in scores.items():
            line = f"📌 {category}: {score:.2f}점"
            print(line)
            prompt_lines.append(line)
            total += score
        
        total_line = f"📊 총점: {total:.2f}점"
        print(total_line)
        prompt_lines.append(total_line)
        print("=" * 30)
        
        # 프롬프트용 텍스트 생성
        prompt_text = "\n=== 📊 현재 계산된 점수 ===\n"
        prompt_text += "\n".join(prompt_lines)
        prompt_text += "\n" + "=" * 30
        
        return prompt_text
    
    @staticmethod
    def print_normalized_scores(normalized_scores: Dict[str, float]) -> str:
        """100점 만점 기준 점수 출력 및 프롬프트용 텍스트 반환"""
        # 콘솔 출력
        print("\n=== 🎯 100점 만점 기준 점수 ===")
        category_names = {
            "academic": "학력",
            "certification": "자격증", 
            "workExperience": "경력",
            "languageProficiency": "어학",
            "extracurricular": "활동"
        }
        
        prompt_lines = []
        for category, score in normalized_scores.items():
            line = f"📌 {category_names[category]}: {score:.2f}/100점"
            print(line)
            prompt_lines.append(line)
        print("=" * 30)
        
        # 프롬프트용 텍스트 생성
        prompt_text = "\n=== 🎯 100점 만점 기준 점수 ===\n"
        prompt_text += "\n".join(prompt_lines)
        prompt_text += "\n" + "=" * 30
        
        return prompt_text
    
    @staticmethod
    def create_score_summary_for_prompt(scores: Dict[str, float], 
                                      normalized_scores: Dict[str, float]) -> str:
        """프롬프트에 포함할 점수 요약 생성"""
        total = sum(scores.values())
        
        summary = f"""
=== 📊 점수 계산 결과 요약 ===
• 현재 총점: {total:.2f}점
• 기본 점수: {scores.get('기본 점수', 40.0):.2f}점
• 전공 점수: {scores.get('전공', 0.0):.2f}점  
• 학교 점수: {scores.get('학교', 0.0):.2f}점
• 자격증 점수: {scores.get('자격증', 0.0):.2f}점
• 경력 점수: {scores.get('경력', 0.0):.2f}점
• 어학 점수: {scores.get('어학', 0.0):.2f}점
• 활동 점수: {scores.get('활동', 0.0):.2f}점

=== 📈 100점 기준 환산 ==="""
        
        category_names = {
            "academic": "학력",
            "certification": "자격증",
            "workExperience": "경력", 
            "languageProficiency": "어학",
            "extracurricular": "활동"
        }
        
        for category, score in normalized_scores.items():
            summary += f"\n• {category_names[category]}: {score:.2f}/100점"
        
        summary += "\n" + "=" * 40
        
        return summary


class PromptBuilder:
    """프롬프트 생성 전담 클래스"""
    
    def __init__(self, weight_config: WeightConfig):
        self.weights = weight_config
    
    def build_basic_prompt(self, job_field: str, criteria: str) -> str:
        """기본 프롬프트 생성"""
        return f"""
당신은 HR 전문가로서 이력서를 간결하게 평가하는 AI입니다.

다음 이력서를 {job_field} 분야 기준으로 평가하세요.

평가기준: {criteria}

아래 JSON 형식으로만 응답하세요:
{{"totalscore": 숫자, "assessment": 핵심평가내용}}

평가 시 고려사항:
- 전공 일치도, 성적, 관련 경험 중심 평가
- 신입 기준으로 평가 (경력 부족은 감점 안함)
- assessment는 간결하게 작성

**평가 공식:**
[주요강점] + [개선필요영역] + [결론]

**강점 키워드:** 전공일치, 우수성적, 관련경력, 적합자격증, 어학우수, 리더십경험
**개선 키워드:** 어학부족, 전공미일치, 경험부족, 자격증부족, 활동경험 없음

**응답 형식:**
{{"totalscore": 점수, "assessment": 50자 이내 핵심평가}}

**절대 준수사항:**
- assessment는 반드시 50자(공백포함) 이내
- "이력서는", "지원자는" 같은 주어 생략
- 구체적 키워드 위주 작성
- 이력서 내용 요약/반복 절대 금지
- "~을 가지고 있습니다", "~를 전공했습니다" 같은 서술 금지"""
    
    def add_rag_context(self, base_prompt: str, rag_context: Dict, 
                       score_calculator: ScoreCalculator) -> str:
        """RAG 컨텍스트를 프롬프트에 추가"""
        if not rag_context:
            return base_prompt
        
        rag_section = "\n\n=== 벡터 검색 기반 점수 가이드 ==="
        
        # 전공 분석
        if rag_context.get('education_matches'):
            match = rag_context['education_matches'][0]
            similarity = match.get('similarity', 0)
            major_score = score_calculator.calculate_major_score(similarity)
            score_calculator.scores[ScoreCategory.MAJOR.value] = major_score
            rag_section += f"\n전공 점수: {major_score:.2f}점 (유사도: {similarity:.2f})"
        
        # 대학교 분석
        if rag_context.get('university_matches'):
            match = rag_context['university_matches'][0]
            rank = match.get('rank_position', 999)
            university_name = match.get('university_name', '')
            uni_score = score_calculator.calculate_university_score(rank)
            score_calculator.scores[ScoreCategory.UNIVERSITY.value] = uni_score
            rag_section += f"\n🎓 {university_name} (순위: {rank}위): {uni_score:.2f}점"
        
        # 자격증 분석
        if rag_context.get('certificate_matches'):
            cert_score = score_calculator.calculate_certification_score(
                rag_context['certificate_matches']
            )
            score_calculator.scores[ScoreCategory.CERTIFICATION.value] = cert_score
            rag_section += f"\n📜 자격증 총점: {cert_score:.2f}점"
        
        # 경력 분석
        if rag_context.get('company_matches'):
            exp_score = score_calculator.calculate_experience_score(
                rag_context['company_matches']
            )
            score_calculator.scores[ScoreCategory.EXPERIENCE.value] = exp_score
            rag_section += f"\n💼 경력 총점: {exp_score:.2f}점"
            
        # 어학 점수 분석
        if rag_context.get('language_scores'):
            language_score = rag_context.get('average_language_score', 0.0)
            # 어학 점수를 weights.language_max에 맞게 조정
            adjusted_language_score = (language_score / 100.0) * score_calculator.weights.language_max
            score_calculator.scores[ScoreCategory.LANGUAGE.value] = adjusted_language_score
            
            # 유효한 점수와 무효한 점수 정보 추가
            valid_scores = [f"{score['test']} {score['score']}" 
                          for score in rag_context['language_scores'] 
                          if score['is_valid']]
            invalid_scores = [f"{score['test']} {score['score']}" 
                            for score in rag_context['language_scores'] 
                            if not score['is_valid']]
            
            rag_section += f"\n🌐 어학 총점: {adjusted_language_score:.2f}점"
            if valid_scores:
                rag_section += f"\n   ✅ 유효한 점수: {', '.join(valid_scores)}"
            if invalid_scores:
                rag_section += f"\n   ❌ 무효 처리된 점수: {', '.join(invalid_scores)}"
        
        # 활동 분석
        if rag_context.get('activity_matches'):
            activity_score = score_calculator.calculate_activity_score(
                rag_context['activity_matches']
            )
            score_calculator.scores[ScoreCategory.ACTIVITY.value] = activity_score
            rag_section += f"\n🏃 활동 총점: {activity_score:.2f}점"
        
        return base_prompt + rag_section


class PromptGenerator:
    """메인 프롬프트 생성기 클래스 - 기존 인터페이스 유지"""
    
    def __init__(self):
        """초기화"""
        self.similarity_threshold = 0.7
        self.max_rag_examples = 3
        self.score_calculator = None
        self.reporter = ScoreReporter()
        self.score_breakdown = {
            "normalized_scores": {
                "academic": 0.0,
                "workExperience": 0.0,
                "certification": 0.0,
                "languageProficiency": 0.0,
                "extracurricular": 0.0
            }
        }
    
    def create_rag_enhanced_prompt(self, job_field: str, weights: Tuple, 
                                 criteria: str, rag_context: Dict) -> str:
        """RAG 향상 프롬프트 생성 - 기존 인터페이스 유지"""
        weight_config = WeightConfig.from_tuple(weights)
        self.score_calculator = ScoreCalculator(weight_config)
        prompt_builder = PromptBuilder(weight_config)
        
        # 기본 프롬프트 생성
        base_prompt = prompt_builder.build_basic_prompt(job_field, criteria)
        
        # RAG 컨텍스트 추가
        enhanced_prompt = prompt_builder.add_rag_context(
            base_prompt, rag_context, self.score_calculator
        )
        
        # 점수 출력 및 프롬프트용 텍스트 생성
        score_text = self.reporter.print_score_breakdown(self.score_calculator.scores)
        normalized_scores = self.score_calculator.normalize_to_100()
        normalized_text = self.reporter.print_normalized_scores(normalized_scores)
        
        # 점수 요약을 프롬프트에 추가
        score_summary = self.reporter.create_score_summary_for_prompt(
            self.score_calculator.scores, normalized_scores
        )
        
        # 최종 프롬프트에 점수 정보 포함
        final_prompt = enhanced_prompt + score_summary
        
        return final_prompt
    
    def create_job_specific_prompt(self, job_field: str, weights: Tuple, criteria: str) -> str:
        """직무별 특화 프롬프트 생성 - 기존 인터페이스 유지"""
        weight_config = WeightConfig.from_tuple(weights)
        prompt_builder = PromptBuilder(weight_config)
        return prompt_builder.build_basic_prompt(job_field, criteria)
    
    def create_chat_format(self, system_prompt: str, user_resume: str) -> List[Dict]:
        """채팅 형식 구성 - 기존 인터페이스 유지"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_resume}
        ]