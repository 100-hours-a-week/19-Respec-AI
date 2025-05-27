from typing import Dict, List, Tuple, Optional
import json

class PromptGenerator:
    """프롬프트 생성을 담당하는 클래스 - 기존 메서드 수정 최소화"""
    
    def __init__(self):
        # RAG 검색 결과 통합을 위한 설정
        self.similarity_threshold = 0.7
        self.max_rag_examples = 3
        self.score_breakdown = {
            "기본 점수": 40.0,
            "전공": 0.0,
            "학교": 0.0,
            "자격증": 0.0,
            "경력": 0.0,
            "어학": 0.0,
            "활동": 0.0
        }

    def print_score_breakdown(self):
        """점수 분석 결과를 콘솔에 출력"""
        print("\n=== 🎯 이력서 점수 분석 결과 ===")
        total = 0.0
        for category, score in self.score_breakdown.items():
            print(f"📌 {category}: {score:.2f}점")
            total += score
        print("=" * 30)
        print(f"📊 총점: {total:.2f}점")
        print("=" * 30)

    def create_job_specific_prompt(self, job_field, weights, criteria):
        """기존 직무별 특화 프롬프트 생성 (변경 없음)"""
        system_prompt = f"""{job_field} 지원분야 이력서를 벡터 검색 결과를 활용하여 100점 만점으로 평가하세요.
=== 평가 기준 ===
{job_field} 분야 요구사항: {criteria}
총점 = 40 + 학력점수

=== 출력 규칙 ===
1. 반드시 JSON 형식으로만 출력: {{"totalScore": XX.XX}}
2. 소수점 둘째 자리까지 정확히 계산
3. 어떤 설명이나 추가 텍스트도 금지
"""

        return system_prompt
    
    def create_rag_enhanced_prompt(self, job_field: str, weights: Tuple, criteria: str, 
                                 rag_context: Dict) -> str:
        """RAG 검색 결과를 반영한 향상된 프롬프트 - 핵심 수정"""
        
        # 기본 프롬프트 시작
        system_prompt = f"""{job_field} 지원분야 이력서를 벡터 검색 결과를 활용하여 100점 만점으로 평가하세요.

=== 평가 기준 ===
{job_field} 분야 요구사항: {criteria}

=== 정량적 점수 계산 방식 ===
각 영역별 점수 = (영역별 세부점수 X 가중치)의 합계

• 학력 영역 ({weights[0]}점 만점)
  - 전공 적합성: 0~{float(weights[0])*float(weights[1]):.2f}점
  - 학점/학교: 0~{float(weights[0])*float(weights[2]):.2f}점
  * 학점 점수 = 학점/학교 만점 × (실제 학점/만점 학점)
  * 학교 점수 계산 기준:
    - 상위 1-50위: 만점의 100%
    - 상위 51-100위: 만점의 90%
    - 상위 101-200위: 만점의 80%
    - 그 외: 만점의 70%

• 자격증 영역 ({weights[3]}점 만점)
  - 직무 관련성: 0~{float(weights[3])*float(weights[4]):.2f}점
  - 개수/난이도: 0~{float(weights[3])*float(weights[5]):.2f}점

• 경력 영역 ({weights[6]}점 만점)
  - 직무 관련성: 0~{float(weights[6])*float(weights[7]):.2f}점
  - 경력 기간: 0~{float(weights[6])*float(weights[8]):.2f}점

• 어학 영역 ({weights[9]}점 만점)
  - 점수/등급: 0~{float(weights[9])}점

• 활동 영역 ({weights[10]}점 만점)
  - 직무 관련성: 0~{float(weights[10])*float(weights[11]):.2f}점
  - 역할/성과: 0~{float(weights[10])*float(weights[12]):.2f}점

=== 점수 계산 주의사항 ===
1. 기본 점수는 정확히 40점입니다.
2. 각 영역의 점수는 해당 영역의 만점을 초과할 수 없습니다.
3. 최종 점수는 반드시 아래 공식으로 계산하세요:
   최종 점수 = 40 + min(학력점수, {weights[0]}) + min(자격증점수, {weights[3]}) + min(경력점수, {weights[6]}) + min(어학점수, {weights[9]}) + min(활동점수, {weights[10]})
4. 계산기를 사용하여 정확한 덧셈을 수행하세요.
5. 계산 과정을 단계별로 나누어 진행하세요.

반드시 아래 JSON 형식으로만 답변하세요:
{{"totalScore": 85.75}}

설명이나 다른 텍스트는 절대 포함하지 마세요. 
총점 = 40 + 학력점수 + 자격증점수 + 경력점수 + 어학점수 + 활동점수 """

        if rag_context:
            system_prompt += f"\n\n=== 벡터 검색 기반 점수 가이드 ==="
        
        # 전공 유사도 분석
        if rag_context.get('education_matches'):
            for match in rag_context['education_matches'][:2]:
                similarity = match.get('similarity', 0)
                
                if similarity >= 0.9:
                    major_score = float(weights[0]) * float(weights[1])  # 만점
                    system_prompt += f"\n전공 점수: {major_score:.2f}점 (완벽 매칭)"
                    self.score_breakdown["전공"] = major_score
                elif similarity >= 0.7:
                    major_score = float(weights[0]) * float(weights[1]) * 0.8  # 80%
                    system_prompt += f"\n전공 점수: {major_score:.2f}점 (높은 적합성)"
                    self.score_breakdown["전공"] = major_score
                elif similarity >= 0.5:
                    major_score = float(weights[0]) * float(weights[1]) * 0.5  # 50%
                    system_prompt += f"\n전공 점수: {major_score:.2f}점 (보통 적합성)"
                    self.score_breakdown["전공"] = major_score
                else:
                    major_score = float(weights[0]) * float(weights[1]) * 0.2  # 20%
                    system_prompt += f"\n전공 점수: {major_score:.2f}점 (낮은 적합성)"
                    self.score_breakdown["전공"] = major_score
                break
        else:
            system_prompt += f"\n📚 전공 점수: 0점 (전공 정보 없음)"
            self.score_breakdown["전공"] = 0.0
        
        # 대학교 랭킹 기반 점수 분석
        if rag_context.get('university_matches'):
            system_prompt += f"\n🎓 대학교 분석:"
            for match in rag_context['university_matches'][:1]:  # top_k=1로 설정했으므로
                rank = match.get('rank_position', 0)
                university_name = match.get('university_name', '')
                
                if rank <= 50:
                    uni_score = float(weights[0]) * float(weights[2])  # 만점
                    system_prompt += f"\n {university_name} (상위 50위권): {uni_score:.2f}점"
                    self.score_breakdown["학교"] = uni_score
                elif rank <= 100:
                    uni_score = float(weights[0]) * float(weights[2]) * 0.9
                    system_prompt += f"\n {university_name} (상위 51-100위권): {uni_score:.2f}점"
                    self.score_breakdown["학교"] = uni_score
                elif rank <= 200:
                    uni_score = float(weights[0]) * float(weights[2]) * 0.8
                    system_prompt += f"\n {university_name} (상위 101-200위권): {uni_score:.2f}점"
                    self.score_breakdown["학교"] = uni_score
                else:
                    uni_score = float(weights[0]) * float(weights[2]) * 0.7
                    system_prompt += f"\n {university_name}: {uni_score:.2f}점"
                    self.score_breakdown["학교"] = uni_score
        else:
            system_prompt += f"\n🎓 대학교 점수: 0점 (대학교 정보 없음)"
            self.score_breakdown["학교"] = 0.0
        
        # 자격증 유사도 분석  
        if rag_context.get('certificate_matches'):
            system_prompt += f"\n자격증 분석:"
            total_cert_score = 0
            for match in rag_context['certificate_matches'][:3]:
                similarity = match.get('similarity', 0)
                weight_score = match.get('weight_score', 0)
                
                if similarity >= 0.8:
                    cert_score = float(weight_score) * float(similarity)
                    total_cert_score += cert_score
                    system_prompt += f"\n  ✅ 관련 자격증: +{cert_score:.2f}점"
                elif similarity >= 0.6:
                    cert_score = float(weight_score) * float(similarity) * 0.7
                    total_cert_score += cert_score
                    system_prompt += f"\n  ⚠️ 부분 관련: +{cert_score:.2f}점"
            
            max_cert_score = min(total_cert_score, float(weights[3]))
            system_prompt += f"\n  📊 자격증 총점: {max_cert_score:.2f}점 (상한: {weights[3]}점)"
            self.score_breakdown["자격증"] = max_cert_score
        else:
            system_prompt += f"\n🏆 자격증 점수: 0점 (자격증 없음)"
            self.score_breakdown["자격증"] = 0.0
        
        # 활동 유사도 분석
        if rag_context.get('activity_matches'):
            system_prompt += f"\n🎯 활동 분석:"
            total_activity_score = 0
            for match in rag_context['activity_matches'][:3]:
                similarity = match.get('similarity', 0)
                relevance = match.get('relevance_score', 0.5)
                
                if similarity >= 0.8:
                    activity_score = float(weights[10]) * float(weights[11])* float(similarity)
                    total_activity_score += activity_score
                    system_prompt += f"\n  ✅ 관련 활동: +{activity_score:.2f}점"
                elif similarity >= 0.6:
                    activity_score = float(weights[10]) * float(weights[11]) * float(similarity) * 0.8
                    total_activity_score += activity_score
                    system_prompt += f"\n  ⚠️ 부분 관련: +{activity_score:.2f}점"
            
            max_activity_score = min(total_activity_score, float(weights[10]) * 0.7)
            system_prompt += f"\n  📊 활동 관련성: {max_activity_score:.2f}점"
            self.score_breakdown["활동"] = max_activity_score
        else:
            system_prompt += f"\n🎯 활동 점수: 0점 (활동 내역 없음)"
            self.score_breakdown["활동"] = 0.0

        # 점수 분석 결과 출력
        self.print_score_breakdown()

        # 평가 지시사항
        system_prompt += f"""
=== 점수 계산 예시 ===
만약 지원자가:
- 전공 적합성 높음: {float(weights[0])*float(weights[1]):.2f}점
- 학점 우수(3.2/4.5): {float(weights[0])*float(weights[2])*0.7:.2f}점  
- 자격증 없음: 0점
- 경력 없음: 0점
- 어학 없음: 0점
- 관련 활동 1개: {float(weights[10])*float(weights[11])*0.8:.2f}점

총점 = 40 + {float(weights[0])*float(weights[1]):.2f} + {float(weights[0])*float(weights[2])*0.7:.2f} + 0 + 0 + 0 + {float(weights[10])*float(weights[11])*0.8:.2f} = {40 + float(weights[0])*float(weights[1]) + float(weights[0])*float(weights[2])*0.7 + float(weights[10])*float(weights[11])*0.8}점

=== 출력 규칙 ===
1. 위 계산 방식에 따라 정확한 점수 산출
2. 반드시 JSON 형식으로만 출력: {{"totalScore": XX.XX}}
3. 소수점 둘째 자리까지 정확히 계산
4. 어떤 설명이나 추가 텍스트도 금지 """

        return system_prompt
    
    def create_chat_format(self, system_prompt, user_resume):
        """채팅 형식 구성 (변경 없음)"""
        user_resume += """
Calculate resume score and return ONLY this format:
{{"totalScore": XX.XX}}"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_resume}
        ]