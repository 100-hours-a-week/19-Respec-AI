class PromptGenerator:
    """프롬프트 생성을 담당하는 클래스"""
    
    def create_job_specific_prompt(self, job_field, weights, few_shot_examples, criteria):
        """직무별 특화 프롬프트 생성"""
        system_prompt = f"""{job_field} 지원분야를 고려하여 100점 만점으로 소수점 2자리까지의 점수만 출력하세요.

{job_field} 분야는 다음 요소를 중요시합니다:
{criteria}

평가 가중치:
- 학력: {weights[0]}%
- 자격증: {weights[1]}%
- 경력: {weights[2]}%
- 어학: {weights[3]}%
- 활동: {weights[4]}%

어떠한 설명도 덧붙이지 말고 숫자만 답변하세요. 예시: 89.75"""

        # Few-shot 예제 추가
        if few_shot_examples:
            examples = ""
            for i, (example, score) in enumerate(few_shot_examples):
                examples += f"\n\n예시 이력서 {i+1}:\n{example}\n점수: {score}"
            system_prompt += examples

        return system_prompt
    
    def create_chat_format(self, system_prompt, user_resume):
        """채팅 형식 구성"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_resume}
        ]