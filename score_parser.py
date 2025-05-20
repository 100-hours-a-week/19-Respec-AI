import re

class ScoreParser:
    """모델 출력에서 점수를 추출하는 클래스"""
    
    def extract_score(self, full_output):
        """출력 텍스트에서 점수 추출"""
        print(f"모델 출력 전체: {full_output}")
        
        # 먼저 assistant 응답 부분 추출 후 숫자 찾기
        assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|<\|endofturn\|>)'
        assistant_match = re.search(assistant_pattern, full_output, re.DOTALL)

        if assistant_match:
            assistant_text = assistant_match.group(1).strip()
            score_match = re.search(r'\b\d{1,3}(?:\.\d{1,2})?\b', assistant_text)

            if score_match:
                return score_match.group(0)
            else:
                # 마지막 시도: 응답 텍스트에서 숫자만 추출
                return re.sub(r'[^\d.]', '', assistant_text)
        else:
            # 백업: 전체 출력에서 숫자 찾기
            score_pattern = r'\b\d{1,3}(?:\.\d{1,2})?\b'
            all_numbers = re.findall(score_pattern, full_output)

            # 시스템 프롬프트의 가중치 숫자들을 제외하기 위해 후반부의 숫자를 선택
            if len(all_numbers) > 10:  # 시스템 프롬프트의 숫자들이 많이 있을 것
                return all_numbers[-1]  # 마지막 숫자 선택
            
        return "점수 추출 실패"