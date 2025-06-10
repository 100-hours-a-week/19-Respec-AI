import re
import json
from typing import Dict, Optional, Tuple

class ScoreParser:
    """모델 출력에서 점수를 추출하는 클래스"""
    
    def __init__(self):
        self.required_fields = ["totalscore", "assessment"]

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

    def parse_response(self, response: str) -> Tuple[int, str]:
        """
        모델의 JSON 응답을 파싱하여 점수와 평가를 반환합니다.
        
        Args:
            response (str): 모델의 JSON 형식 응답 문자열
            
        Returns:
            Tuple[int, str]: (총점, 평가) 튜플
            
        Raises:
            ValueError: JSON 파싱 실패 또는 필수 필드 누락 시
        """
        try:
            # JSON 부분만 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
            
            json_str = response[json_start:json_end]
            
            # JSON 문자열을 파싱
            data = json.loads(json_str)
            
            # 필수 필드 확인
            for field in self.required_fields:
                if field not in data:
                    raise ValueError(f"필수 필드 '{field}'가 응답에 없습니다.")
            
            # 점수가 정수인지 확인
            if not isinstance(data["totalscore"], int):
                raise ValueError("totalscore는 정수여야 합니다.")
            
            # 점수가 0-100 범위인지 확인
            if not 0 <= data["totalscore"] <= 100:
                raise ValueError("totalscore는 0에서 100 사이여야 합니다.")
            
            return data["totalscore"], data["assessment"]
            
        except json.JSONDecodeError:
            raise ValueError("유효하지 않은 JSON 형식입니다.")

class LanguageScoreValidator:
    """어학점수 검증 클래스"""
    
    # 각 시험별 유효 점수 범위 정의
    VALID_RANGES = {
        "TOEIC": (0, 990),
        "TOEFL": (0, 120),
        "TEPS": (0, 990),
        "IELTS": (0, 9.0),
        "OPIC": ["NL", "NM", "NH", "IL", "IM", "IH", "AL", "AM", "AH"],
        "JLPT": ["N5", "N4", "N3", "N2", "N1"],
        "HSK": ["1급", "2급", "3급", "4급", "5급", "6급", "1", "2", "3", "4", "5", "6"]
    }
    
    @classmethod
    def validate_score(cls, test_type: str, score: str) -> Tuple[bool, float]:
        """
        어학점수 검증 및 정규화된 점수 반환
        Returns:
            Tuple[bool, float]: (유효성 여부, 정규화된 점수)
        """
        if test_type not in cls.VALID_RANGES:
            return False, 0.0
            
        try:
            if isinstance(cls.VALID_RANGES[test_type], tuple):
                # 숫자 점수 검증
                min_score, max_score = cls.VALID_RANGES[test_type]
                numeric_score = float(score)
                if min_score <= numeric_score <= max_score:
                    # 100점 만점으로 정규화
                    normalized_score = (numeric_score - min_score) / (max_score - min_score) * 100
                    return True, normalized_score
                return False, 0.0
            else:
                # 등급 검증
                valid_grades = cls.VALID_RANGES[test_type]
                if score.upper() in [str(grade).upper() for grade in valid_grades]:
                    # 등급을 점수로 변환
                    grade_index = len(valid_grades) - [str(grade).upper() for grade in valid_grades].index(score.upper())
                    normalized_score = (grade_index / len(valid_grades)) * 100
                    return True, normalized_score
                return False, 0.0
        except (ValueError, AttributeError):
            return False, 0.0