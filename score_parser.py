import re
import json
from typing import Dict, Optional, Tuple, List, Union

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
        # 영어 시험
        "TOEIC": (0, 990),
        "TOEFL": (0, 120),
        "TEPS": (0, 990),
        "G_TELP": (0, 100),
        "TOEIC_SPEAKING": (0, 200),
        "TEPS_SPEAKING": (0, 8),
        "G_TELP_SPEAKING": (0, 100),
        "IELTS": (0, 9.0),
        "FLEX": (0, 800),
        "OPIC": ["NL", "NM", "NH", "IL", "IM", "IH", "AL", "AM", "AH"],
        
        # 독일어 시험
        "SNULT": (0, 200),  # 독일어, 프랑스어, 러시아어, 중국어, 일본어, 스페인어 공통
        
        # 중국어 시험
        "NEW_HSK": ["1급", "2급", "3급", "4급", "5급", "6급", "1", "2", "3", "4", "5", "6"],
        "HSK": ["1급", "2급", "3급", "4급", "5급", "6급", "1", "2", "3", "4", "5", "6"],
        
        # 일본어 시험
        "JPT": (0, 990),
        "JLPT": ["N5", "N4", "N3", "N2", "N1"],
    }
    
    @classmethod
    def extract_test_type(cls, full_test_type: str) -> str:
        """
        전체 시험 유형에서 기본 시험명만 추출
        예: TOEIC_ENGLISH -> TOEIC, NEW_HSK_CHINESE -> NEW_HSK
        """
        # 언어 접미사 제거
        language_suffixes = ['_ENGLISH', '_GERMAN', '_FRENCH', '_RUSSIAN', 
                           '_CHINESE', '_JAPANESE', '_SPANISH', '_VIETNAMESE']
        
        test_type = full_test_type
        for suffix in language_suffixes:
            if test_type.endswith(suffix):
                test_type = test_type[:-len(suffix)]
                break
        
        return test_type
    
    @classmethod
    def validate_score(cls, test_type: str, score: str) -> Tuple[bool, float]:
        """
        어학점수 검증 및 정규화된 점수 반환
        Args:
            test_type: 시험 유형 (예: TOEIC_ENGLISH, NEW_HSK_CHINESE 등)
            score: 입력된 점수
        Returns:
            Tuple[bool, float]: (유효성 여부, 정규화된 점수)
        """
        # 기본 시험 유형 추출
        base_test_type = cls.extract_test_type(test_type)
        
        if base_test_type not in cls.VALID_RANGES:
            return False, 0.0
            
        try:
            range_info = cls.VALID_RANGES[base_test_type]
            
            if isinstance(range_info, tuple):
                # 숫자 점수 검증
                min_score, max_score = range_info
                numeric_score = float(score)
                if min_score <= numeric_score <= max_score:
                    # 100점 만점으로 정규화
                    normalized_score = (numeric_score - min_score) / (max_score - min_score) * 100
                    return True, normalized_score
                return False, 0.0
            else:
                # 등급 검증 (리스트 타입)
                valid_grades = range_info
                if score.upper() in [str(grade).upper() for grade in valid_grades]:
                    # 등급을 점수로 변환 (높은 등급일수록 높은 점수)
                    grade_index = len(valid_grades) - [str(grade).upper() for grade in valid_grades].index(score.upper())
                    normalized_score = (grade_index / len(valid_grades)) * 100
                    return True, normalized_score
                return False, 0.0
                
        except (ValueError, AttributeError):
            return False, 0.0
    
    @classmethod
    def get_valid_range_info(cls, test_type: str) -> Union[Tuple[int, int], List[str], None]:
        """
        특정 시험의 유효 범위 정보 반환
        """
        base_test_type = cls.extract_test_type(test_type)
        return cls.VALID_RANGES.get(base_test_type)
    
    @classmethod
    def is_supported_test(cls, test_type: str) -> bool:
        """
        지원하는 시험인지 확인
        """
        base_test_type = cls.extract_test_type(test_type)
        return base_test_type in cls.VALID_RANGES