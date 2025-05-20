from database_connector import DatabaseConnector
from model_manager import ModelManager
from prompt_generator import PromptGenerator
from score_parser import ScoreParser
from resume_evaluator import ResumeEvaluator

class ResumeEvaluationSystem:
    """전체 이력서 평가 시스템을 관리하는 메인 클래스"""
    
    def __init__(self):
        # 각 컴포넌트 초기화
        self.db_connector = DatabaseConnector()
        self.model_manager = ModelManager()
        self.prompt_generator = PromptGenerator()
        self.score_parser = ScoreParser()
        
        # 평가기 초기화
        self.evaluator = ResumeEvaluator(
            self.db_connector, 
            self.model_manager, 
            self.prompt_generator,
            self.score_parser
        )
    
    def evaluate_resume(self, resume_text, job_field):
        """이력서 평가 수행"""
        try:
            score = self.evaluator.evaluate(resume_text, job_field)
            return score
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return "평가 실패"