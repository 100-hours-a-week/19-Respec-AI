from resume_evaluation_system import ResumeEvaluationSystem
from database_connector import DatabaseConnector
from model_manager import ModelManager
from prompt_generator import PromptGenerator
from score_parser import ScoreParser
from resume_evaluator import ResumeEvaluator

class SpecEvaluator:
    """
    스펙 평가를 담당하는 클래스
    app.py와 ResumeEvaluationSystem을 연결하는 역할
    """
    
    def __init__(self):
        """초기화 함수, Redis 캐싱 사용 여부를 설정할 수 있음"""
        
        # 평가 시스템 초기화
        self.db_connector = DatabaseConnector()
        self.model_manager = ModelManager()
        self.prompt_generator = PromptGenerator()
        self.score_parser = ScoreParser()
        
        self.evaluator = ResumeEvaluator(
            self.db_connector, 
            self.model_manager, 
            self.prompt_generator,
            self.score_parser
        )
    
    def _format_resume_text(self, spec_data):
        """SpecV1 API 데이터를 이력서 텍스트 형식으로 변환"""
        resume_text = "" 
        resume_text += f"최종학력: {spec_data['final_edu']} ({spec_data['final_status']}), "
        resume_text += f"지원직종: {spec_data['desired_job']}, "
        
        # 대학 정보
        if spec_data.get('universities'):
            univ_list = []
            for univ in spec_data['universities']:
                univ_text = f"{univ['school_name']}"
                if univ.get('major'):
                    univ_text += f" {univ.get('major')}"
                if univ.get('degree'):
                    univ_text += f" ({univ.get('degree')})"
                if univ.get('gpa') and univ.get('gpa_max'):
                    univ_text += f" 학점:{univ['gpa']}/{univ['gpa_max']}"
                univ_list.append(univ_text)
            resume_text += f"학력: {', '.join(univ_list)}, "
        else:
            resume_text += "학력: 대학 정보 없음, "
        
        # 경력 정보
        if spec_data.get('careers'):
            career_list = []
            for career in spec_data['careers']:
                career_text = f"{career['company']}"
                if career.get('role'):
                    career_text += f" {career.get('role')}"
                if career.get('work_month'):
                    career_text += f" {career['work_month']}개월"
                career_list.append(career_text)
            resume_text += f"경력: {', '.join(career_list)}, "
        else:
            resume_text += "경력: 경력 없음, "
        
        # 자격증 정보
        if spec_data.get('certificates'):
            resume_text += f"자격증: {', '.join(spec_data['certificates'])}, "
        else:
            resume_text += "자격증: 자격증 없음, "
        
        # 언어 능력
        if spec_data.get('languages'):
            lang_list = []
            for lang in spec_data['languages']:
                lang_list.append(f"{lang['test']} {lang['score_or_grade']}")
            resume_text += f"어학: {', '.join(lang_list)}, "
        else:
            resume_text += "어학: 공인 어학 점수 없음, "
        
        # 활동 정보
        if spec_data.get('activities'):
            activity_list = []
            for activity in spec_data['activities']:
                activity_text = f"{activity['name']}"
                if activity.get('role'):
                    activity_text += f" {activity.get('role')}"
                if activity.get('award') and activity['award']:
                    activity_text += f" (수상: {activity['award']})"
                activity_list.append(activity_text)
            resume_text += f"활동: {', '.join(activity_list)}"
        else:
            resume_text += "활동: 활동 내역 없음"
        
        return resume_text
    
    def predict(self, spec_data):
        """
        스펙 데이터를 평가하여 점수 반환
        
        Args:
            spec_data (dict): SpecV1 API 형식의 스펙 데이터
            
        Returns:
            dict: 닉네임과 총점을 포함한 결과
        """
        try:
            # 이력서 텍스트 형식으로 변환
            resume_text = self._format_resume_text(spec_data)
            
            # 지원 직종
            job_field = spec_data['desired_job']
            
            # 평가 실행
            score_result = self.evaluator.evaluate(resume_text, job_field)
            
            # 점수가 문자열이면 실수로 변환
            try:
                total_score = float(score_result)
            except:
                total_score = 50.0  # 기본값
            
            # 결과 포맷
            result = {
                "nickname": spec_data['nickname'],
                "totalScore": total_score
            }            
            return result
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            # 오류 발생 시 기본 점수 반환
            return {
                "nickname": spec_data['nickname'],
                "totalScore": 10.04
            }