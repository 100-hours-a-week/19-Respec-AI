from typing import Dict, Optional
from database_connector import DatabaseConnector
from model_manager import ModelManager
from prompt_generator import PromptGenerator
from score_parser import ScoreParser
from vector_database import VectorDatabase

class ResumeEvaluationSystem:
    """전체 이력서 평가 시스템을 관리하는 메인 클래스"""
    
    def __init__(self):
        # 각 컴포넌트 초기화
        self.db_connector = DatabaseConnector()
        self.model_manager = ModelManager()
        self.prompt_generator = PromptGenerator()
        self.score_parser = ScoreParser()
        
        # 벡터 DB 초기화
        try:
            self.vector_db = VectorDatabase()
            self.rag_enabled = True
        except Exception as e:
            print(f"⚠️ RAG 기능 비활성화: {e}")
            self.vector_db = None
            self.rag_enabled = False
    
    def evaluate_resume(self, spec_data: Dict) -> Dict:
        """이력서 평가 실행"""
        try:
            # 기본 정보 준비
            job_field = spec_data['desired_job']
            univ_name = spec_data.get('universities', [{}])[0].get('school_name', '')
            
            # DB에서 평가 데이터 로드
            weights, examples, criteria, ranking = self.db_connector.load_job_specific_data(
                job_field, univ_name
            )
            
            # RAG 컨텍스트 준비 (가능한 경우)
            rag_context = self._prepare_rag_context(spec_data, job_field) if self.rag_enabled else {}
            
            # 프롬프트 생성
            system_prompt = (
                self.prompt_generator.create_rag_enhanced_prompt(job_field, weights, criteria, rag_context)
                if rag_context else
                self.prompt_generator.create_job_specific_prompt(job_field, weights, examples, criteria)
            )
            
            # 이력서 텍스트 생성 및 채팅 포맷 준비
            resume_text = self._format_resume_text(spec_data)
            chat = self.prompt_generator.create_chat_format(system_prompt, resume_text)
            
            # 모델 평가 실행
            if not self.model_manager.model and not self.model_manager.load_model():
                return self._create_default_response(spec_data['nickname'])
                
            response = self.model_manager.generate_response(chat)
            if not response:
                return self._create_default_response(spec_data['nickname'])
            
            # 점수 추출 및 검증
            score = self._validate_score(self.score_parser.extract_score(response))
            
            return {
                "nickname": spec_data['nickname'],
                "totalScore": score,
                "evaluation_type": "RAG" if (rag_context and self.rag_enabled) else "Basic"
            }
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return self._create_default_response(spec_data['nickname'])
    
    def _prepare_rag_context(self, spec_data: Dict, job_field: str) -> Dict:
        """RAG 컨텍스트 준비"""
        if not self.rag_enabled:
            return {}
            
        context = {}
        
        # 전공 검색
        if spec_data.get('universities'):
            for univ in spec_data['universities']:
                if univ.get('major'):
                    matches = self.vector_db.search_similar_majors(univ['major'], job_field, top_k=1)
                    if matches:
                        context['education_matches'] = matches
                        break
        
        # 자격증 검색
        if spec_data.get('certificates'):
            cert_matches = []
            for cert in spec_data['certificates'][:3]:
                matches = self.vector_db.search_similar_certificates(cert, job_field, top_k=1)
                if matches:
                    cert_matches.extend(matches)
            if cert_matches:
                context['certificate_matches'] = cert_matches
        
        # 활동 검색
        if spec_data.get('activities'):
            activity_matches = []
            for activity in spec_data['activities'][:3]:
                activity_text = f"{activity.get('name', '')} {activity.get('role', '')}"
                matches = self.vector_db.search_similar_activities(activity_text, job_field, top_k=1)
                if matches:
                    activity_matches.extend(matches)
            if activity_matches:
                context['activity_matches'] = activity_matches
        
        return context
    
    def _format_resume_text(self, spec_data: Dict) -> str:
        """이력서 텍스트 포맷팅"""
        sections = []
        
        # 기본 정보
        sections.append(f"최종학력: {spec_data['final_edu']} ({spec_data['final_status']})")
        sections.append(f"지원직종: {spec_data['desired_job']}")
        
        # 대학 정보
        if spec_data.get('universities'):
            univ_texts = []
            for univ in spec_data['universities']:
                parts = [univ['school_name']]
                if univ.get('major'): parts.append(univ['major'])
                if univ.get('degree'): parts.append(f"({univ['degree']})")
                if univ.get('gpa') and univ.get('gpa_max'):
                    parts.append(f"학점:{univ['gpa']}/{univ['gpa_max']}")
                univ_texts.append(' '.join(parts))
            sections.append(f"학력: {', '.join(univ_texts)}")
        else:
            sections.append("학력: 대학 정보 없음")
        
        # 경력 정보
        if spec_data.get('careers'):
            career_texts = []
            for career in spec_data['careers']:
                parts = [career['company']]
                if career.get('role'): parts.append(career['role'])
                if career.get('work_month'): parts.append(f"{career['work_month']}개월")
                career_texts.append(' '.join(parts))
            sections.append(f"경력: {', '.join(career_texts)}")
        else:
            sections.append("경력: 경력 없음")
        
        # 자격증
        sections.append(
            f"자격증: {', '.join(spec_data.get('certificates', []))}" if spec_data.get('certificates')
            else "자격증: 자격증 없음"
        )
        
        # 어학
        if spec_data.get('languages'):
            lang_texts = [f"{lang['test']} {lang['score_or_grade']}" for lang in spec_data['languages']]
            sections.append(f"어학: {', '.join(lang_texts)}")
        else:
            sections.append("어학: 공인 어학 점수 없음")
        
        # 활동
        if spec_data.get('activities'):
            activity_texts = []
            for activity in spec_data['activities']:
                parts = [activity['name']]
                if activity.get('role'): parts.append(activity['role'])
                if activity.get('award'): parts.append(f"(수상: {activity['award']})")
                activity_texts.append(' '.join(parts))
            sections.append(f"활동: {', '.join(activity_texts)}")
        else:
            sections.append("활동: 활동 내역 없음")
        
        return ', '.join(sections)
    
    def _validate_score(self, score: Optional[str]) -> float:
        """점수 검증 및 변환"""
        try:
            score = float(score)
            return max(0.0, min(100.0, score))
        except:
            return 50.0
    
    def _create_default_response(self, nickname: str) -> Dict:
        """기본 응답 생성"""
        return {
            "nickname": nickname,
            "totalScore": 50.0,
            "evaluation_type": "Default"
        }
    
    def get_system_status(self) -> Dict:
        """시스템 상태 확인"""
        return {
            "rag_enabled": self.rag_enabled,
            "model_loaded": self.model_manager.model is not None,
            "vector_db_stats": self.vector_db.get_statistics() if self.rag_enabled else None
        }