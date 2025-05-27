from database_connector import DatabaseConnector
from model_manager import ModelManager
from prompt_generator import PromptGenerator
from score_parser import ScoreParser
from resume_evaluator import ResumeEvaluator
from vector_database import VectorDatabase
import os
from dotenv import load_dotenv

class SpecEvaluator:
    """
    RAG 기능이 통합된 간단하고 효과적인 스펙 평가기
    기존 구조를 유지하면서 벡터 검색만 추가
    """
    
    def __init__(self):
        """기존 구조 유지하면서 벡터 DB만 추가"""
        # 기존 시스템 초기화
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
        
        # 벡터 데이터베이스 추가 (선택적)
        try:
            load_dotenv()
            db_config = {
                'host': os.getenv('HOST'),
                'database': os.getenv('DATABASE'),
                'user': os.getenv('USER'),
                'password': os.getenv('PASSWORD'),
                'port': 5432
            }
            self.vector_db = VectorDatabase(db_config)
            self.rag_enabled = True
            print("✅ RAG 기능 활성화")
        except Exception as e:
            print(f"⚠️ RAG 기능 비활성화 (기본 평가 사용): {e}")
            self.vector_db = None
            self.rag_enabled = False
    
    def _get_rag_context(self, spec_data, job_field):
        """벡터 검색으로 RAG 컨텍스트 생성 (간단 버전)"""
        if not self.rag_enabled:
            return {}
        
        try:
            rag_context = {}
            
            # 전공 유사도 검색
            if spec_data.get('universities'):
                for univ in spec_data['universities']:
                    if univ.get('major'):
                        matches = self.vector_db.search_similar_majors(
                            univ['major'], job_field, top_k=1
                        )
                        if matches:
                            rag_context['education_matches'] = matches
                            break
            
            # 자격증 유사도 검색  
            if spec_data.get('certificates'):
                cert_matches = []
                for cert in spec_data['certificates'][:3]:  # 최대 3개만
                    matches = self.vector_db.search_similar_certificates(
                        cert, job_field, top_k=1
                    )
                    if matches:
                        cert_matches.extend(matches)
                if cert_matches:
                    rag_context['certificate_matches'] = cert_matches
            
            # 활동 유사도 검색
            if spec_data.get('activities'):
                activity_matches = []
                for activity in spec_data['activities'][:3]:  # 최대 3개만
                    activity_text = f"{activity.get('name', '')} {activity.get('role', '')}"
                    matches = self.vector_db.search_similar_activities(
                        activity_text, job_field, top_k=1
                    )
                    if matches:
                        activity_matches.extend(matches)
                if activity_matches:
                    rag_context['activity_matches'] = activity_matches
            
            return rag_context
            
        except Exception as e:
            print(f"RAG 컨텍스트 생성 오류: {e}")
            return {}
    
    def _format_resume_text(self, spec_data):
        """기존 이력서 텍스트 포맷팅 (변경 없음)"""
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
        핵심 수정: 기존 evaluator를 그대로 사용하되 
        프롬프트만 RAG 강화 버전으로 교체
        """
        try:
            # 1. 기본 정보 추출
            resume_text = self._format_resume_text(spec_data)
            job_field = spec_data['desired_job']
            univ_name = ""
            if spec_data.get('universities') and len(spec_data['universities']) > 0:
                univ_name = spec_data['universities'][0].get('school_name', '')
            
            # 2. 기존 DB에서 가중치와 기준 로드
            weights, few_shot_examples, criteria, university_ranking = \
                self.db_connector.load_job_specific_data(job_field, univ_name)
            
            # 3. RAG 컨텍스트 생성 (있으면 사용, 없으면 무시)
            rag_context = self._get_rag_context(spec_data, job_field) if self.rag_enabled else {}
            
            # 4. 프롬프트 생성 (RAG 컨텍스트가 있으면 강화 버전, 없으면 기본 버전)
            if rag_context and self.rag_enabled:
                # RAG 강화 프롬프트 사용
                system_prompt = self.prompt_generator.create_rag_enhanced_prompt(
                    job_field, weights, criteria, rag_context
                )
                print("📊 RAG 강화 프롬프트 사용")
            else:
                # 기본 프롬프트 사용
                system_prompt = self.prompt_generator.create_job_specific_prompt(
                    job_field, weights, few_shot_examples, criteria
                )
                print("📝 기본 프롬프트 사용")
            
            # 5. 기존 평가 로직 그대로 사용
            chat = self.prompt_generator.create_chat_format(system_prompt, resume_text)
            
            # 모델 로드 확인
            if not self.model_manager.model:
                if not self.model_manager.load_model():
                    return {"nickname": spec_data['nickname'], "totalScore": 50.0}
            
            # 모델 추론
            full_output = self.model_manager.generate_response(chat)
            if not full_output:
                return {"nickname": spec_data['nickname'], "totalScore": 50.0}
            
            # 결과 파싱
            final_score = self.score_parser.extract_score(full_output)
            
            # 점수 검증 및 변환
            try:
                total_score = float(final_score)
                total_score = max(0.0, min(100.0, total_score))  # 0-100 범위 제한
            except:
                total_score = 50.0  # 기본값
            
            result = {
                "nickname": spec_data['nickname'],
                "totalScore": total_score
            }
            
            # RAG 사용 여부 로깅
            rag_status = "RAG 활성" if (rag_context and self.rag_enabled) else "기본 평가"
            print(f"✅ 평가 완료: {spec_data['nickname']} -> {total_score:.2f}점 ({rag_status})")
            
            return result
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return {
                "nickname": spec_data['nickname'],
                "totalScore": 50.0
            }
    
    def get_system_status(self):
        """시스템 상태 확인"""
        return {
            "rag_enabled": self.rag_enabled,
            "model_loaded": self.model_manager.model is not None,
            "vector_db_stats": self.vector_db.get_statistics() if self.rag_enabled else None
        }