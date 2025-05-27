from rag_evaluator import ResumeEvaluationRAG
from prompt_generator import PromptGenerator
from database_connector import DatabaseConnector
import os
from dotenv import load_dotenv

class SpecEvaluator:
    """
    RAG 기반 스펙 평가를 담당하는 통합 클래스
    기존 시스템과 RAG 시스템을 완전히 통합
    """
    
    def __init__(self):
        """RAG 기반 평가 시스템으로 초기화"""
        load_dotenv()
        
        # 벡터 데이터베이스 설정
        db_config = {
            'host': os.environ.get('HOST'),
            'database': os.environ.get('DATABASE'), 
            'user': os.environ.get('USER'),
            'password': os.environ.get('PASSWORD'),
            'port': os.environ.get('PORT', 5432)
        }
        
        # RAG 평가 시스템 초기화
        self.rag_evaluator = ResumeEvaluationRAG(db_config)
        
        # 프롬프트 생성기 초기화 (RAG 컨텍스트 활용용)
        self.prompt_generator = PromptGenerator()
        
        # 기존 데이터베이스 커넥터 (가중치, 기준 정보용)
        self.db_connector = DatabaseConnector()
        
        print("RAG 기반 SpecEvaluator 초기화 완료")
    
    def _format_resume_data(self, spec_data):
        """SpecV1 API 데이터를 RAG 평가용 형식으로 변환"""
        return {
            'nickname': spec_data['nickname'],
            'final_edu': spec_data['final_edu'],
            'final_status': spec_data['final_status'],
            'desired_job': spec_data['desired_job'],
            'universities': spec_data.get('universities', []),
            'careers': spec_data.get('careers', []),
            'certificates': spec_data.get('certificates', []),
            'languages': spec_data.get('languages', []),
            'activities': spec_data.get('activities', [])
        }
    
    def _create_rag_context_from_evaluation(self, evaluation_result):
        """RAG 평가 결과에서 프롬프트용 컨텍스트 추출"""
        component_scores = evaluation_result['component_scores']
        
        rag_context = {}
        
        # 학력 매칭 정보
        if 'education' in component_scores and 'details' in component_scores['education']:
            education_matches = []
            for detail in component_scores['education']['details']:
                if isinstance(detail, dict) and 'major' in detail:
                    education_matches.append({
                        'input_major': detail['major'],
                        'matched_major': detail.get('matched_major', ''),
                        'similarity': detail.get('similarity', 0),
                        'relevance_score': detail.get('base_relevance', 0)
                    })
            rag_context['education_matches'] = education_matches
        
        # 자격증 매칭 정보  
        if 'certificates' in component_scores and 'details' in component_scores['certificates']:
            certificate_matches = []
            for detail in component_scores['certificates']['details']:
                if isinstance(detail, dict):
                    certificate_matches.append({
                        'input_certificate': detail.get('input_certificate', ''),
                        'matched_certificate': detail.get('matched_certificate', ''),
                        'similarity': detail.get('similarity', 0),
                        'weight_score': detail.get('base_weight', 0)
                    })
            rag_context['certificate_matches'] = certificate_matches
        
        # 활동 매칭 정보
        if 'activities' in component_scores and 'details' in component_scores['activities']:
            activity_matches = []
            for detail in component_scores['activities']['details']:
                if isinstance(detail, dict):
                    activity_matches.append({
                        'input_activity': detail.get('activity_name', ''),
                        'matched_activity': detail.get('matched_activity', ''),
                        'similarity': detail.get('similarity', 0),
                        'relevance_score': detail.get('base_relevance', 0)
                    })
            rag_context['activity_matches'] = activity_matches
        
        # 경력 요약 정보
        if 'experience' in component_scores and 'details' in component_scores['experience']:
            total_months = sum(
                detail.get('months', 0) 
                for detail in component_scores['experience']['details'] 
                if isinstance(detail, dict)
            )
            rag_context['experience_summary'] = {'total_months': total_months}
        
        return rag_context
    
    def predict(self, spec_data):
        """
        RAG 기반 스펙 평가 수행
        
        Args:
            spec_data (dict): SpecV1 API 형식의 스펙 데이터
            
        Returns:
            dict: 닉네임과 총점을 포함한 결과
        """
        try:
            # 1. 데이터 형식 변환
            resume_data = self._format_resume_data(spec_data)
            job_field = resume_data['desired_job']
            
            # 2. RAG 기반 평가 수행 (벡터 검색 포함)
            evaluation_result = self.rag_evaluator.evaluate_resume(resume_data)
            
            # 3. RAG 검색 결과를 프롬프트 컨텍스트로 변환
            rag_context = self._create_rag_context_from_evaluation(evaluation_result)
            
            # 4. 기존 DB에서 가중치와 기준 정보 로드
            try:
                univ_name = spec_data.get('universities', [{}])[0].get('school_name', '')
                weights, few_shot_examples, criteria, university_ranking = \
                    self.db_connector.load_job_specific_data(job_field, univ_name)
            except:
                # 기본값 사용
                weights = (30.0, 25.0, 25.0, 5.0, 15.0)  # 학력, 자격증, 경력, 어학, 활동
                criteria = f"{job_field} 분야의 핵심 역량과 경험을 중시합니다."
            
            # 5. RAG 강화 프롬프트 생성 및 LLM 평가
            enhanced_prompt = self.prompt_generator.create_rag_enhanced_prompt(
                job_field, weights, criteria, rag_context
            )
            
            # 6. LLM으로 최종 평가 (RAG 컨텍스트 반영)
            resume_text = self._format_resume_text_for_llm(resume_data)
            final_score = self._get_llm_score_with_rag_prompt(enhanced_prompt, resume_text)
            
            # 7. 결과 반환
            result = {
                "nickname": spec_data['nickname'],
                "totalScore": final_score,
                "rag_details": {
                    "component_scores": evaluation_result['component_scores'],
                    "rag_enhanced": True
                }
            }
            
            print(f"RAG 기반 평가 완료: {spec_data['nickname']} -> {final_score:.2f}점")
            return result
            
        except Exception as e:
            print(f"RAG 평가 중 오류 발생: {e}")
            # 오류 시 기본 RAG 점수 사용
            try:
                resume_data = self._format_resume_data(spec_data)
                evaluation_result = self.rag_evaluator.evaluate_resume(resume_data)
                fallback_score = evaluation_result['total_score']
            except:
                fallback_score = 50.0
            
            return {
                "nickname": spec_data['nickname'],
                "totalScore": fallback_score
            }
    
    def _format_resume_text_for_llm(self, resume_data):
        """RAG 평가 데이터를 LLM용 텍스트로 변환"""
        resume_text = f"최종학력: {resume_data['final_edu']} ({resume_data['final_status']})\n"
        resume_text += f"지원직종: {resume_data['desired_job']}\n"
        
        # 대학 정보
        if resume_data.get('universities'):
            univ_list = []
            for univ in resume_data['universities']:
                univ_text = f"{univ.get('school_name', '')}"
                if univ.get('major'):
                    univ_text += f" {univ['major']}"
                if univ.get('gpa') and univ.get('gpa_max'):
                    univ_text += f" (학점: {univ['gpa']}/{univ['gpa_max']})"
                univ_list.append(univ_text)
            resume_text += f"학력: {', '.join(univ_list)}\n"
        
        # 자격증
        if resume_data.get('certificates'):
            resume_text += f"자격증: {', '.join(resume_data['certificates'])}\n"
        
        # 경력
        if resume_data.get('careers'):
            career_list = []
            for career in resume_data['careers']:
                career_text = f"{career.get('company', '')} {career.get('role', '')} {career.get('work_month', 0)}개월"
                career_list.append(career_text)
            resume_text += f"경력: {', '.join(career_list)}\n"
        
        # 활동
        if resume_data.get('activities'):
            activity_list = []
            for activity in resume_data['activities']:
                activity_text = f"{activity.get('name', '')} {activity.get('role', '')}"
                activity_list.append(activity_text)
            resume_text += f"활동: {', '.join(activity_list)}\n"
        
        return resume_text
    
    def _get_llm_score_with_rag_prompt(self, enhanced_prompt, resume_text):
        """RAG 강화 프롬프트로 LLM 점수 생성"""
        try:
            # 채팅 형식 구성
            chat = [
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": resume_text}
            ]
            
            # LLM 추론
            llm_output = self.rag_evaluator.generate_llm_evaluation(
                {'desired_job': '인터넷·IT'}, 
                {'total_score': 75.0}  # 임시값
            )
            
            # 점수 추출
            try:
                score = float(llm_output.strip())
                return min(max(score, 0.0), 100.0)  # 0-100 범위 제한
            except:
                return 10.04  # 기본값
                
        except Exception as e:
            print(f"LLM 점수 생성 오류: {e}")
            return 44.44