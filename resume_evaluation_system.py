import json
import re
from typing import Dict, Optional
from database_connector import DatabaseConnector
from model_manager import ModelManager
from prompt_generator import PromptGenerator
from score_parser import ScoreParser, LanguageScoreValidator
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
            
            # DB에서 평가 데이터 로드
            weights, criteria = self.db_connector.load_job_specific_data(job_field)
            
            # RAG 컨텍스트 준비 (가능한 경우)
            rag_context = self._prepare_rag_context(spec_data, job_field) if self.rag_enabled else {}
            
            # 프롬프트 생성
            system_prompt = (
                self.prompt_generator.create_rag_enhanced_prompt(job_field, weights, criteria, rag_context)
                if rag_context else
                self.prompt_generator.create_job_specific_prompt(job_field, weights, criteria)
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
            
            # ===== 🔧 핵심 수정 부분 =====
            # JSON에서 assessment 값만 추출
            assessment_text = self._extract_assessment_from_response(response)
            
            # score = self._validate_score(self.score_parser.extract_score(response))
            score = self.prompt_generator.score_calculator.get_total_score()
            # PromptGenerator에서 계산된 정규화된 점수 가져오기
            normalized_scores = {}
            if self.prompt_generator.score_calculator:
                normalized_scores = self.prompt_generator.score_calculator.normalize_to_100()
            
            return {
                "nickname": spec_data['nickname'],
                "totalScore": score,
                "academicScore": min(normalized_scores.get("academic", 0.0),100),
                "workExperienceScore": min(normalized_scores.get("workExperience", 0.0),100),
                "certificationScore": min(normalized_scores.get("certification", 0.0),100),
                "languageProficiencyScore": min(normalized_scores.get("languageProficiency", 0.0),100),
                "extracurricularScore": min(normalized_scores.get("extracurricular", 0.0),100),
                "assessment": assessment_text  # 📌 assessment 값만 포함
            }
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return self._create_default_response(spec_data['nickname'])
    
    def _extract_assessment_from_response(self, response: str) -> str:
        """LLM 응답에서 assessment 값만 추출하는 함수"""
        try:
            # print(f"🔍 원본 응답: {response[:200]}...")  # 디버깅용
            
            # 방법 1: assessment": "내용" 패턴으로 직접 추출 (가장 안전)
            assessment_patterns = [
                r'"assessment":\s*"([^"]*)"',  # 기본 패턴
                r'"assessment"\s*:\s*"([^"]*)"',  # 공백 포함
                r'assessment":\s*"([^"]*)"',  # 앞의 따옴표 없는 경우
                r'"assessment":\s*\'([^\']*)\'',  # 작은따옴표 사용
            ]
            
            for pattern in assessment_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    assessment_text = match.group(1)
                    # print(f"✅ Pattern으로 추출 성공: {assessment_text}")
                    return assessment_text
            
            # 방법 2: JSON 블록 전체 추출 후 파싱
            json_patterns = [
                r'\{[^}]*"assessment"[^}]*\}',  # 한 줄 JSON
                r'\{[\s\S]*?"assessment"[\s\S]*?\}',  # 여러 줄 JSON
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response)
                if json_match:
                    json_str = json_match.group()
                    try:
                        # JSON 문자열 정제
                        json_str = json_str.replace('\n', '').replace('\r', '')
                        parsed = json.loads(json_str)
                        if "assessment" in parsed:
                            assessment_text = parsed["assessment"]
                            # print(f"✅ JSON 파싱으로 추출 성공: {assessment_text}")
                            return assessment_text
                    except json.JSONDecodeError as je:
                        print(f"⚠️ JSON 파싱 실패: {je}")
                        continue
            
            # 방법 3: 라인별 분석 (assessment가 포함된 라인 찾기)
            lines = response.split('\n')
            for line in lines:
                if 'assessment' in line.lower():
                    # 콜론 뒤의 내용 추출
                    if ':' in line:
                        after_colon = line.split(':', 1)[1].strip()
                        # 따옴표와 특수문자 제거
                        cleaned = re.sub(r'^["\'\s,{]+|["\'\s,}]+$', '', after_colon)
                        if cleaned and len(cleaned) > 5:  # 의미있는 길이의 텍스트
                            # print(f"✅ 라인 분석으로 추출: {cleaned}")
                            return cleaned
            
            # 방법 4: 전체 응답에서 의미있는 한국어 문장 추출
            korean_sentences = re.findall(r'[가-힣\s]{10,}', response)
            if korean_sentences:
                # 가장 긴 한국어 문장을 선택
                longest_sentence = max(korean_sentences, key=len).strip()
                if len(longest_sentence) > 10:
                    # print(f"✅ 한국어 문장 추출: {longest_sentence[:50]}...")
                    return longest_sentence[:100]  # 100자로 제한
            
            # print("❌ 모든 추출 방법 실패")
            return "구체적인 스펙 분석 후 개선방안을 제시드릴 수 있습니다."
            
        except Exception as e:
            print(f"❌ Assessment 추출 중 예외 발생: {e}")
            print(f"❌ 응답 내용: {response}")
            return "평가 내용 추출 중 오류가 발생했습니다."
    
    def _prepare_rag_context(self, spec_data: Dict, job_field: str) -> Dict:
        """RAG 컨텍스트 준비"""
        if not self.rag_enabled:
            return {}
            
        context = {
            'education_matches': [],
            'university_matches': [],
            'company_matches': [],
            'certificate_matches': [],
            'activity_matches': [],
            'language_scores': []
        }
        
        # 전공과 대학교 정보 수집
        if spec_data.get('universities'):
            for univ in spec_data['universities']:
                # 대학교 검색
                if univ.get('school_name'):
                    uni_matches = self.vector_db.search_similar_universities(univ['school_name'], top_k=1)
                    if uni_matches:
                        context['university_matches'].extend(uni_matches)
                
                # 전공 검색
                if univ.get('major'):
                    matches = self.vector_db.search_similar_majors(univ['major'], job_field, top_k=1)
                    if matches:
                        context['education_matches'].extend(matches)
        
        # 경력 정보 수집
        if spec_data.get('careers'):
            for career in spec_data['careers']:
                if career.get('company') and career.get('role'):
                    company_matches = self.vector_db.search_similar_companies(
                        career['company'],
                        job_field,
                        career['role'],
                        top_k=1
                    )
                    if company_matches:
                        # 근무 기간 정보 추가
                        for match in company_matches:
                            match['work_month'] = career.get('work_month', 0)
                        context['company_matches'].extend(company_matches)
        
        # 자격증 정보 수집
        if spec_data.get('certificates'):
            context['certificate_matches'] = []
            for certificate in spec_data['certificates']:
                cert_matches = self.vector_db.search_similar_certificates(certificate, job_field, top_k=1)
                if cert_matches:
                    context['certificate_matches'].extend(cert_matches)
        
        # 활동 정보 수집
        if spec_data.get('activities'):
            for activity in spec_data['activities']:
                if activity.get('name'):
                    activity_matches = self.vector_db.search_similar_activities(activity['name'], job_field, top_k=1)
                    if activity_matches:
                        context['activity_matches'].extend(activity_matches)
        
        # 어학 정보 수집 및 검증
        if spec_data.get('languages'):
            context['language_scores'] = []
            language_scores = []  # 개별 어학 점수 저장
            
            for lang in spec_data['languages']:
                is_valid, normalized_score = LanguageScoreValidator.validate_score(
                    lang['test'], 
                    lang['score_or_grade']
                )
                
                context['language_scores'].append({
                    'test': lang['test'],
                    'score': lang['score_or_grade'],
                    'is_valid': is_valid,
                    'normalized_score': normalized_score
                })
                
                if is_valid:
                    language_scores.append(normalized_score)
            
            # 가장 높은 점수에 나머지는 10%만 추가하는 방식으로 계산
            if language_scores:
                max_score = max(language_scores)
                remaining_scores = [score for score in language_scores if score != max_score]
                bonus_from_others = sum(remaining_scores) * 0.1
                context['average_language_score'] = max_score + bonus_from_others
            else:
                context['average_language_score'] = 0.0
        
        return context
    
    def _format_resume_text(self, spec_data: Dict) -> str:
        """이력서 텍스트 포맷팅"""
        sections = []
        
        # 기본 정보
        sections.append(f"""
=== 이력서 내용 ===
최종학력: {spec_data['final_edu']} ({spec_data['final_status']})""")
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
            "academicScore": 0.0,
            "workExperienceScore": 0.0,
            "certificationScore": 0.0,
            "languageProficiencyScore": 0.0,
            "extracurricularScore": 0.0,
            "assessment": "기본 평가: 추가 정보 입력 후 재평가를 권장합니다.",
            "evaluation_type": "Default"
        }
    
    def get_system_status(self) -> Dict:
        """시스템 상태 확인"""
        return {
            "rag_enabled": self.rag_enabled,
            "model_loaded": self.model_manager.model is not None,
            "vector_db_stats": self.vector_db.get_statistics() if self.rag_enabled else None
        }