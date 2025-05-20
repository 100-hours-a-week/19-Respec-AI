import time

class ResumeEvaluator:
    """이력서 평가 과정을 관리하는 클래스"""
    
    def __init__(self, db_connector, model_manager, prompt_generator, score_parser):
        self.db_connector = db_connector
        self.model_manager = model_manager
        self.prompt_generator = prompt_generator
        self.score_parser = score_parser
    
    def evaluate(self, user_resume, job_field):
        """이력서 평가 수행"""
        start_time = time.time()
        
        # 1. 모델 로드 확인
        if not self.model_manager.model:
            if not self.model_manager.load_model():
                return "모델 로드 실패"
        
        # 2. 직무별 데이터 로드
        weights, few_shot_examples, criteria = self.db_connector.load_job_specific_data(job_field)
        
        # 3. 직무별 프롬프트 생성
        system_prompt = self.prompt_generator.create_job_specific_prompt(
            job_field, weights, few_shot_examples, criteria
        )
        
        # 4. 채팅 형식 구성
        chat = self.prompt_generator.create_chat_format(system_prompt, user_resume)
        
        # 5. 모델 추론
        full_output = self.model_manager.generate_response(chat)
        if not full_output:
            return "모델 추론 실패"
        
        # 6. 결과 파싱
        final_score = self.score_parser.extract_score(full_output)
        
        end_time = time.time()
        print(f"평가 소요 시간: {end_time - start_time:.2f}초")
        
        return final_score