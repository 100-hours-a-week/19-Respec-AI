
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ModelManager:
    """AI 모델 로드 및 추론 작업을 담당하는 클래스 (파인튜닝 모델 지원)"""
    
    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct", peft_model_path="syahaeun/qwen2-resume-evaluator"):
        self.model_name = model_name
        self.peft_model_path = peft_model_path  # 파인튜닝된 모델 경로
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 중인 장치: {self.device}")
    
    def load_model(self):
        """모델과 토크나이저 로드 (파인튜닝 모델 우선)"""
        try:
            # 1. 토크나이저 로드 (베이스 모델에서)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 2. 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # 3. 파인튜닝 모델이 있으면 로드, 없으면 베이스 모델 사용
            if self.peft_model_path:
                print(f"파인튜닝 모델 로드 중: {self.peft_model_path}")
                self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
                print("파인튜닝 모델 로드 완료")
            else:
                print("베이스 모델 사용")
                self.model = base_model
            
            # 4. GPU로 이동
            self.model = self.model.to(device=self.device)
            return True
            
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False