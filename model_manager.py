import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelManager:
    """AI 모델 로드 및 추론 작업을 담당하는 클래스"""
    
    def __init__(self, model_name="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 중인 장치: {self.device}")
    
    def load_model(self):
        """모델과 토크나이저 로드"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return True
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False
    
    def generate_response(self, chat):
        """모델을 사용하여 응답 생성"""
        try:
            inputs = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(device=self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=17,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.2,
                tokenizer=self.tokenizer
            )

            return self.tokenizer.batch_decode(output_ids)[0]
        except Exception as e:
            print(f"모델 추론 오류: {e}")
            return None