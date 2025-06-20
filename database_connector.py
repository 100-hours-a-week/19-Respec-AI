import psycopg2
from dotenv import load_dotenv
import os

class DatabaseConnector:
    """데이터베이스 연결 및 쿼리 실행을 담당하는 클래스"""
    def __init__(self):
        load_dotenv()
        self.host = os.environ.get('HOST')
        self.database = os.environ.get('DATABASE')
        self.user = os.environ.get('USER')
        self.password = os.environ.get('PASSWORD')
    
    def connect(self):
        """데이터베이스에 연결하고 커넥션 객체 반환"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            return conn
        except Exception as e:
            print(f"데이터베이스 연결 오류: {e}")
            return None
    
    def load_job_specific_data(self, job_field):
        """직무별 가중치, Few-shot 예제, 평가 기준을 로드"""
        try:
            conn = self.connect()
            if not conn:
                return self.get_default_data()
                
            cursor = conn.cursor()

            # 1. 직무별 가중치 로드
            cursor.execute("""
                SELECT education_weight, certification_weight, experience_weight,
                    language_weight, activity_weight
                FROM job_weights
                WHERE job_field = %s
            """, (job_field,))
            weights = cursor.fetchone()

            # 2. 직무별 Few-shot 예제 로드
            cursor.execute("""
                SELECT resume_text, score
                FROM few_shot_examples
                WHERE job_field = %s
                ORDER BY score
            """, (job_field,))
            few_shot_examples = cursor.fetchall()

            # 3. 직무별 평가 기준 로드
            cursor.execute("""
                SELECT evaluation_criteria
                FROM job_criteria
                WHERE job_field = %s
            """, (job_field,))
            criteria = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return weights, few_shot_examples, criteria
            
        except Exception as e:
            print(f"데이터베이스 쿼리 오류: {e}")
            return self.get_default_data()
    
    def get_default_data(self):
        """데이터베이스 연결 실패 시 기본값 반환"""
        return (15.0, 20.0, 40.0, 10.0, 15.0), [], "기술적 전문성, 프로그래밍 능력, 문제 해결 능력"