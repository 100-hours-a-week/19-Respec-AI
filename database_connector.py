import psycopg2
from database_config import DatabaseConfig

class DatabaseConnector:
    """데이터베이스 연결 및 쿼리 실행을 담당하는 클래스"""
    def __init__(self):
        self.db_config = DatabaseConfig().get_config()
    
    def connect(self):
        """데이터베이스에 연결하고 커넥션 객체 반환"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"데이터베이스 연결 오류: {e}")
            return None
    
    def load_job_specific_data(self, job_field):
        """직무별 가중치, Few-shot 예제, 평가 기준을 로드"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            # 1. 직무별 가중치 로드
            cursor.execute("""
                SELECT education_score, education_major_weight, education_school_weight, certification_score, certification_major_weight, certification_level_weight, 
                    experience_score, experience_major_weight, experience_term_weight, language_score, activity_score, activity_major_weight, activity_result_weight
                FROM job_weights
                WHERE job_field = %s
            """, (job_field,))
            weights = cursor.fetchone()


            # 2. 직무별 평가 기준 로드
            cursor.execute("""
                SELECT evaluation_criteria
                FROM job_criteria
                WHERE job_field = %s
            """, (job_field,))
            criteria_result = cursor.fetchone()
            criteria = criteria_result[0] if criteria_result else "기본 평가 기준"

            cursor.close()
            conn.close()

            return weights, criteria
            
        except Exception as e:
            print(f"데이터베이스 쿼리 오류: {e}")
            return None,None
