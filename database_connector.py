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
    
    def load_job_specific_data(self, job_field, univ_name):
        """직무별 가중치, Few-shot 예제, 평가 기준을 로드"""
        try:
            conn = self.connect()
            if not conn:
                return self.get_default_data()
                
            cursor = conn.cursor()

            # 1. 직무별 가중치 로드
            cursor.execute("""
                SELECT education_score, education_major_weight, education_school_weight, certification_score, certification_major_weight, certification_level_weight, 
                    experience_score, experience_major_weight, experience_term_weight, language_score, activity_score, activity_major_weight, activity_result_weight
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
            criteria_result = cursor.fetchone()
            criteria = criteria_result[0] if criteria_result else "기본 평가 기준"

            # 4. QS대학 순위 로드
            cursor.execute("""
                SELECT international_rank, domestic_rank
                FROM university_rankings
                WHERE university like %s
            """, (f"%{univ_name}%",))
            university_ranking = cursor.fetchall()

            cursor.close()
            conn.close()

            return weights, few_shot_examples, criteria, university_ranking
            
        except Exception as e:
            print(f"데이터베이스 쿼리 오류: {e}")
            return self.get_default_data()
    
    def get_default_data(self):
        """데이터베이스 연결 실패 시 기본값 반환"""
        default_weights = (30.0, 25.0, 25.0, 5.0, 15.0)  # 학력, 자격증, 경력, 어학, 활동
        default_examples = []
        default_criteria = "기술적 전문성, 프로그래밍 능력, 문제 해결 능력을 중시합니다."
        default_ranking = []
        
        return default_weights, default_examples, default_criteria, default_ranking