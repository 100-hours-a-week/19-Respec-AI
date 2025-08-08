import psycopg2
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import json
from database_config import DatabaseConfig

class VectorDatabase:
    """벡터 데이터베이스 연결 및 검색 관리 클래스"""
    
    def __init__(self):
        self.db_config = DatabaseConfig().get_config()
        self.conn = None
        self._connect()
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    def _connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("벡터 데이터베이스 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 오류: {e}")
            raise
    
    def search_similar_majors(self, major: str, job_category: str, top_k: int = 3) -> List[Dict]:
        """유사한 전공 검색"""
        try:
            query_embedding = self.embedding_model.encode(major)
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT major_name, job_category, relevance_score,
                       1 - (embedding <=> %s::vector) as similarity
                FROM major_job_mapping
                WHERE job_category = %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), job_category, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "major_name": row[0],
                    "job_category": row[1],
                    "relevance_score": float(row[2]),
                    "similarity": float(row[3])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"전공 검색 오류: {e}")
            return []
    
    def search_similar_certificates(self, certificate: str, job_category: str, top_k: int = 5) -> List[Dict]:
        """유사한 자격증 검색"""
        try:
            query_embedding = self.embedding_model.encode(certificate)
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT certificate_name, job_category, weight_score,
                       1 - (embedding <=> %s::vector) as similarity
                FROM certificate_job_mapping
                WHERE job_category = %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), job_category, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "certificate_name": row[0],
                    "job_category": row[1],
                    "weight_score": float(row[2]),
                    "similarity": float(row[3])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"자격증 검색 오류: {e}")
            return []
    
    def search_similar_activities(self, activity: str, job_category: str, top_k: int = 5) -> List[Dict]:
        """유사한 활동 검색"""
        try:
            query_embedding = self.embedding_model.encode(activity)
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT activity_keyword, job_category, relevance_score,
                       1 - (embedding <=> %s::vector) as similarity
                FROM activity_job_mapping
                WHERE job_category = %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), job_category, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "activity_keyword": row[0],
                    "job_category": row[1],
                    "relevance_score": float(row[2]),
                    "similarity": float(row[3])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"활동 검색 오류: {e}")
            return []
    def search_similar_universities(self, query: str, top_k: int = 1) -> List[Tuple]:
        """유사한 대학교 검색 (임베딩 기반)"""
        try:
            query_embedding = self.embedding_model.encode(query)
            cursor = self.conn.cursor()
            # 코사인 유사도 기반 검색
            search_query = """
            SELECT university_name, score, rank_position,
                  1 -  (name_embedding <=> %s::vector) as similarity
            FROM university_rankings
            ORDER BY similarity DESC
            LIMIT %s;
            """
            
            cursor.execute(search_query, (query_embedding.tolist(), top_k))
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "university_name": row[0],
                    "score": float(row[1]),
                    "rank_position": row[2],
                    "similarity": float(row[3])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"대학 검색 오류: {e}")
            return []
    def search_similar_companies(self, company: str, job_category: str, role: str, top_k: int = 5) -> List[Dict]:
        """유사한 회사 및 직무 검색"""
        try:
            # 회사명과 직무를 결합하여 임베딩
            query_embedding = self.embedding_model.encode(company)
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT company_name, industry_group, recognition_score,
                       1 - (name_embedding <=> %s::vector) as similarity
                FROM company_recognition
                WHERE industry_group = %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), job_category, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "company_name": row[0],
                    "industry_group": row[1],
                    "recognition_score": float(row[2]),
                    "similarity": float(row[3])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"회사 및 직무 검색 오류: {e}")
            return []
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            print("벡터 데이터베이스 연결 종료")