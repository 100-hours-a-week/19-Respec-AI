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
    
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            print("벡터 데이터베이스 연결 종료")