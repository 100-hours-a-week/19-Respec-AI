import json
import time
import concurrent.futures
from tqdm import tqdm

class BatchProcessor:
    """스펙 평가를 위한 배치 처리 클래스"""
    
    def __init__(self, evaluator, batch_size=10, max_workers=4):
        """
        배치 처리 초기화
        
        Args:
            evaluator: SpecEvaluator 인스턴스
            batch_size: 배치 당 처리할 항목 수
            max_workers: 병렬 처리 시 최대 워커 수
        """
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batch_sequential(self, spec_data_list):
        """
        순차적으로 배치 처리
        
        Args:
            spec_data_list: 스펙 데이터 리스트
            
        Returns:
            results: 평가 결과 리스트
        """
        results = []
        
        print(f"총 {len(spec_data_list)}개 항목을 처리합니다...")
        start_time = time.time()
        
        for i, spec_data in enumerate(tqdm(spec_data_list)):
            result = self.evaluator.predict(spec_data)
            results.append(result)
            
            # 배치 크기마다 진행 상황 출력
            if (i + 1) % self.batch_size == 0:
                elapsed = time.time() - start_time
                items_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"처리 속도: {items_per_sec:.2f} 항목/초")
        
        total_time = time.time() - start_time
        print(f"총 처리 시간: {total_time:.2f}초")
        print(f"평균 처리 속도: {len(spec_data_list)/total_time:.2f} 항목/초")
        
        # 캐시 통계 출력
        cache_stats = self.evaluator.get_cache_stats()
        print(f"캐시 통계: {json.dumps(cache_stats, indent=2)}")
        
        return results
    
    def process_batch_parallel(self, spec_data_list):
        """
        병렬로 배치 처리 (주의: 이 함수는 KV 캐시가 여러 프로세스에서 공유되지 않을 수 있음)
        
        Args:
            spec_data_list: 스펙 데이터 리스트
            
        Returns:
            results: 평가 결과 리스트
        """
        results = []
        
        print(f"총 {len(spec_data_list)}개 항목을 {self.max_workers}개 워커로 병렬 처리합니다...")
        start_time = time.time()
        
        # ThreadPoolExecutor 사용 (GPU 공유를 위해)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_spec = {executor.submit(self.evaluator.predict, spec_data): spec_data for spec_data in spec_data_list}
            
            # 결과 수집
            for future in tqdm(concurrent.futures.as_completed(future_to_spec), total=len(spec_data_list)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"처리 중 오류 발생: {e}")
        
        total_time = time.time() - start_time
        print(f"총 처리 시간: {total_time:.2f}초")
        print(f"평균 처리 속도: {len(spec_data_list)/total_time:.2f} 항목/초")
        
        # 캐시 통계 출력
        cache_stats = self.evaluator.get_cache_stats()
        print(f"캐시 통계: {json.dumps(cache_stats, indent=2)}")
        
        return results
    
    def process_files(self, file_paths, output_path=None):
        """
        여러 파일을 일괄 처리
        
        Args:
            file_paths: 처리할 파일 경로 리스트
            output_path: 결과 저장 경로 (선택 사항)
            
        Returns:
            results: 평가 결과 리스트
        """
        all_spec_data = []
        
        # 파일에서 스펙 데이터 로드
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    spec_data = json.load(f)
                    
                    # 단일 객체 또는 배열 처리
                    if isinstance(spec_data, list):
                        all_spec_data.extend(spec_data)
                    else:
                        all_spec_data.append(spec_data)
            except Exception as e:
                print(f"파일 {path} 로드 중 오류 발생: {e}")
        
        # 배치 처리
        print(f"{len(all_spec_data)}개 항목을 배치 처리합니다...")
        results = self.process_batch_sequential(all_spec_data)
        
        # 결과 저장 (선택 사항)
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"결과가 {output_path}에 저장되었습니다.")
            except Exception as e:
                print(f"결과 저장 중 오류 발생: {e}")
        
        return results