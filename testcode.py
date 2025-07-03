from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import asyncio
import traceback
import time

# ====== 테스트용 모델들 직접 정의 (순환 참조 방지) ======
class TestUniversity(BaseModel):
    school_name: str
    degree: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[float] = None
    gpa_max: Optional[float] = None

class TestCareer(BaseModel):
    company: str
    role: Optional[str] = None
    work_month: Optional[int] = None

class TestLanguage(BaseModel):
    test: str
    score_or_grade: str

class TestActivity(BaseModel):
    name: str
    role: Optional[str] = None
    award: Optional[str] = None

class TestResumeData(BaseModel):
    nickname: str
    final_edu: str
    final_status: str
    desired_job: str
    universities: Optional[List[TestUniversity]] = []
    careers: Optional[List[TestCareer]] = []
    certificates: Optional[List[str]] = []
    languages: Optional[List[TestLanguage]] = []
    activities: Optional[List[TestActivity]] = []


@dataclass
class TestResult:
    """테스트 결과를 담는 데이터 클래스"""
    test_name: str
    description: str
    status: str  # PASS, FAIL, SKIP
    execution_time: float
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None


class SpecV2Tester:
    """스펙 V2 엔드포인트 테스트 전담 클래스"""
    
    def __init__(self, evaluation_system):
        self.evaluation_system = evaluation_system
        self.test_cases = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행 및 결과 반환"""
        start_time = time.time()
        
        test_summary = {
            "test_name": "Spec V2 Endpoint Test Suite",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time": 0.0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_cases": [],
            "overall_status": "UNKNOWN",
            "system_info": {}
        }
        
        try:
            print("🔧 테스트 시스템 시작...")
            
            # 시스템 상태 먼저 확인
            system_status = await self._check_system_status()
            test_summary["system_info"] = system_status
            
            # 개별 테스트 케이스들 실행
            test_cases = [
                await self._test_normal_case(),
                await self._test_minimal_case(), 
                await self._test_full_data_case(),
                await self._test_edge_cases(),
                await self._test_error_handling(),
                await self._test_performance()
            ]
            
            # 결과 집계
            test_summary["test_cases"] = [case.__dict__ for case in test_cases]
            test_summary["total_tests"] = len(test_cases)
            test_summary["passed_tests"] = sum(1 for case in test_cases if case.status == "PASS")
            test_summary["failed_tests"] = sum(1 for case in test_cases if case.status == "FAIL")
            test_summary["skipped_tests"] = sum(1 for case in test_cases if case.status == "SKIP")
            
            test_summary["overall_status"] = self._determine_overall_status(test_summary)
            
        except Exception as e:
            test_summary["overall_status"] = "SYSTEM_ERROR"
            test_summary["error_message"] = f"테스트 실행 중 시스템 오류: {str(e)}"
            
        finally:
            test_summary["total_execution_time"] = round(time.time() - start_time, 3)
            
        return test_summary
    
    async def _check_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        try:
            status = self.evaluation_system.get_system_status()
            return {
                "model_loaded": status.get("model_loaded", False),
                "rag_enabled": status.get("rag_enabled", False),
                "system_healthy": True
            }
        except Exception as e:
            return {
                "model_loaded": False,
                "rag_enabled": False, 
                "system_healthy": False,
                "error": str(e)
            }
    
    async def _test_normal_case(self) -> TestResult:
        """표준적인 이력서 데이터로 정상 동작 테스트"""
        test_case = TestResult(
            test_name="Normal Case Test",
            description="일반적인 대학생/신입사원 이력서로 정상 동작 확인",
            status="UNKNOWN",
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # 표준 테스트 데이터 (TestResumeData 사용)
            test_data = TestResumeData(
                nickname="김개발",
                final_edu="대학교",
                final_status="졸업",
                desired_job="인터넷_IT",
                universities=[
                    TestUniversity(
                        school_name="서울대학교",
                        degree="학사",
                        major="컴퓨터공학과",
                        gpa=3.5,
                        gpa_max=4.5
                    )
                ],
                careers=[
                    TestCareer(
                        company="네이버",
                        role="인턴",
                        work_month=3
                    )
                ],
                certificates=["정보처리기사", "SQLD"],
                languages=[
                    TestLanguage(test="TOEIC_ENGLISH", score_or_grade="820")
                ],
                activities=[
                    TestActivity(
                        name="프로그래밍 동아리",
                        role="회장",
                        award="우수상"
                    )
                ]
            )
            
            # dict()로 변환하여 평가 시스템에 전달
            result = self.evaluation_system.evaluate_resume(test_data.dict())
            
            # 결과 검증
            validation = self._validate_v2_response(result)
            
            if validation["valid"]:
                test_case.status = "PASS"
                test_case.response_data = {
                    "total_score": result.get("totalScore"),
                    "assessment_length": len(result.get("assessment", "")),
                    "all_scores_present": all(
                        field in result for field in [
                            "academicScore", "workExperienceScore", 
                            "certificationScore", "languageProficiencyScore", 
                            "extracurricularScore"
                        ]
                    )
                }
            else:
                test_case.status = "FAIL"
                test_case.error_message = validation["error"]
                
        except Exception as e:
            test_case.status = "FAIL"
            test_case.error_message = f"실행 오류: {str(e)}"
            
        test_case.execution_time = round(time.time() - start_time, 3)
        return test_case
    
    async def _test_minimal_case(self) -> TestResult:
        """최소한의 필수 데이터만으로 테스트"""
        test_case = TestResult(
            test_name="Minimal Data Test",
            description="필수 필드만 포함된 최소 데이터로 동작 확인",
            status="UNKNOWN",
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # 최소 데이터
            test_data = TestResumeData(
                nickname="최소유저",
                final_edu="고등학교",
                final_status="졸업",
                desired_job="경영_사무"
            )
            
            result = self.evaluation_system.evaluate_resume(test_data.dict())
            validation = self._validate_v2_response(result)
            
            if validation["valid"]:
                test_case.status = "PASS"
                test_case.response_data = {
                    "total_score": result.get("totalScore"),
                    "handles_empty_data": True
                }
            else:
                test_case.status = "FAIL"
                test_case.error_message = validation["error"]
                
        except Exception as e:
            test_case.status = "FAIL"
            test_case.error_message = f"최소 데이터 처리 오류: {str(e)}"
            
        test_case.execution_time = round(time.time() - start_time, 3)
        return test_case
    
    async def _test_full_data_case(self) -> TestResult:
        """모든 필드가 채워진 완전한 데이터로 테스트"""
        test_case = TestResult(
            test_name="Full Data Test", 
            description="모든 선택적 필드까지 포함된 완전한 이력서로 테스트",
            status="UNKNOWN",
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # 완전한 데이터
            test_data = TestResumeData(
                nickname="풀스펙유저",
                final_edu="대학원",
                final_status="졸업",
                desired_job="인터넷_IT",
                universities=[
                    TestUniversity(
                        school_name="카이스트",
                        degree="석사", 
                        major="인공지능학과",
                        gpa=4.3,
                        gpa_max=4.5
                    ),
                    TestUniversity(
                        school_name="서울대학교",
                        degree="학사",
                        major="컴퓨터공학과", 
                        gpa=3.9,
                        gpa_max=4.5
                    )
                ],
                careers=[
                    TestCareer(company="삼성전자", role="정규직", work_month=36),
                    TestCareer(company="구글코리아", role="인턴", work_month=6)
                ],
                certificates=[
                    "정보처리기사", "SQLD", "AWS Solution Architect"
                ],
                languages=[
                    TestLanguage(test="TOEIC_ENGLISH", score_or_grade="990"),
                    TestLanguage(test="OPIC_ENGLISH", score_or_grade="IH")
                ],
                activities=[
                    TestActivity(name="AI 경진대회", role="팀장", award="1등"),
                    TestActivity(name="오픈소스 기여", role="메인테이너", award=""),
                    TestActivity(name="대학 동아리", role="회장", award="우수상")
                ]
            )
            
            result = self.evaluation_system.evaluate_resume(test_data.dict())
            validation = self._validate_v2_response(result)
            
            if validation["valid"]:
                test_case.status = "PASS"
                test_case.response_data = {
                    "total_score": result.get("totalScore"),
                    "handles_complex_data": True,
                    "score_distribution": {
                        "academic": result.get("academicScore"),
                        "experience": result.get("workExperienceScore"),
                        "certification": result.get("certificationScore"),
                        "language": result.get("languageProficiencyScore"),
                        "activity": result.get("extracurricularScore")
                    }
                }
            else:
                test_case.status = "FAIL"
                test_case.error_message = validation["error"]
                
        except Exception as e:
            test_case.status = "FAIL"
            test_case.error_message = f"복잡한 데이터 처리 오류: {str(e)}"
            
        test_case.execution_time = round(time.time() - start_time, 3)
        return test_case
    
    async def _test_edge_cases(self) -> TestResult:
        """경계값 및 특수 케이스 테스트"""
        test_case = TestResult(
            test_name="Edge Cases Test",
            description="경계값 및 특수한 입력 데이터에 대한 처리 확인",
            status="UNKNOWN",
            execution_time=0.0
        )
        
        start_time = time.time()
        edge_tests_passed = 0
        total_edge_tests = 0
        
        try:
            # 1. 매우 높은 학점 테스트
            total_edge_tests += 1
            try:
                test_data = TestResumeData(
                    nickname="고학점유저",
                    final_edu="대학교",
                    final_status="졸업",
                    desired_job="인터넷_IT",
                    universities=[
                        TestUniversity(
                            school_name="서울대학교",
                            degree="학사",
                            major="컴퓨터사이언스",
                            gpa=4.5,
                            gpa_max=4.5
                        )
                    ]
                )
                result = self.evaluation_system.evaluate_resume(test_data.dict())
                if self._validate_v2_response(result)["valid"]:
                    edge_tests_passed += 1
            except Exception:
                pass
            
            # 2. 긴 경력 기간 테스트
            total_edge_tests += 1
            try:
                test_data = TestResumeData(
                    nickname="장기경력유저",
                    final_edu="대학교",
                    final_status="졸업",
                    desired_job="인터넷_IT",
                    careers=[
                        TestCareer(company="IBM", role="정규직", work_month=120)
                    ]
                )
                result = self.evaluation_system.evaluate_resume(test_data.dict())
                if self._validate_v2_response(result)["valid"]:
                    edge_tests_passed += 1
            except Exception:
                pass
            
            # 3. 매우 많은 자격증 테스트
            total_edge_tests += 1
            try:
                many_certs = [f"자격증{i}" for i in range(1, 21)]
                test_data = TestResumeData(
                    nickname="자격증왕",
                    final_edu="대학교",
                    final_status="졸업",
                    desired_job="인터넷_IT",
                    certificates=many_certs
                )
                result = self.evaluation_system.evaluate_resume(test_data.dict())
                if self._validate_v2_response(result)["valid"]:
                    edge_tests_passed += 1
            except Exception:
                pass
            
            # 결과 판정
            pass_rate = edge_tests_passed / total_edge_tests if total_edge_tests > 0 else 0
            test_case.status = "PASS" if pass_rate >= 0.67 else "FAIL"
            test_case.response_data = {
                "passed_edge_tests": edge_tests_passed,
                "total_edge_tests": total_edge_tests,
                "pass_rate": f"{pass_rate:.1%}"
            }
            
        except Exception as e:
            test_case.status = "FAIL"
            test_case.error_message = f"경계값 테스트 오류: {str(e)}"
            
        test_case.execution_time = round(time.time() - start_time, 3)
        return test_case
    
    async def _test_error_handling(self) -> TestResult:
        """에러 처리 능력 테스트"""
        test_case = TestResult(
            test_name="Error Handling Test",
            description="잘못된 입력에 대한 에러 처리 및 복구 능력 확인",
            status="UNKNOWN",
            execution_time=0.0
        )
        
        start_time = time.time()
        error_tests_passed = 0
        total_error_tests = 0
        
        try:            
            # 잘못된 직종 테스트
            total_error_tests += 1
            try:
                test_data = TestResumeData(
                    nickname="에러테스트",
                    final_edu="대학교", 
                    final_status="졸업",
                    desired_job="존재하지않는직종"
                )
                result = self.evaluation_system.evaluate_resume(test_data.dict())
                if result.get("totalScore") is not None:
                    error_tests_passed += 1
            except Exception:
                pass
            
            # 결과 판정
            pass_rate = error_tests_passed / total_error_tests if total_error_tests > 0 else 0
            test_case.status = "PASS" if pass_rate >= 0.5 else "FAIL"
            test_case.response_data = {
                "passed_error_tests": error_tests_passed,
                "total_error_tests": total_error_tests,
                "pass_rate": f"{pass_rate:.1%}"
            }
            
        except Exception as e:
            test_case.status = "FAIL"
            test_case.error_message = f"에러 처리 테스트 오류: {str(e)}"
            
        test_case.execution_time = round(time.time() - start_time, 3)
        return test_case
    
    async def _test_performance(self) -> TestResult:
        """성능 테스트"""
        test_case = TestResult(
            test_name="Performance Test",
            description="응답 시간 및 성능 벤치마크 테스트",
            status="UNKNOWN",
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # 표준 데이터로 여러 번 실행하여 평균 응답시간 측정
            test_data = TestResumeData(
                nickname="성능테스트",
                final_edu="대학교",
                final_status="졸업",
                desired_job="인터넷_IT",
                universities=[
                    TestUniversity(
                        school_name="연세대학교",
                        degree="학사",
                        major="경영학과",
                        gpa=3.2,
                        gpa_max=4.5
                    )
                ]
            )
            
            execution_times = []
            successful_runs = 0
            
            # 3회 실행 (시간 단축)
            for i in range(3):
                try:
                    run_start = time.time()
                    result = self.evaluation_system.evaluate_resume(test_data.dict())
                    run_time = time.time() - run_start
                    
                    if self._validate_v2_response(result)["valid"]:
                        execution_times.append(run_time)
                        successful_runs += 1
                        
                except Exception:
                    continue
            
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                
                # 성능 기준: 평균 10초 이내, 최대 15초 이내 (좀 더 관대하게)
                performance_ok = avg_time <= 10.0 and max_time <= 15.0
                
                test_case.status = "PASS" if performance_ok and successful_runs >= 2 else "FAIL"
                test_case.response_data = {
                    "successful_runs": successful_runs,
                    "total_runs": 3,
                    "avg_execution_time": round(avg_time, 3),
                    "min_execution_time": round(min_time, 3),
                    "max_execution_time": round(max_time, 3),
                    "performance_ok": performance_ok
                }
            else:
                test_case.status = "FAIL"
                test_case.error_message = "모든 성능 테스트 실행 실패"
                
        except Exception as e:
            test_case.status = "FAIL"
            test_case.error_message = f"성능 테스트 오류: {str(e)}"
            
        test_case.execution_time = round(time.time() - start_time, 3)
        return test_case
    
    def _validate_v2_response(self, response: Dict) -> Dict[str, Any]:
        """V2 응답 구조 및 내용 검증"""
        required_fields = [
            "nickname", "totalScore", "academicScore",
            "workExperienceScore", "certificationScore", 
            "languageProficiencyScore", "extracurricularScore", "assessment"
        ]
        
        validation_result = {"valid": True, "error": None}
        
        try:
            # 응답이 딕셔너리인지 확인
            if not isinstance(response, dict):
                return {"valid": False, "error": "응답이 딕셔너리 형태가 아님"}
            
            # 필수 필드 존재 확인
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                return {"valid": False, "error": f"필수 필드 누락: {missing_fields}"}
            
            # assessment 필드 확인
            assessment = response["assessment"]
            if not isinstance(assessment, str):
                return {"valid": False, "error": "assessment는 문자열이어야 함"}
            if len(assessment.strip()) == 0:
                return {"valid": False, "error": "assessment가 비어있음"}
                
        except Exception as e:
            return {"valid": False, "error": f"검증 중 예외 발생: {str(e)}"}
        
        return validation_result
    
    def _determine_overall_status(self, test_summary: Dict) -> str:
        """전체 테스트 상태 결정"""
        total = test_summary["total_tests"]
        passed = test_summary["passed_tests"]
        failed = test_summary["failed_tests"]
        
        if total == 0:
            return "NO_TESTS"
        elif failed == 0:
            return "ALL_PASS"
        elif passed == 0:
            return "ALL_FAIL"
        elif passed / total >= 0.8:
            return "MOSTLY_PASS"
        elif passed / total >= 0.5:
            return "PARTIAL_PASS"
        else:
            return "MOSTLY_FAIL"


# 편의 함수
async def run_spec_v2_tests(evaluation_system) -> Dict[str, Any]:
    """스펙 V2 테스트 실행 편의 함수"""
    tester = SpecV2Tester(evaluation_system)
    return await tester.run_all_tests()