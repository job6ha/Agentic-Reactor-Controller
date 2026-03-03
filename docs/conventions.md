# 프로젝트 컨벤션

> 글로벌 CLAUDE.md의 컨벤션을 기본으로 따르며, 이 문서는 프로젝트 고유 규칙을 보완한다.

## 네이밍 규칙

- **케이스 ID**: `case_XXXX` (4자리 0-패딩, 예: `case_0001`)
- **필드 경로**: 도트 표기법 (`materials.fuel_enrichment`, `geometry.pitch`)
- **모듈 파일**: 기능 단위 (`input_generator.py`, `result_parser.py`)
- **테스트 파일**: `test_{모듈명}.py` (모듈과 1:1 대응)
- **XML 파일**: OpenMC 규약 (`geometry.xml`, `materials.xml`, `settings.xml`)
- **JSON 메타**: `config.json`, `status.json`, `kpi.json`

## 모듈 구성 규칙

- **데이터 모델**: `src/armi_layer/models.py`에 집중. 모든 모듈이 공유.
- **레이어 독립성**: 각 레이어(`armi_layer`, `openmc_layer`, `run_manager`, `controller`)는 최소한의 교차 의존만 허용.
- **모듈 간 결합**: Pydantic 모델을 통한 데이터 교환만 허용. 직접 함수 호출보다 데이터 전달 선호.
- **독립 유틸리티**: `cpu_detect.py`, `nucdata.py`는 다른 모듈에 의존하지 않는다.

## 설정 관리

- 환경변수: `.env` 파일로 관리, `.env.example`에 키 목록 유지
- 핵데이터 경로: 3단계 우선순위 (명시 > 환경변수 > 자동 탐색)
- OpenMP 스레드: `OMP_NUM_THREADS` 환경변수, 미설정 시 P-core 수 자동 감지

## 로깅

- `logging` 표준 라이브러리 사용
- 모듈별 `logger = logging.getLogger(__name__)`
- 레벨 가이드:
  - `DEBUG`: 내부 상태 추적 (파라미터 값, 파일 경로)
  - `INFO`: 주요 동작 (케이스 생성, 시뮬레이션 시작/완료, 파일 저장)
  - `WARNING`: 무시 가능한 문제 (SHA-256 미등록, 기본값 적용)
  - `ERROR`: 복구 불가능한 실패 (핵데이터 미발견, 실행 타임아웃)

## 에러 핸들링

- **커스텀 예외**: 도메인별 정의 (`NucdataNotFoundError`, `StatepointNotFoundError`, `StatepointParseError`)
- **예외 체이닝**: `raise ... from e`로 원인 보존
- **실패 격리**: 서브프로세스 실패는 `RunResult`로 캡슐화, 상태는 `status.json`에 기록
- **재시도**: `RunConfig.max_retries`로 설정. 각 시도는 독립 로그 (`run_1.log`, `run_2.log`)
- **타임아웃**: `RunConfig.timeout` (초). 서브프로세스에 적용.

## 직렬화 규약

- 모든 Pydantic 모델: `model_dump_json()` / `model_validate_json()` 사용
- JSON 파일: `indent=2`, `ensure_ascii=False`, 줄바꿈 마무리 (`+ "\n"`)
- 인코딩: `utf-8` 고정
