# Agentic-Reactor-Controller

## 프로젝트 개요
- **목적**: ARMI를 오케스트레이터로, OpenMC를 계산 엔진으로 사용하는 원자로 시뮬레이션 시스템
- **아키텍처**: 파일/프로세스 기반 느슨한 결합 (loose coupling)
- **기술 스택**: Python 3.11+, ARMI, OpenMC, Pydantic
- **실행 환경**: macOS (OpenMP), Docker (단일 or 2컨테이너)

## 문서
- [아키텍처](docs/architecture.md) — 시스템 구조, 디렉토리 구조, 주요 모듈
- [컨벤션](docs/conventions.md) — 프로젝트 전용 코딩 컨벤션
- [API 설계](docs/api-design.md) — 핵심 인터페이스 설계
- [테스트 전략](docs/testing.md) — 테스트 전략, 커버리지 기준

## 핵심 모듈
- `src/armi_layer/` — 실험 설계, 케이스 생성, 결과 수집, **Pydantic 모델 정의**
- `src/openmc_layer/` — OpenMC 입력 생성, 결과 파싱, KPI 계산
- `src/run_manager/` — 실행 래퍼 (환경변수, 로그, 재시도, 타임아웃)
- `src/controller/` — 제어/의사결정 (BaseController ABC → 에이전트로 교체 가능)

## Pydantic 데이터 모델 (`src/armi_layer/models.py`)
- `CaseConfig` — 케이스 설정 (geometry params, material params, settings)
- `RunConfig` — 실행 설정 (omp_threads, cross_sections_path, timeout, max_retries)
- `RunStatus` — 실행 상태 (queued/running/done/failed + timestamps)
- `SimulationResult` — 결과 (keff, keff_std, tallies, runtime)
- `ReactorState` — 원자로 상태 (현재 설계, 운전 이력, KPI)

## 핵심 인터페이스 (`src/controller/base.py`)
```python
class BaseController(ABC):
    def propose_actions(self, state: ReactorState) -> list[Action]
    def apply_actions_to_case(self, actions: list[Action]) -> CaseConfig
    def evaluate_results(self, result: SimulationResult) -> dict
    def update_state(self, metrics: dict) -> ReactorState
```

## 데이터 흐름
```
ReactorState → propose_actions() → Actions
    Actions → apply_actions_to_case() → CaseConfig
    CaseConfig → generate_input() → OpenMC XML (input/)
    OpenMC XML → run_openmc() → statepoint.h5 (output/)
    statepoint → parse_results() → SimulationResult
    SimulationResult → evaluate_results() → Metrics
    Metrics → update_state() → ReactorState (루프)
```

## 프로젝트 구조
```
Agentic-Reactor-Controller/
├── src/
│   ├── armi_layer/
│   │   ├── models.py           # Pydantic 데이터 모델 (전 모듈 공유)
│   │   ├── case_manager.py     # 케이스 생성/조회/관리
│   │   ├── case_folder.py      # runs/case_XXXX 디렉토리 생성
│   │   ├── sweep.py            # 파라미터 스윕 엔진
│   │   └── collector.py        # 결과 수집기
│   ├── openmc_layer/
│   │   ├── input_generator.py  # CaseConfig → OpenMC XML
│   │   ├── result_parser.py    # statepoint.h5 → SimulationResult
│   │   └── kpi.py              # KPI 계산기
│   ├── run_manager/
│   │   ├── runner.py           # OpenMC 프로세스 실행
│   │   ├── status.py           # 상태 관리 (status.json)
│   │   └── omp.py              # OMP_NUM_THREADS 자동 감지
│   └── controller/
│       ├── base.py             # BaseController ABC
│       └── simple.py           # SimpleController (룰 기반)
├── tests/
├── runs/                       # 시뮬레이션 케이스 (git 미추적)
│   └── case_XXXX/{input,output,meta}/
├── nucdata/                    # 핵데이터 (호스트 고정, git 미추적)
├── docs/
├── pyproject.toml
└── README.md
```

## Docker 배치
- **기본**: 단일 컨테이너 (ARMI + OpenMC)
- **옵션**: 2컨테이너 분리 (공유볼륨: /work/runs, /data/nucdata)
- docker-compose.yml로 관리

## 프로젝트 고유 규칙
- 케이스 폴더: `runs/case_XXXX/{input,output,meta}/`
- 케이스 메타: `meta/config.json` (CaseConfig), `meta/status.json` (RunStatus)
- 핵데이터: `nucdata/` (호스트 고정, git 미추적)
- OpenMP 스레드: `OMP_NUM_THREADS` = P-core 수 기준 (자동 감지)
- 모듈 간 결합은 파일/프로세스 기반으로 유지
- 모든 데이터 교환은 Pydantic 모델을 통해 타입 안전하게 수행
