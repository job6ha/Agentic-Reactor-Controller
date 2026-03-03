# 아키텍처

## 시스템 개요

ARMI를 오케스트레이터로, OpenMC를 몬테카를로 중성자 수송 계산 엔진으로 사용하는
원자로 시뮬레이션 제어 시스템. 파일/프로세스 기반 느슨한 결합(loose coupling)
아키텍처로, 각 레이어가 독립적으로 교체 가능하다.

### 레이어 구조

```
┌──────────────────────────────────────────────────────┐
│                    Controller                         │
│  의사결정 엔진 (룰 기반 → AI 에이전트 교체 가능)      │
│  propose_actions() / evaluate_results()               │
└────────────────────────┬─────────────────────────────┘
                         │ Action[]
┌────────────────────────▼─────────────────────────────┐
│                    ARMI Layer                          │
│  실험 설계 / 케이스 생성 / 파라미터 스윕 / 결과 수집   │
│  CaseManager / SweepEngine / ResultCollector          │
└──────────┬─────────────────────────────┬─────────────┘
           │ CaseConfig                  │ case_path
┌──────────▼──────────┐    ┌─────────────▼─────────────┐
│    OpenMC Layer      │    │       Run Manager          │
│  XML 입력 생성       │    │  프로세스 실행 / 상태 관리  │
│  statepoint 파싱     │    │  타임아웃 / 재시도 / 로그   │
│  KPI 계산            │    │  CPU 감지 / 핵데이터 경로   │
└──────────────────────┘    └───────────────────────────┘
```

## 프로젝트 구조

```
Agentic-Reactor-Controller/
├── src/
│   ├── armi_layer/                 # ARMI 오케스트레이션
│   │   ├── models.py               #   Pydantic 데이터 모델 (전 모듈 공유)
│   │   ├── case_folder.py          #   runs/case_XXXX 디렉토리 생성/로드
│   │   ├── case_manager.py         #   케이스 생성/조회/상태 관리
│   │   ├── sweep.py                #   파라미터 그리드 스윕 엔진
│   │   └── result_collector.py     #   다수 케이스 결과 수집 → DataFrame
│   ├── openmc_layer/               # OpenMC 입출력
│   │   ├── input_generator.py      #   CaseConfig → OpenMC XML
│   │   ├── result_parser.py        #   statepoint.h5 → SimulationResult
│   │   └── kpi_calculator.py       #   KPI 계산 (keff, peaking factor)
│   ├── run_manager/                # 실행 관리
│   │   ├── runner.py               #   OpenMC 서브프로세스 실행
│   │   ├── status.py               #   상태 전이 + 재시도 오케스트레이션
│   │   ├── cpu_detect.py           #   OMP_NUM_THREADS 자동 감지
│   │   └── nucdata.py              #   핵데이터 경로 해석/검증
│   └── controller/                 # 제어 로직
│       ├── base.py                 #   BaseController ABC
│       └── simple.py               #   SimpleController (룰 기반 스윕)
├── scripts/
│   └── download_xs.py              # 핵데이터 다운로드 스크립트
├── tests/                          # pytest 테스트 (318개)
├── runs/                           # 시뮬레이션 케이스 (git 미추적)
│   └── case_XXXX/
│       ├── input/                  #   geometry.xml, materials.xml, settings.xml
│       ├── output/                 #   statepoint.h5, run.log
│       └── meta/                   #   config.json, status.json, kpi.json
├── nucdata/                        # 핵데이터 (git 미추적)
├── docs/                           # 프로젝트 문서
└── pyproject.toml
```

## 모듈 의존 관계

```
models.py ─────────────── 의존성 없음 (모든 모듈의 기반)
    │
    ├──→ case_folder.py
    │        └──→ case_manager.py ←── result_parser.py
    │                  │               kpi_calculator.py
    │                  └──→ result_collector.py
    │
    ├──→ sweep.py
    │
    ├──→ input_generator.py
    │
    ├──→ result_parser.py ←── h5py
    │
    ├──→ kpi_calculator.py
    │
    ├──→ runner.py ←── cpu_detect.py
    │        └──→ status.py
    │
    ├──→ base.py (ABC)
    │        └──→ simple.py
    │
    └── nucdata.py ─── 독립 유틸리티
        cpu_detect.py ── 독립 유틸리티
```

**핵심 원칙**: `models.py`가 유일한 공유 계약. 모듈 간 직접 의존은 최소화하고,
Pydantic 모델을 통한 데이터 교환으로 타입 안전성을 보장한다.

## 데이터 모델 관계

```
ReactorState
├── current_config: CaseConfig
│   ├── geometry: GeometryParams
│   │   └── fuel_radius, clad_inner_radius, clad_outer_radius, pitch, fuel_height
│   ├── materials: MaterialParams
│   │   └── fuel_enrichment, fuel_density, coolant_temperature, coolant_density
│   └── settings: SimulationSettings
│       └── batches, inactive, particles, source_type
├── history: list[SimulationResult]
│   ├── keff, keff_std, runtime, batches_completed
│   ├── statepoint_path: Path | None
│   └── tallies: list[TallyResult]
│       └── name, scores, mean, std_dev
├── kpi: dict[str, float]
└── iteration: int

RunConfig ──── 실행 환경 설정 (omp_threads, cross_sections_path, timeout, max_retries)
RunStatus ──── 실행 상태 추적 (queued → running → done/failed, timestamps, attempt)
Action ──────── 제어 명령 (MODIFY_PARAM | SET_CONFIG | STOP)
```

## 데이터 흐름

시스템은 피드백 루프로 동작한다. Controller가 상태를 분석하여 다음 행동을 제안하고,
시뮬레이션 결과를 평가하여 상태를 갱신한다.

```
1. ReactorState
   │
   ▼ propose_actions(state)
2. Action[]  ─── MODIFY_PARAM("materials.fuel_enrichment", 4.5)
   │              STOP("수렴 달성")
   ▼ apply_actions_to_case(actions, state)
3. CaseConfig ── 새 파라미터가 반영된 케이스 설정
   │
   ▼ case_folder.create_case(config)
4. runs/case_XXXX/  ── 케이스 디렉토리 생성
   │                    meta/config.json, meta/status.json
   ▼ input_generator.generate_input(config, case_path)
5. input/*.xml  ── geometry.xml, materials.xml, settings.xml
   │
   ▼ status.run_case(run_config, case_dir)
6. openmc 서브프로세스 실행  ── OMP_NUM_THREADS, OPENMC_CROSS_SECTIONS 설정
   │                            타임아웃/재시도 적용, 로그 캡처
   ▼
7. output/statepoint.*.h5  ── 시뮬레이션 결과
   │
   ▼ result_parser.parse_results(case_path)
8. SimulationResult  ── keff, tallies, runtime
   │
   ▼ kpi_calculator.calculate_kpi(result)
9. KPI dict  ── keff, criticality, peaking_factor
   │             meta/kpi.json에 저장
   ▼ evaluate_results(result) → update_state(state, metrics)
10. ReactorState (갱신)  ── 다시 1번으로 (루프)
```

### 종료 조건

- keff가 목표값에 수렴 (|keff - target| < tolerance)
- 스윕 파라미터 소진
- 최대 반복 횟수 도달

## 케이스 디렉토리 규약

```
runs/
└── case_0001/
    ├── input/
    │   ├── geometry.xml        # OpenMC 기하 구조
    │   ├── materials.xml       # OpenMC 재료 정의
    │   └── settings.xml        # OpenMC 시뮬레이션 설정
    ├── output/
    │   ├── statepoint.110.h5   # OpenMC 결과 (배치 번호)
    │   └── run.log             # 실행 로그 (stdout + stderr)
    └── meta/
        ├── config.json         # CaseConfig 직렬화
        ├── status.json         # RunStatus (상태 전이 이력)
        └── kpi.json            # KPI 계산 결과
```

- 케이스 번호: 4자리 0-패딩 (`case_0001`, `case_0002`, ...)
- 기존 최대 번호 + 1로 자동 채번
- `runs/` 디렉토리는 git 미추적

## 설계 패턴

| 패턴 | 적용 위치 | 설명 |
|------|-----------|------|
| **Template Method** | `BaseController` ABC | 4개 추상 메서드로 제어 로직 교체 가능 |
| **Adapter** | `input_generator` | CaseConfig → OpenMC XML 변환 |
| **Pipeline** | 데이터 흐름 전체 | config → input → run → parse → KPI 순차 처리 |
| **Factory** | `CaseManager.create()` | 케이스 디렉토리 자동 생성 |
| **DTO** | Pydantic 모델 전체 | 모듈 간 타입 안전한 데이터 교환 |

## Controller 인터페이스

```python
class BaseController(ABC):
    """제어기 추상 인터페이스. 구현체를 교체하여 다양한 전략 적용."""

    @abstractmethod
    def propose_actions(self, state: ReactorState) -> list[Action]:
        """현재 상태에서 다음 행동 제안."""

    @abstractmethod
    def apply_actions_to_case(
        self, actions: list[Action], state: ReactorState
    ) -> CaseConfig:
        """행동을 적용하여 새 케이스 설정 생성."""

    @abstractmethod
    def evaluate_results(self, result: SimulationResult) -> dict[str, float]:
        """시뮬레이션 결과 평가."""

    @abstractmethod
    def update_state(
        self, state: ReactorState, metrics: dict[str, float]
    ) -> ReactorState:
        """평가 결과로 상태 갱신."""
```

**현재 구현체**:
- `SimpleController`: 파라미터 순차 스윕 + keff 수렴 판정

**확장 계획**:
- AI 에이전트 Controller (LLM 기반 파라미터 최적화)
- 베이지안 최적화 Controller

## 실행 환경

### 로컬 (macOS)

- OpenMP 병렬화: `OMP_NUM_THREADS` = Apple Silicon P-core 수 자동 감지
- 핵데이터: `nucdata/` 디렉토리 또는 `OPENMC_CROSS_SECTIONS` 환경변수

### Docker

- **기본**: 단일 컨테이너 (`openmc/openmc:latest` 기반)
- **옵션**: 2컨테이너 분리 (공유볼륨: `/work/runs`, `/data/nucdata`)
- `docker-compose.yml`로 관리
- amd64 이미지 → Apple Silicon에서 Rosetta 에뮬레이션

## 핵데이터 관리

3단계 우선순위로 핵데이터 경로를 해석한다:

1. **명시적 경로** (`RunConfig.cross_sections_path`)
2. **환경변수** (`OPENMC_CROSS_SECTIONS`)
3. **자동 탐색** (`{프로젝트루트}/nucdata/` 하위 재귀 검색)

지원 라이브러리:
- ENDF/B-VIII.0 (기본, 권장)
- ENDF/B-VII.1

## 외부 의존성

| 패키지 | 용도 |
|--------|------|
| **pydantic** | 데이터 모델 정의, 유효성 검증, JSON 직렬화 |
| **h5py** | HDF5 statepoint 파일 파싱 |
| **pandas** | 다수 케이스 결과 집계/분석 |
| **OpenMC** | 몬테카를로 중성자 수송 계산 (서브프로세스) |
