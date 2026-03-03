# API 설계

> 이 프로젝트는 REST API가 아닌 Python 모듈 인터페이스로 구성된다.
> 여기서 "API"는 모듈 간 공개 인터페이스를 의미한다.

## 설계 원칙

- **Pydantic-first**: 모든 입출력은 Pydantic 모델로 타입 정의
- **명시적 에러**: 커스텀 예외로 실패 원인 전달 (절대 삼키지 않음)
- **불변 결과**: `SimulationResult`, `TallyResult`은 `frozen=True`
- **선택적 필드**: `None`으로 표현, 기본값은 도메인 표준 (PWR pin cell)

## Controller 인터페이스

시스템의 핵심 확장점. `BaseController` ABC를 구현하여 제어 전략을 교체한다.

```python
# src/controller/base.py
class BaseController(ABC):
    def propose_actions(self, state: ReactorState) -> list[Action]
    def apply_actions_to_case(self, actions: list[Action], state: ReactorState) -> CaseConfig
    def evaluate_results(self, result: SimulationResult) -> dict[str, float]
    def update_state(self, state: ReactorState, metrics: dict[str, float]) -> ReactorState
```

### Action 타입

| ActionType | 용도 | field_path | value |
|------------|------|------------|-------|
| `MODIFY_PARAM` | 단일 파라미터 변경 | `"materials.fuel_enrichment"` | `4.5` |
| `SET_CONFIG` | 전체 설정 교체 | 미사용 | 미사용 |
| `STOP` | 루프 종료 | 미사용 | 미사용 |

## ARMI Layer 인터페이스

### CaseManager

```python
# src/armi_layer/case_manager.py
class CaseManager:
    def __init__(self, runs_dir: Path)
    def create(self, config: CaseConfig) -> str           # case_id 반환
    def list(self, *, status_filter: StatusType | None = None) -> list[str]
    def get(self, case_id: str) -> CaseInfo               # 전체 정보
    def get_by_status(self, status: StatusType) -> list[CaseInfo]
    def count(self) -> dict[str, int]                     # 상태별 카운트
```

### SweepEngine

```python
# src/armi_layer/sweep.py
def generate_sweep_configs(sweep_config: SweepConfig) -> list[CaseConfig]
```

### ResultCollector

```python
# src/armi_layer/result_collector.py
def collect_results(manager: CaseManager, case_ids: list[str] | None = None,
                    *, failed_mode: str = "include") -> pd.DataFrame
def export_csv(df: pd.DataFrame, path: Path) -> Path
def export_json(df: pd.DataFrame, path: Path) -> Path
```

## OpenMC Layer 인터페이스

```python
# 입력 생성
def generate_input(config: CaseConfig, case_path: Path) -> list[Path]

# 결과 파싱
def parse_results(case_path: Path) -> SimulationResult

# KPI 계산
def calculate_kpi(result: SimulationResult, *, keff_margin: float = 0.05) -> dict
def save_kpi(kpi: dict, case_path: Path) -> Path
def load_kpi(case_path: Path) -> dict
```

## Run Manager 인터페이스

```python
# 케이스 실행 (상태 전이 + 재시도 포함)
def run_case(run_config: RunConfig, case_dir: Path) -> RunResult

# 상태 관리
def save_status(case_dir: Path, status: RunStatus) -> Path
def get_status(case_dir: Path) -> RunStatus
def update_status(case_dir: Path, new_state: StatusType, **kwargs) -> RunStatus

# 유틸리티
def detect_physical_cores() -> int
def resolve_cross_sections_path(explicit_path: Path | None = None) -> Path
```

## 에러 코드

| 예외 | 모듈 | 의미 |
|------|------|------|
| `NucdataNotFoundError` | `nucdata` | 핵데이터 경로 해석 실패 |
| `StatepointNotFoundError` | `result_parser` | statepoint.h5 파일 없음 |
| `StatepointParseError` | `result_parser` | statepoint 파싱 실패 |
| `ValidationError` | Pydantic | 모델 유효성 검증 실패 |
| `FileNotFoundError` | 여러 모듈 | 필수 파일 미존재 |
