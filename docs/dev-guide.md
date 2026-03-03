# 개발 가이드

이 문서는 프로젝트 환경 세팅부터 첫 시뮬레이션 케이스 실행까지 안내한다.

## 사전 요구사항

| 도구 | 버전 | 용도 |
|------|------|------|
| Python | 3.11+ | 런타임 |
| uv | 최신 | 패키지 관리 |
| OpenMC | 0.14+ | 몬테카를로 시뮬레이션 (로컬 실행 시) |
| Docker | 최신 | 컨테이너 실행 (Docker 실행 시) |
| Git | 최신 | 버전 관리 |

> OpenMC를 로컬에 설치하지 않아도 Docker로 실행 가능하다.

## 로컬 개발 환경 세팅

### 1. 저장소 클론

```bash
git clone https://github.com/job6ha/Agentic-Reactor-Controller.git
cd Agentic-Reactor-Controller
```

### 2. 의존성 설치

```bash
# uv가 없으면 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치 (개발 의존성 포함)
uv sync
```

### 3. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 환경에 맞게 설정:

```bash
# OpenMP 스레드 수 (미설정 시 P-core 수 자동 감지)
OMP_NUM_THREADS=8

# 핵데이터 경로 (미설정 시 nucdata/에서 자동 탐색)
OPENMC_CROSS_SECTIONS=nucdata/cross_sections.xml
```

### 4. 핵데이터 설치

OpenMC 시뮬레이션에는 핵단면적(Cross Section) 데이터가 필요하다.

```bash
# ENDF/B-VIII.0 다운로드 (기본, ~3GB)
uv run python scripts/download_xs.py

# 다른 라이브러리 선택
uv run python scripts/download_xs.py --library endfb-vii.1

# 사용 가능한 라이브러리 목록
uv run python scripts/download_xs.py --list
```

다운로드 완료 후 `nucdata/` 디렉토리에 설치되며, 자동 탐색으로 인해 별도 설정 없이 사용 가능하다.

### 5. 테스트 실행

```bash
# 전체 테스트 (318개)
uv run pytest

# 상세 출력
uv run pytest -v

# 특정 모듈만
uv run pytest tests/test_models.py
```

### 6. 정적 분석

```bash
# Lint
uv run ruff check src/ tests/ scripts/

# Format
uv run ruff format src/ tests/ scripts/

# Type check (strict 모드)
uv run mypy src/ scripts/
```

## Docker 실행

OpenMC를 로컬에 설치하지 않고 Docker로 실행할 수 있다.

### 빌드

```bash
docker compose build
```

> Apple Silicon (M1/M2/M3)에서는 Rosetta 에뮬레이션으로 amd64 이미지를 실행한다.
> 첫 빌드 시 시간이 다소 걸릴 수 있다.

### 컨테이너 시작

```bash
# 인터랙티브 셸
docker compose run --rm openmc

# 백그라운드 실행
docker compose up -d
docker compose exec openmc bash
```

### 컨테이너 내부에서 실행

```bash
# 테스트
uv run pytest

# 시뮬레이션 실행
uv run python -c "
from src.armi_layer.case_folder import create_case
from src.armi_layer.models import CaseConfig
case_path = create_case(CaseConfig(name='docker-test'), runs_dir=Path('/work/runs'))
print(f'케이스 생성: {case_path}')
"
```

### 볼륨 마운트

| 호스트 경로 | 컨테이너 경로 | 용도 |
|-------------|--------------|------|
| `./runs` | `/work/runs` | 시뮬레이션 결과 공유 |
| `./nucdata` | `/data/nucdata` | 커스텀 핵데이터 (옵션) |

커스텀 핵데이터를 사용하려면 `docker-compose.yml`에서 nucdata 볼륨 주석을 해제하고
`OPENMC_CROSS_SECTIONS` 환경변수를 설정한다.

## 첫 시뮬레이션 케이스 실행

### Step 1: 케이스 설정 생성

```python
from src.armi_layer.models import CaseConfig, MaterialParams, SimulationSettings

config = CaseConfig(
    name="my-first-case",
    description="PWR pin cell 기본 시뮬레이션",
    materials=MaterialParams(fuel_enrichment=3.0),
    settings=SimulationSettings(
        batches=110,
        inactive=10,
        particles=1000,
    ),
)
```

### Step 2: 케이스 디렉토리 생성

```python
from src.armi_layer.case_folder import create_case

case_path = create_case(config)
print(f"케이스 경로: {case_path}")
# 출력: 케이스 경로: /path/to/runs/case_0001
```

이 단계에서 다음 구조가 생성된다:

```
runs/case_0001/
├── input/
├── output/
└── meta/
    ├── config.json     # CaseConfig 직렬화
    └── status.json     # RunStatus (queued)
```

### Step 3: OpenMC 입력 파일 생성

```python
from src.openmc_layer.input_generator import generate_input

xml_files = generate_input(config, case_path)
print(f"생성된 파일: {[f.name for f in xml_files]}")
# 출력: ['geometry.xml', 'materials.xml', 'settings.xml']
```

### Step 4: 시뮬레이션 실행

```python
from src.armi_layer.models import RunConfig
from src.run_manager.status import run_case

run_config = RunConfig(timeout=600.0)  # 10분 타임아웃
result = run_case(run_config, case_path)

print(f"성공: {result.success}")
print(f"실행 시간: {result.runtime:.1f}초")
```

### Step 5: 결과 파싱 및 KPI 계산

```python
from src.openmc_layer.result_parser import parse_results
from src.openmc_layer.kpi_calculator import calculate_kpi, save_kpi

# statepoint.h5 → SimulationResult
sim_result = parse_results(case_path)
print(f"keff = {sim_result.keff:.5f} +/- {sim_result.keff_std:.5f}")

# KPI 계산 및 저장
kpi = calculate_kpi(sim_result)
save_kpi(kpi, case_path)
print(f"Criticality: {kpi['criticality']}")
```

### 파라미터 스윕 예제

여러 농축도를 한번에 시뮬레이션하려면:

```python
from src.armi_layer.sweep import SweepConfig, SweepParam, generate_sweep_configs

sweep = SweepConfig(
    base_config=CaseConfig(name="enrichment-sweep"),
    params=[
        SweepParam(
            field_path="materials.fuel_enrichment",
            values=[2.0, 3.0, 4.0, 5.0],
        ),
    ],
)

configs = generate_sweep_configs(sweep)
print(f"생성된 케이스 수: {len(configs)}")
# 출력: 4
```

## 브랜치 전략

```
main ← develop ← feature/*, fix/*, refactor/*
```

- `main`: 릴리즈 브랜치
- `develop`: 개발 통합 브랜치
- `feature/*`: 기능 개발 (예: `feature/user-auth`)
- `fix/*`: 버그 수정 (예: `fix/keff-rounding`)
- `refactor/*`: 리팩토링 (예: `refactor/sweep-engine`)
- `docs/*`: 문서 작업 (예: `docs/architecture`)

PR은 squash merge를 기본으로 한다.

## 트러블슈팅

### "OPENMC_CROSS_SECTIONS 경로가 존재하지 않습니다"

핵데이터가 설치되지 않았거나 경로가 잘못되었다.

```bash
# 핵데이터 설치
uv run python scripts/download_xs.py

# 설치 확인
python -c "from src.run_manager.nucdata import is_installed; print(is_installed())"
```

### "openmc: command not found"

OpenMC가 로컬에 설치되지 않았다. Docker를 사용하거나 OpenMC를 빌드/설치한다.

```bash
# Docker로 실행
docker compose run --rm openmc
```

### Docker 빌드 실패 (Apple Silicon)

Rosetta가 활성화되어 있는지 확인:

```bash
# Rosetta 설치
softwareupdate --install-rosetta

# Docker Desktop 설정에서 "Use Rosetta for x86_64/amd64 emulation" 활성화
```

### pytest 실행 시 import 에러

```bash
# 의존성 재설치
uv sync

# 캐시 정리 후 재실행
uv run pytest --cache-clear
```

### mypy 에러: "Cannot find implementation or library stub"

```bash
# 타입 스텁 설치 확인
uv sync  # pandas-stubs 포함

# h5py는 mypy overrides로 처리됨 (pyproject.toml 참고)
```

### OMP_NUM_THREADS 자동 감지가 1을 반환

macOS에서 `sysctl` 명령이 실패할 수 있다. 명시적으로 설정:

```bash
# .env
OMP_NUM_THREADS=8
```

또는 `RunConfig`에서 직접 지정:

```python
run_config = RunConfig(omp_threads=8)
```
