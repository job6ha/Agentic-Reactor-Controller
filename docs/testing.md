# 테스트 전략

## 테스트 프레임워크

- **pytest** (v9+)
- 실행: `uv run pytest`
- 설정: `pyproject.toml`의 `[tool.pytest.ini_options]`

## 테스트 구조

```
tests/
├── test_models.py              # Pydantic 모델 (39개)
├── test_case_folder.py         # 케이스 디렉토리 생성/로드 (12개)
├── test_case_manager.py        # 케이스 생명주기 관리 (22개)
├── test_sweep.py               # 파라미터 스윕 (25개)
├── test_result_collector.py    # 결과 수집/내보내기 (19개)
├── test_input_generator.py     # XML 입력 생성 (21개)
├── test_result_parser.py       # statepoint 파싱 (30개)
├── test_kpi_calculator.py      # KPI 계산 (23개)
├── test_runner.py              # OpenMC 실행 래퍼 (16개)
├── test_status.py              # 상태 관리/재시도 (22개)
├── test_cpu_detect.py          # CPU 코어 감지 (14개)
├── test_nucdata.py             # 핵데이터 경로 관리 (23개)
├── test_base_controller.py     # Controller ABC (17개)
└── test_simple_controller.py   # SimpleController (35개)
```

총 318개 테스트.

## 실행 방법

```bash
# 전체 테스트
uv run pytest

# 상세 출력
uv run pytest -v

# 특정 모듈
uv run pytest tests/test_models.py

# 특정 클래스/함수
uv run pytest tests/test_models.py::TestCaseConfig::test_json_roundtrip

# 키워드 필터
uv run pytest -k "keff"
```

## 테스트 원칙

### 독립성

- 테스트 간 공유 상태 없음
- 각 테스트는 `tmp_path` 또는 `monkeypatch`로 격리
- 외부 의존성(OpenMC, 핵데이터)은 mock/fixture로 대체

### Fixture 활용

```python
@pytest.fixture()
def case_dir(tmp_path: Path) -> Path:
    """기본 케이스 디렉토리 구조."""
    ...

@pytest.fixture()
def sample_result() -> SimulationResult:
    """테스트용 시뮬레이션 결과."""
    ...
```

- `tmp_path`: 임시 디렉토리 (pytest 내장)
- `monkeypatch`: 환경변수, 모듈 속성 패치

### Mock 전략

- **OpenMC 서브프로세스**: `subprocess.run` 모킹, exit code + stdout 시뮬레이션
- **statepoint.h5**: h5py로 최소 HDF5 파일 직접 생성
- **핵데이터**: 임시 디렉토리에 `cross_sections.xml` 파일 생성
- **CPU 감지**: `platform.system()` 모킹으로 macOS/Linux 테스트

### 검증 범위

| 카테고리 | 테스트 내용 |
|----------|------------|
| 모델 생성 | 기본값, 커스텀 값, extra 파라미터 |
| 유효성 검증 | 범위 초과, 순서 위반, 필수 필드 누락 |
| 직렬화 | JSON 라운드트립 (dump → validate) |
| 에러 처리 | 커스텀 예외, 파일 미존재, 잘못된 입력 |
| 상태 전이 | queued → running → done/failed |
| 경계값 | 허용치 경계, 빈 리스트, None 처리 |
| 통합 | 전체 제어 루프 (스윕 → 수렴 → 종료) |

## 정적 분석

```bash
# Lint
uv run ruff check src/ tests/ scripts/

# Format 확인
uv run ruff format --check src/ tests/ scripts/

# Type check (strict)
uv run mypy src/ scripts/
```

## CI/CD

GitHub Actions (`ci.yml`)에서 PR/push 시 자동 실행:
- **lint** job: ruff check + format
- **typecheck** job: mypy strict
- **test** job: pytest
