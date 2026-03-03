# Agentic-Reactor-Controller

## 프로젝트 개요
- **목적**: ARMI를 오케스트레이터로, OpenMC를 계산 엔진으로 사용하는 원자로 시뮬레이션 시스템
- **아키텍처**: 파일/프로세스 기반 느슨한 결합 (loose coupling)
- **기술 스택**: Python 3.11+, ARMI, OpenMC
- **실행 환경**: macOS (OpenMP), Docker (단일 or 2컨테이너)

## 문서
- [아키텍처](docs/architecture.md) — 시스템 구조, 디렉토리 구조, 주요 모듈
- [컨벤션](docs/conventions.md) — 프로젝트 전용 코딩 컨벤션
- [API 설계](docs/api-design.md) — 핵심 인터페이스 설계
- [테스트 전략](docs/testing.md) — 테스트 전략, 커버리지 기준

## 핵심 모듈
- `src/armi_layer/` — 실험 설계, 케이스 생성, 결과 수집
- `src/openmc_layer/` — OpenMC 입력 생성, 결과 파싱
- `src/run_manager/` — 실행 래퍼 (환경변수, 로그, 재시도)
- `src/controller/` — 제어/의사결정 (나중에 에이전트로 교체 가능)

## 핵심 인터페이스
1. `propose_actions(state) -> actions` — 현재 상태에서 다음 행동 제안
2. `apply_actions_to_case(actions) -> openmc_input` — 행동을 OpenMC 입력으로 변환
3. `evaluate_results(openmc_output) -> metrics` — 계산 결과 평가
4. `update_state(metrics) -> new_state` — 상태 갱신

## 프로젝트 구조
```
Agentic-Reactor-Controller/
├── src/
│   ├── armi_layer/         # ARMI 오케스트레이션
│   ├── openmc_layer/       # OpenMC 입출력
│   ├── run_manager/        # 실행 관리
│   └── controller/         # 제어 로직
├── tests/
├── runs/                   # 시뮬레이션 케이스 (git 미추적)
│   └── case_XXXX/{input,output,meta}/
├── nucdata/                # 핵데이터 (호스트 고정, git 미추적)
├── docs/
├── pyproject.toml
└── README.md
```

## 프로젝트 고유 규칙
- 케이스 폴더: `runs/case_XXXX/{input,output,meta}/`
- 핵데이터: `nucdata/` (호스트 고정, git 미추적)
- OpenMP 스레드: `OMP_NUM_THREADS` = P-core 수 기준
- 모듈 간 결합은 파일/프로세스 기반으로 유지
