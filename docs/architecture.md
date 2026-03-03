# 아키텍처

## 시스템 개요
ARMI를 오케스트레이터로, OpenMC를 몬테카를로 중성자 수송 계산 엔진으로 사용하는 원자로 시뮬레이션 시스템.
파일/프로세스 기반 느슨한 결합 아키텍처로, 각 레이어가 독립적으로 교체 가능하다.

```
┌─────────────┐
│  Controller  │  제어/의사결정 (→ 에이전트 교체 가능)
└──────┬───────┘
       │ propose_actions(state) -> actions
       ▼
┌─────────────┐
│  ARMI Layer  │  실험 설계, 케이스 생성, 결과 수집
└──────┬───────┘
       │ apply_actions_to_case(actions) -> openmc_input
       ▼
┌──────────────┐
│ OpenMC Layer  │  OpenMC 입력 생성, 실행, 결과 파싱
└──────┬────────┘
       │ evaluate_results(openmc_output) -> metrics
       ▼
┌──────────────┐
│  Run Manager  │  실행 래퍼 (환경변수, 로그, 재시도)
└──────────────┘
```

## 디렉토리 구조
```
src/
├── __init__.py
├── armi_layer/             # ARMI 오케스트레이션
│   └── __init__.py
├── openmc_layer/           # OpenMC 입출력
│   └── __init__.py
├── run_manager/            # 실행 관리
│   └── __init__.py
└── controller/             # 제어 로직
    └── __init__.py
```

## 주요 모듈

### armi_layer
- 실험 설계 및 파라미터 스위프
- 케이스 디렉토리 생성 (`runs/case_XXXX/`)
- OpenMC 계산 결과 수집 및 정리

### openmc_layer
- ARMI 케이스로부터 OpenMC 입력 파일 생성
- OpenMC 출력(statepoint 등) 파싱
- 지오메트리/재료/설정 변환

### run_manager
- OpenMC 프로세스 실행 래퍼
- 환경변수 관리 (`OMP_NUM_THREADS` 등)
- 로깅, 에러 핸들링, 재시도 로직

### controller
- 현재 상태 기반 다음 행동 결정
- 수렴 판정, 최적화 루프
- (향후) AI 에이전트로 교체 가능한 인터페이스

## 핵심 인터페이스 (데이터 흐름)
```
State ──→ propose_actions() ──→ Actions
Actions ──→ apply_actions_to_case() ──→ OpenMC Input
OpenMC Input ──→ [OpenMC 실행] ──→ OpenMC Output
OpenMC Output ──→ evaluate_results() ──→ Metrics
Metrics ──→ update_state() ──→ New State (루프)
```

## 실행 환경
- **로컬**: macOS, OpenMP 기반 단일 실행 가속 (P-core 기준)
- **Docker**: 단일 컨테이너 또는 2컨테이너 (ARMI + OpenMC 분리)

## 외부 의존성
- **ARMI**: 원자로 분석 통합 프레임워크
- **OpenMC**: 몬테카를로 중성자/광자 수송 코드
- **핵데이터**: `nucdata/` 디렉토리 (호스트 고정, git 미추적)
