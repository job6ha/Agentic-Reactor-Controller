# Agentic-Reactor-Controller

ARMI를 오케스트레이터로, OpenMC를 계산 엔진으로 사용하는 원자로 시뮬레이션 시스템.
파일/프로세스 기반 느슨한 결합 아키텍처.

## 설치

```bash
# 저장소 클론
git clone https://github.com/job6ha/Agentic-Reactor-Controller.git
cd Agentic-Reactor-Controller

# 의존성 설치
uv sync
```

## 환경 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 값을 설정
```

## 실행

```bash
uv run python -m src
```

## 테스트

```bash
uv run pytest
```

## 프로젝트 구조

```
Agentic-Reactor-Controller/
├── src/
│   ├── armi_layer/         # ARMI 오케스트레이션 (실험 설계, 케이스 생성, 결과 수집)
│   ├── openmc_layer/       # OpenMC 입출력 (입력 생성, 결과 파싱)
│   ├── run_manager/        # 실행 관리 (환경변수, 로그, 재시도)
│   └── controller/         # 제어/의사결정 (에이전트 교체 가능)
├── tests/
├── runs/                   # 시뮬레이션 케이스 (git 미추적)
├── nucdata/                # 핵데이터 (호스트 고정, git 미추적)
├── docs/                   # 프로젝트 문서
├── pyproject.toml
└── README.md
```
