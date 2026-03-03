# Agentic-Reactor-Controller

(TODO: 프로젝트 설명)

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
├── src/                    # 소스 코드
├── tests/                  # 테스트 코드
├── docs/                   # 프로젝트 문서
├── pyproject.toml          # 프로젝트 메타데이터 및 의존성
└── README.md               # 프로젝트 소개
```
