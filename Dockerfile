# OpenMC 공식 이미지는 amd64만 지원
# Apple Silicon에서는 Rosetta 에뮬레이션으로 실행됨
FROM --platform=linux/amd64 openmc/openmc:latest

WORKDIR /app

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 프로젝트 파일 복사
COPY pyproject.toml README.md uv.lock* ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/

# 의존성 설치
RUN uv sync

# runs 디렉토리 생성
RUN mkdir -p /work/runs

# 기본 환경변수
ENV OMP_NUM_THREADS=4
ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-m", "src", "--config", "configs/default.json"]
