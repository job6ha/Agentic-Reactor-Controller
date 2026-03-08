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

# scripts 복사
COPY scripts/ ./scripts/

# openmc venv에 프로젝트 의존성 설치
ENV VIRTUAL_ENV=/openmc_venv
ENV PATH="/openmc_venv/bin:$PATH"
RUN pip install httpx "pydantic>=2" scikit-optimize scikit-learn numpy pandas h5py

# runs 디렉토리 생성
RUN mkdir -p /app/runs

# 기본 환경변수
ENV OMP_NUM_THREADS=4
ENV PYTHONPATH=/app

CMD ["/openmc_venv/bin/python3", "-m", "src", "--config", "configs/default.json"]
