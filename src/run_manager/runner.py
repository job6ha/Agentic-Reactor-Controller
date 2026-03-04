"""OpenMC 프로세스 실행기.

RunConfig를 기반으로 환경변수를 세팅하고, OpenMC를 서브프로세스로
실행하며, stdout/stderr를 로그 파일에 캡처한다.
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.armi_layer.models import RunConfig
from src.run_manager.cpu_detect import detect_physical_cores

logger = logging.getLogger(__name__)

# OpenMC 실행 바이너리 기본값
OPENMC_COMMAND = "openmc"


@dataclass(frozen=True)
class RunResult:
    """OpenMC 실행 결과.

    Attributes:
        exit_code: 프로세스 종료 코드.
        runtime: 실행 시간 (초).
        log_path: 로그 파일 경로.
        success: 성공 여부 (exit_code == 0).
    """

    exit_code: int
    runtime: float
    log_path: Path
    success: bool


def _build_env(run_config: RunConfig) -> dict[str, str]:
    """RunConfig를 기반으로 OpenMC 실행 환경변수를 구성한다.

    현재 프로세스 환경변수를 복사한 뒤, RunConfig의 설정값으로
    OMP_NUM_THREADS와 OPENMC_CROSS_SECTIONS를 오버라이드한다.
    omp_threads가 None이면 자동 감지된 물리 코어 수를 사용한다.

    Args:
        run_config: 실행 설정.

    Returns:
        환경변수 딕셔너리.
    """
    env = os.environ.copy()

    if run_config.omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(run_config.omp_threads)
    else:
        detected = detect_physical_cores()
        env["OMP_NUM_THREADS"] = str(detected)
        logger.info("OMP_NUM_THREADS 자동 감지: %d", detected)

    if run_config.cross_sections_path is not None:
        env["OPENMC_CROSS_SECTIONS"] = str(run_config.cross_sections_path)

    return env


def run_openmc(
    run_config: RunConfig,
    case_path: Path,
    *,
    openmc_command: str = OPENMC_COMMAND,
    attempt: int | None = None,
) -> RunResult:
    """OpenMC를 서브프로세스로 실행한다.

    case_path/input/ 디렉토리를 작업 디렉토리로 사용하여 OpenMC를 실행하고,
    stdout/stderr를 case_path/output/run.log에 기록한다.

    Args:
        run_config: 실행 설정 (환경변수, 타임아웃 등).
        case_path: 케이스 폴더 경로.
        openmc_command: OpenMC 실행 명령어.
        attempt: 시도 번호. 지정하면 run_{attempt}.log로 저장.

    Returns:
        RunResult 실행 결과.
    """
    output_dir = (case_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = (case_path / "input").resolve()
    log_filename = f"run_{attempt}.log" if attempt is not None else "run.log"
    log_path = output_dir / log_filename

    # 작업 디렉토리: output/ (statepoint가 여기에 생성됨)
    # OpenMC에 input/ 경로를 인자로 전달하여 XML을 읽도록 함
    cwd = run_config.working_dir or output_dir

    env = _build_env(run_config)

    logger.info(
        "OpenMC 실행 시작: case=%s, threads=%s, timeout=%s",
        case_path.name,
        env.get("OMP_NUM_THREADS", "default"),
        run_config.timeout,
    )

    start_time = time.monotonic()

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                [openmc_command, str(input_dir)],
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=run_config.timeout,
            )
        elapsed = time.monotonic() - start_time
        exit_code = result.returncode

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start_time
        exit_code = -1

        # 타임아웃 메시지를 로그에 추가
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[TIMEOUT] 실행 시간 {elapsed:.1f}초 초과 "
                f"(제한: {run_config.timeout}초)\n"
            )
        logger.warning(
            "OpenMC 타임아웃: case=%s, elapsed=%.1fs",
            case_path.name,
            elapsed,
        )

    success = exit_code == 0

    if success:
        logger.info(
            "OpenMC 실행 완료: case=%s, runtime=%.1fs",
            case_path.name,
            elapsed,
        )
    else:
        logger.error(
            "OpenMC 실행 실패: case=%s, exit_code=%d, runtime=%.1fs",
            case_path.name,
            exit_code,
            elapsed,
        )

    return RunResult(
        exit_code=exit_code,
        runtime=elapsed,
        log_path=log_path,
        success=success,
    )
