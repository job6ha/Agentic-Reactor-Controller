"""케이스 실행 상태 관리.

meta/status.json을 통해 케이스의 실행 상태를 추적한다.
상태 전이: queued → running → done/failed.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from src.armi_layer.models import RunConfig, RunStatus, StatusType
from src.run_manager.runner import RunResult, run_openmc

logger = logging.getLogger(__name__)

STATUS_FILENAME = "status.json"


def _status_path(case_dir: Path) -> Path:
    """meta/status.json 경로를 반환한다."""
    return case_dir / "meta" / STATUS_FILENAME


def save_status(case_dir: Path, status: RunStatus) -> Path:
    """RunStatus를 meta/status.json에 저장한다.

    Args:
        case_dir: 케이스 폴더 경로.
        status: 저장할 RunStatus 객체.

    Returns:
        저장된 status.json 파일 경로.

    Raises:
        FileNotFoundError: meta 디렉토리가 없을 때.
    """
    path = _status_path(case_dir)
    if not path.parent.exists():
        raise FileNotFoundError(f"meta 디렉토리가 없습니다: {path.parent}")

    path.write_text(status.model_dump_json(indent=2), encoding="utf-8")
    logger.debug("상태 저장: case=%s, status=%s", case_dir.name, status.status.value)
    return path


def get_status(case_dir: Path) -> RunStatus:
    """케이스의 현재 실행 상태를 조회한다.

    Args:
        case_dir: 케이스 폴더 경로.

    Returns:
        현재 RunStatus.

    Raises:
        FileNotFoundError: status.json이 없을 때.
    """
    path = _status_path(case_dir)
    if not path.exists():
        raise FileNotFoundError(f"status.json이 없습니다: {path}")

    return RunStatus.model_validate_json(path.read_text(encoding="utf-8"))


def update_status(
    case_dir: Path,
    new_state: StatusType,
    *,
    error_message: str | None = None,
    increment_attempt: bool = False,
) -> RunStatus:
    """케이스 상태를 업데이트한다.

    현재 상태를 읽어 새 상태로 전이하고, 타임스탬프를 자동 기록한다.

    Args:
        case_dir: 케이스 폴더 경로.
        new_state: 전이할 상태.
        error_message: 실패 시 에러 메시지.
        increment_attempt: True면 attempt를 1 증가.

    Returns:
        업데이트된 RunStatus.
    """
    current = get_status(case_dir)
    now = datetime.now(UTC)

    update_fields: dict = {
        "status": new_state,
    }

    if new_state == StatusType.RUNNING:
        update_fields["started_at"] = now

    if new_state in (StatusType.DONE, StatusType.FAILED):
        update_fields["completed_at"] = now

    if error_message is not None:
        update_fields["error_message"] = error_message

    if increment_attempt:
        update_fields["attempt"] = current.attempt + 1

    updated = current.model_copy(update=update_fields)
    save_status(case_dir, updated)

    logger.info(
        "상태 전이: case=%s, %s → %s",
        case_dir.name,
        current.status.value,
        new_state.value,
    )
    return updated


def run_case(
    run_config: RunConfig,
    case_dir: Path,
    *,
    openmc_command: str = "openmc",
) -> RunResult:
    """상태 관리와 재시도를 포함한 케이스 실행.

    status.json을 queued→running→done/failed로 전이하면서
    OpenMC를 실행한다. max_retries > 0이면 실패 시 자동 재시도하며,
    각 시도의 로그를 run_1.log, run_2.log 등으로 별도 보존한다.

    Args:
        run_config: 실행 설정.
        case_dir: 케이스 폴더 경로.
        openmc_command: OpenMC 실행 명령어.

    Returns:
        최종 RunResult 실행 결과.
    """
    max_attempts = run_config.max_retries + 1
    result: RunResult | None = None

    for attempt_idx in range(1, max_attempts + 1):
        # running 상태로 전이
        status = update_status(
            case_dir,
            StatusType.RUNNING,
            increment_attempt=True,
        )
        current_attempt = status.attempt

        logger.info(
            "실행 시도 %d/%d: case=%s",
            attempt_idx,
            max_attempts,
            case_dir.name,
        )

        # OpenMC 실행 (attempt 번호로 로그 분리)
        result = run_openmc(
            run_config,
            case_dir,
            openmc_command=openmc_command,
            attempt=current_attempt,
        )

        if result.success:
            update_status(case_dir, StatusType.DONE)
            return result

        # 마지막 시도가 아니면 재시도를 위해 failed 전이 후 계속
        if attempt_idx < max_attempts:
            update_status(
                case_dir,
                StatusType.FAILED,
                error_message=(
                    f"exit_code={result.exit_code}, "
                    f"시도 {current_attempt}/{max_attempts} 실패, 재시도 예정"
                ),
            )
            logger.warning(
                "재시도 예정: case=%s, attempt=%d/%d",
                case_dir.name,
                attempt_idx,
                max_attempts,
            )

    # 모든 시도 실패
    assert result is not None  # noqa: S101
    update_status(
        case_dir,
        StatusType.FAILED,
        error_message=(
            f"exit_code={result.exit_code}, "
            f"총 {max_attempts}회 시도 후 최종 실패"
        ),
    )
    return result
