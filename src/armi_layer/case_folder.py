"""케이스 폴더 생성 유틸리티.

runs/case_XXXX/{input, output, meta}/ 구조를 자동 생성하고,
meta/config.json에 CaseConfig를 직렬화하여 저장한다.
"""

import json
import re
import threading
from pathlib import Path

from src.armi_layer.models import CaseConfig, RunStatus

# 케이스 폴더명 패턴: case_0001, case_0002, ...
CASE_DIR_PATTERN = re.compile(r"^case_(\d{4})$")

# 케이스 내부 서브 디렉토리
CASE_SUBDIRS = ("input", "output", "meta")

# 병렬 실행 시 케이스 번호 할당 경합 방지용 락
_case_number_lock = threading.Lock()

# 동일 프로세스 내 race condition 재시도 상한
_MAX_CREATE_RETRIES = 10


def _find_next_case_number(runs_dir: Path) -> int:
    """기존 케이스 폴더를 스캔하여 다음 번호를 반환한다.

    Args:
        runs_dir: 케이스들이 저장되는 상위 디렉토리.

    Returns:
        다음 케이스 번호 (1부터 시작).
    """
    max_number = 0
    if runs_dir.exists():
        for entry in runs_dir.iterdir():
            if entry.is_dir():
                match = CASE_DIR_PATTERN.match(entry.name)
                if match:
                    max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def create_case(
    config: CaseConfig,
    runs_dir: Path | None = None,
) -> Path:
    """케이스 폴더를 생성하고 설정을 저장한다.

    runs_dir 아래에 case_XXXX/{input, output, meta}/ 구조를 생성하고,
    meta/config.json에 CaseConfig를, meta/status.json에 RunStatus(queued)를
    저장한다.

    Args:
        config: 케이스 설정.
        runs_dir: 케이스 상위 디렉토리. None이면 현재 디렉토리의 runs/.

    Returns:
        생성된 케이스 폴더의 Path.

    Raises:
        FileExistsError: 케이스 폴더가 이미 존재하는 경우 (경합 조건).
    """
    if runs_dir is None:
        runs_dir = Path("runs")

    runs_dir.mkdir(parents=True, exist_ok=True)

    # 병렬 실행 시 TOCTOU race condition 방지: 락 + 재시도
    with _case_number_lock:
        for _ in range(_MAX_CREATE_RETRIES):
            case_number = _find_next_case_number(runs_dir)
            case_dir = runs_dir / f"case_{case_number:04d}"

            if case_dir.exists():
                continue

            # 서브 디렉토리 생성 (mkdir로 원자적 점유)
            try:
                (case_dir / CASE_SUBDIRS[0]).mkdir(parents=True)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(
                f"케이스 폴더 생성 실패: {_MAX_CREATE_RETRIES}회 재시도 초과 ({runs_dir})"
            )

    # 나머지 서브 디렉토리 생성 (input은 이미 생성됨)
    for subdir in CASE_SUBDIRS[1:]:
        (case_dir / subdir).mkdir(parents=True, exist_ok=True)

    # config.json 저장
    config_path = case_dir / "meta" / "config.json"
    config_path.write_text(
        json.dumps(
            config.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # status.json 초기 상태 저장
    status = RunStatus()
    status_path = case_dir / "meta" / "status.json"
    status_path.write_text(
        json.dumps(
            status.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return case_dir


def load_case_config(case_dir: Path) -> CaseConfig:
    """케이스 폴더에서 CaseConfig를 로드한다.

    Args:
        case_dir: 케이스 폴더 경로.

    Returns:
        로드된 CaseConfig.

    Raises:
        FileNotFoundError: config.json이 존재하지 않는 경우.
    """
    config_path = case_dir / "meta" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json을 찾을 수 없습니다: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    return CaseConfig.model_validate(data)


def load_run_status(case_dir: Path) -> RunStatus:
    """케이스 폴더에서 RunStatus를 로드한다.

    Args:
        case_dir: 케이스 폴더 경로.

    Returns:
        로드된 RunStatus.

    Raises:
        FileNotFoundError: status.json이 존재하지 않는 경우.
    """
    status_path = case_dir / "meta" / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"status.json을 찾을 수 없습니다: {status_path}")

    data = json.loads(status_path.read_text(encoding="utf-8"))
    return RunStatus.model_validate(data)
