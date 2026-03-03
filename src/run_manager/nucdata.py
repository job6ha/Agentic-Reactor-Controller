"""핵데이터(Cross Section) 경로 관리.

OPENMC_CROSS_SECTIONS 환경변수를 기반으로 핵데이터 경로를 해석하고
유효성을 검증한다. 버전 메타데이터 조회 기능도 제공한다.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# cross_sections.xml 기본 탐색 경로 (프로젝트 루트 기준)
DEFAULT_NUCDATA_DIR = "nucdata"
CROSS_SECTIONS_FILENAME = "cross_sections.xml"
VERSION_FILENAME = "version.json"


class NucdataNotFoundError(Exception):
    """핵데이터를 찾을 수 없을 때 발생하는 예외."""


def _find_project_root() -> Path:
    """프로젝트 루트 디렉토리를 탐색한다.

    pyproject.toml이 있는 디렉토리를 프로젝트 루트로 판단한다.
    현재 파일 위치에서 상위로 탐색한다.

    Returns:
        프로젝트 루트 경로.

    Raises:
        FileNotFoundError: pyproject.toml을 찾을 수 없을 때.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # 최대 10단계
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다 (pyproject.toml 없음)")


def resolve_cross_sections_path(
    explicit_path: Path | None = None,
) -> Path:
    """핵데이터 cross_sections.xml 경로를 해석한다.

    우선순위:
    1. explicit_path (RunConfig.cross_sections_path에서 전달)
    2. OPENMC_CROSS_SECTIONS 환경변수
    3. {프로젝트루트}/nucdata/ 하위에서 자동 탐색

    Args:
        explicit_path: 명시적으로 지정된 경로.

    Returns:
        cross_sections.xml 절대 경로.

    Raises:
        NucdataNotFoundError: 핵데이터를 찾을 수 없을 때.
    """
    # 1. 명시적 경로
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        if resolved.exists():
            logger.info("핵데이터 경로 (명시): %s", resolved)
            return resolved
        raise NucdataNotFoundError(
            f"명시된 핵데이터 경로가 존재하지 않습니다: {resolved}"
        )

    # 2. 환경변수
    env_path = os.environ.get("OPENMC_CROSS_SECTIONS")
    if env_path:
        resolved = Path(env_path).resolve()
        if resolved.exists():
            logger.info("핵데이터 경로 (환경변수): %s", resolved)
            return resolved
        raise NucdataNotFoundError(
            f"OPENMC_CROSS_SECTIONS 경로가 존재하지 않습니다: {resolved}"
        )

    # 3. 프로젝트 nucdata/ 하위 자동 탐색
    try:
        project_root = _find_project_root()
    except FileNotFoundError as e:
        raise NucdataNotFoundError(
            "핵데이터를 찾을 수 없습니다. "
            "OPENMC_CROSS_SECTIONS 환경변수를 설정하거나 "
            "scripts/download_xs.py를 실행하세요."
        ) from e

    nucdata_dir = project_root / DEFAULT_NUCDATA_DIR
    if not nucdata_dir.exists():
        raise NucdataNotFoundError(
            f"nucdata/ 디렉토리가 없습니다: {nucdata_dir}. "
            "scripts/download_xs.py를 실행하여 핵데이터를 설치하세요."
        )

    # nucdata/ 하위에서 cross_sections.xml 재귀 탐색
    for xs_path in nucdata_dir.rglob(CROSS_SECTIONS_FILENAME):
        logger.info("핵데이터 경로 (자동 탐색): %s", xs_path)
        return xs_path

    raise NucdataNotFoundError(
        f"nucdata/ 디렉토리에 {CROSS_SECTIONS_FILENAME}이 없습니다: {nucdata_dir}. "
        "scripts/download_xs.py를 실행하여 핵데이터를 설치하세요."
    )


def get_version_info(nucdata_dir: Path | None = None) -> dict[str, str]:
    """설치된 핵데이터의 버전 정보를 반환한다.

    Args:
        nucdata_dir: nucdata 디렉토리 경로.
            None이면 프로젝트 기본 경로를 사용.

    Returns:
        버전 메타데이터 딕셔너리.

    Raises:
        FileNotFoundError: version.json이 없을 때.
    """
    if nucdata_dir is None:
        project_root = _find_project_root()
        nucdata_dir = project_root / DEFAULT_NUCDATA_DIR

    version_path = nucdata_dir / VERSION_FILENAME

    if not version_path.exists():
        raise FileNotFoundError(f"버전 정보를 찾을 수 없습니다: {version_path}")

    data = json.loads(version_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {k: str(v) for k, v in data.items()}
    return result


def is_installed() -> bool:
    """핵데이터가 설치되어 있는지 확인한다.

    resolve_cross_sections_path 우선순위에 따라
    명시경로, 환경변수, nucdata/ 자동탐색을 시도한다.

    Returns:
        설치 여부.
    """
    try:
        resolve_cross_sections_path()
        return True
    except NucdataNotFoundError:
        return False
