"""케이스 매니저.

케이스 생성, 목록 조회, 상태/결과 조회, 필터링을 통합 관리한다.
case_folder, status, result_parser, kpi_calculator 모듈을 조합하여
케이스 라이프사이클 전체를 관리하는 고수준 인터페이스를 제공한다.
"""

from __future__ import annotations

import builtins
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.armi_layer.case_folder import (
    CASE_DIR_PATTERN,
    create_case,
    load_case_config,
    load_run_status,
)
from src.armi_layer.models import CaseConfig, RunStatus, SimulationResult, StatusType
from src.openmc_layer.kpi_calculator import load_kpi
from src.openmc_layer.result_parser import StatepointNotFoundError, parse_results

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseInfo:
    """케이스 상세 정보.

    Attributes:
        case_id: 케이스 폴더명 (예: case_0001).
        case_dir: 케이스 폴더 절대 경로.
        config: 케이스 설정.
        status: 실행 상태.
        result: 시뮬레이션 결과. 완료되지 않았으면 None.
        kpi: KPI 딕셔너리. 없으면 None.
    """

    case_id: str
    case_dir: Path
    config: CaseConfig
    status: RunStatus
    result: SimulationResult | None = field(default=None)
    kpi: dict[str, float | str | None] | None = field(default=None)


class CaseManager:
    """케이스 라이프사이클 통합 관리자.

    runs 디렉토리 내의 모든 케이스를 생성, 조회, 필터링한다.

    Args:
        runs_dir: 케이스들이 저장되는 상위 디렉토리.
    """

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    @property
    def runs_dir(self) -> Path:
        """케이스 상위 디렉토리."""
        return self._runs_dir

    def create(self, config: CaseConfig) -> str:
        """새 케이스를 생성한다.

        Args:
            config: 케이스 설정.

        Returns:
            생성된 케이스 ID (예: case_0001).
        """
        case_dir = create_case(config, self._runs_dir)
        case_id = case_dir.name

        logger.info("케이스 생성: %s", case_id)
        return case_id

    def list(self, *, status_filter: StatusType | None = None) -> list[str]:
        """케이스 목록을 반환한다.

        Args:
            status_filter: 특정 상태의 케이스만 필터링. None이면 전체.

        Returns:
            케이스 ID 목록 (정렬됨).
        """
        if not self._runs_dir.exists():
            return []

        case_ids: list[str] = []

        for entry in sorted(self._runs_dir.iterdir()):
            if entry.is_dir() and CASE_DIR_PATTERN.match(entry.name):
                if status_filter is not None:
                    try:
                        run_status = load_run_status(entry)
                        if run_status.status != status_filter:
                            continue
                    except FileNotFoundError:
                        continue

                case_ids.append(entry.name)

        return case_ids

    def get(self, case_id: str) -> CaseInfo:
        """케이스 상세 정보를 반환한다.

        config, status를 필수로 로드하고,
        result와 kpi는 존재할 때만 로드한다.

        Args:
            case_id: 케이스 ID (예: case_0001).

        Returns:
            케이스 상세 정보.

        Raises:
            FileNotFoundError: 케이스가 존재하지 않을 때.
        """
        case_dir = self._runs_dir / case_id

        if not case_dir.is_dir():
            raise FileNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

        config = load_case_config(case_dir)
        status = load_run_status(case_dir)

        # 결과 로드 시도 (없으면 None)
        result = self._try_load_result(case_dir)

        # KPI 로드 시도 (없으면 None)
        kpi = self._try_load_kpi(case_dir)

        return CaseInfo(
            case_id=case_id,
            case_dir=case_dir,
            config=config,
            status=status,
            result=result,
            kpi=kpi,
        )

    def get_by_status(self, status: StatusType) -> builtins.list[CaseInfo]:
        """특정 상태의 케이스 상세 목록을 반환한다.

        Args:
            status: 필터링할 상태.

        Returns:
            CaseInfo 목록.
        """
        case_ids = self.list(status_filter=status)
        return [self.get(cid) for cid in case_ids]

    def count(self) -> dict[str, int]:
        """상태별 케이스 수를 반환한다.

        Returns:
            {상태: 개수} 딕셔너리.
        """
        counts: dict[str, int] = {s.value: 0 for s in StatusType}

        for case_id in self.list():
            case_dir = self._runs_dir / case_id
            try:
                status = load_run_status(case_dir)
                counts[status.status.value] += 1
            except FileNotFoundError:
                pass

        return counts

    @staticmethod
    def _try_load_result(case_dir: Path) -> SimulationResult | None:
        """시뮬레이션 결과 로드를 시도한다. 실패 시 None."""
        try:
            return parse_results(case_dir)
        except (StatepointNotFoundError, ValueError):
            return None

    @staticmethod
    def _try_load_kpi(case_dir: Path) -> dict[str, float | str | None] | None:
        """KPI 로드를 시도한다. 실패 시 None."""
        try:
            return load_kpi(case_dir)
        except FileNotFoundError:
            return None
