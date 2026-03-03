"""결과 수집기.

다수 케이스의 SimulationResult와 KPI를 일괄 수집하여
파라미터 vs KPI 비교 DataFrame을 생성하고 CSV/JSON으로 내보낸다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.armi_layer.case_manager import CaseInfo, CaseManager
from src.armi_layer.models import StatusType

logger = logging.getLogger(__name__)

# 실패 케이스 처리 모드
INCLUDE_FAILED = "include"
EXCLUDE_FAILED = "exclude"


def _extract_row(info: CaseInfo) -> dict:
    """CaseInfo에서 DataFrame 한 행의 데이터를 추출한다.

    Args:
        info: 케이스 상세 정보.

    Returns:
        파라미터 + 결과 + KPI 딕셔너리.
    """
    row: dict = {
        "case_id": info.case_id,
        "status": info.status.status.value,
        "name": info.config.name,
    }

    # 주요 파라미터
    row["fuel_enrichment"] = info.config.materials.fuel_enrichment
    row["fuel_density"] = info.config.materials.fuel_density
    row["coolant_density"] = info.config.materials.coolant_density
    row["fuel_radius"] = info.config.geometry.fuel_radius
    row["pitch"] = info.config.geometry.pitch
    row["batches"] = info.config.settings.batches
    row["particles"] = info.config.settings.particles

    # 시뮬레이션 결과
    if info.result is not None:
        row["keff"] = info.result.keff
        row["keff_std"] = info.result.keff_std
        row["runtime"] = info.result.runtime
        row["batches_completed"] = info.result.batches_completed
    else:
        row["keff"] = None
        row["keff_std"] = None
        row["runtime"] = None
        row["batches_completed"] = None

    # KPI
    if info.kpi is not None:
        row["criticality"] = info.kpi.get("criticality")
        row["peaking_factor"] = info.kpi.get("peaking_factor")
    else:
        row["criticality"] = None
        row["peaking_factor"] = None

    return row


def collect_results(
    manager: CaseManager,
    case_ids: list[str] | None = None,
    *,
    failed_mode: str = INCLUDE_FAILED,
) -> pd.DataFrame:
    """케이스 결과를 DataFrame으로 수집한다.

    Args:
        manager: 케이스 매니저.
        case_ids: 수집할 케이스 ID 목록. None이면 전체.
        failed_mode: 실패 케이스 처리.
            "include"면 포함, "exclude"면 제외.

    Returns:
        파라미터 vs 결과 비교 DataFrame.
    """
    if case_ids is None:
        case_ids = manager.list()

    rows: list[dict] = []

    for case_id in case_ids:
        try:
            info = manager.get(case_id)
        except FileNotFoundError:
            logger.warning("케이스 로드 실패: %s", case_id)
            continue

        if (
            failed_mode == EXCLUDE_FAILED
            and info.status.status == StatusType.FAILED
        ):
            continue

        rows.append(_extract_row(info))

    df = pd.DataFrame(rows)

    logger.info("결과 수집 완료: %d건", len(df))
    return df


def export_csv(df: pd.DataFrame, path: Path) -> Path:
    """DataFrame을 CSV 파일로 내보낸다.

    Args:
        df: 결과 DataFrame.
        path: 저장 경로.

    Returns:
        저장된 파일 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("CSV 내보내기: %s (%d행)", path, len(df))
    return path


def export_json(df: pd.DataFrame, path: Path) -> Path:
    """DataFrame을 JSON 파일로 내보낸다.

    Args:
        df: 결과 DataFrame.
        path: 저장 경로.

    Returns:
        저장된 파일 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2, force_ascii=False)
    logger.info("JSON 내보내기: %s (%d행)", path, len(df))
    return path
