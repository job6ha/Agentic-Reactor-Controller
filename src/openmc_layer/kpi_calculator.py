"""핵심 성능 지표(KPI) 계산기.

SimulationResult에서 keff criticality 판정, peaking factor 등
핵심 성능 지표를 추출하고, JSON 파일로 저장한다.
"""

import json
import logging
from pathlib import Path

from src.armi_layer.models import SimulationResult

logger = logging.getLogger(__name__)

# keff criticality 판정 기본 마진
DEFAULT_KEFF_MARGIN = 0.05

# KPI JSON 파일명
KPI_FILENAME = "kpi.json"


def calculate_keff_kpi(
    result: SimulationResult,
    *,
    margin: float = DEFAULT_KEFF_MARGIN,
) -> dict[str, float | str]:
    """keff 관련 KPI를 계산한다.

    Args:
        result: 시뮬레이션 결과.
        margin: criticality 판정 마진. |keff - 1.0| <= margin이면 critical.

    Returns:
        keff KPI 딕셔너리.
    """
    keff = result.keff
    keff_std = result.keff_std
    deviation = keff - 1.0

    if abs(deviation) <= margin:
        criticality = "critical"
    elif deviation > margin:
        criticality = "supercritical"
    else:
        criticality = "subcritical"

    return {
        "keff": keff,
        "keff_std": keff_std,
        "keff_deviation": round(deviation, 6),
        "criticality": criticality,
    }


def calculate_peaking_factor(result: SimulationResult) -> dict[str, float | None]:
    """탈리 데이터에서 peaking factor를 계산한다.

    power/fission 관련 탈리의 mean 값에서
    최대값/평균값 비율로 peaking factor를 산출한다.

    Args:
        result: 시뮬레이션 결과.

    Returns:
        peaking factor KPI 딕셔너리.
    """
    # fission 또는 flux 탈리에서 peaking factor 계산
    target_scores = {"fission", "nu-fission", "flux"}
    best_tally = None

    for tally in result.tallies:
        matching = [s for s in tally.scores if s in target_scores]
        if matching and tally.mean:
            best_tally = tally
            # fission 우선
            if "fission" in tally.scores or "nu-fission" in tally.scores:
                break

    if best_tally is None or not best_tally.mean:
        logger.debug("peaking factor 계산 불가: 적합한 탈리 없음")
        return {
            "peaking_factor": None,
            "peaking_factor_tally": None,
        }

    mean_values = best_tally.mean
    avg = sum(mean_values) / len(mean_values)

    if avg <= 0:
        return {
            "peaking_factor": None,
            "peaking_factor_tally": best_tally.name,
        }

    peaking = max(mean_values) / avg

    return {
        "peaking_factor": round(peaking, 6),
        "peaking_factor_tally": best_tally.name,
    }


def calculate_kpi(
    result: SimulationResult,
    *,
    keff_margin: float = DEFAULT_KEFF_MARGIN,
) -> dict[str, float | str | None]:
    """SimulationResult에서 전체 KPI를 계산한다.

    Args:
        result: 시뮬레이션 결과.
        keff_margin: keff criticality 판정 마진.

    Returns:
        전체 KPI 딕셔너리.
    """
    kpi: dict[str, float | str | None] = {}

    # keff KPI
    keff_kpi = calculate_keff_kpi(result, margin=keff_margin)
    kpi.update(keff_kpi)

    # peaking factor KPI
    peaking_kpi = calculate_peaking_factor(result)
    kpi.update(peaking_kpi)

    # 메타 정보
    kpi["batches_completed"] = result.batches_completed
    kpi["runtime"] = result.runtime

    logger.info(
        "KPI 계산 완료: keff=%.5f, criticality=%s, peaking=%s",
        kpi["keff"],
        kpi["criticality"],
        kpi.get("peaking_factor"),
    )

    return kpi


def save_kpi(kpi: dict, case_path: Path) -> Path:
    """KPI를 JSON 파일로 저장한다.

    case_path/meta/kpi.json에 저장한다.

    Args:
        kpi: KPI 딕셔너리.
        case_path: 케이스 폴더 경로.

    Returns:
        저장된 JSON 파일 경로.
    """
    meta_dir = case_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    kpi_path = meta_dir / KPI_FILENAME
    kpi_path.write_text(
        json.dumps(kpi, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("KPI 저장: %s", kpi_path)
    return kpi_path


def load_kpi(case_path: Path) -> dict:
    """저장된 KPI를 읽어온다.

    Args:
        case_path: 케이스 폴더 경로.

    Returns:
        KPI 딕셔너리.

    Raises:
        FileNotFoundError: kpi.json이 없을 때.
    """
    kpi_path = case_path / "meta" / KPI_FILENAME

    if not kpi_path.exists():
        raise FileNotFoundError(f"KPI 파일을 찾을 수 없습니다: {kpi_path}")

    return json.loads(kpi_path.read_text(encoding="utf-8"))
