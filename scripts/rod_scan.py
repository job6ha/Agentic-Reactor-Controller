"""제어봉 위치별 keff 스캔 스크립트.

각 블루프린트에 대해 여러 제어봉 위치에서 keff를 측정하여
임계 위치(keff≈1.0)를 찾는다.

Usage (Docker):
    docker compose run --rm openmc uv run python scripts/rod_scan.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.armi_layer.case_folder import create_case
from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    RunConfig,
    SimulationSettings,
)
from src.openmc_layer.input_generator import generate_input
from src.openmc_layer.result_parser import parse_results
from src.run_manager.status import run_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rod_scan")

# 빠른 스캔용 설정 (정밀도 약간 낮지만 빠름)
SCAN_SETTINGS = SimulationSettings(batches=80, inactive=20, particles=5000)
RUNS_DIR = Path("runs/rod_scan")
RUN_CONFIG = RunConfig(omp_threads=4, timeout=300, max_retries=1)

# 스캔할 제어봉 위치들
SCAN_POSITIONS = [0, 15, 30, 50, 75, 100, 125, 150, 175, 200, 228]

# 블루프린트 로드
BLUEPRINTS_PATH = Path("configs/blueprints.json")


def scan_blueprint(
    name: str,
    geometry_overrides: dict,
    material_overrides: dict,
) -> list[dict]:
    """한 블루프린트에 대해 여러 제어봉 위치를 스캔한다."""
    results = []

    for rod_pos in SCAN_POSITIONS:
        logger.info("[%s] rod_position=%d 시뮬레이션 시작", name, rod_pos)

        geo_params = {**geometry_overrides, "extra_params": {"rod_position": rod_pos}}
        config = CaseConfig(
            name=f"scan_{name}_rod{rod_pos}",
            description=f"{name} rod scan at position {rod_pos}",
            geometry=GeometryParams(**{
                k: v for k, v in geo_params.items()
                if k in GeometryParams.model_fields
            }),
            materials=MaterialParams(**material_overrides),
            settings=SCAN_SETTINGS,
        )
        # extra_params는 별도 설정
        config.geometry.extra_params = {"rod_position": rod_pos}

        case_dir = create_case(config, runs_dir=RUNS_DIR)
        generate_input(config, case_dir)

        run_result = run_case(RUN_CONFIG, case_dir)
        if not run_result.success:
            logger.warning("[%s] rod=%d 실행 실패, 건너뜀", name, rod_pos)
            continue

        result = parse_results(case_dir)
        logger.info(
            "[%s] rod=%d → keff=%.5f±%.5f",
            name, rod_pos, result.keff, result.keff_std,
        )
        results.append({
            "rod_position": rod_pos,
            "keff": result.keff,
            "keff_std": result.keff_std,
        })

    return results


def find_critical_position(scan_results: list[dict]) -> int:
    """keff가 1.0에 가장 가까운 위치를 찾고, 보간하여 추정한다."""
    # keff=1.0과의 차이가 가장 작은 포인트
    closest = min(scan_results, key=lambda r: abs(r["keff"] - 1.0))

    # 1.0을 사이에 두는 두 포인트를 찾아 선형 보간
    sorted_results = sorted(scan_results, key=lambda r: r["rod_position"])
    for i in range(len(sorted_results) - 1):
        k1 = sorted_results[i]["keff"]
        k2 = sorted_results[i + 1]["keff"]
        p1 = sorted_results[i]["rod_position"]
        p2 = sorted_results[i + 1]["rod_position"]

        if (k1 - 1.0) * (k2 - 1.0) <= 0:  # 1.0을 사이에 둠
            # 선형 보간
            if abs(k2 - k1) > 1e-6:
                frac = (1.0 - k1) / (k2 - k1)
                critical_pos = int(round(p1 + frac * (p2 - p1)))
                return max(0, min(228, critical_pos))

    return closest["rod_position"]


def main() -> None:
    """메인 실행."""
    with open(BLUEPRINTS_PATH) as f:
        bp_data = json.load(f)

    all_results = {}
    critical_positions = {}

    for bp in bp_data["blueprints"]:
        name = bp["name"]
        geo = {k: v for k, v in bp.get("geometry", {}).items()}
        mat = {k: v for k, v in bp.get("materials", {}).items()}

        logger.info("=" * 60)
        logger.info("블루프린트 스캔: %s", name)
        logger.info("=" * 60)

        scan_results = scan_blueprint(name, geo, mat)
        all_results[name] = scan_results

        critical_pos = find_critical_position(scan_results)
        critical_positions[name] = critical_pos

        logger.info(
            "[%s] 추정 임계 위치: rod_position=%d",
            name, critical_pos,
        )

    # 결과 저장
    output = {
        "scan_results": all_results,
        "critical_positions": critical_positions,
        "scan_positions": SCAN_POSITIONS,
    }
    output_path = RUNS_DIR / "rod_scan_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("스캔 완료! 결과: %s", output_path)
    logger.info("=" * 60)
    for name, pos in critical_positions.items():
        bp_scan = all_results[name]
        keffs = {r["rod_position"]: r["keff"] for r in bp_scan}
        logger.info(
            "  %s: 임계위치=%d (keff@0=%.4f, keff@228=%.4f)",
            name, pos,
            keffs.get(0, 0),
            keffs.get(228, 0),
        )


if __name__ == "__main__":
    main()
