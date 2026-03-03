"""KPI 계산기 테스트.

SimulationResult → KPI 변환 로직을 검증한다.
"""

import json
from pathlib import Path

import pytest

from src.armi_layer.models import SimulationResult, TallyResult
from src.openmc_layer.kpi_calculator import (
    calculate_keff_kpi,
    calculate_kpi,
    calculate_peaking_factor,
    load_kpi,
    save_kpi,
)


@pytest.fixture()
def critical_result() -> SimulationResult:
    """keff ≈ 1.0인 critical 결과."""
    return SimulationResult(
        keff=1.00234,
        keff_std=0.00056,
        runtime=45.3,
        batches_completed=100,
        tallies=[
            TallyResult(
                name="flux_tally",
                scores=["flux"],
                mean=[5.0e12, 4.5e12, 5.5e12, 4.0e12],
                std_dev=[1e10, 1e10, 1e10, 1e10],
            ),
            TallyResult(
                name="fission_tally",
                scores=["fission", "nu-fission"],
                mean=[1.2e13, 2.9e13],
                std_dev=[1e11, 2e11],
            ),
        ],
    )


@pytest.fixture()
def supercritical_result() -> SimulationResult:
    """keff > 1.0 + margin인 supercritical 결과."""
    return SimulationResult(
        keff=1.08,
        keff_std=0.001,
        runtime=30.0,
        batches_completed=50,
    )


@pytest.fixture()
def subcritical_result() -> SimulationResult:
    """keff < 1.0 - margin인 subcritical 결과."""
    return SimulationResult(
        keff=0.92,
        keff_std=0.002,
        runtime=60.0,
        batches_completed=80,
    )


@pytest.fixture()
def no_tally_result() -> SimulationResult:
    """탈리 없는 결과."""
    return SimulationResult(
        keff=1.01,
        keff_std=0.001,
        runtime=20.0,
        batches_completed=50,
    )


class TestCalculateKeffKpi:
    """calculate_keff_kpi 함수 테스트."""

    def test_critical(self, critical_result: SimulationResult) -> None:
        kpi = calculate_keff_kpi(critical_result)
        assert kpi["criticality"] == "critical"
        assert kpi["keff"] == critical_result.keff
        assert kpi["keff_std"] == critical_result.keff_std

    def test_supercritical(self, supercritical_result: SimulationResult) -> None:
        kpi = calculate_keff_kpi(supercritical_result)
        assert kpi["criticality"] == "supercritical"

    def test_subcritical(self, subcritical_result: SimulationResult) -> None:
        kpi = calculate_keff_kpi(subcritical_result)
        assert kpi["criticality"] == "subcritical"

    def test_deviation_calculated(self, critical_result: SimulationResult) -> None:
        kpi = calculate_keff_kpi(critical_result)
        expected = round(critical_result.keff - 1.0, 6)
        assert kpi["keff_deviation"] == expected

    def test_custom_margin(self) -> None:
        """마진 조정 시 판정이 변경됨."""
        result = SimulationResult(
            keff=1.08, keff_std=0.001, runtime=10.0,
        )
        # 기본 마진(0.05)에서는 supercritical
        kpi_default = calculate_keff_kpi(result)
        assert kpi_default["criticality"] == "supercritical"

        # 마진 0.1으로 확대하면 critical
        kpi_wide = calculate_keff_kpi(result, margin=0.1)
        assert kpi_wide["criticality"] == "critical"

    def test_within_boundary(self) -> None:
        """마진 내 값은 critical로 판정."""
        result = SimulationResult(
            keff=1.049, keff_std=0.001, runtime=10.0,
        )
        kpi = calculate_keff_kpi(result, margin=0.05)
        assert kpi["criticality"] == "critical"

    def test_outside_boundary(self) -> None:
        """마진을 초과하면 supercritical."""
        result = SimulationResult(
            keff=1.06, keff_std=0.001, runtime=10.0,
        )
        kpi = calculate_keff_kpi(result, margin=0.05)
        assert kpi["criticality"] == "supercritical"


class TestCalculatePeakingFactor:
    """calculate_peaking_factor 함수 테스트."""

    def test_peaking_from_fission_tally(
        self, critical_result: SimulationResult,
    ) -> None:
        kpi = calculate_peaking_factor(critical_result)
        assert kpi["peaking_factor"] is not None
        assert kpi["peaking_factor"] > 0
        # fission 탈리가 우선 선택됨
        assert kpi["peaking_factor_tally"] == "fission_tally"

    def test_peaking_value(self) -> None:
        """peaking factor = max / avg 검증."""
        result = SimulationResult(
            keff=1.0, keff_std=0.001, runtime=10.0,
            tallies=[
                TallyResult(
                    name="power",
                    scores=["fission"],
                    mean=[1.0, 2.0, 3.0, 4.0],
                    std_dev=[0.1, 0.1, 0.1, 0.1],
                ),
            ],
        )
        kpi = calculate_peaking_factor(result)
        # avg = 2.5, max = 4.0 → peaking = 1.6
        assert kpi["peaking_factor"] == pytest.approx(1.6, rel=1e-5)

    def test_uniform_distribution(self) -> None:
        """균일 분포 시 peaking factor = 1.0."""
        result = SimulationResult(
            keff=1.0, keff_std=0.001, runtime=10.0,
            tallies=[
                TallyResult(
                    name="power",
                    scores=["fission"],
                    mean=[5.0, 5.0, 5.0, 5.0],
                    std_dev=[0.1, 0.1, 0.1, 0.1],
                ),
            ],
        )
        kpi = calculate_peaking_factor(result)
        assert kpi["peaking_factor"] == pytest.approx(1.0, rel=1e-5)

    def test_no_tallies(self, no_tally_result: SimulationResult) -> None:
        kpi = calculate_peaking_factor(no_tally_result)
        assert kpi["peaking_factor"] is None
        assert kpi["peaking_factor_tally"] is None

    def test_flux_tally_used_when_no_fission(self) -> None:
        """fission 탈리가 없으면 flux 탈리 사용."""
        result = SimulationResult(
            keff=1.0, keff_std=0.001, runtime=10.0,
            tallies=[
                TallyResult(
                    name="flux_only",
                    scores=["flux"],
                    mean=[3.0, 6.0],
                    std_dev=[0.1, 0.1],
                ),
            ],
        )
        kpi = calculate_peaking_factor(result)
        assert kpi["peaking_factor_tally"] == "flux_only"
        # avg = 4.5, max = 6.0 → peaking ≈ 1.333
        assert kpi["peaking_factor"] == pytest.approx(6.0 / 4.5, rel=1e-5)

    def test_irrelevant_tally_ignored(self) -> None:
        """관련 없는 스코어만 있으면 peaking factor 없음."""
        result = SimulationResult(
            keff=1.0, keff_std=0.001, runtime=10.0,
            tallies=[
                TallyResult(
                    name="other",
                    scores=["heating"],
                    mean=[1.0, 2.0],
                    std_dev=[0.1, 0.1],
                ),
            ],
        )
        kpi = calculate_peaking_factor(result)
        assert kpi["peaking_factor"] is None


class TestCalculateKpi:
    """calculate_kpi 통합 함수 테스트."""

    def test_all_kpis_present(self, critical_result: SimulationResult) -> None:
        kpi = calculate_kpi(critical_result)
        assert "keff" in kpi
        assert "keff_std" in kpi
        assert "keff_deviation" in kpi
        assert "criticality" in kpi
        assert "peaking_factor" in kpi
        assert "batches_completed" in kpi
        assert "runtime" in kpi

    def test_meta_info(self, critical_result: SimulationResult) -> None:
        kpi = calculate_kpi(critical_result)
        assert kpi["batches_completed"] == 100
        assert kpi["runtime"] == 45.3

    def test_custom_margin_passed(self) -> None:
        result = SimulationResult(
            keff=1.08, keff_std=0.001, runtime=10.0,
        )
        kpi = calculate_kpi(result, keff_margin=0.1)
        assert kpi["criticality"] == "critical"

    def test_no_tallies_still_works(
        self, no_tally_result: SimulationResult,
    ) -> None:
        kpi = calculate_kpi(no_tally_result)
        assert kpi["peaking_factor"] is None
        assert kpi["keff"] == 1.01


class TestSaveAndLoadKpi:
    """save_kpi / load_kpi 테스트."""

    def test_save_creates_file(
        self, tmp_path: Path, critical_result: SimulationResult,
    ) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        kpi = calculate_kpi(critical_result)
        kpi_path = save_kpi(kpi, case_dir)
        assert kpi_path.exists()
        assert kpi_path.name == "kpi.json"

    def test_save_creates_meta_dir(
        self, tmp_path: Path, critical_result: SimulationResult,
    ) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        kpi = calculate_kpi(critical_result)
        save_kpi(kpi, case_dir)
        assert (case_dir / "meta").is_dir()

    def test_saved_content_is_valid_json(
        self, tmp_path: Path, critical_result: SimulationResult,
    ) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        kpi = calculate_kpi(critical_result)
        kpi_path = save_kpi(kpi, case_dir)
        loaded = json.loads(kpi_path.read_text(encoding="utf-8"))
        assert loaded["keff"] == kpi["keff"]

    def test_roundtrip(
        self, tmp_path: Path, critical_result: SimulationResult,
    ) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        kpi = calculate_kpi(critical_result)
        save_kpi(kpi, case_dir)
        loaded = load_kpi(case_dir)
        assert loaded == kpi

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_empty"
        case_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="KPI 파일"):
            load_kpi(case_dir)

    def test_overwrite_existing(
        self, tmp_path: Path, critical_result: SimulationResult,
    ) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        kpi1 = calculate_kpi(critical_result)
        save_kpi(kpi1, case_dir)

        new_result = SimulationResult(
            keff=0.95, keff_std=0.002, runtime=80.0, batches_completed=200,
        )
        kpi2 = calculate_kpi(new_result)
        save_kpi(kpi2, case_dir)

        loaded = load_kpi(case_dir)
        assert loaded["keff"] == 0.95
