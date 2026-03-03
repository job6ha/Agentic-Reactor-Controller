"""결과 수집기 테스트.

다수 케이스 결과를 DataFrame으로 수집하고 CSV/JSON 내보내기를 검증한다.
"""

import json
from pathlib import Path

import h5py
import pandas as pd
import pytest

from src.armi_layer.case_manager import CaseManager
from src.armi_layer.models import (
    CaseConfig,
    MaterialParams,
    SimulationResult,
    StatusType,
)
from src.armi_layer.result_collector import (
    EXCLUDE_FAILED,
    INCLUDE_FAILED,
    collect_results,
    export_csv,
    export_json,
)
from src.openmc_layer.kpi_calculator import calculate_kpi, save_kpi
from src.run_manager.status import update_status


def _create_mock_statepoint(case_dir: Path, keff: float = 1.00234) -> None:
    """테스트용 mock statepoint 생성."""
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    with h5py.File(output_dir / "statepoint.100.h5", "w") as f:
        f.create_dataset("k_combined", data=[keff, 0.00056])
        f.attrs["n_batches"] = 100
        f.attrs["runtime"] = 45.3


@pytest.fixture()
def manager(tmp_path: Path) -> CaseManager:
    """3개 케이스가 있는 CaseManager."""
    runs_dir = tmp_path / "runs"
    mgr = CaseManager(runs_dir)

    # case_0001: done + statepoint + kpi
    c1 = mgr.create(
        CaseConfig(
            name="enr_2.0",
            materials=MaterialParams(fuel_enrichment=2.0),
        )
    )
    case_dir1 = runs_dir / c1
    update_status(case_dir1, StatusType.DONE)
    _create_mock_statepoint(case_dir1, keff=0.98)
    save_kpi(
        calculate_kpi(SimulationResult(keff=0.98, keff_std=0.001, runtime=30.0)),
        case_dir1,
    )

    # case_0002: done + statepoint + kpi
    c2 = mgr.create(
        CaseConfig(
            name="enr_4.5",
            materials=MaterialParams(fuel_enrichment=4.5),
        )
    )
    case_dir2 = runs_dir / c2
    update_status(case_dir2, StatusType.DONE)
    _create_mock_statepoint(case_dir2, keff=1.05)
    save_kpi(
        calculate_kpi(SimulationResult(keff=1.05, keff_std=0.0008, runtime=40.0)),
        case_dir2,
    )

    # case_0003: failed, no statepoint
    c3 = mgr.create(
        CaseConfig(
            name="enr_failed",
            materials=MaterialParams(fuel_enrichment=5.0),
        )
    )
    case_dir3 = runs_dir / c3
    update_status(case_dir3, StatusType.FAILED, error_message="timeout")

    return mgr


class TestCollectResults:
    """collect_results 함수 테스트."""

    def test_returns_dataframe(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        assert isinstance(df, pd.DataFrame)

    def test_all_cases_collected(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        assert len(df) == 3

    def test_columns_present(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        expected_cols = [
            "case_id",
            "status",
            "name",
            "fuel_enrichment",
            "pitch",
            "keff",
            "keff_std",
            "criticality",
            "peaking_factor",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_parameter_values(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        enrichments = df["fuel_enrichment"].tolist()
        assert 2.0 in enrichments
        assert 4.5 in enrichments
        assert 5.0 in enrichments

    def test_keff_from_statepoint(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        done_df = df[df["status"] == "done"]
        keff_values = done_df["keff"].tolist()
        assert any(abs(k - 0.98) < 0.01 for k in keff_values)
        assert any(abs(k - 1.05) < 0.01 for k in keff_values)

    def test_failed_case_has_none_results(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        failed_row = df[df["status"] == "failed"].iloc[0]
        assert pd.isna(failed_row["keff"])

    def test_exclude_failed(self, manager: CaseManager) -> None:
        df = collect_results(manager, failed_mode=EXCLUDE_FAILED)
        assert len(df) == 2
        assert "failed" not in df["status"].values

    def test_include_failed(self, manager: CaseManager) -> None:
        df = collect_results(manager, failed_mode=INCLUDE_FAILED)
        assert len(df) == 3

    def test_specific_case_ids(self, manager: CaseManager) -> None:
        df = collect_results(manager, case_ids=["case_0001"])
        assert len(df) == 1
        assert df.iloc[0]["case_id"] == "case_0001"

    def test_kpi_columns(self, manager: CaseManager) -> None:
        df = collect_results(manager)
        done_df = df[df["status"] == "done"]
        assert done_df["criticality"].notna().all()

    def test_empty_manager(self, tmp_path: Path) -> None:
        mgr = CaseManager(tmp_path / "empty_runs")
        df = collect_results(mgr)
        assert len(df) == 0

    def test_nonexistent_case_id_skipped(self, manager: CaseManager) -> None:
        df = collect_results(manager, case_ids=["case_9999", "case_0001"])
        assert len(df) == 1


class TestExportCsv:
    """export_csv 함수 테스트."""

    def test_creates_file(self, manager: CaseManager, tmp_path: Path) -> None:
        df = collect_results(manager)
        path = tmp_path / "results.csv"
        export_csv(df, path)
        assert path.exists()

    def test_csv_readable(self, manager: CaseManager, tmp_path: Path) -> None:
        df = collect_results(manager)
        path = tmp_path / "results.csv"
        export_csv(df, path)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(df)

    def test_csv_columns_match(
        self,
        manager: CaseManager,
        tmp_path: Path,
    ) -> None:
        df = collect_results(manager)
        path = tmp_path / "results.csv"
        export_csv(df, path)
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(df.columns)

    def test_creates_parent_dirs(
        self,
        manager: CaseManager,
        tmp_path: Path,
    ) -> None:
        df = collect_results(manager)
        path = tmp_path / "sub" / "dir" / "results.csv"
        export_csv(df, path)
        assert path.exists()


class TestExportJson:
    """export_json 함수 테스트."""

    def test_creates_file(self, manager: CaseManager, tmp_path: Path) -> None:
        df = collect_results(manager)
        path = tmp_path / "results.json"
        export_json(df, path)
        assert path.exists()

    def test_json_valid(self, manager: CaseManager, tmp_path: Path) -> None:
        df = collect_results(manager)
        path = tmp_path / "results.json"
        export_json(df, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == len(df)

    def test_json_contains_case_ids(
        self,
        manager: CaseManager,
        tmp_path: Path,
    ) -> None:
        df = collect_results(manager)
        path = tmp_path / "results.json"
        export_json(df, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        case_ids = [r["case_id"] for r in data]
        assert "case_0001" in case_ids
