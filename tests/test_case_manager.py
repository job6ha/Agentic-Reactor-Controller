"""케이스 매니저 테스트.

CaseManager의 생성, 조회, 필터링 기능을 검증한다.
"""

import json
from pathlib import Path

import h5py
import pytest

from src.armi_layer.case_manager import CaseInfo, CaseManager
from src.armi_layer.models import (
    CaseConfig,
    MaterialParams,
    RunStatus,
    SimulationSettings,
    StatusType,
)
from src.openmc_layer.kpi_calculator import calculate_kpi, save_kpi
from src.armi_layer.models import SimulationResult


def _set_case_status(case_dir: Path, status: StatusType) -> None:
    """테스트용: 케이스 상태를 직접 변경."""
    status_path = case_dir / "meta" / "status.json"
    run_status = RunStatus(status=status)
    status_path.write_text(
        run_status.model_dump_json(indent=2), encoding="utf-8",
    )


def _create_mock_statepoint(case_dir: Path) -> None:
    """테스트용: mock statepoint 생성."""
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    sp_path = output_dir / "statepoint.100.h5"
    with h5py.File(sp_path, "w") as f:
        f.create_dataset("k_combined", data=[1.00234, 0.00056])
        f.attrs["n_batches"] = 100
        f.attrs["runtime"] = 45.3


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    """테스트용 runs 디렉토리."""
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture()
def manager(runs_dir: Path) -> CaseManager:
    """CaseManager 인스턴스."""
    return CaseManager(runs_dir)


class TestCaseManagerCreate:
    """CaseManager.create 테스트."""

    def test_create_returns_case_id(self, manager: CaseManager) -> None:
        case_id = manager.create(CaseConfig())
        assert case_id == "case_0001"

    def test_create_increments(self, manager: CaseManager) -> None:
        manager.create(CaseConfig())
        case_id = manager.create(CaseConfig())
        assert case_id == "case_0002"

    def test_create_with_custom_config(self, manager: CaseManager) -> None:
        config = CaseConfig(
            name="test_case",
            materials=MaterialParams(fuel_enrichment=4.5),
        )
        case_id = manager.create(config)
        info = manager.get(case_id)
        assert info.config.name == "test_case"
        assert info.config.materials.fuel_enrichment == 4.5

    def test_create_sets_queued_status(self, manager: CaseManager) -> None:
        case_id = manager.create(CaseConfig())
        info = manager.get(case_id)
        assert info.status.status == StatusType.QUEUED


class TestCaseManagerList:
    """CaseManager.list 테스트."""

    def test_empty_list(self, manager: CaseManager) -> None:
        assert manager.list() == []

    def test_list_created_cases(self, manager: CaseManager) -> None:
        manager.create(CaseConfig())
        manager.create(CaseConfig())
        manager.create(CaseConfig())
        cases = manager.list()
        assert cases == ["case_0001", "case_0002", "case_0003"]

    def test_list_sorted(self, manager: CaseManager) -> None:
        for _ in range(5):
            manager.create(CaseConfig())
        cases = manager.list()
        assert cases == sorted(cases)

    def test_list_filter_by_status(self, manager: CaseManager) -> None:
        c1 = manager.create(CaseConfig())
        c2 = manager.create(CaseConfig())
        c3 = manager.create(CaseConfig())

        # c1=done, c2=failed, c3=queued
        case_dir1 = manager.runs_dir / c1
        case_dir2 = manager.runs_dir / c2
        _set_case_status(case_dir1, StatusType.DONE)
        _set_case_status(case_dir2, StatusType.FAILED)

        done_cases = manager.list(status_filter=StatusType.DONE)
        assert done_cases == [c1]

        failed_cases = manager.list(status_filter=StatusType.FAILED)
        assert failed_cases == [c2]

        queued_cases = manager.list(status_filter=StatusType.QUEUED)
        assert queued_cases == [c3]

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        mgr = CaseManager(tmp_path / "nonexistent")
        assert mgr.list() == []


class TestCaseManagerGet:
    """CaseManager.get 테스트."""

    def test_get_returns_case_info(self, manager: CaseManager) -> None:
        case_id = manager.create(CaseConfig())
        info = manager.get(case_id)
        assert isinstance(info, CaseInfo)
        assert info.case_id == case_id

    def test_get_includes_config(self, manager: CaseManager) -> None:
        config = CaseConfig(
            name="enrichment_test",
            settings=SimulationSettings(batches=200),
        )
        case_id = manager.create(config)
        info = manager.get(case_id)
        assert info.config.name == "enrichment_test"
        assert info.config.settings.batches == 200

    def test_get_includes_status(self, manager: CaseManager) -> None:
        case_id = manager.create(CaseConfig())
        info = manager.get(case_id)
        assert info.status.status == StatusType.QUEUED

    def test_get_result_none_when_no_statepoint(
        self, manager: CaseManager,
    ) -> None:
        case_id = manager.create(CaseConfig())
        info = manager.get(case_id)
        assert info.result is None

    def test_get_result_when_statepoint_exists(
        self, manager: CaseManager,
    ) -> None:
        case_id = manager.create(CaseConfig())
        case_dir = manager.runs_dir / case_id
        _create_mock_statepoint(case_dir)

        info = manager.get(case_id)
        assert info.result is not None
        assert abs(info.result.keff - 1.00234) < 1e-10

    def test_get_kpi_none_when_no_kpi_file(
        self, manager: CaseManager,
    ) -> None:
        case_id = manager.create(CaseConfig())
        info = manager.get(case_id)
        assert info.kpi is None

    def test_get_kpi_when_saved(self, manager: CaseManager) -> None:
        case_id = manager.create(CaseConfig())
        case_dir = manager.runs_dir / case_id

        # KPI 저장
        sim_result = SimulationResult(
            keff=1.00234, keff_std=0.00056, runtime=45.3,
        )
        kpi = calculate_kpi(sim_result)
        save_kpi(kpi, case_dir)

        info = manager.get(case_id)
        assert info.kpi is not None
        assert info.kpi["keff"] == 1.00234

    def test_get_nonexistent_raises(self, manager: CaseManager) -> None:
        with pytest.raises(FileNotFoundError, match="케이스를 찾을 수 없습니다"):
            manager.get("case_9999")

    def test_case_info_frozen(self, manager: CaseManager) -> None:
        case_id = manager.create(CaseConfig())
        info = manager.get(case_id)
        with pytest.raises(AttributeError):
            info.case_id = "other"  # type: ignore[misc]


class TestCaseManagerGetByStatus:
    """CaseManager.get_by_status 테스트."""

    def test_get_by_status(self, manager: CaseManager) -> None:
        c1 = manager.create(CaseConfig())
        c2 = manager.create(CaseConfig())
        manager.create(CaseConfig())

        _set_case_status(manager.runs_dir / c1, StatusType.DONE)
        _set_case_status(manager.runs_dir / c2, StatusType.DONE)

        done_infos = manager.get_by_status(StatusType.DONE)
        assert len(done_infos) == 2
        assert all(i.status.status == StatusType.DONE for i in done_infos)

    def test_get_by_status_empty(self, manager: CaseManager) -> None:
        manager.create(CaseConfig())
        done_infos = manager.get_by_status(StatusType.DONE)
        assert done_infos == []


class TestCaseManagerCount:
    """CaseManager.count 테스트."""

    def test_count_empty(self, manager: CaseManager) -> None:
        counts = manager.count()
        assert all(v == 0 for v in counts.values())

    def test_count_by_status(self, manager: CaseManager) -> None:
        c1 = manager.create(CaseConfig())
        c2 = manager.create(CaseConfig())
        c3 = manager.create(CaseConfig())
        manager.create(CaseConfig())

        _set_case_status(manager.runs_dir / c1, StatusType.DONE)
        _set_case_status(manager.runs_dir / c2, StatusType.DONE)
        _set_case_status(manager.runs_dir / c3, StatusType.FAILED)

        counts = manager.count()
        assert counts["done"] == 2
        assert counts["failed"] == 1
        assert counts["queued"] == 1
        assert counts["running"] == 0
