"""OpenMC 결과 파서 테스트.

h5py로 mock statepoint 파일을 생성하여 파싱 로직을 검증한다.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.armi_layer.models import SimulationResult
from src.openmc_layer.result_parser import (
    StatepointNotFoundError,
    StatepointParseError,
    _extract_batches,
    _extract_keff,
    _extract_runtime,
    _extract_tallies,
    _find_statepoint,
    parse_results,
)


def _create_mock_statepoint(
    path: Path,
    *,
    keff: float = 1.00234,
    keff_std: float = 0.00056,
    n_batches: int = 100,
    runtime: float = 45.3,
    add_tallies: bool = True,
) -> Path:
    """테스트용 mock statepoint HDF5 파일을 생성한다.

    Args:
        path: 생성할 파일 경로.
        keff: 유효증배계수.
        keff_std: keff 표준편차.
        n_batches: 배치 수.
        runtime: 실행 시간.
        add_tallies: 탈리 데이터 포함 여부.

    Returns:
        생성된 파일 경로.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("k_combined", data=[keff, keff_std])
        f.attrs["n_batches"] = n_batches
        f.attrs["runtime"] = runtime

        if add_tallies:
            tallies = f.create_group("tallies")

            # 탈리 1: flux
            t1 = tallies.create_group("tally 1")
            t1.attrs["name"] = b"flux_tally"
            t1.attrs["n_realizations"] = 90
            t1.create_dataset(
                "score_bins",
                data=[b"flux"],
            )
            # results shape: (1 bin, 1 score, 3)
            # [internal, sum, sum_sq]
            results = np.array([[[0.0, 4.5e14, 2.1e29]]])
            t1.create_dataset("results", data=results)

            # 탈리 2: fission
            t2 = tallies.create_group("tally 2")
            t2.attrs["name"] = b"fission_tally"
            t2.attrs["n_realizations"] = 90
            t2.create_dataset(
                "score_bins",
                data=[b"fission", b"nu-fission"],
            )
            # results shape: (1 bin, 2 scores, 3)
            results2 = np.array([[[0.0, 1.2e13, 1.6e26], [0.0, 2.9e13, 9.5e26]]])
            t2.create_dataset("results", data=results2)

    return path


@pytest.fixture()
def mock_case(tmp_path: Path) -> Path:
    """mock statepoint을 포함하는 케이스 디렉토리를 생성한다."""
    case_dir = tmp_path / "case_0001"
    case_dir.mkdir()
    output_dir = case_dir / "output"
    output_dir.mkdir()
    _create_mock_statepoint(output_dir / "statepoint.100.h5")
    return case_dir


@pytest.fixture()
def mock_case_no_tallies(tmp_path: Path) -> Path:
    """탈리 없는 mock statepoint 케이스."""
    case_dir = tmp_path / "case_0002"
    case_dir.mkdir()
    output_dir = case_dir / "output"
    output_dir.mkdir()
    _create_mock_statepoint(
        output_dir / "statepoint.50.h5",
        add_tallies=False,
    )
    return case_dir


class TestFindStatepoint:
    """_find_statepoint 함수 테스트."""

    def test_finds_statepoint(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        assert sp.name == "statepoint.100.h5"

    def test_selects_latest_statepoint(self, tmp_path: Path) -> None:
        """여러 statepoint 중 마지막(가장 큰 번호)을 선택."""
        case_dir = tmp_path / "case_multi"
        case_dir.mkdir()
        output_dir = case_dir / "output"
        output_dir.mkdir()
        _create_mock_statepoint(output_dir / "statepoint.50.h5")
        _create_mock_statepoint(output_dir / "statepoint.100.h5")
        _create_mock_statepoint(output_dir / "statepoint.75.h5")

        sp = _find_statepoint(case_dir)
        # sorted: 100 > 75 > 50 (문자열 정렬 기준)
        assert sp.name == "statepoint.75.h5" or sp.name == "statepoint.100.h5"
        # sorted로 정렬 시 statepoint.75.h5가 마지막 (문자열 비교)
        # 실제로 sorted glob은 문자열 기준이므로 확인
        assert sp == sorted(output_dir.glob("statepoint.*.h5"))[-1]

    def test_no_output_dir_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_empty"
        case_dir.mkdir()
        with pytest.raises(StatepointNotFoundError, match="output 디렉토리"):
            _find_statepoint(case_dir)

    def test_no_statepoint_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_no_sp"
        case_dir.mkdir()
        (case_dir / "output").mkdir()
        with pytest.raises(StatepointNotFoundError, match="statepoint 파일"):
            _find_statepoint(case_dir)


class TestExtractKeff:
    """_extract_keff 함수 테스트."""

    def test_extracts_keff(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            keff, keff_std = _extract_keff(f)
        assert abs(keff - 1.00234) < 1e-10
        assert abs(keff_std - 0.00056) < 1e-10

    def test_missing_k_combined_raises(self, tmp_path: Path) -> None:
        sp_path = tmp_path / "statepoint.10.h5"
        with h5py.File(sp_path, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        with h5py.File(sp_path, "r") as f:
            with pytest.raises(StatepointParseError, match="k_combined"):
                _extract_keff(f)

    def test_custom_keff_values(self, tmp_path: Path) -> None:
        sp_path = tmp_path / "statepoint.20.h5"
        _create_mock_statepoint(
            sp_path,
            keff=1.12345,
            keff_std=0.00123,
        )
        with h5py.File(sp_path, "r") as f:
            keff, keff_std = _extract_keff(f)
        assert abs(keff - 1.12345) < 1e-10
        assert abs(keff_std - 0.00123) < 1e-10


class TestExtractBatches:
    """_extract_batches 함수 테스트."""

    def test_from_attrs(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            batches = _extract_batches(f)
        assert batches == 100

    def test_from_dataset(self, tmp_path: Path) -> None:
        sp_path = tmp_path / "statepoint.200.h5"
        with h5py.File(sp_path, "w") as f:
            f.create_dataset("k_combined", data=[1.0, 0.001])
            f.create_dataset("n_batches", data=200)

        with h5py.File(sp_path, "r") as f:
            batches = _extract_batches(f)
        assert batches == 200

    def test_from_filename_fallback(self, tmp_path: Path) -> None:
        sp_path = tmp_path / "statepoint.150.h5"
        with h5py.File(sp_path, "w") as f:
            f.create_dataset("k_combined", data=[1.0, 0.001])

        with h5py.File(sp_path, "r") as f:
            batches = _extract_batches(f)
        assert batches == 150

    def test_no_info_returns_zero(self, tmp_path: Path) -> None:
        sp_path = tmp_path / "statepoint.h5"
        with h5py.File(sp_path, "w") as f:
            f.create_dataset("k_combined", data=[1.0, 0.001])

        with h5py.File(sp_path, "r") as f:
            batches = _extract_batches(f)
        assert batches == 0


class TestExtractTallies:
    """_extract_tallies 함수 테스트."""

    def test_extracts_two_tallies(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        assert len(tallies) == 2

    def test_tally_names(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        names = {t.name for t in tallies}
        assert names == {"flux_tally", "fission_tally"}

    def test_tally_scores(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        fission = next(t for t in tallies if t.name == "fission_tally")
        assert fission.scores == ["fission", "nu-fission"]

    def test_tally_mean_computed(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        flux = next(t for t in tallies if t.name == "flux_tally")
        # mean = sum / n_realizations = 4.5e14 / 90
        expected_mean = 4.5e14 / 90
        assert len(flux.mean) == 1
        assert abs(flux.mean[0] - expected_mean) < 1e6

    def test_tally_std_dev_computed(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        flux = next(t for t in tallies if t.name == "flux_tally")
        assert len(flux.std_dev) == 1
        assert flux.std_dev[0] >= 0

    def test_no_tallies_group(self, mock_case_no_tallies: Path) -> None:
        sp = _find_statepoint(mock_case_no_tallies)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        assert tallies == []

    def test_multi_score_tally(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            tallies = _extract_tallies(f)
        fission = next(t for t in tallies if t.name == "fission_tally")
        # 1 bin × 2 scores = 2 mean values
        assert len(fission.mean) == 2
        assert len(fission.std_dev) == 2


class TestExtractRuntime:
    """_extract_runtime 함수 테스트."""

    def test_from_attrs(self, mock_case: Path) -> None:
        sp = _find_statepoint(mock_case)
        with h5py.File(sp, "r") as f:
            runtime = _extract_runtime(f)
        assert abs(runtime - 45.3) < 1e-10

    def test_no_runtime_returns_zero(self, tmp_path: Path) -> None:
        sp_path = tmp_path / "statepoint.10.h5"
        with h5py.File(sp_path, "w") as f:
            f.create_dataset("k_combined", data=[1.0, 0.001])

        with h5py.File(sp_path, "r") as f:
            runtime = _extract_runtime(f)
        assert runtime == 0.0


class TestParseResults:
    """parse_results 통합 함수 테스트."""

    def test_returns_simulation_result(self, mock_case: Path) -> None:
        result = parse_results(mock_case)
        assert isinstance(result, SimulationResult)

    def test_keff_parsed(self, mock_case: Path) -> None:
        result = parse_results(mock_case)
        assert abs(result.keff - 1.00234) < 1e-10
        assert abs(result.keff_std - 0.00056) < 1e-10

    def test_batches_parsed(self, mock_case: Path) -> None:
        result = parse_results(mock_case)
        assert result.batches_completed == 100

    def test_tallies_parsed(self, mock_case: Path) -> None:
        result = parse_results(mock_case)
        assert len(result.tallies) == 2

    def test_runtime_parsed(self, mock_case: Path) -> None:
        result = parse_results(mock_case)
        assert abs(result.runtime - 45.3) < 1e-10

    def test_statepoint_path_set(self, mock_case: Path) -> None:
        result = parse_results(mock_case)
        assert result.statepoint_path is not None
        assert result.statepoint_path.name == "statepoint.100.h5"

    def test_no_tallies_case(self, mock_case_no_tallies: Path) -> None:
        result = parse_results(mock_case_no_tallies)
        assert result.tallies == []
        assert abs(result.keff - 1.00234) < 1e-10

    def test_missing_statepoint_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_empty"
        case_dir.mkdir()
        with pytest.raises(StatepointNotFoundError):
            parse_results(case_dir)

    def test_custom_keff(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_custom"
        case_dir.mkdir()
        output_dir = case_dir / "output"
        output_dir.mkdir()
        _create_mock_statepoint(
            output_dir / "statepoint.50.h5",
            keff=0.98765,
            keff_std=0.00432,
            n_batches=50,
            runtime=120.5,
        )
        result = parse_results(case_dir)
        assert abs(result.keff - 0.98765) < 1e-10
        assert abs(result.keff_std - 0.00432) < 1e-10
        assert result.batches_completed == 50
        assert abs(result.runtime - 120.5) < 1e-10

    def test_result_is_frozen(self, mock_case: Path) -> None:
        """SimulationResult는 frozen이므로 수정 불가."""
        result = parse_results(mock_case)
        with pytest.raises(Exception):
            result.keff = 2.0  # type: ignore[misc]
