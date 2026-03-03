"""케이스 폴더 생성기 테스트."""

import json
from pathlib import Path

import pytest

from src.armi_layer.case_folder import (
    create_case,
    load_case_config,
    load_run_status,
)
from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    StatusType,
)


class TestCreateCase:
    """create_case 함수 테스트."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        config = CaseConfig(name="test-case")
        case_dir = create_case(config, runs_dir=tmp_path)

        assert case_dir.exists()
        assert (case_dir / "input").is_dir()
        assert (case_dir / "output").is_dir()
        assert (case_dir / "meta").is_dir()

    def test_first_case_is_0001(self, tmp_path: Path) -> None:
        config = CaseConfig(name="first")
        case_dir = create_case(config, runs_dir=tmp_path)
        assert case_dir.name == "case_0001"

    def test_auto_increment(self, tmp_path: Path) -> None:
        config = CaseConfig(name="case-a")
        dir1 = create_case(config, runs_dir=tmp_path)
        dir2 = create_case(config, runs_dir=tmp_path)
        dir3 = create_case(config, runs_dir=tmp_path)

        assert dir1.name == "case_0001"
        assert dir2.name == "case_0002"
        assert dir3.name == "case_0003"

    def test_detects_existing_cases(self, tmp_path: Path) -> None:
        """기존 케이스 폴더가 있으면 그 다음 번호를 사용한다."""
        (tmp_path / "case_0005").mkdir()
        config = CaseConfig(name="after-gap")
        case_dir = create_case(config, runs_dir=tmp_path)
        assert case_dir.name == "case_0006"

    def test_ignores_non_case_dirs(self, tmp_path: Path) -> None:
        """case_XXXX 패턴이 아닌 디렉토리는 무시한다."""
        (tmp_path / "other_folder").mkdir()
        (tmp_path / "case_abc").mkdir()
        config = CaseConfig(name="first")
        case_dir = create_case(config, runs_dir=tmp_path)
        assert case_dir.name == "case_0001"

    def test_saves_config_json(self, tmp_path: Path) -> None:
        config = CaseConfig(
            name="pwr-test",
            geometry=GeometryParams(fuel_radius=0.35, clad_inner_radius=0.38,
                                    clad_outer_radius=0.42, pitch=1.2),
            materials=MaterialParams(fuel_enrichment=4.5),
        )
        case_dir = create_case(config, runs_dir=tmp_path)

        config_path = case_dir / "meta" / "config.json"
        assert config_path.exists()

        data = json.loads(config_path.read_text())
        assert data["name"] == "pwr-test"
        assert data["geometry"]["fuel_radius"] == 0.35
        assert data["materials"]["fuel_enrichment"] == 4.5

    def test_saves_status_json(self, tmp_path: Path) -> None:
        config = CaseConfig(name="status-test")
        case_dir = create_case(config, runs_dir=tmp_path)

        status_path = case_dir / "meta" / "status.json"
        assert status_path.exists()

        data = json.loads(status_path.read_text())
        assert data["status"] == "queued"
        assert data["attempt"] == 0

    def test_creates_runs_dir_if_missing(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "nested" / "runs"
        config = CaseConfig(name="nested")
        case_dir = create_case(config, runs_dir=runs_dir)
        assert case_dir.exists()
        assert runs_dir.exists()


class TestLoadCaseConfig:
    """load_case_config 함수 테스트."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = CaseConfig(
            name="roundtrip-test",
            materials=MaterialParams(fuel_enrichment=5.0),
            tags=["test", "roundtrip"],
        )
        case_dir = create_case(original, runs_dir=tmp_path)

        loaded = load_case_config(case_dir)
        assert loaded.name == "roundtrip-test"
        assert loaded.materials.fuel_enrichment == 5.0
        assert loaded.tags == ["test", "roundtrip"]

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        fake_case = tmp_path / "case_9999"
        fake_case.mkdir()
        with pytest.raises(FileNotFoundError, match="config.json"):
            load_case_config(fake_case)


class TestLoadRunStatus:
    """load_run_status 함수 테스트."""

    def test_initial_status(self, tmp_path: Path) -> None:
        config = CaseConfig(name="status-test")
        case_dir = create_case(config, runs_dir=tmp_path)

        status = load_run_status(case_dir)
        assert status.status == StatusType.QUEUED
        assert status.attempt == 0

    def test_missing_status_raises(self, tmp_path: Path) -> None:
        fake_case = tmp_path / "case_9999"
        fake_case.mkdir()
        with pytest.raises(FileNotFoundError, match="status.json"):
            load_run_status(fake_case)
