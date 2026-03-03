"""파라미터 스윕 엔진 테스트.

격자 조합 생성, CaseConfig 변환, JSON 저장/로드를 검증한다.
"""

from pathlib import Path

import pytest

from src.armi_layer.models import CaseConfig, MaterialParams, SimulationSettings
from src.armi_layer.sweep import (
    SweepConfig,
    SweepParam,
    _build_case_name,
    _set_nested_field,
    generate_sweep_configs,
    load_sweep_config,
    save_sweep_config,
)


class TestSetNestedField:
    """_set_nested_field 함수 테스트."""

    def test_single_level(self) -> None:
        data = {"name": "test"}
        _set_nested_field(data, "name", "changed")
        assert data["name"] == "changed"

    def test_nested_level(self) -> None:
        data = {"materials": {"fuel_enrichment": 3.0}}
        _set_nested_field(data, "materials.fuel_enrichment", 4.5)
        assert data["materials"]["fuel_enrichment"] == 4.5

    def test_deep_nested(self) -> None:
        data = {"a": {"b": {"c": 1.0}}}
        _set_nested_field(data, "a.b.c", 2.0)
        assert data["a"]["b"]["c"] == 2.0

    def test_invalid_path_raises(self) -> None:
        data = {"materials": {"fuel_enrichment": 3.0}}
        with pytest.raises(KeyError, match="유효하지 않은 경로"):
            _set_nested_field(data, "geometry.fuel_radius", 0.5)

    def test_invalid_field_raises(self) -> None:
        data = {"materials": {"fuel_enrichment": 3.0}}
        with pytest.raises(KeyError, match="유효하지 않은 필드"):
            _set_nested_field(data, "materials.nonexistent", 1.0)


class TestBuildCaseName:
    """_build_case_name 함수 테스트."""

    def test_single_param(self) -> None:
        name = _build_case_name("sweep", [("materials.fuel_enrichment", 3.0)])
        assert name == "sweep_fuel_enrichment=3.0"

    def test_multi_params(self) -> None:
        name = _build_case_name(
            "study",
            [
                ("materials.fuel_enrichment", 4.5),
                ("geometry.pitch", 1.3),
            ],
        )
        assert name == "study_fuel_enrichment=4.5_pitch=1.3"

    def test_custom_template(self) -> None:
        name = _build_case_name("enr", [("materials.fuel_enrichment", 5.0)])
        assert name == "enr_fuel_enrichment=5.0"


class TestSweepParam:
    """SweepParam 모델 테스트."""

    def test_creation(self) -> None:
        param = SweepParam(
            field_path="materials.fuel_enrichment",
            values=[2.0, 3.0, 4.0],
        )
        assert param.field_path == "materials.fuel_enrichment"
        assert param.values == [2.0, 3.0, 4.0]

    def test_empty_values_rejected(self) -> None:
        with pytest.raises(Exception):
            SweepParam(field_path="materials.fuel_enrichment", values=[])


class TestSweepConfig:
    """SweepConfig 모델 테스트."""

    def test_creation(self) -> None:
        config = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0],
                ),
            ],
        )
        assert len(config.params) == 1

    def test_empty_params_rejected(self) -> None:
        with pytest.raises(Exception):
            SweepConfig(params=[])


class TestGenerateSweepConfigs:
    """generate_sweep_configs 함수 테스트."""

    def test_single_param_sweep(self) -> None:
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0, 4.5],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        assert len(configs) == 3

    def test_single_param_values(self) -> None:
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0, 4.5],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        enrichments = [c.materials.fuel_enrichment for c in configs]
        assert enrichments == [2.0, 3.0, 4.5]

    def test_two_param_grid(self) -> None:
        """2개 파라미터의 격자 조합 (3 x 2 = 6개)."""
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0, 4.5],
                ),
                SweepParam(
                    field_path="geometry.pitch",
                    values=[1.2, 1.3],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        assert len(configs) == 6

    def test_grid_combinations_correct(self) -> None:
        """격자 조합이 올바르게 생성되는지 확인."""
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 4.0],
                ),
                SweepParam(
                    field_path="geometry.pitch",
                    values=[1.2, 1.4],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        combos = [(c.materials.fuel_enrichment, c.geometry.pitch) for c in configs]
        assert (2.0, 1.2) in combos
        assert (2.0, 1.4) in combos
        assert (4.0, 1.2) in combos
        assert (4.0, 1.4) in combos

    def test_three_param_grid(self) -> None:
        """3개 파라미터 (2 x 2 x 2 = 8개)."""
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 4.0],
                ),
                SweepParam(
                    field_path="materials.coolant_density",
                    values=[0.7, 0.8],
                ),
                SweepParam(
                    field_path="settings.particles",
                    values=[1000, 5000],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        assert len(configs) == 8

    def test_base_config_preserved(self) -> None:
        """base_config의 비스윕 파라미터는 보존."""
        base = CaseConfig(
            description="base case",
            settings=SimulationSettings(batches=200),
        )
        sweep = SweepConfig(
            base_config=base,
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[3.0, 4.0],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        for c in configs:
            assert c.description == "base case"
            assert c.settings.batches == 200

    def test_case_names_generated(self) -> None:
        sweep = SweepConfig(
            name_template="enr_study",
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 4.0],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        assert configs[0].name == "enr_study_fuel_enrichment=2.0"
        assert configs[1].name == "enr_study_fuel_enrichment=4.0"

    def test_invalid_field_path_raises(self) -> None:
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="nonexistent.field",
                    values=[1.0],
                ),
            ],
        )
        with pytest.raises(KeyError):
            generate_sweep_configs(sweep)

    def test_returns_valid_case_configs(self) -> None:
        """생성된 CaseConfig가 유효한지 검증."""
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0, 5.0],
                ),
            ],
        )
        configs = generate_sweep_configs(sweep)
        for c in configs:
            assert isinstance(c, CaseConfig)


class TestSaveLoadSweepConfig:
    """save_sweep_config / load_sweep_config 테스트."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0],
                ),
            ],
        )
        path = tmp_path / "sweep.json"
        save_sweep_config(sweep, path)
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        sweep = SweepConfig(
            name_template="test_sweep",
            base_config=CaseConfig(
                description="roundtrip test",
                materials=MaterialParams(fuel_enrichment=3.5),
            ),
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0, 3.0, 4.5],
                ),
                SweepParam(
                    field_path="geometry.pitch",
                    values=[1.2, 1.3],
                ),
            ],
        )
        path = tmp_path / "sweep.json"
        save_sweep_config(sweep, path)
        loaded = load_sweep_config(path)

        assert loaded.name_template == "test_sweep"
        assert loaded.base_config.description == "roundtrip test"
        assert len(loaded.params) == 2
        assert loaded.params[0].values == [2.0, 3.0, 4.5]

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="스윕 설정 파일"):
            load_sweep_config(tmp_path / "nonexistent.json")

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        sweep = SweepConfig(
            params=[
                SweepParam(
                    field_path="materials.fuel_enrichment",
                    values=[2.0],
                ),
            ],
        )
        path = tmp_path / "sub" / "dir" / "sweep.json"
        save_sweep_config(sweep, path)
        assert path.exists()
