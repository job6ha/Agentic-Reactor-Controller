"""PipelineConfig 모델 테스트."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pipeline_config import OutputConfig, PipelineConfig, load_config


class TestPipelineConfigDefaults:
    """PipelineConfig 기본값 테스트."""

    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.controller.sweep_field == "materials.fuel_enrichment"
        assert config.controller.sweep_values == [2.0, 3.0, 4.0, 5.0]
        assert config.controller.target_keff == 1.0
        assert config.run.omp_threads is None
        assert config.run.max_retries == 0
        assert config.output.export_csv is True
        assert config.output.export_json is False

    def test_output_config_defaults(self) -> None:
        config = OutputConfig()
        assert config.runs_dir == Path("runs")
        assert config.export_csv is True
        assert config.export_json is False


class TestPipelineConfigValidation:
    """PipelineConfig 검증 테스트."""

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"unknown_field": 123})

    def test_invalid_keff_tolerance(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({
                "controller": {"keff_tolerance": -0.01},
            })

    def test_invalid_max_iterations(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({
                "controller": {"max_iterations": 0},
            })


class TestLoadConfig:
    """load_config 함수 테스트."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        config_data = {
            "controller": {
                "sweep_field": "materials.fuel_enrichment",
                "sweep_values": [1.0, 2.0, 3.0],
                "target_keff": 1.0,
                "keff_tolerance": 0.02,
                "max_iterations": 5,
            },
            "run": {
                "omp_threads": 8,
                "timeout": 600,
                "max_retries": 2,
            },
            "output": {
                "runs_dir": "custom_runs",
                "export_csv": True,
                "export_json": True,
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        config = load_config(config_path)

        assert config.controller.sweep_values == [1.0, 2.0, 3.0]
        assert config.controller.keff_tolerance == 0.02
        assert config.run.omp_threads == 8
        assert config.run.timeout == 600
        assert config.output.runs_dir == Path("custom_runs")
        assert config.output.export_json is True

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")

        config = load_config(config_path)
        assert config.controller.sweep_values == [2.0, 3.0, 4.0, 5.0]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text("not json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_config(config_path)

    def test_load_invalid_fields(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"controller": {"max_iterations": -1}}),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            load_config(config_path)
