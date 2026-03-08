"""컨트롤러 팩토리 테스트 (KAE-85)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.controller.base import BaseController
from src.controller.factory import create_controller
from src.controller.llm.models import LLMControllerConfig
from src.controller.simple import SimpleController, SimpleControllerConfig


class TestCreateController:
    """create_controller 팩토리 함수 테스트."""

    def test_creates_simple_controller(self) -> None:
        config = SimpleControllerConfig()
        ctrl = create_controller(config)

        assert isinstance(ctrl, SimpleController)

    def test_creates_llm_controller(self, tmp_path: Path) -> None:
        config = LLMControllerConfig(log_dir=tmp_path / "logs")

        with patch("src.controller.llm.controller.LLMPlanner"):
            ctrl = create_controller(config)

        assert isinstance(ctrl, BaseController)
        assert type(ctrl).__name__ == "LLMController"

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError, match="지원하지 않는"):
            create_controller("invalid")  # type: ignore[arg-type]


class TestPipelineConfigIntegration:
    """PipelineConfig discriminated union 테스트."""

    def test_simple_config_default(self) -> None:
        from src.pipeline_config import PipelineConfig

        config = PipelineConfig()
        assert isinstance(config.controller, SimpleControllerConfig)
        assert config.controller.controller_type == "simple"

    def test_simple_config_explicit(self) -> None:
        from src.pipeline_config import PipelineConfig

        config = PipelineConfig.model_validate(
            {
                "controller": {
                    "controller_type": "simple",
                    "sweep_values": [2.0, 3.0],
                },
            }
        )
        assert isinstance(config.controller, SimpleControllerConfig)

    def test_llm_config(self) -> None:
        from src.pipeline_config import PipelineConfig

        config = PipelineConfig.model_validate(
            {
                "controller": {
                    "controller_type": "llm",
                    "n_candidates": 5,
                    "max_iterations": 6,
                },
            }
        )
        assert isinstance(config.controller, LLMControllerConfig)
        assert config.controller.n_candidates == 5

    def test_backward_compatible_no_controller_type(self) -> None:
        """controller_type 없는 기존 JSON도 동작."""
        from src.pipeline_config import PipelineConfig

        config = PipelineConfig.model_validate(
            {
                "controller": {
                    "sweep_values": [2.0, 3.0, 4.0],
                    "target_keff": 1.0,
                },
            }
        )
        assert isinstance(config.controller, SimpleControllerConfig)
        assert config.controller.sweep_values == [2.0, 3.0, 4.0]

    def test_load_config_json(self, tmp_path: Path) -> None:
        """JSON 파일에서 LLM 설정 로드."""
        import json

        from src.pipeline_config import load_config

        config_data = {
            "controller": {
                "controller_type": "llm",
                "n_candidates": 8,
                "max_iterations": 24,
            },
            "output": {"runs_dir": str(tmp_path / "runs")},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        config = load_config(config_path)

        assert isinstance(config.controller, LLMControllerConfig)
        assert config.controller.n_candidates == 8
