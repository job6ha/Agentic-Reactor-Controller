"""SimulationPipeline 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.armi_layer.models import SimulationResult
from src.pipeline import PipelineResult, SimulationPipeline
from src.pipeline_config import PipelineConfig
from src.run_manager.runner import RunResult


@pytest.fixture()
def pipeline_config() -> PipelineConfig:
    """테스트용 파이프라인 설정."""
    return PipelineConfig.model_validate({
        "controller": {
            "sweep_values": [2.0, 3.0],
            "target_keff": 1.0,
            "keff_tolerance": 0.01,
            "max_iterations": 5,
        },
        "run": {
            "omp_threads": 1,
            "timeout": 60,
            "max_retries": 0,
        },
        "output": {
            "export_csv": False,
            "export_json": False,
        },
    })


@pytest.fixture()
def pipeline(pipeline_config: PipelineConfig, tmp_path: Path) -> SimulationPipeline:
    """tmp_path 기반 파이프라인."""
    config = pipeline_config.model_copy(
        update={"output": pipeline_config.output.model_copy(
            update={"runs_dir": tmp_path / "runs"},
        )},
    )
    return SimulationPipeline(config)


def _mock_sim_result(keff: float = 1.05) -> SimulationResult:
    """테스트용 SimulationResult 생성."""
    return SimulationResult(
        keff=keff,
        keff_std=0.001,
        runtime=10.0,
        batches_completed=100,
    )


def _mock_run_result(success: bool = True) -> RunResult:
    """테스트용 RunResult 생성."""
    return RunResult(
        exit_code=0 if success else 1,
        runtime=10.0,
        log_path=Path("/dev/null"),
        success=success,
    )


class TestSimulationPipelineInit:
    """파이프라인 초기화 테스트."""

    def test_init(self, pipeline_config: PipelineConfig) -> None:
        pipeline = SimulationPipeline(pipeline_config)
        assert pipeline._config == pipeline_config

    def test_from_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "output": {"runs_dir": str(tmp_path / "runs")},
        }), encoding="utf-8")

        pipeline = SimulationPipeline.from_config(config_path)
        assert isinstance(pipeline, SimulationPipeline)


class TestSimulationPipelineRun:
    """파이프라인 실행 테스트."""

    @patch("src.pipeline.run_case")
    @patch("src.pipeline.parse_results")
    def test_full_sweep(
        self,
        mock_parse: object,
        mock_run: object,
        pipeline: SimulationPipeline,
    ) -> None:
        """전체 스윕: 2개 값 → 2회 실행 → 값 소진으로 종료."""
        mock_run.return_value = _mock_run_result(success=True)  # type: ignore[attr-defined]
        mock_parse.return_value = _mock_sim_result(keff=1.05)  # type: ignore[attr-defined]

        result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert result.total_iterations == 2
        assert result.successful_cases == 2
        assert result.failed_cases == 0
        assert "소진" in result.stop_reason

    @patch("src.pipeline.run_case")
    @patch("src.pipeline.parse_results")
    def test_keff_convergence(
        self,
        mock_parse: object,
        mock_run: object,
        pipeline: SimulationPipeline,
    ) -> None:
        """keff 수렴 시 즉시 종료."""
        mock_run.return_value = _mock_run_result(success=True)  # type: ignore[attr-defined]
        # 첫 번째 실행에서 바로 수렴
        mock_parse.return_value = _mock_sim_result(keff=1.005)  # type: ignore[attr-defined]

        result = pipeline.run()

        assert result.total_iterations == 1
        assert result.successful_cases == 1
        assert "수렴" in result.stop_reason

    @patch("src.pipeline.run_case")
    def test_failed_case_skip(
        self,
        mock_run: object,
        pipeline: SimulationPipeline,
    ) -> None:
        """실패한 케이스 스킵 후 계속 진행."""
        mock_run.return_value = _mock_run_result(success=False)  # type: ignore[attr-defined]

        result = pipeline.run()

        assert result.total_iterations == 2
        assert result.successful_cases == 0
        assert result.failed_cases == 2
        assert result.skipped_cases == 2

    @patch("src.pipeline.run_case")
    @patch("src.pipeline.parse_results")
    def test_exception_in_case_skips(
        self,
        mock_parse: object,
        mock_run: object,
        pipeline: SimulationPipeline,
    ) -> None:
        """케이스 실행 중 예외 발생 시 스킵."""
        mock_run.return_value = _mock_run_result(success=True)  # type: ignore[attr-defined]
        mock_parse.side_effect = Exception("statepoint 파싱 실패")  # type: ignore[attr-defined]

        result = pipeline.run()

        assert result.failed_cases == 2
        assert result.skipped_cases == 2


class TestPipelineResult:
    """PipelineResult 테스트."""

    def test_default_values(self) -> None:
        result = PipelineResult()
        assert result.total_iterations == 0
        assert result.successful_cases == 0
        assert result.failed_cases == 0
        assert result.stop_reason == ""
        assert result.case_ids == []
