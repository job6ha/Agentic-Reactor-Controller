"""Pydantic 데이터 모델 테스트.

모든 모델의 생성, 직렬화/역직렬화, 유효성 검증을 테스트한다.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    ReactorState,
    RunConfig,
    RunStatus,
    SimulationResult,
    SimulationSettings,
    StatusType,
    TallyResult,
)


class TestGeometryParams:
    """GeometryParams 모델 테스트."""

    def test_default_pwr_pin_cell(self) -> None:
        geo = GeometryParams()
        assert geo.fuel_radius == pytest.approx(0.39218)
        assert geo.clad_inner_radius == pytest.approx(0.40005)
        assert geo.clad_outer_radius == pytest.approx(0.45720)
        assert geo.pitch == pytest.approx(1.25984)

    def test_custom_values(self) -> None:
        geo = GeometryParams(
            fuel_radius=0.35,
            clad_inner_radius=0.38,
            clad_outer_radius=0.42,
            pitch=1.5,
        )
        assert geo.fuel_radius == 0.35
        assert geo.pitch == 1.5

    def test_extra_params(self) -> None:
        geo = GeometryParams(extra_params={"gap_thickness": 0.01})
        assert geo.extra_params["gap_thickness"] == 0.01

    def test_invalid_geometry_ordering(self) -> None:
        with pytest.raises(ValidationError, match="fuel_radius < clad_inner_radius"):
            GeometryParams(fuel_radius=0.5, clad_inner_radius=0.4)

    def test_clad_exceeds_half_pitch(self) -> None:
        with pytest.raises(ValidationError, match="pitch/2보다 작아야"):
            GeometryParams(
                fuel_radius=0.3,
                clad_inner_radius=0.4,
                clad_outer_radius=0.65,
                pitch=1.2,
            )

    def test_negative_radius_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GeometryParams(fuel_radius=-0.1)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GeometryParams(unknown_field=1.0)


class TestMaterialParams:
    """MaterialParams 모델 테스트."""

    def test_default_values(self) -> None:
        mat = MaterialParams()
        assert mat.fuel_enrichment == 3.0
        assert mat.fuel_density == pytest.approx(10.29769)
        assert mat.coolant_temperature == 600.0

    def test_custom_enrichment(self) -> None:
        mat = MaterialParams(fuel_enrichment=4.5)
        assert mat.fuel_enrichment == 4.5

    def test_negative_density_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MaterialParams(fuel_density=-1.0)

    def test_enrichment_over_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MaterialParams(fuel_enrichment=101.0)


class TestSimulationSettings:
    """SimulationSettings 모델 테스트."""

    def test_default_values(self) -> None:
        settings = SimulationSettings()
        assert settings.batches == 110
        assert settings.inactive == 10
        assert settings.particles == 1000

    def test_extra_settings(self) -> None:
        settings = SimulationSettings(
            extra_settings={"entropy_mesh": True, "photon_transport": False}
        )
        assert settings.extra_settings["entropy_mesh"] is True

    def test_inactive_exceeds_batches_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inactive.*batches"):
            SimulationSettings(batches=10, inactive=10)

    def test_zero_particles_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationSettings(particles=0)


class TestCaseConfig:
    """CaseConfig 모델 테스트."""

    def test_default_creation(self) -> None:
        config = CaseConfig()
        assert config.name == ""
        assert isinstance(config.geometry, GeometryParams)
        assert isinstance(config.materials, MaterialParams)
        assert isinstance(config.settings, SimulationSettings)

    def test_named_case(self) -> None:
        config = CaseConfig(
            name="pwr-pin-3pct",
            description="PWR pin cell with 3% enrichment",
            tags=["pwr", "pin-cell", "baseline"],
        )
        assert config.name == "pwr-pin-3pct"
        assert len(config.tags) == 3

    def test_json_roundtrip(self) -> None:
        config = CaseConfig(
            name="test-case",
            geometry=GeometryParams(
                fuel_radius=0.35,
                clad_inner_radius=0.38,
                clad_outer_radius=0.42,
                pitch=1.2,
            ),
            materials=MaterialParams(fuel_enrichment=4.5),
            settings=SimulationSettings(batches=200, particles=5000),
        )
        json_str = config.model_dump_json()
        restored = CaseConfig.model_validate_json(json_str)
        assert restored.name == "test-case"
        assert restored.geometry.fuel_radius == 0.35
        assert restored.materials.fuel_enrichment == 4.5
        assert restored.settings.batches == 200


class TestRunConfig:
    """RunConfig 모델 테스트."""

    def test_default_creation(self) -> None:
        rc = RunConfig()
        assert rc.omp_threads is None
        assert rc.cross_sections_path is None
        assert rc.timeout is None
        assert rc.max_retries == 0

    def test_custom_config(self) -> None:
        rc = RunConfig(
            omp_threads=8,
            cross_sections_path=Path("/data/nucdata/cross_sections.xml"),
            timeout=3600.0,
            max_retries=2,
        )
        assert rc.omp_threads == 8
        assert rc.cross_sections_path == Path("/data/nucdata/cross_sections.xml")
        assert rc.max_retries == 2

    def test_json_roundtrip(self) -> None:
        rc = RunConfig(omp_threads=4, timeout=600.0)
        json_str = rc.model_dump_json()
        restored = RunConfig.model_validate_json(json_str)
        assert restored.omp_threads == 4
        assert restored.timeout == 600.0


class TestRunStatus:
    """RunStatus 모델 테스트."""

    def test_default_queued(self) -> None:
        status = RunStatus()
        assert status.status == StatusType.QUEUED
        assert isinstance(status.created_at, datetime)
        assert status.created_at.tzinfo is not None  # UTC aware
        assert status.started_at is None
        assert status.completed_at is None
        assert status.attempt == 0

    def test_running_state(self) -> None:
        now = datetime.now()
        status = RunStatus(
            status=StatusType.RUNNING,
            started_at=now,
            attempt=1,
        )
        assert status.status == StatusType.RUNNING
        assert status.started_at == now

    def test_failed_state(self) -> None:
        status = RunStatus(
            status=StatusType.FAILED,
            error_message="Segmentation fault",
            attempt=3,
        )
        assert status.error_message == "Segmentation fault"
        assert status.attempt == 3

    def test_json_roundtrip(self) -> None:
        status = RunStatus(
            status=StatusType.DONE,
            started_at=datetime(2026, 3, 3, 12, 0, 0),
            completed_at=datetime(2026, 3, 3, 12, 30, 0),
        )
        json_str = status.model_dump_json()
        restored = RunStatus.model_validate_json(json_str)
        assert restored.status == StatusType.DONE
        assert restored.completed_at is not None

    def test_status_enum_values(self) -> None:
        assert StatusType.QUEUED.value == "queued"
        assert StatusType.RUNNING.value == "running"
        assert StatusType.DONE.value == "done"
        assert StatusType.FAILED.value == "failed"


class TestTallyResult:
    """TallyResult 모델 테스트."""

    def test_creation(self) -> None:
        tally = TallyResult(
            name="flux",
            scores=["flux"],
            mean=[1.23e14],
            std_dev=[1.5e12],
        )
        assert tally.name == "flux"
        assert len(tally.mean) == 1

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="길이가 일치"):
            TallyResult(
                name="flux",
                scores=["flux"],
                mean=[1.0, 2.0],
                std_dev=[0.1],
            )

    def test_frozen(self) -> None:
        tally = TallyResult(name="flux", scores=["flux"], mean=[1.0], std_dev=[0.1])
        with pytest.raises(ValidationError):
            tally.name = "changed"


class TestSimulationResult:
    """SimulationResult 모델 테스트."""

    def test_basic_result(self) -> None:
        result = SimulationResult(
            keff=1.06532,
            keff_std=0.00045,
            runtime=120.5,
            batches_completed=110,
        )
        assert result.keff == pytest.approx(1.06532)
        assert result.keff_std == pytest.approx(0.00045)
        assert result.tallies == []

    def test_negative_runtime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationResult(keff=1.0, keff_std=0.001, runtime=-1.0)

    def test_negative_keff_std_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationResult(keff=1.0, keff_std=-0.001, runtime=60.0)

    def test_frozen(self) -> None:
        result = SimulationResult(keff=1.0, keff_std=0.001, runtime=60.0)
        with pytest.raises(ValidationError):
            result.keff = 2.0

    def test_with_tallies(self) -> None:
        tally = TallyResult(
            name="fission-rate",
            scores=["fission"],
            mean=[2.5e13, 3.1e13],
            std_dev=[1.0e11, 1.2e11],
        )
        result = SimulationResult(
            keff=1.05,
            keff_std=0.001,
            runtime=60.0,
            tallies=[tally],
        )
        assert len(result.tallies) == 1
        assert result.tallies[0].name == "fission-rate"

    def test_json_roundtrip(self) -> None:
        result = SimulationResult(
            keff=1.06532,
            keff_std=0.00045,
            runtime=120.5,
            statepoint_path=Path("/work/runs/case_0001/output/statepoint.100.h5"),
            batches_completed=110,
            tallies=[
                TallyResult(
                    name="flux",
                    scores=["flux"],
                    mean=[1.23e14],
                    std_dev=[1.5e12],
                )
            ],
        )
        json_str = result.model_dump_json()
        restored = SimulationResult.model_validate_json(json_str)
        assert restored.keff == pytest.approx(1.06532)
        assert len(restored.tallies) == 1
        assert restored.statepoint_path is not None


class TestReactorState:
    """ReactorState 모델 테스트."""

    def test_default_creation(self) -> None:
        state = ReactorState()
        assert isinstance(state.current_config, CaseConfig)
        assert state.history == []
        assert state.kpi == {}
        assert state.iteration == 0

    def test_with_history(self) -> None:
        results = [
            SimulationResult(keff=1.05, keff_std=0.001, runtime=60.0),
            SimulationResult(keff=1.06, keff_std=0.0008, runtime=55.0),
        ]
        state = ReactorState(
            current_config=CaseConfig(name="iteration-2"),
            history=results,
            kpi={"best_keff": 1.06, "convergence": 0.01},
            iteration=2,
        )
        assert len(state.history) == 2
        assert state.kpi["best_keff"] == 1.06
        assert state.iteration == 2

    def test_json_roundtrip(self) -> None:
        state = ReactorState(
            current_config=CaseConfig(name="test"),
            history=[SimulationResult(keff=1.05, keff_std=0.001, runtime=60.0)],
            kpi={"keff": 1.05},
            iteration=1,
        )
        json_str = state.model_dump_json()
        restored = ReactorState.model_validate_json(json_str)
        assert restored.current_config.name == "test"
        assert len(restored.history) == 1
        assert restored.kpi["keff"] == 1.05


class TestCrossModelIntegration:
    """모델 간 통합 테스트."""

    def test_full_workflow_serialization(self) -> None:
        """전체 워크플로우 데이터를 JSON으로 직렬화/역직렬화."""
        config = CaseConfig(
            name="pwr-sweep-001",
            geometry=GeometryParams(fuel_radius=0.4),
            materials=MaterialParams(fuel_enrichment=3.5),
            settings=SimulationSettings(batches=150, particles=5000),
        )
        run_config = RunConfig(omp_threads=4, timeout=1800.0)
        status = RunStatus(status=StatusType.DONE)
        result = SimulationResult(
            keff=1.06532,
            keff_std=0.00045,
            runtime=120.5,
            batches_completed=150,
        )
        state = ReactorState(
            current_config=config,
            history=[result],
            kpi={"keff": 1.06532},
            iteration=1,
        )

        # 각 모델을 dict로 변환 후 JSON으로 직렬화
        payload = {
            "config": config.model_dump(mode="json"),
            "run_config": run_config.model_dump(mode="json"),
            "status": status.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "state": state.model_dump(mode="json"),
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        restored = json.loads(json_str)

        # 복원 검증
        assert CaseConfig.model_validate(restored["config"]).name == "pwr-sweep-001"
        assert RunConfig.model_validate(restored["run_config"]).omp_threads == 4
        assert RunStatus.model_validate(restored["status"]).status == StatusType.DONE
        restored_result = SimulationResult.model_validate(
            restored["result"],
        )
        assert restored_result.keff == pytest.approx(1.06532)
        assert ReactorState.model_validate(restored["state"]).iteration == 1
