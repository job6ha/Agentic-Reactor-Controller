"""SimpleController 테스트.

단순 파라미터 스윕 기반 룰 컨트롤러의 동작을 검증한다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult
from src.controller.base import Action, ActionType
from src.controller.simple import SimpleController, SimpleControllerConfig


class TestSimpleControllerConfig:
    """SimpleControllerConfig 설정 모델 테스트."""

    def test_defaults(self) -> None:
        config = SimpleControllerConfig()
        assert config.sweep_field == "materials.fuel_enrichment"
        assert config.sweep_values == [2.0, 3.0, 4.0, 5.0]
        assert config.target_keff == 1.0
        assert config.keff_tolerance == 0.01
        assert config.max_iterations == 10

    def test_custom_values(self) -> None:
        config = SimpleControllerConfig(
            sweep_field="geometry.pitch",
            sweep_values=[1.2, 1.3, 1.4],
            target_keff=1.05,
            keff_tolerance=0.02,
            max_iterations=5,
        )
        assert config.sweep_field == "geometry.pitch"
        assert config.sweep_values == [1.2, 1.3, 1.4]
        assert config.target_keff == 1.05
        assert config.keff_tolerance == 0.02
        assert config.max_iterations == 5

    def test_tolerance_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimpleControllerConfig(keff_tolerance=0)

    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimpleControllerConfig(max_iterations=0)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SimpleControllerConfig(unknown_field="value")


class TestSimpleControllerInit:
    """SimpleController 초기화 테스트."""

    def test_default_config(self) -> None:
        ctrl = SimpleController()
        assert ctrl.config.sweep_field == "materials.fuel_enrichment"

    def test_custom_config(self) -> None:
        config = SimpleControllerConfig(max_iterations=3)
        ctrl = SimpleController(config)
        assert ctrl.config.max_iterations == 3


class TestProposeActions:
    """propose_actions 테스트."""

    def test_first_iteration_proposes_first_value(self) -> None:
        ctrl = SimpleController()
        state = ReactorState(iteration=0)
        actions = ctrl.propose_actions(state)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.MODIFY_PARAM
        assert actions[0].value == 2.0

    def test_second_iteration_proposes_second_value(self) -> None:
        ctrl = SimpleController()
        state = ReactorState(iteration=1)
        actions = ctrl.propose_actions(state)
        assert actions[0].value == 3.0

    def test_sweep_field_in_action(self) -> None:
        ctrl = SimpleController()
        state = ReactorState(iteration=0)
        actions = ctrl.propose_actions(state)
        assert actions[0].field_path == "materials.fuel_enrichment"

    def test_reason_includes_progress(self) -> None:
        ctrl = SimpleController()
        state = ReactorState(iteration=0)
        actions = ctrl.propose_actions(state)
        assert "1/4" in actions[0].reason

    def test_stop_at_max_iterations(self) -> None:
        config = SimpleControllerConfig(max_iterations=3)
        ctrl = SimpleController(config)
        state = ReactorState(iteration=3)
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.STOP
        assert "최대 반복" in actions[0].reason

    def test_stop_when_sweep_exhausted(self) -> None:
        config = SimpleControllerConfig(sweep_values=[2.0, 3.0])
        ctrl = SimpleController(config)
        state = ReactorState(iteration=2)
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.STOP
        assert "소진" in actions[0].reason

    def test_stop_when_keff_converged(self) -> None:
        ctrl = SimpleController()
        state = ReactorState(
            iteration=1,
            kpi={"keff": 1.005},
        )
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.STOP
        assert "수렴" in actions[0].reason

    def test_no_convergence_without_kpi(self) -> None:
        """kpi가 없으면 수렴 체크를 건너뛴다."""
        ctrl = SimpleController()
        state = ReactorState(iteration=0)
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.MODIFY_PARAM

    def test_no_convergence_when_deviation_large(self) -> None:
        """keff 편차가 허용치보다 크면 계속 진행."""
        ctrl = SimpleController()
        state = ReactorState(
            iteration=1,
            kpi={"keff": 1.05},
        )
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.MODIFY_PARAM

    def test_max_iterations_checked_first(self) -> None:
        """최대 반복이 수렴보다 먼저 체크된다."""
        config = SimpleControllerConfig(max_iterations=1)
        ctrl = SimpleController(config)
        state = ReactorState(
            iteration=1,
            kpi={"keff": 1.0},  # 수렴 상태
        )
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.STOP
        assert "최대 반복" in actions[0].reason

    def test_custom_sweep_field(self) -> None:
        config = SimpleControllerConfig(
            sweep_field="geometry.pitch",
            sweep_values=[1.2, 1.3],
        )
        ctrl = SimpleController(config)
        state = ReactorState(iteration=0)
        actions = ctrl.propose_actions(state)
        assert actions[0].field_path == "geometry.pitch"
        assert actions[0].value == 1.2


class TestApplyActionsToCase:
    """apply_actions_to_case 테스트."""

    def test_applies_enrichment_change(self) -> None:
        ctrl = SimpleController()
        state = ReactorState()
        actions = [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path="materials.fuel_enrichment",
                value=4.5,
            ),
        ]
        config = ctrl.apply_actions_to_case(actions, state)
        assert config.materials.fuel_enrichment == 4.5

    def test_preserves_other_params(self) -> None:
        ctrl = SimpleController()
        state = ReactorState()
        actions = [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path="materials.fuel_enrichment",
                value=4.5,
            ),
        ]
        config = ctrl.apply_actions_to_case(actions, state)
        assert config.geometry.pitch == state.current_config.geometry.pitch

    def test_stop_action_no_change(self) -> None:
        """STOP 액션은 파라미터를 변경하지 않는다."""
        ctrl = SimpleController()
        state = ReactorState()
        actions = [Action(action_type=ActionType.STOP)]
        config = ctrl.apply_actions_to_case(actions, state)
        assert config == state.current_config

    def test_returns_valid_case_config(self) -> None:
        ctrl = SimpleController()
        state = ReactorState()
        actions = [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path="materials.fuel_enrichment",
                value=3.0,
            ),
        ]
        config = ctrl.apply_actions_to_case(actions, state)
        assert isinstance(config, CaseConfig)


class TestEvaluateResults:
    """evaluate_results 테스트."""

    def test_returns_keff(self) -> None:
        ctrl = SimpleController()
        result = SimulationResult(keff=1.02, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert metrics["keff"] == 1.02

    def test_returns_keff_std(self) -> None:
        ctrl = SimpleController()
        result = SimulationResult(keff=1.02, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert metrics["keff_std"] == 0.001

    def test_deviation_calculation(self) -> None:
        ctrl = SimpleController()
        result = SimulationResult(keff=1.02, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert abs(metrics["keff_deviation"] - 0.02) < 1e-10

    def test_converged_when_within_tolerance(self) -> None:
        ctrl = SimpleController()
        result = SimulationResult(keff=1.005, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert metrics["converged"] == 1.0

    def test_not_converged_when_outside_tolerance(self) -> None:
        ctrl = SimpleController()
        result = SimulationResult(keff=1.05, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert metrics["converged"] == 0.0

    def test_converged_at_near_boundary(self) -> None:
        """deviation이 tolerance에 매우 가까울 때 수렴으로 판정."""
        config = SimpleControllerConfig(target_keff=1.0, keff_tolerance=0.02)
        ctrl = SimpleController(config)
        # 0.019 < 0.02 이므로 수렴
        result = SimulationResult(keff=1.019, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert metrics["converged"] == 1.0

    def test_custom_target_keff(self) -> None:
        config = SimpleControllerConfig(target_keff=1.1)
        ctrl = SimpleController(config)
        result = SimulationResult(keff=1.105, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert abs(metrics["keff_deviation"] - 0.005) < 1e-10
        assert metrics["converged"] == 1.0


class TestUpdateState:
    """update_state 테스트."""

    def test_increments_iteration(self) -> None:
        ctrl = SimpleController()
        state = ReactorState(iteration=0)
        metrics = {"keff": 1.02, "keff_std": 0.001}
        new_state = ctrl.update_state(state, metrics)
        assert new_state.iteration == 1

    def test_updates_kpi(self) -> None:
        ctrl = SimpleController()
        state = ReactorState()
        metrics = {"keff": 1.02, "keff_std": 0.001}
        new_state = ctrl.update_state(state, metrics)
        assert new_state.kpi["keff"] == 1.02

    def test_preserves_config(self) -> None:
        ctrl = SimpleController()
        state = ReactorState()
        metrics = {"keff": 1.02}
        new_state = ctrl.update_state(state, metrics)
        assert new_state.current_config == state.current_config


class TestControllerLoop:
    """전체 제어 루프 통합 테스트."""

    def test_sweep_to_completion(self) -> None:
        """스윕 값 목록이 소진될 때까지 루프 실행."""
        config = SimpleControllerConfig(
            sweep_values=[2.0, 3.0, 4.0],
            max_iterations=10,
        )
        ctrl = SimpleController(config)
        state = ReactorState()

        iterations = 0
        for _ in range(20):
            actions = ctrl.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break

            new_config = ctrl.apply_actions_to_case(actions, state)
            result = SimulationResult(
                keff=0.9 + 0.05 * state.iteration,
                keff_std=0.001,
                runtime=30.0,
            )
            metrics = ctrl.evaluate_results(result)
            state = ctrl.update_state(state, metrics)
            state = state.model_copy(update={"current_config": new_config})
            iterations += 1

        assert iterations == 3

    def test_convergence_stops_early(self) -> None:
        """keff 수렴 시 조기 종료."""
        config = SimpleControllerConfig(
            sweep_values=[2.0, 3.0, 4.0, 5.0],
            target_keff=1.0,
            keff_tolerance=0.01,
        )
        ctrl = SimpleController(config)
        state = ReactorState()

        iterations = 0
        for _ in range(20):
            actions = ctrl.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break

            new_config = ctrl.apply_actions_to_case(actions, state)
            # 2번째 반복에서 수렴하도록 설정
            keff = 1.005 if state.iteration == 1 else 0.9
            result = SimulationResult(
                keff=keff,
                keff_std=0.001,
                runtime=30.0,
            )
            metrics = ctrl.evaluate_results(result)
            state = ctrl.update_state(state, metrics)
            state = state.model_copy(update={"current_config": new_config})
            iterations += 1

        # iter 0 실행 → iter 1 실행(keff=1.005) → iter 2에서 수렴 감지 → STOP
        assert iterations == 2

    def test_max_iterations_stops(self) -> None:
        """최대 반복 횟수 도달 시 종료."""
        config = SimpleControllerConfig(
            sweep_values=[2.0, 3.0, 4.0, 5.0],
            max_iterations=2,
        )
        ctrl = SimpleController(config)
        state = ReactorState()

        iterations = 0
        for _ in range(20):
            actions = ctrl.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break

            new_config = ctrl.apply_actions_to_case(actions, state)
            result = SimulationResult(
                keff=0.8,
                keff_std=0.001,
                runtime=30.0,
            )
            metrics = ctrl.evaluate_results(result)
            state = ctrl.update_state(state, metrics)
            state = state.model_copy(update={"current_config": new_config})
            iterations += 1

        assert iterations == 2
