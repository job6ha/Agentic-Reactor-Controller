"""BaseController 인터페이스 테스트.

ABC 인터페이스 준수, Action 모델, 구현체 패턴을 검증한다.
"""

from __future__ import annotations

import pytest

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult
from src.controller.base import Action, ActionType, BaseController


class _MockController(BaseController):
    """테스트용 최소 구현체."""

    def propose_actions(self, state: ReactorState) -> list[Action]:
        if state.iteration >= 3:
            return [Action(action_type=ActionType.STOP, reason="max iterations")]
        return [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path="materials.fuel_enrichment",
                value=3.0 + state.iteration * 0.5,
                reason=f"iteration {state.iteration}",
            ),
        ]

    def apply_actions_to_case(
        self,
        actions: list[Action],
        state: ReactorState,
    ) -> CaseConfig:
        data = state.current_config.model_dump(mode="json")
        for action in actions:
            if action.action_type == ActionType.MODIFY_PARAM and action.field_path:
                keys = action.field_path.split(".")
                current = data
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = action.value
        return CaseConfig.model_validate(data)

    def evaluate_results(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        return {
            "keff": result.keff,
            "keff_std": result.keff_std,
            "keff_deviation": abs(result.keff - 1.0),
        }

    def update_state(
        self,
        state: ReactorState,
        metrics: dict[str, float],
    ) -> ReactorState:
        return state.model_copy(
            update={
                "kpi": metrics,
                "iteration": state.iteration + 1,
            },
        )


class TestActionType:
    """ActionType enum 테스트."""

    def test_values(self) -> None:
        assert ActionType.MODIFY_PARAM == "modify_param"
        assert ActionType.SET_CONFIG == "set_config"
        assert ActionType.STOP == "stop"

    def test_all_types(self) -> None:
        assert len(ActionType) == 3


class TestAction:
    """Action 모델 테스트."""

    def test_modify_param_action(self) -> None:
        action = Action(
            action_type=ActionType.MODIFY_PARAM,
            field_path="materials.fuel_enrichment",
            value=4.5,
            reason="increase enrichment",
        )
        assert action.action_type == ActionType.MODIFY_PARAM
        assert action.field_path == "materials.fuel_enrichment"
        assert action.value == 4.5

    def test_stop_action(self) -> None:
        action = Action(
            action_type=ActionType.STOP,
            reason="converged",
        )
        assert action.action_type == ActionType.STOP
        assert action.field_path is None
        assert action.value is None

    def test_frozen(self) -> None:
        action = Action(action_type=ActionType.STOP)
        with pytest.raises(Exception):
            action.reason = "changed"  # type: ignore[misc]

    def test_default_reason(self) -> None:
        action = Action(action_type=ActionType.STOP)
        assert action.reason == ""


class TestBaseControllerABC:
    """BaseController ABC 테스트."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseController()  # type: ignore[abstract]

    def test_mock_controller_instantiates(self) -> None:
        ctrl = _MockController()
        assert isinstance(ctrl, BaseController)


class TestMockControllerPropose:
    """propose_actions 테스트."""

    def test_proposes_modify_action(self) -> None:
        ctrl = _MockController()
        state = ReactorState()
        actions = ctrl.propose_actions(state)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.MODIFY_PARAM

    def test_proposes_stop_at_max_iterations(self) -> None:
        ctrl = _MockController()
        state = ReactorState(iteration=3)
        actions = ctrl.propose_actions(state)
        assert actions[0].action_type == ActionType.STOP

    def test_enrichment_increases_per_iteration(self) -> None:
        ctrl = _MockController()
        values = []
        for i in range(3):
            state = ReactorState(iteration=i)
            actions = ctrl.propose_actions(state)
            values.append(actions[0].value)
        assert values == [3.0, 3.5, 4.0]


class TestMockControllerApply:
    """apply_actions_to_case 테스트."""

    def test_applies_param_change(self) -> None:
        ctrl = _MockController()
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
        ctrl = _MockController()
        state = ReactorState()
        actions = [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path="materials.fuel_enrichment",
                value=4.5,
            ),
        ]
        config = ctrl.apply_actions_to_case(actions, state)
        # 기본값 유지
        assert config.geometry.pitch == state.current_config.geometry.pitch


class TestMockControllerEvaluate:
    """evaluate_results 테스트."""

    def test_returns_metrics(self) -> None:
        ctrl = _MockController()
        result = SimulationResult(keff=1.02, keff_std=0.001, runtime=30.0)
        metrics = ctrl.evaluate_results(result)
        assert "keff" in metrics
        assert "keff_deviation" in metrics
        assert abs(metrics["keff_deviation"] - 0.02) < 1e-10


class TestMockControllerUpdateState:
    """update_state 테스트."""

    def test_increments_iteration(self) -> None:
        ctrl = _MockController()
        state = ReactorState(iteration=0)
        metrics = {"keff": 1.02, "keff_std": 0.001, "keff_deviation": 0.02}
        new_state = ctrl.update_state(state, metrics)
        assert new_state.iteration == 1

    def test_updates_kpi(self) -> None:
        ctrl = _MockController()
        state = ReactorState()
        metrics = {"keff": 1.02, "keff_std": 0.001, "keff_deviation": 0.02}
        new_state = ctrl.update_state(state, metrics)
        assert new_state.kpi["keff"] == 1.02


class TestControllerLoop:
    """전체 제어 루프 패턴 테스트."""

    def test_loop_runs_to_completion(self) -> None:
        """propose → apply → evaluate → update 루프가 정상 동작."""
        ctrl = _MockController()
        state = ReactorState()

        iterations = 0
        for _ in range(10):  # 안전 가드
            actions = ctrl.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break

            ctrl.apply_actions_to_case(actions, state)
            # 시뮬레이션 결과를 가정
            result = SimulationResult(
                keff=1.0 + 0.01 * state.iteration,
                keff_std=0.001,
                runtime=30.0,
            )
            metrics = ctrl.evaluate_results(result)
            state = ctrl.update_state(state, metrics)
            iterations += 1

        assert iterations == 3
        assert state.iteration == 3
