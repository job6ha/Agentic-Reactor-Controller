"""LLMController 통합 테스트 (KAE-90)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult
from src.controller.base import ActionType
from src.controller.llm.controller import LLMController
from src.controller.llm.models import LLMControllerConfig


@pytest.fixture()
def llm_config(tmp_path: Path) -> LLMControllerConfig:
    """테스트용 LLMController 설정."""
    return LLMControllerConfig(
        n_candidates=5,
        initial_rod_position=228,
        max_iterations=3,
        target_keff=1.0,
        keff_tolerance=0.01,
        keff_critical_deviation=0.05,
        log_dir=tmp_path / "logs",
    )


@pytest.fixture()
def controller(llm_config: LLMControllerConfig) -> LLMController:
    """LLM mock이 적용된 LLMController."""
    with patch(
        "src.controller.llm.controller.LLMPlanner"
    ) as MockPlanner:
        mock_planner = MockPlanner.return_value
        mock_planner.generate.return_value = [218, 223, 225, 228, 233]

        ctrl = LLMController(llm_config)
        ctrl._planner = mock_planner
        return ctrl


class TestLLMControllerInit:
    """LLMController 초기화 테스트."""

    def test_initial_state(self, controller: LLMController) -> None:
        assert controller.rod_position == 228
        assert controller.step == 0

    def test_config_accessible(self, controller: LLMController) -> None:
        assert controller.config.n_candidates == 5
        assert controller.config.max_iterations == 3


class TestProposeActions:
    """propose_actions 테스트."""

    def test_returns_modify_param_action(self, controller: LLMController) -> None:
        state = ReactorState()
        actions = controller.propose_actions(state)

        assert len(actions) == 1
        assert actions[0].action_type == ActionType.MODIFY_PARAM
        assert actions[0].field_path == "geometry.extra_params.rod_position"

    def test_stop_at_max_iterations(self, controller: LLMController) -> None:
        controller._step = 3  # max_iterations = 3
        state = ReactorState()

        actions = controller.propose_actions(state)

        assert actions[0].action_type == ActionType.STOP
        assert "최대 스텝" in actions[0].reason

    def test_stop_on_keff_deviation(self, controller: LLMController) -> None:
        state = ReactorState(kpi={"keff": 1.08})  # deviation=0.08 > 0.05

        actions = controller.propose_actions(state)

        assert actions[0].action_type == ActionType.STOP
        assert "이탈" in actions[0].reason

    def test_no_stop_within_tolerance(self, controller: LLMController) -> None:
        state = ReactorState(kpi={"keff": 1.03})  # deviation=0.03 < 0.05

        actions = controller.propose_actions(state)

        assert actions[0].action_type == ActionType.MODIFY_PARAM

    def test_selects_best_candidate(self, controller: LLMController) -> None:
        """LLM 후보 중 최고 안전 점수 후보를 선택."""
        state = ReactorState()
        controller.propose_actions(state)

        # _last_best가 설정되어야 함
        assert controller._last_best is not None
        assert controller._last_best.safety_score >= 0


class TestApplyActionsToCase:
    """apply_actions_to_case 테스트."""

    def test_applies_rod_position(self, controller: LLMController) -> None:
        state = ReactorState()
        actions = controller.propose_actions(state)

        config = controller.apply_actions_to_case(actions, state)

        assert isinstance(config, CaseConfig)
        assert "rod_position" in config.geometry.extra_params

    def test_preserves_other_params(self, controller: LLMController) -> None:
        state = ReactorState()
        actions = controller.propose_actions(state)

        config = controller.apply_actions_to_case(actions, state)

        assert config.geometry.fuel_radius == state.current_config.geometry.fuel_radius
        assert (
            config.materials.fuel_enrichment
            == state.current_config.materials.fuel_enrichment
        )


class TestEvaluateResults:
    """evaluate_results 테스트."""

    def test_returns_metrics(self, controller: LLMController) -> None:
        state = ReactorState()
        controller.propose_actions(state)

        result = SimulationResult(keff=1.002, keff_std=0.001, runtime=30.0)
        metrics = controller.evaluate_results(result)

        assert metrics["keff"] == 1.002
        assert metrics["keff_std"] == 0.001
        assert "safety_score" in metrics
        assert "rod_position" in metrics
        assert "converged" in metrics

    def test_logs_entry(self, controller: LLMController) -> None:
        state = ReactorState()
        controller.propose_actions(state)

        result = SimulationResult(keff=1.002, keff_std=0.001, runtime=30.0)
        controller.evaluate_results(result)

        assert controller._log_store.count() == 1

    def test_increments_step(self, controller: LLMController) -> None:
        state = ReactorState()
        controller.propose_actions(state)

        result = SimulationResult(keff=1.002, keff_std=0.001, runtime=30.0)
        controller.evaluate_results(result)

        assert controller.step == 1

    def test_convergence_detected(self, controller: LLMController) -> None:
        state = ReactorState()
        controller.propose_actions(state)

        result = SimulationResult(keff=1.005, keff_std=0.001, runtime=30.0)
        metrics = controller.evaluate_results(result)

        assert metrics["converged"] == 1.0

    def test_no_convergence(self, controller: LLMController) -> None:
        state = ReactorState()
        controller.propose_actions(state)

        result = SimulationResult(keff=1.03, keff_std=0.001, runtime=30.0)
        metrics = controller.evaluate_results(result)

        assert metrics["converged"] == 0.0


class TestUpdateState:
    """update_state 테스트."""

    def test_increments_iteration(self, controller: LLMController) -> None:
        state = ReactorState(iteration=0)
        metrics = {"keff": 1.002, "keff_std": 0.001}

        new_state = controller.update_state(state, metrics)

        assert new_state.iteration == 1

    def test_updates_kpi(self, controller: LLMController) -> None:
        state = ReactorState()
        metrics = {"keff": 1.002, "rod_position": 220.0}

        new_state = controller.update_state(state, metrics)

        assert new_state.kpi["keff"] == 1.002
        assert new_state.kpi["rod_position"] == 220.0


class TestControllerLoop:
    """전체 제어 루프 통합 테스트."""

    def test_full_loop(self, controller: LLMController) -> None:
        """3 iteration 루프 실행."""
        state = ReactorState()
        iterations = 0

        for _ in range(10):
            actions = controller.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break

            config = controller.apply_actions_to_case(actions, state)
            result = SimulationResult(
                keff=1.01 - 0.003 * iterations,
                keff_std=0.001,
                runtime=30.0,
            )
            metrics = controller.evaluate_results(result)
            state = controller.update_state(state, metrics)
            state = state.model_copy(update={"current_config": config})
            iterations += 1

        # max_iterations=3이므로 3회 실행 후 종료
        assert iterations == 3
        assert controller._log_store.count() == 3

    def test_keff_deviation_stops_early(self, controller: LLMController) -> None:
        """keff 이탈 시 조기 종료."""
        state = ReactorState()
        iterations = 0

        for _ in range(10):
            actions = controller.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break

            config = controller.apply_actions_to_case(actions, state)
            # 2번째에서 keff가 크게 이탈
            keff = 1.002 if iterations == 0 else 1.08
            result = SimulationResult(keff=keff, keff_std=0.001, runtime=30.0)
            metrics = controller.evaluate_results(result)
            state = controller.update_state(state, metrics)
            state = state.model_copy(update={"current_config": config})
            iterations += 1

        assert iterations == 2  # 2번째 평가 후 keff=1.08 → 3번째에서 STOP


class TestLogRestoration:
    """기존 로그 복원 테스트."""

    def test_restores_from_existing_log(self, tmp_path: Path) -> None:
        """기존 로그가 있으면 상태를 복원한다."""
        config = LLMControllerConfig(
            n_candidates=5,
            max_iterations=10,
            log_dir=tmp_path / "logs",
        )

        # 1차 컨트롤러: 2 스텝 실행
        with patch("src.controller.llm.controller.LLMPlanner") as MockPlanner:
            mock = MockPlanner.return_value
            mock.generate.return_value = [-5, -3, 0, 3, 5]
            ctrl1 = LLMController(config)
            ctrl1._planner = mock

            state = ReactorState()
            for i in range(2):
                actions = ctrl1.propose_actions(state)
                ctrl1.apply_actions_to_case(actions, state)
                result = SimulationResult(keff=1.005, keff_std=0.001, runtime=30.0)
                metrics = ctrl1.evaluate_results(result)
                state = ctrl1.update_state(state, metrics)

            assert ctrl1.step == 2

        # 2차 컨트롤러: 기존 로그에서 복원
        with patch("src.controller.llm.controller.LLMPlanner") as MockPlanner2:
            mock2 = MockPlanner2.return_value
            mock2.generate.return_value = [-5, -3, 0, 3, 5]
            ctrl2 = LLMController(config)
            ctrl2._planner = mock2

            assert ctrl2.step == 2
            assert ctrl2._optimizer.is_fitted
