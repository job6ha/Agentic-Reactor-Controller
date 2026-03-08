"""PID / P-only 제어봉 컨트롤러 테스트.

float position, gain scheduling, dead-band 기능을 포함한
PID/P-only 컨트롤러의 동작을 검증한다.
"""

from __future__ import annotations

import pytest

from src.armi_layer.models import ReactorState, SimulationResult
from src.controller.base import ActionType
from src.controller.pid import (
    NOMINAL_ROD_WORTH,
    PIDController,
    PIDControllerConfig,
    ProportionalController,
    ProportionalControllerConfig,
    _estimate_rod_worth,
)

# ---------------------------------------------------------------------------
# Config 테스트
# ---------------------------------------------------------------------------


class TestPIDControllerConfig:
    """PIDControllerConfig 검증."""

    def test_defaults(self) -> None:
        config = PIDControllerConfig()
        assert config.controller_type == "pid"
        assert config.kp == 40.0
        assert config.ki == 8.0
        assert config.kd == 3.0
        assert config.initial_rod_position == 14.0
        assert config.dead_band == 0.008
        assert config.gain_scheduling is True
        assert config.max_iterations == 50

    def test_float_positions(self) -> None:
        config = PIDControllerConfig(initial_rod_position=14.5)
        assert config.initial_rod_position == 14.5

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="rod_position_min"):
            PIDControllerConfig(rod_position_min=228.0, rod_position_max=100.0)

    def test_initial_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_rod_position"):
            PIDControllerConfig(initial_rod_position=300.0)


class TestProportionalControllerConfig:
    """ProportionalControllerConfig 검증."""

    def test_defaults(self) -> None:
        config = ProportionalControllerConfig()
        assert config.controller_type == "proportional"
        assert config.kp == 40.0
        assert config.dead_band == 0.008
        assert config.gain_scheduling is True


# ---------------------------------------------------------------------------
# Rod worth 추정 테스트
# ---------------------------------------------------------------------------


class TestEstimateRodWorth:
    """_estimate_rod_worth 함수 검증."""

    def test_insufficient_history_returns_default(self) -> None:
        assert _estimate_rod_worth([]) == NOMINAL_ROD_WORTH
        assert _estimate_rod_worth([(14.0, 0.85)]) == NOMINAL_ROD_WORTH

    def test_basic_gradient(self) -> None:
        history = [(14.0, 0.85), (21.0, 0.97)]
        # Δkeff=0.12, Δpos=7 → gradient = 0.12/7 ≈ 0.0171
        worth = _estimate_rod_worth(history)
        assert 0.015 < worth < 0.020

    def test_no_position_change_returns_default(self) -> None:
        history = [(14.0, 0.85), (14.0, 0.86)]
        assert _estimate_rod_worth(history) == NOMINAL_ROD_WORTH

    def test_clamped_to_physical_range(self) -> None:
        # 극단적 gradient
        history = [(14.0, 0.5), (14.1, 1.5)]
        worth = _estimate_rod_worth(history)
        assert worth <= 0.1  # 상한

        history2 = [(14.0, 0.85), (114.0, 0.86)]
        worth2 = _estimate_rod_worth(history2)
        assert worth2 >= 0.001  # 하한


# ---------------------------------------------------------------------------
# PID Controller 동작 테스트
# ---------------------------------------------------------------------------


class TestPIDController:
    """PIDController 동작 검증."""

    @pytest.fixture
    def controller(self) -> PIDController:
        config = PIDControllerConfig(
            kp=40.0,
            ki=8.0,
            kd=3.0,
            initial_rod_position=14.0,
            dead_band=0.008,
            gain_scheduling=False,  # 테스트 단순화
        )
        return PIDController(config)

    def test_initial_action(self, controller: PIDController) -> None:
        """첫 스텝: 초기 위치 설정."""
        state = ReactorState()
        actions = controller.propose_actions(state)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.MODIFY_PARAM
        assert actions[0].value == 14.0

    def test_float_movement(self, controller: PIDController) -> None:
        """이동량이 float으로 유지되는지 확인."""
        state = ReactorState(kpi={"keff": 0.90})
        actions = controller.propose_actions(state)
        assert len(actions) == 1
        # error=0.10, Kp=40 → raw_movement=4.0 + ki*0.1 + kd*0.1
        value = actions[0].value
        assert isinstance(value, float)
        # 이동 후 위치가 14.0보다 커야 함 (keff < 1.0 → 인출)
        assert value > 14.0

    def test_convergence_holds_position(self, controller: PIDController) -> None:
        """수렴 시 위치를 유지한다."""
        state = ReactorState(kpi={"keff": 0.98})  # deviation=0.02 < 0.05
        actions = controller.propose_actions(state)
        assert actions[0].value == 14.0
        assert "수렴 유지" in actions[0].reason

    def test_dead_band_prevents_movement(self, controller: PIDController) -> None:
        """Dead-band 내에서는 이동하지 않는다."""
        # deviation=0.006 < dead_band=0.008, but > tolerance=0.05 아님
        # 실제로 deviation=0.006은 tolerance 0.05 이내이므로 수렴으로 판정됨
        # dead-band가 작동하려면 tolerance < deviation < dead_band가 아니라
        # dead_band < deviation 일때만 이동, 아니면 정지
        # 테스트: dead_band=0.05, tolerance=0.01로 설정
        config = PIDControllerConfig(
            kp=40.0,
            initial_rod_position=14.0,
            keff_tolerance=0.005,
            dead_band=0.01,
            gain_scheduling=False,
        )
        ctrl = PIDController(config)
        # deviation=0.008: tolerance(0.005) < 0.008 < dead_band(0.01)
        state = ReactorState(kpi={"keff": 0.992})
        actions = ctrl.propose_actions(state)
        assert "dead-band" in actions[0].reason
        assert actions[0].value == 14.0

    def test_max_movement_clamp(self, controller: PIDController) -> None:
        """이동량이 max_single_movement로 제한된다."""
        state = ReactorState(kpi={"keff": 0.50})  # huge error=0.50
        actions = controller.propose_actions(state)
        # raw_movement = 40*0.5 + ... = ~20+ → clamped to 7.0
        movement = actions[0].value - 14.0
        assert abs(movement) <= 7.0

    def test_max_iterations_stop(self, controller: PIDController) -> None:
        """최대 반복 도달 시 STOP 액션을 반환한다."""
        controller._step = 50
        state = ReactorState(kpi={"keff": 0.90})
        actions = controller.propose_actions(state)
        assert actions[0].action_type == ActionType.STOP

    def test_evaluate_results_tracks_history(self, controller: PIDController) -> None:
        """evaluate_results가 이력을 저장한다."""
        result = SimulationResult(keff=0.95, keff_std=0.001, runtime=1.0)
        metrics = controller.evaluate_results(result)
        assert metrics["keff"] == 0.95
        assert metrics["rod_position"] == 14.0
        assert len(controller._history) == 1

    def test_direction_correctness(self, controller: PIDController) -> None:
        """keff < 1.0이면 인출(위치 증가), keff > 1.0이면 삽입(위치 감소)."""
        # keff 낮음 → 인출
        state_low = ReactorState(kpi={"keff": 0.80})
        actions_low = controller.propose_actions(state_low)
        assert actions_low[0].value > 14.0

        # 위치 리셋
        controller._rod_position = 20.0
        controller._integral = 0.0
        controller._prev_error = 0.0
        state_high = ReactorState(kpi={"keff": 1.10})
        actions_high = controller.propose_actions(state_high)
        assert actions_high[0].value < 20.0


# ---------------------------------------------------------------------------
# Gain Scheduling 테스트
# ---------------------------------------------------------------------------


class TestGainScheduling:
    """Gain scheduling 동작 검증."""

    def test_high_gradient_reduces_gain(self) -> None:
        """Rod worth gradient가 높으면 gain이 감소한다."""
        config = PIDControllerConfig(
            kp=40.0,
            gain_scheduling=True,
            initial_rod_position=14.0,
        )
        ctrl = PIDController(config)
        # 높은 gradient: 2 steps에서 0.05 keff 변화 → 0.025/step
        ctrl._history = [(14.0, 0.85), (16.0, 0.90)]
        kp_eff = ctrl._effective_kp()
        # ratio = 0.025/0.017 ≈ 1.47 → Kp_eff = 40/1.47 ≈ 27
        assert kp_eff < 40.0

    def test_low_gradient_increases_gain(self) -> None:
        """Rod worth gradient가 낮으면 gain이 증가한다."""
        config = PIDControllerConfig(
            kp=40.0,
            gain_scheduling=True,
            initial_rod_position=100.0,
        )
        ctrl = PIDController(config)
        # 낮은 gradient: 20 steps에서 0.01 keff 변화 → 0.0005/step
        ctrl._history = [(100.0, 1.34), (120.0, 1.35)]
        kp_eff = ctrl._effective_kp()
        # ratio = 0.0005/0.017 ≈ 0.03 → clamped to 0.3 → Kp_eff = 40/0.3 ≈ 133
        assert kp_eff > 40.0

    def test_disabled_returns_base_kp(self) -> None:
        """gain_scheduling=False이면 기본 Kp를 반환한다."""
        config = PIDControllerConfig(kp=40.0, gain_scheduling=False)
        ctrl = PIDController(config)
        ctrl._history = [(14.0, 0.85), (16.0, 0.90)]
        assert ctrl._effective_kp() == 40.0


# ---------------------------------------------------------------------------
# P-only Controller 동작 테스트
# ---------------------------------------------------------------------------


class TestProportionalController:
    """ProportionalController 동작 검증."""

    @pytest.fixture
    def controller(self) -> ProportionalController:
        config = ProportionalControllerConfig(
            kp=40.0,
            initial_rod_position=14.0,
            dead_band=0.008,
            gain_scheduling=False,
        )
        return ProportionalController(config)

    def test_initial_action(self, controller: ProportionalController) -> None:
        state = ReactorState()
        actions = controller.propose_actions(state)
        assert actions[0].value == 14.0

    def test_proportional_movement(self, controller: ProportionalController) -> None:
        """P-only: movement = Kp * error (I, D 없음)."""
        state = ReactorState(kpi={"keff": 0.90})
        actions = controller.propose_actions(state)
        # error=0.10, Kp=40 → movement=4.0
        expected_pos = 14.0 + 4.0
        assert actions[0].value == expected_pos

    def test_no_integral_drift(self, controller: ProportionalController) -> None:
        """P-only는 적분 누적이 없다."""
        state1 = ReactorState(kpi={"keff": 0.90})
        controller.propose_actions(state1)
        controller.evaluate_results(
            SimulationResult(keff=0.90, keff_std=0.001, runtime=1.0),
        )

        # 같은 error로 다시 호출해도 적분 효과 없음
        controller._rod_position = 14.0  # 위치 리셋
        state2 = ReactorState(kpi={"keff": 0.90})
        actions2 = controller.propose_actions(state2)
        # 동일한 movement
        assert actions2[0].value == 14.0 + 4.0

    def test_convergence_holds(self, controller: ProportionalController) -> None:
        state = ReactorState(kpi={"keff": 1.01})
        actions = controller.propose_actions(state)
        assert "수렴 유지" in actions[0].reason


# ---------------------------------------------------------------------------
# 통합 시뮬레이션 루프 테스트
# ---------------------------------------------------------------------------


class TestControlLoop:
    """제어 루프 통합 테스트 (mock 없이 순수 로직)."""

    def test_pid_converges_on_linear_plant(self) -> None:
        """선형 plant 모델에서 PID가 수렴하는지 검증.

        Mock plant: keff = 0.5 + 0.02 * rod_position (선형)
        Critical position: rod = 25.0 (keff=1.0)
        """
        config = PIDControllerConfig(
            kp=30.0,
            ki=5.0,
            kd=2.0,
            initial_rod_position=14.0,
            keff_tolerance=0.01,
            dead_band=0.005,
            gain_scheduling=False,
            max_iterations=30,
        )
        ctrl = PIDController(config)
        state = ReactorState()

        for step in range(30):
            actions = ctrl.propose_actions(state)
            if actions[0].action_type == ActionType.STOP:
                break

            rod_pos = actions[0].value
            # Linear plant model
            keff = 0.5 + 0.02 * rod_pos
            keff_std = 0.001

            result = SimulationResult(keff=keff, keff_std=keff_std, runtime=1.0)
            case_config = ctrl.apply_actions_to_case(actions, state)
            metrics = ctrl.evaluate_results(result)
            state = ctrl.update_state(state, metrics)
            state = state.model_copy(update={"current_config": case_config})

            if metrics["converged"] > 0.5:
                break

        # 수렴 확인: rod ≈ 25.0, keff ≈ 1.0
        assert abs(state.kpi["keff"] - 1.0) < 0.01
        assert abs(state.kpi["rod_position"] - 25.0) < 2.0

    def test_proportional_approaches_target(self) -> None:
        """P-only가 목표에 접근하는지 검증 (정상상태 오차 허용)."""
        config = ProportionalControllerConfig(
            kp=30.0,
            initial_rod_position=14.0,
            keff_tolerance=0.05,
            dead_band=0.005,
            gain_scheduling=False,
            max_iterations=20,
        )
        ctrl = ProportionalController(config)
        state = ReactorState()

        final_keff = 0.0
        for step in range(20):
            actions = ctrl.propose_actions(state)
            if actions[0].action_type == ActionType.STOP:
                break

            rod_pos = actions[0].value
            keff = 0.5 + 0.02 * rod_pos
            keff_std = 0.001

            result = SimulationResult(keff=keff, keff_std=keff_std, runtime=1.0)
            case_config = ctrl.apply_actions_to_case(actions, state)
            metrics = ctrl.evaluate_results(result)
            state = ctrl.update_state(state, metrics)
            state = state.model_copy(update={"current_config": case_config})
            final_keff = keff

            if metrics["converged"] > 0.5:
                break

        # P-only는 offset이 있을 수 있지만 tolerance 이내
        assert abs(final_keff - 1.0) < 0.05
