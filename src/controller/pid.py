"""PID / P-only 제어봉 컨트롤러.

원자력 발전소 제어봉 제어 시스템의 표준 방식을 구현한다.
keff 편차를 입력으로 받아 제어봉 이동량을 PID 공식으로 계산한다.

PID 제어 공식:
    movement = Kp * error + Ki * integral(error) + Kd * d(error)/dt

여기서:
    error = target_keff - current_keff
    양수 error → keff가 낮음 → 제어봉 인출 (위치 증가)
    음수 error → keff가 높음 → 제어봉 삽입 (위치 감소)

Gain Scheduling:
    Rod worth curve는 S자 형태로, 임계점 근처(rod≈15-25)에서 기울기가
    매우 가파르다 (~0.017 keff/step). 고정 게인은 이 비선형성을 처리할 수
    없으므로, 관측된 rod worth gradient를 기반으로 게인을 자동 조정한다.

    Kp_effective = Kp_base / (estimated_gradient / nominal_gradient)

Dead-band:
    임계점 근처에서 1 step 이동이 ~0.017 keff 변화를 유발하여 limit cycle이
    발생한다. Dead-band 내에서는 이동하지 않아 진동을 방지한다.

References:
    - Duderstadt & Hamilton, "Nuclear Reactor Analysis", Ch. 15
    - Glasstone & Sesonske, "Nuclear Reactor Engineering", Ch. 7
    - Åström & Hägglund, "Advanced PID Control", 2006 (gain scheduling)
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult
from src.controller.base import Action, ActionType, BaseController
from src.controller.utils import set_nested

logger = logging.getLogger(__name__)

ROD_FIELD_PATH = "geometry.extra_params.rod_position"

# Rod worth curve 기반 상수
# Rod scan 결과: 임계점(rod≈15-25) 근처에서 ~0.017 keff/step
NOMINAL_ROD_WORTH = 0.017  # keff per step (임계 영역 기준)


def _estimate_rod_worth(
    history: list[tuple[float, float]],
    default: float = NOMINAL_ROD_WORTH,
) -> float:
    """최근 관측 이력에서 local rod worth gradient를 추정한다.

    연속된 두 관측 (position, keff) 쌍에서 Δkeff/Δposition을 계산한다.

    Args:
        history: (rod_position, keff) 튜플 리스트 (최근순).
        default: 추정 불가 시 기본값.

    Returns:
        추정된 rod worth (keff per step, 양수).
    """
    if len(history) < 2:
        return default

    # 최근 2개 관측에서 gradient 계산
    pos1, keff1 = history[-2]
    pos2, keff2 = history[-1]
    delta_pos = abs(pos2 - pos1)
    delta_keff = abs(keff2 - keff1)

    if delta_pos < 0.1:  # 위치 변화 없으면 추정 불가
        return default

    estimated = delta_keff / delta_pos
    # 물리적 범위 제한: 0.001 ~ 0.1 keff/step
    return max(0.001, min(0.1, estimated))


class PIDControllerConfig(BaseModel):
    """PID 제어봉 컨트롤러 설정.

    Attributes:
        controller_type: 컨트롤러 유형 식별자.
        kp: 비례 게인 (steps per unit keff error).
        ki: 적분 게인 (steps per unit keff·step).
        kd: 미분 게인 (steps per unit keff/step).
        target_keff: 목표 keff 값.
        keff_tolerance: keff 수렴 허용 오차.
        initial_rod_position: 초기 제어봉 위치.
        rod_position_min: 제어봉 최소 위치.
        rod_position_max: 제어봉 최대 위치.
        max_single_movement: 단일 스텝 최대 이동량 (안전 제한).
        max_iterations: 최대 반복 횟수.
        integral_windup_limit: 적분 와인드업 제한.
        dead_band: dead-band 폭 (keff). 이 범위 내에서는 이동하지 않음.
        gain_scheduling: True이면 관측된 rod worth로 게인 자동 조정.
    """

    model_config = ConfigDict(extra="forbid")

    controller_type: Literal["pid"] = "pid"
    kp: float = Field(
        default=40.0,
        description="비례 게인 (steps per unit keff error)",
    )
    ki: float = Field(
        default=8.0,
        ge=0,
        description="적분 게인",
    )
    kd: float = Field(
        default=3.0,
        ge=0,
        description="미분 게인",
    )
    target_keff: float = Field(default=1.0, description="목표 keff")
    keff_tolerance: float = Field(default=0.05, gt=0, description="수렴 허용 오차")
    keff_critical_deviation: float = Field(
        default=0.50,
        gt=0,
        description="keff 이탈 판정 기준",
    )
    initial_rod_position: float = Field(default=14.0, ge=0)
    rod_position_min: float = Field(default=0.0, ge=0)
    rod_position_max: float = Field(default=228.0, gt=0)
    max_single_movement: float = Field(default=7.0, gt=0)
    max_iterations: int = Field(default=50, gt=0)
    integral_windup_limit: float = Field(
        default=0.1,
        gt=0,
        description="적분 누적 상한 (anti-windup)",
    )
    dead_band: float = Field(
        default=0.008,
        ge=0,
        description="dead-band 폭 (keff). limit cycle 방지.",
    )
    gain_scheduling: bool = Field(
        default=True,
        description="True이면 관측된 rod worth gradient로 게인 자동 조정.",
    )

    @model_validator(mode="after")
    def validate_rod_range(self) -> PIDControllerConfig:
        """제어봉 위치 범위 유효성 검증."""
        if self.rod_position_min >= self.rod_position_max:
            msg = (
                f"rod_position_min({self.rod_position_min})은 "
                f"rod_position_max({self.rod_position_max})보다 작아야 합니다"
            )
            raise ValueError(msg)
        in_range = (
            self.rod_position_min <= self.initial_rod_position <= self.rod_position_max
        )
        if not in_range:
            msg = (
                f"initial_rod_position({self.initial_rod_position})은 "
                f"[{self.rod_position_min}, {self.rod_position_max}] 범위여야 합니다"
            )
            raise ValueError(msg)
        return self


class ProportionalControllerConfig(BaseModel):
    """P-only 제어봉 컨트롤러 설정.

    PID에서 적분/미분 항을 제거한 비례 제어만 사용.

    Attributes:
        controller_type: 컨트롤러 유형 식별자.
        kp: 비례 게인.
        target_keff: 목표 keff.
        keff_tolerance: 수렴 허용 오차.
        initial_rod_position: 초기 제어봉 위치.
        rod_position_min: 제어봉 최소 위치.
        rod_position_max: 제어봉 최대 위치.
        max_single_movement: 단일 스텝 최대 이동량.
        max_iterations: 최대 반복 횟수.
        dead_band: dead-band 폭 (keff).
        gain_scheduling: 게인 스케줄링 사용 여부.
    """

    model_config = ConfigDict(extra="forbid")

    controller_type: Literal["proportional"] = "proportional"
    kp: float = Field(
        default=40.0,
        description="비례 게인 (steps per unit keff error)",
    )
    target_keff: float = Field(default=1.0)
    keff_tolerance: float = Field(default=0.05, gt=0)
    keff_critical_deviation: float = Field(default=0.50, gt=0)
    initial_rod_position: float = Field(default=14.0, ge=0)
    rod_position_min: float = Field(default=0.0, ge=0)
    rod_position_max: float = Field(default=228.0, gt=0)
    max_single_movement: float = Field(default=7.0, gt=0)
    max_iterations: int = Field(default=50, gt=0)
    dead_band: float = Field(
        default=0.008,
        ge=0,
        description="dead-band 폭 (keff).",
    )
    gain_scheduling: bool = Field(
        default=True,
        description="True이면 관측된 rod worth gradient로 게인 자동 조정.",
    )

    @model_validator(mode="after")
    def validate_rod_range(self) -> ProportionalControllerConfig:
        """제어봉 위치 범위 유효성 검증."""
        if self.rod_position_min >= self.rod_position_max:
            msg = (
                f"rod_position_min({self.rod_position_min})은 "
                f"rod_position_max({self.rod_position_max})보다 작아야 합니다"
            )
            raise ValueError(msg)
        in_range = (
            self.rod_position_min <= self.initial_rod_position <= self.rod_position_max
        )
        if not in_range:
            msg = (
                f"initial_rod_position({self.initial_rod_position})은 "
                f"[{self.rod_position_min}, {self.rod_position_max}] 범위여야 합니다"
            )
            raise ValueError(msg)
        return self


class PIDController(BaseController):
    """PID 제어봉 컨트롤러.

    원전 PID 피드백 제어를 구현한다.
    keff 편차를 기반으로 제어봉 위치를 조절한다.

    Float rod position, gain scheduling, dead-band을 지원하여
    rod worth curve의 비선형성과 limit cycle을 처리한다.

    Args:
        config: PID 컨트롤러 설정.
    """

    def __init__(self, config: PIDControllerConfig | None = None) -> None:
        self._config = config or PIDControllerConfig()
        self._rod_position: float = float(self._config.initial_rod_position)
        self._step = 0
        self._integral = 0.0
        self._prev_error = 0.0
        self._history: list[tuple[float, float]] = []  # (position, keff)

    @property
    def config(self) -> PIDControllerConfig:
        """컨트롤러 설정."""
        return self._config

    def _effective_kp(self) -> float:
        """현재 관측 기반 effective Kp를 계산한다.

        Gain scheduling: rod worth gradient가 높으면 gain을 낮추고,
        낮으면 gain을 높인다. 이로써 비선형 rod worth curve에 적응한다.
        """
        if not self._config.gain_scheduling:
            return self._config.kp

        estimated_worth = _estimate_rod_worth(self._history)
        ratio = estimated_worth / NOMINAL_ROD_WORTH
        # ratio > 1: gradient가 가파름 → gain 감소
        # ratio < 1: gradient가 완만 → gain 증가
        # 비율을 0.3~3.0으로 제한하여 과도한 조정 방지
        clamped_ratio = max(0.3, min(3.0, ratio))
        effective = self._config.kp / clamped_ratio

        logger.debug(
            "Gain scheduling: worth=%.4f, ratio=%.2f, Kp=%.1f→%.1f",
            estimated_worth,
            ratio,
            self._config.kp,
            effective,
        )
        return effective

    def propose_actions(self, state: ReactorState) -> list[Action]:
        """PID 공식으로 제어봉 이동을 제안한다."""
        if self._step >= self._config.max_iterations:
            return [
                Action(
                    action_type=ActionType.STOP,
                    reason=f"최대 반복 {self._config.max_iterations}회 도달",
                )
            ]

        # 첫 스텝: 초기 위치로 설정
        if self._step == 0 and not state.kpi:
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason=f"PID 초기 위치: {self._rod_position:.1f}",
                )
            ]

        if not state.kpi or "keff" not in state.kpi:
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason="keff 정보 없음, 현재 위치 유지",
                )
            ]

        keff = state.kpi["keff"]
        error = self._config.target_keff - keff
        deviation = abs(error)

        # 수렴 판정: tolerance 이내이면 위치 유지
        if deviation <= self._config.keff_tolerance:
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason=(
                        f"PID 수렴 유지: keff={keff:.5f}, rod={self._rod_position:.1f}"
                    ),
                )
            ]

        # Dead-band: 작은 오차에서는 이동하지 않음 (limit cycle 방지)
        if deviation <= self._config.dead_band:
            logger.info(
                "PID dead-band: error=%.5f <= dead_band=%.4f, 이동 안 함",
                error,
                self._config.dead_band,
            )
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason=f"PID dead-band: error={error:.5f} 내",
                )
            ]

        # Gain scheduling 적용
        kp_eff = self._effective_kp()

        # 적분 (conditional anti-windup: dead-band 밖에서만 적분)
        self._integral += error
        self._integral = max(
            -self._config.integral_windup_limit,
            min(self._config.integral_windup_limit, self._integral),
        )

        # 미분
        derivative = error - self._prev_error
        self._prev_error = error

        # PID 출력: 제어봉 이동량 (float)
        raw_movement = (
            kp_eff * error
            + self._config.ki * self._integral
            + self._config.kd * derivative
        )

        # 이동량 제한 (float 유지, 정수화하지 않음)
        movement = max(
            -self._config.max_single_movement,
            min(self._config.max_single_movement, raw_movement),
        )

        # 소수점 1자리로 반올림 (OpenMC 입력 정밀도)
        movement = round(movement, 1)

        # 위치 클램핑
        new_position = round(
            max(
                self._config.rod_position_min,
                min(self._config.rod_position_max, self._rod_position + movement),
            ),
            1,
        )

        logger.info(
            "PID 제어: error=%.5f, Kp_eff=%.1f, P=%.2f, I=%.2f, D=%.2f, "
            "raw=%.2f, movement=%.1f, position=%.1f→%.1f",
            error,
            kp_eff,
            kp_eff * error,
            self._config.ki * self._integral,
            self._config.kd * derivative,
            raw_movement,
            movement,
            self._rod_position,
            new_position,
        )

        self._rod_position = new_position
        return [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path=ROD_FIELD_PATH,
                value=new_position,
                reason=(
                    f"PID: error={error:.5f}, Kp_eff={kp_eff:.1f}, "
                    f"movement={movement:.1f}, position={new_position:.1f}"
                ),
            )
        ]

    def apply_actions_to_case(
        self,
        actions: list[Action],
        state: ReactorState,
    ) -> CaseConfig:
        """액션을 적용하여 새 CaseConfig를 생성한다."""
        data = state.current_config.model_dump(mode="json")
        for action in actions:
            if action.action_type == ActionType.MODIFY_PARAM and action.field_path:
                set_nested(data, action.field_path, action.value)
        return CaseConfig.model_validate(data)

    def evaluate_results(self, result: SimulationResult) -> dict[str, float]:
        """keff 기준으로 결과를 평가한다."""
        keff = result.keff
        deviation = abs(keff - self._config.target_keff)
        converged = 1.0 if deviation <= self._config.keff_tolerance else 0.0
        self._step += 1

        # 이력 저장 (gain scheduling용)
        self._history.append((self._rod_position, keff))

        return {
            "keff": keff,
            "keff_std": result.keff_std,
            "keff_deviation": deviation,
            "converged": converged,
            "rod_position": self._rod_position,
        }

    def update_state(
        self,
        state: ReactorState,
        metrics: dict[str, float],
    ) -> ReactorState:
        """메트릭으로 상태를 업데이트한다."""
        return state.model_copy(
            update={
                "kpi": metrics,
                "iteration": state.iteration + 1,
            },
        )


class ProportionalController(BaseController):
    """비례(P-only) 제어봉 컨트롤러.

    PID에서 I, D 항을 제거한 비례 제어만 사용.
    Float position, gain scheduling, dead-band을 지원한다.

    Args:
        config: P-only 컨트롤러 설정.
    """

    def __init__(self, config: ProportionalControllerConfig | None = None) -> None:
        self._config = config or ProportionalControllerConfig()
        self._rod_position: float = float(self._config.initial_rod_position)
        self._step = 0
        self._history: list[tuple[float, float]] = []  # (position, keff)

    @property
    def config(self) -> ProportionalControllerConfig:
        """컨트롤러 설정."""
        return self._config

    def _effective_kp(self) -> float:
        """현재 관측 기반 effective Kp를 계산한다."""
        if not self._config.gain_scheduling:
            return self._config.kp

        estimated_worth = _estimate_rod_worth(self._history)
        ratio = estimated_worth / NOMINAL_ROD_WORTH
        clamped_ratio = max(0.3, min(3.0, ratio))
        return self._config.kp / clamped_ratio

    def propose_actions(self, state: ReactorState) -> list[Action]:
        """비례 제어로 제어봉 이동을 제안한다."""
        if self._step >= self._config.max_iterations:
            return [
                Action(
                    action_type=ActionType.STOP,
                    reason=f"최대 반복 {self._config.max_iterations}회 도달",
                )
            ]

        # 첫 스텝 또는 keff 없음: 초기 위치
        if not state.kpi or "keff" not in state.kpi:
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason=f"P-only 초기 위치: {self._rod_position:.1f}",
                )
            ]

        keff = state.kpi["keff"]
        error = self._config.target_keff - keff
        deviation = abs(error)

        # 수렴 시 위치 유지
        if deviation <= self._config.keff_tolerance:
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason=(
                        f"P-only 수렴 유지: keff={keff:.5f}, "
                        f"rod={self._rod_position:.1f}"
                    ),
                )
            ]

        # Dead-band
        if deviation <= self._config.dead_band:
            return [
                Action(
                    action_type=ActionType.MODIFY_PARAM,
                    field_path=ROD_FIELD_PATH,
                    value=self._rod_position,
                    reason=f"P-only dead-band: error={error:.5f} 내",
                )
            ]

        # Gain scheduling 적용
        kp_eff = self._effective_kp()

        # P 제어: movement = Kp * error (float)
        raw_movement = kp_eff * error

        movement = max(
            -self._config.max_single_movement,
            min(self._config.max_single_movement, raw_movement),
        )
        movement = round(movement, 1)

        new_position = round(
            max(
                self._config.rod_position_min,
                min(self._config.rod_position_max, self._rod_position + movement),
            ),
            1,
        )

        logger.info(
            "P-only 제어: error=%.5f, Kp_eff=%.1f, raw=%.2f, "
            "movement=%.1f, position=%.1f→%.1f",
            error,
            kp_eff,
            raw_movement,
            movement,
            self._rod_position,
            new_position,
        )

        self._rod_position = new_position
        return [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path=ROD_FIELD_PATH,
                value=new_position,
                reason=(
                    f"P-only: error={error:.5f}, Kp_eff={kp_eff:.1f}, "
                    f"movement={movement:.1f}, position={new_position:.1f}"
                ),
            )
        ]

    def apply_actions_to_case(
        self,
        actions: list[Action],
        state: ReactorState,
    ) -> CaseConfig:
        """액션을 적용하여 새 CaseConfig를 생성한다."""
        data = state.current_config.model_dump(mode="json")
        for action in actions:
            if action.action_type == ActionType.MODIFY_PARAM and action.field_path:
                set_nested(data, action.field_path, action.value)
        return CaseConfig.model_validate(data)

    def evaluate_results(self, result: SimulationResult) -> dict[str, float]:
        """keff 기준으로 결과를 평가한다."""
        keff = result.keff
        deviation = abs(keff - self._config.target_keff)
        converged = 1.0 if deviation <= self._config.keff_tolerance else 0.0
        self._step += 1

        # 이력 저장 (gain scheduling용)
        self._history.append((self._rod_position, keff))

        return {
            "keff": keff,
            "keff_std": result.keff_std,
            "keff_deviation": deviation,
            "converged": converged,
            "rod_position": self._rod_position,
        }

    def update_state(
        self,
        state: ReactorState,
        metrics: dict[str, float],
    ) -> ReactorState:
        """메트릭으로 상태를 업데이트한다."""
        return state.model_copy(
            update={
                "kpi": metrics,
                "iteration": state.iteration + 1,
            },
        )


__all__ = [
    "NOMINAL_ROD_WORTH",
    "PIDController",
    "PIDControllerConfig",
    "ProportionalController",
    "ProportionalControllerConfig",
]
