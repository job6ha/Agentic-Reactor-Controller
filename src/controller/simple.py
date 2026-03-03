"""단순 룰 기반 컨트롤러.

미리 정의된 파라미터 값 목록을 순차 탐색하면서
keff 수렴 여부를 판정하는 베이스라인 컨트롤러.
에이전트 교체 전 기본 동작을 검증하는 용도로 사용한다.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult
from src.controller.base import Action, ActionType, BaseController

logger = logging.getLogger(__name__)

# keff 수렴 기본 기준
DEFAULT_TARGET_KEFF = 1.0
DEFAULT_KEFF_TOLERANCE = 0.01
DEFAULT_MAX_ITERATIONS = 10


class SimpleControllerConfig(BaseModel):
    """SimpleController 설정.

    Attributes:
        sweep_field: 스윕할 파라미터 경로 (dot notation).
        sweep_values: 순차 탐색할 값 목록.
        target_keff: 목표 keff 값.
        keff_tolerance: keff 수렴 허용 오차.
        max_iterations: 최대 반복 횟수.
    """

    model_config = ConfigDict(extra="forbid")

    sweep_field: str = Field(
        default="materials.fuel_enrichment",
        description="스윕할 파라미터 경로",
    )
    sweep_values: list[float] = Field(
        default_factory=lambda: [2.0, 3.0, 4.0, 5.0],
        description="순차 탐색할 값 목록",
    )
    target_keff: float = Field(
        default=DEFAULT_TARGET_KEFF,
        description="목표 keff 값",
    )
    keff_tolerance: float = Field(
        default=DEFAULT_KEFF_TOLERANCE,
        gt=0,
        description="keff 수렴 허용 오차",
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        gt=0,
        description="최대 반복 횟수",
    )


class SimpleController(BaseController):
    """단순 파라미터 스윕 기반 룰 컨트롤러.

    설정된 파라미터 값 목록을 순차적으로 탐색하면서
    keff가 목표값에 수렴하면 종료한다.

    Args:
        config: 컨트롤러 설정.
    """

    def __init__(self, config: SimpleControllerConfig | None = None) -> None:
        self._config = config or SimpleControllerConfig()

    @property
    def config(self) -> SimpleControllerConfig:
        """컨트롤러 설정."""
        return self._config

    def propose_actions(self, state: ReactorState) -> list[Action]:
        """다음 스윕 값을 제안하거나 종료를 결정한다.

        종료 조건:
        1. 최대 반복 횟수 도달
        2. 스윕 값 목록 소진
        3. keff가 목표값에 수렴

        Args:
            state: 현재 원자로 상태.

        Returns:
            Action 목록.
        """
        # 최대 반복 체크
        if state.iteration >= self._config.max_iterations:
            logger.info("최대 반복 도달: %d", state.iteration)
            return [
                Action(
                    action_type=ActionType.STOP,
                    reason=f"최대 반복 {self._config.max_iterations}회 도달",
                )
            ]

        # 스윕 값 소진 체크
        if state.iteration >= len(self._config.sweep_values):
            logger.info("스윕 값 소진: %d개", len(self._config.sweep_values))
            return [
                Action(
                    action_type=ActionType.STOP,
                    reason="스윕 값 목록 소진",
                )
            ]

        # keff 수렴 체크 (첫 반복이 아닌 경우)
        if state.kpi and "keff" in state.kpi:
            keff = state.kpi["keff"]
            deviation = abs(keff - self._config.target_keff)
            if deviation <= self._config.keff_tolerance:
                logger.info(
                    "keff 수렴: keff=%.5f, target=%.5f, tol=%.5f",
                    keff,
                    self._config.target_keff,
                    self._config.keff_tolerance,
                )
                return [
                    Action(
                        action_type=ActionType.STOP,
                        reason=(
                            f"keff 수렴 (|{keff:.5f} - {self._config.target_keff}|"
                            f" <= {self._config.keff_tolerance})"
                        ),
                    )
                ]

        # 다음 스윕 값 제안
        next_value = self._config.sweep_values[state.iteration]
        return [
            Action(
                action_type=ActionType.MODIFY_PARAM,
                field_path=self._config.sweep_field,
                value=next_value,
                reason=(
                    f"스윕 탐색 [{state.iteration + 1}"
                    f"/{len(self._config.sweep_values)}]"
                ),
            )
        ]

    def apply_actions_to_case(
        self,
        actions: list[Action],
        state: ReactorState,
    ) -> CaseConfig:
        """액션을 적용하여 새 CaseConfig를 생성한다.

        Args:
            actions: 적용할 액션 목록.
            state: 현재 원자로 상태.

        Returns:
            수정된 CaseConfig.
        """
        data = state.current_config.model_dump(mode="json")

        for action in actions:
            if action.action_type == ActionType.MODIFY_PARAM and action.field_path:
                _set_nested(data, action.field_path, action.value)

        config = CaseConfig.model_validate(data)

        logger.info(
            "CaseConfig 생성: actions=%d",
            len(actions),
        )
        return config

    def evaluate_results(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """keff 기준으로 결과를 평가한다.

        Args:
            result: 시뮬레이션 결과.

        Returns:
            메트릭 딕셔너리.
        """
        keff = result.keff
        deviation = abs(keff - self._config.target_keff)
        converged = 1.0 if deviation <= self._config.keff_tolerance else 0.0

        return {
            "keff": keff,
            "keff_std": result.keff_std,
            "keff_deviation": deviation,
            "converged": converged,
        }

    def update_state(
        self,
        state: ReactorState,
        metrics: dict[str, float],
    ) -> ReactorState:
        """메트릭으로 상태를 업데이트한다.

        Args:
            state: 현재 상태.
            metrics: 평가 메트릭.

        Returns:
            업데이트된 ReactorState.
        """
        return state.model_copy(
            update={
                "kpi": metrics,
                "iteration": state.iteration + 1,
            },
        )


def _set_nested(data: dict[str, Any], field_path: str, value: object) -> None:
    """중첩 딕셔너리에서 dot notation 경로로 값을 설정한다."""
    keys = field_path.split(".")
    current = data
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
