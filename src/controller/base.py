"""제어 인터페이스 정의.

BaseController ABC를 정의하여 에이전트 교체 가능한 설계를 제공한다.
룰 기반 컨트롤러, LLM 에이전트 등 다양한 구현체를 교체할 수 있다.

데이터 흐름:
    ReactorState → propose_actions() → Actions
    Actions → apply_actions_to_case() → CaseConfig
    SimulationResult → evaluate_results() → Metrics
    Metrics → update_state() → ReactorState (루프)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult


class ActionType(StrEnum):
    """Action 유형.

    Attributes:
        MODIFY_PARAM: 파라미터 값 변경.
        SET_CONFIG: 설정 값 변경.
        STOP: 루프 종료.
    """

    MODIFY_PARAM = "modify_param"
    SET_CONFIG = "set_config"
    STOP = "stop"


class Action(BaseModel):
    """제어 액션.

    컨트롤러가 제안하는 단일 행동을 표현한다.

    Attributes:
        action_type: 액션 유형.
        field_path: 변경할 파라미터 경로 (dot notation).
        value: 변경할 값. STOP 액션에서는 None.
        reason: 액션의 이유/근거.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: ActionType = Field(description="액션 유형")
    field_path: str | None = Field(
        default=None,
        description="변경할 파라미터 경로",
    )
    value: float | int | str | bool | None = Field(
        default=None,
        description="변경할 값",
    )
    reason: str = Field(default="", description="액션의 이유/근거")


class BaseController(ABC):
    """원자로 제어 추상 기반 클래스.

    시뮬레이션 루프에서 의사결정을 담당하는 인터페이스를 정의한다.
    구현체는 룰 기반, 최적화 알고리즘, LLM 에이전트 등으로 교체 가능하다.

    사용 패턴::

        controller = SomeController()
        state = ReactorState()

        while True:
            actions = controller.propose_actions(state)
            if any(a.action_type == ActionType.STOP for a in actions):
                break
            config = controller.apply_actions_to_case(actions, state)
            # ... run simulation ...
            metrics = controller.evaluate_results(result)
            state = controller.update_state(state, metrics)
    """

    @abstractmethod
    def propose_actions(self, state: ReactorState) -> list[Action]:
        """현재 상태를 분석하여 다음 행동을 제안한다.

        Args:
            state: 현재 원자로 상태 (설계, 이력, KPI 포함).

        Returns:
            제안된 Action 목록. STOP 액션이 포함되면 루프 종료.
        """

    @abstractmethod
    def apply_actions_to_case(
        self,
        actions: list[Action],
        state: ReactorState,
    ) -> CaseConfig:
        """액션 목록을 적용하여 새 CaseConfig를 생성한다.

        현재 상태의 설정을 기반으로 액션에 따라
        파라미터를 변경한 새 케이스 설정을 반환한다.

        Args:
            actions: 적용할 액션 목록.
            state: 현재 원자로 상태.

        Returns:
            액션이 반영된 새 CaseConfig.
        """

    @abstractmethod
    def evaluate_results(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """시뮬레이션 결과를 평가하여 메트릭을 산출한다.

        keff, peaking factor 등의 결과에서
        의사결정에 필요한 메트릭을 추출/계산한다.

        Args:
            result: 시뮬레이션 결과.

        Returns:
            메트릭 딕셔너리.
        """

    @abstractmethod
    def update_state(
        self,
        state: ReactorState,
        metrics: dict[str, float],
    ) -> ReactorState:
        """메트릭을 반영하여 원자로 상태를 업데이트한다.

        이력에 결과를 추가하고, KPI를 갱신하며,
        iteration을 증가시킨다.

        Args:
            state: 현재 원자로 상태.
            metrics: 평가된 메트릭.

        Returns:
            업데이트된 ReactorState.
        """
