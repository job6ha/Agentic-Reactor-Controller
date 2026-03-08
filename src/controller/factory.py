"""컨트롤러 팩토리.

설정 객체의 타입에 따라 적절한 컨트롤러 인스턴스를 생성한다.
"""

from __future__ import annotations

from src.controller.base import BaseController
from src.controller.llm.models import LLMControllerConfig
from src.controller.pid import (
    PIDController,
    PIDControllerConfig,
    ProportionalController,
    ProportionalControllerConfig,
)
from src.controller.simple import SimpleController, SimpleControllerConfig

# 지원하는 컨트롤러 설정 타입 Union
ControllerConfig = (
    SimpleControllerConfig
    | LLMControllerConfig
    | PIDControllerConfig
    | ProportionalControllerConfig
)


def create_controller(config: ControllerConfig) -> BaseController:
    """설정에 맞는 컨트롤러를 생성한다.

    Args:
        config: 컨트롤러 설정.

    Returns:
        초기화된 BaseController 구현체.

    Raises:
        ValueError: 지원하지 않는 설정 타입일 때.
    """
    if isinstance(config, SimpleControllerConfig):
        return SimpleController(config)

    if isinstance(config, LLMControllerConfig):
        from src.controller.llm.controller import LLMController

        return LLMController(config)

    if isinstance(config, PIDControllerConfig):
        return PIDController(config)

    if isinstance(config, ProportionalControllerConfig):
        return ProportionalController(config)

    raise ValueError(f"지원하지 않는 컨트롤러 설정: {type(config).__name__}")
