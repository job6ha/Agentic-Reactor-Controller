"""파이프라인 설정 모델.

JSON 설정 파일을 파싱하여 파이프라인 실행에 필요한
컨트롤러, 실행 환경, 출력 설정을 통합 관리한다.

컨트롤러 설정은 discriminated union으로 관리하며,
``controller_type`` 필드로 컨트롤러 유형을 구분한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

from src.armi_layer.models import RunConfig, SimulationSettings
from src.controller.llm.models import LLMControllerConfig
from src.controller.simple import SimpleControllerConfig

logger = logging.getLogger(__name__)

# Discriminated union: controller_type 필드로 구분
ControllerConfig = Annotated[
    Annotated[SimpleControllerConfig, Tag("simple")]
    | Annotated[LLMControllerConfig, Tag("llm")],
    Discriminator("controller_type"),
]


class OutputConfig(BaseModel):
    """파이프라인 출력 설정.

    Attributes:
        runs_dir: 케이스 폴더 루트 경로.
        export_csv: CSV 보고서 내보내기 여부.
        export_json: JSON 보고서 내보내기 여부.
    """

    model_config = ConfigDict(extra="forbid")

    runs_dir: Path = Field(default=Path("runs"), description="케이스 폴더 루트 경로")
    export_csv: bool = Field(default=True, description="CSV 보고서 내보내기 여부")
    export_json: bool = Field(default=False, description="JSON 보고서 내보내기 여부")


class PipelineConfig(BaseModel):
    """파이프라인 통합 설정.

    컨트롤러, 실행 환경, 출력 설정을 하나의 JSON 파일로 관리한다.
    ``controller_type`` 필드로 SimpleController / LLMController를 구분한다.

    하위 호환성: ``controller_type`` 미지정 시 ``"simple"``로 기본 설정된다.

    Attributes:
        controller: 컨트롤러 설정 (SimpleControllerConfig 또는 LLMControllerConfig).
        simulation: OpenMC 시뮬레이션 설정.
        run: OpenMC 실행 환경 설정.
        output: 출력 및 보고서 설정.
    """

    model_config = ConfigDict(extra="forbid")

    controller: ControllerConfig = Field(
        default_factory=SimpleControllerConfig,
        description="컨트롤러 설정",
    )
    simulation: SimulationSettings = Field(
        default_factory=SimulationSettings,
        description="OpenMC 시뮬레이션 설정 (batches, particles 등)",
    )
    run: RunConfig = Field(
        default_factory=RunConfig,
        description="OpenMC 실행 환경 설정",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="출력 및 보고서 설정",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_default_controller_type(cls, data: object) -> object:
        """controller_type 미지정 시 'simple'로 기본값 주입.

        기존 JSON 설정 파일과의 하위 호환성을 위해,
        controller 객체에 controller_type이 없으면 'simple'을 주입한다.
        """
        if isinstance(data, dict) and "controller" in data:
            ctrl = data["controller"]
            if isinstance(ctrl, dict) and "controller_type" not in ctrl:
                ctrl["controller_type"] = "simple"
        return data


def load_config(config_path: Path) -> PipelineConfig:
    """JSON 설정 파일을 로드하여 PipelineConfig를 반환한다.

    Args:
        config_path: 설정 파일 경로.

    Returns:
        파싱된 PipelineConfig 객체.

    Raises:
        FileNotFoundError: 설정 파일이 없을 때.
        json.JSONDecodeError: JSON 파싱 실패 시.
        pydantic.ValidationError: 필드 검증 실패 시.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    config = PipelineConfig.model_validate(data)

    logger.info("설정 로드 완료: %s", config_path)
    return config
