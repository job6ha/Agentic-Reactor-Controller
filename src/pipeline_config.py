"""파이프라인 설정 모델.

JSON 설정 파일을 파싱하여 파이프라인 실행에 필요한
컨트롤러, 실행 환경, 출력 설정을 통합 관리한다.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.armi_layer.models import RunConfig
from src.controller.simple import SimpleControllerConfig

logger = logging.getLogger(__name__)


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

    Attributes:
        controller: 컨트롤러 설정 (SimpleController 기준).
        run: OpenMC 실행 환경 설정.
        output: 출력 및 보고서 설정.
    """

    model_config = ConfigDict(extra="forbid")

    controller: SimpleControllerConfig = Field(
        default_factory=SimpleControllerConfig,
        description="컨트롤러 설정",
    )
    run: RunConfig = Field(
        default_factory=RunConfig,
        description="OpenMC 실행 환경 설정",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="출력 및 보고서 설정",
    )


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
