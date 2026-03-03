"""파라미터 스윕 엔진.

스윕할 파라미터와 값 목록을 지정하면 모든 격자 조합의
CaseConfig를 자동 생성한다. CaseManager와 연동하여
케이스를 일괄 생성할 수 있다.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.armi_layer.models import CaseConfig

logger = logging.getLogger(__name__)


class SweepParam(BaseModel):
    """스윕할 단일 파라미터 정의.

    Attributes:
        field_path: 파라미터 경로 (예: "materials.fuel_enrichment").
        values: 스윕할 값 목록.
    """

    model_config = ConfigDict(extra="forbid")

    field_path: str = Field(description="파라미터 경로 (dot notation)")
    values: list[float] = Field(min_length=1, description="스윕할 값 목록")


class SweepConfig(BaseModel):
    """스윕 설정.

    Attributes:
        base_config: 기본 케이스 설정. 스윕 파라미터 외 값은 이것을 따른다.
        params: 스윕할 파라미터 목록.
        name_template: 케이스 이름 템플릿. {field}={value} 형태로 치환.
    """

    model_config = ConfigDict(extra="forbid")

    base_config: CaseConfig = Field(
        default_factory=CaseConfig,
        description="기본 케이스 설정",
    )
    params: list[SweepParam] = Field(
        min_length=1,
        description="스윕할 파라미터 목록",
    )
    name_template: str = Field(
        default="sweep",
        description="케이스 이름 접두사",
    )


def _set_nested_field(data: dict[str, Any], field_path: str, value: float) -> None:
    """중첩 딕셔너리에서 dot notation 경로로 값을 설정한다.

    Args:
        data: 대상 딕셔너리.
        field_path: 경로 (예: "materials.fuel_enrichment").
        value: 설정할 값.

    Raises:
        KeyError: 경로가 유효하지 않을 때.
    """
    keys = field_path.split(".")
    current = data

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"유효하지 않은 경로: {field_path}")
        current = current[key]

    if keys[-1] not in current:
        raise KeyError(f"유효하지 않은 필드: {field_path}")

    current[keys[-1]] = value


def _build_case_name(
    template: str,
    combination: list[tuple[str, float]],
) -> str:
    """조합 정보로 케이스 이름을 생성한다.

    Args:
        template: 이름 접두사.
        combination: (field_path, value) 리스트.

    Returns:
        생성된 케이스 이름.
    """
    parts = [template]
    for field_path, value in combination:
        short_name = field_path.split(".")[-1]
        parts.append(f"{short_name}={value}")
    return "_".join(parts)


def generate_sweep_configs(sweep_config: SweepConfig) -> list[CaseConfig]:
    """스윕 설정에서 모든 격자 조합의 CaseConfig를 생성한다.

    Args:
        sweep_config: 스윕 설정.

    Returns:
        생성된 CaseConfig 목록.

    Raises:
        KeyError: 파라미터 경로가 유효하지 않을 때.
    """
    base_data = sweep_config.base_config.model_dump(mode="json")

    # 각 파라미터의 (field_path, value) 리스트를 생성
    param_lists: list[list[tuple[str, float]]] = []
    for param in sweep_config.params:
        param_lists.append([(param.field_path, v) for v in param.values])

    # 격자 조합 생성
    combinations = list(itertools.product(*param_lists))

    configs: list[CaseConfig] = []
    for combo in combinations:
        combo_list = list(combo)
        data = _deep_copy_dict(base_data)

        for field_path, value in combo_list:
            _set_nested_field(data, field_path, value)

        data["name"] = _build_case_name(
            sweep_config.name_template,
            combo_list,
        )

        config = CaseConfig.model_validate(data)
        configs.append(config)

    logger.info(
        "스윕 조합 생성: params=%d, combinations=%d",
        len(sweep_config.params),
        len(configs),
    )

    return configs


def _deep_copy_dict(data: dict[str, Any]) -> dict[str, Any]:
    """딕셔너리를 깊은 복사한다 (JSON 직렬화 가능한 데이터만)."""
    result: dict[str, Any] = json.loads(json.dumps(data))
    return result


def save_sweep_config(sweep_config: SweepConfig, path: Path) -> Path:
    """스윕 설정을 JSON 파일로 저장한다.

    Args:
        sweep_config: 스윕 설정.
        path: 저장할 파일 경로.

    Returns:
        저장된 파일 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        sweep_config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("스윕 설정 저장: %s", path)
    return path


def load_sweep_config(path: Path) -> SweepConfig:
    """스윕 설정을 JSON 파일에서 로드한다.

    Args:
        path: 설정 파일 경로.

    Returns:
        로드된 SweepConfig.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    if not path.exists():
        raise FileNotFoundError(f"스윕 설정 파일을 찾을 수 없습니다: {path}")

    return SweepConfig.model_validate_json(path.read_text(encoding="utf-8"))
