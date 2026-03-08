"""컨트롤러 공유 유틸리티.

여러 컨트롤러 구현체에서 공통으로 사용하는 헬퍼 함수를 정의한다.
"""

from __future__ import annotations

from typing import Any


def set_nested(data: dict[str, Any], field_path: str, value: object) -> None:
    """중첩 딕셔너리에서 dot notation 경로로 값을 설정한다.

    중간 키가 없으면 빈 딕셔너리를 자동 생성한다.

    Args:
        data: 대상 딕셔너리.
        field_path: 변경할 경로 (dot notation).
        value: 설정할 값.

    Raises:
        KeyError: 중간 키가 딕셔너리가 아닐 때.
    """
    keys = field_path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
