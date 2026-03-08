"""LLM 컨트롤러 로그 저장소.

시뮬레이션 스텝별 로그를 JSON 파일로 관리한다.
누적 로그는 BO 모델 재피팅과 LLM 프롬프트 컨텍스트에 사용된다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.controller.llm.models import LogEntry

logger = logging.getLogger(__name__)


class LogStore:
    """JSON 기반 로그 저장소.

    로그 파일은 ``log_dir/log.json`` 에 JSON 배열로 저장된다.

    Args:
        log_dir: 로그 저장 디렉토리.
    """

    LOG_FILENAME = "log.json"

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_path = log_dir / self.LOG_FILENAME

    @property
    def log_path(self) -> Path:
        """로그 파일 경로."""
        return self._log_path

    def append(self, entry: LogEntry) -> None:
        """로그 엔트리를 추가한다.

        Args:
            entry: 추가할 로그 엔트리.
        """
        entries = self.load()
        entries.append(entry)
        self._save(entries)
        logger.debug("로그 엔트리 추가: step=%d", entry.step)

    def load(self) -> list[LogEntry]:
        """저장된 로그를 모두 읽는다.

        Returns:
            LogEntry 목록. 파일이 없으면 빈 리스트.
        """
        if not self._log_path.exists():
            return []

        raw = self._log_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return [LogEntry.model_validate(item) for item in data]

    def count(self) -> int:
        """저장된 로그 수를 반환한다."""
        return len(self.load())

    def get_observations(self) -> list[tuple[int, float]]:
        """BO 재피팅용 (rod_position_after, keff) 쌍을 반환한다.

        Returns:
            (rod_position_after, keff) 튜플 목록.
        """
        entries = self.load()
        return [(e.rod_position_after, e.keff) for e in entries]

    def _save(self, entries: list[LogEntry]) -> None:
        """로그 엔트리 목록을 파일에 저장한다."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        data = [entry.model_dump(mode="json") for entry in entries]
        self._log_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
