"""LogStore 테스트 (KAE-87)."""

from __future__ import annotations

from pathlib import Path

from src.controller.llm.log_store import LogStore
from src.controller.llm.models import CandidateLog, LogEntry


def _make_entry(step: int, rod_pos: int = 220, keff: float = 1.0) -> LogEntry:
    """테스트용 LogEntry 생성."""
    return LogEntry(
        step=step,
        rod_movement=-(228 - rod_pos),
        rod_position_after=rod_pos,
        keff=keff,
        keff_std=0.001,
        safety_score=0.9,
        candidates=[
            CandidateLog(rod_movement=-5, safety_score=0.9),
            CandidateLog(rod_movement=0, safety_score=0.8),
        ],
    )


class TestLogStoreInit:
    """LogStore 초기화 테스트."""

    def test_log_path(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        assert store.log_path == tmp_path / "log.json"


class TestLogStoreLoad:
    """LogStore load 테스트."""

    def test_load_empty_when_no_file(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        assert store.load() == []

    def test_count_zero_when_no_file(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        assert store.count() == 0


class TestLogStoreAppend:
    """LogStore append 테스트."""

    def test_append_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        entry = _make_entry(step=0)

        store.append(entry)
        loaded = store.load()

        assert len(loaded) == 1
        assert loaded[0].step == 0
        assert loaded[0].keff == 1.0

    def test_append_multiple(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)

        store.append(_make_entry(step=0, keff=1.01))
        store.append(_make_entry(step=1, keff=1.005))
        store.append(_make_entry(step=2, keff=1.001))

        assert store.count() == 3
        loaded = store.load()
        assert loaded[0].keff == 1.01
        assert loaded[2].keff == 1.001

    def test_creates_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "nested" / "logs"
        store = LogStore(log_dir)

        store.append(_make_entry(step=0))

        assert log_dir.exists()
        assert store.count() == 1

    def test_preserves_candidates(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        store.append(_make_entry(step=0))

        loaded = store.load()
        assert len(loaded[0].candidates) == 2
        assert loaded[0].candidates[0].rod_movement == -5


class TestLogStoreObservations:
    """LogStore get_observations 테스트."""

    def test_empty_observations(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        assert store.get_observations() == []

    def test_observations_format(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path)
        store.append(_make_entry(step=0, rod_pos=220, keff=1.01))
        store.append(_make_entry(step=1, rod_pos=215, keff=1.005))

        obs = store.get_observations()

        assert len(obs) == 2
        assert obs[0] == (220, 1.01)
        assert obs[1] == (215, 1.005)
