"""스테일 로그 버그 수정 통합 테스트.

버그: LLMController가 이전 벤치마크 실행의 log.json을 복원하여
      즉시 max_iterations에 도달, 시뮬레이션을 한 번도 실행하지 않고 STOP.

수정: resume_from_log=False(기본값)이면 기존 log.json을 삭제하고 step=0으로 시작.
     resume_from_log=True이면 기존 로그를 복원한다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.controller.llm.controller import LLMController
from src.controller.llm.models import LLMControllerConfig


def _make_fake_log(n_entries: int) -> list[dict]:
    """n_entries개의 가짜 로그 엔트리를 생성한다."""
    entries = []
    for i in range(n_entries):
        entries.append(
            {
                "step": i,
                "timestamp": datetime.now(UTC).isoformat(),
                "rod_movement": float(i % 5 - 2),
                "rod_position_after": float(14 + i),
                "keff": 1.0 + 0.001 * i,
                "keff_std": 0.0005,
                "safety_score": 0.8,
                "candidates": [],
            }
        )
    return entries


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    """50개 엔트리가 담긴 가짜 log.json이 있는 임시 디렉토리."""
    log_dir = tmp_path / "llm_logs"
    log_dir.mkdir()
    log_file = log_dir / "log.json"
    log_file.write_text(
        json.dumps(_make_fake_log(50), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return log_dir


class TestStalLogNotRestored:
    """resume_from_log=False(기본값)일 때 스테일 로그를 무시하고 초기 상태로 시작한다."""

    def test_step_starts_at_zero_when_stale_log_exists(self, log_dir: Path) -> None:
        """이전 로그가 있어도 step=0으로 시작해야 한다."""
        config = LLMControllerConfig(
            log_dir=log_dir,
            resume_from_log=False,
        )
        controller = LLMController(config)

        assert controller.step == 0, (
            f"resume_from_log=False인데 step={controller.step}으로 시작함 "
            "(스테일 로그 버그 재발)"
        )

    def test_old_log_file_deleted_when_not_resuming(self, log_dir: Path) -> None:
        """resume_from_log=False이면 기존 log.json이 삭제되어야 한다."""
        log_file = log_dir / "log.json"
        assert log_file.exists(), "전제 조건: 테스트 시작 전 log.json이 존재해야 함"

        config = LLMControllerConfig(
            log_dir=log_dir,
            resume_from_log=False,
        )
        LLMController(config)

        assert not log_file.exists(), (
            "resume_from_log=False인데 기존 log.json이 삭제되지 않음"
        )

    def test_rod_position_at_initial_when_stale_log_exists(
        self, log_dir: Path
    ) -> None:
        """이전 로그가 있어도 제어봉 위치가 initial_rod_position으로 시작해야 한다."""
        initial_position = 14
        config = LLMControllerConfig(
            log_dir=log_dir,
            resume_from_log=False,
            initial_rod_position=initial_position,
        )
        controller = LLMController(config)

        assert controller.rod_position == float(initial_position), (
            f"resume_from_log=False인데 rod_position={controller.rod_position} "
            f"(기대값={initial_position})"
        )

    def test_default_resume_from_log_is_false(self, log_dir: Path) -> None:
        """resume_from_log 기본값이 False여야 한다 (명시적 설정 없이도 버그 안전)."""
        config = LLMControllerConfig(log_dir=log_dir)
        assert config.resume_from_log is False

        controller = LLMController(config)
        assert controller.step == 0


class TestStaleLogRestoredWhenRequested:
    """resume_from_log=True일 때 기존 로그를 복원한다."""

    def test_step_restored_from_50_entry_log(self, log_dir: Path) -> None:
        """50개 엔트리 로그에서 복원 시 step=50이어야 한다."""
        config = LLMControllerConfig(
            log_dir=log_dir,
            resume_from_log=True,
        )
        controller = LLMController(config)

        assert controller.step == 50, (
            f"resume_from_log=True인데 step={controller.step} (기대값=50)"
        )

    def test_log_file_preserved_when_resuming(self, log_dir: Path) -> None:
        """resume_from_log=True이면 기존 log.json이 유지되어야 한다."""
        log_file = log_dir / "log.json"
        config = LLMControllerConfig(
            log_dir=log_dir,
            resume_from_log=True,
        )
        LLMController(config)

        assert log_file.exists(), (
            "resume_from_log=True인데 log.json이 삭제됨"
        )

    def test_rod_position_restored_from_last_log_entry(self, log_dir: Path) -> None:
        """마지막 로그 엔트리의 rod_position_after로 제어봉 위치가 복원되어야 한다."""
        # 마지막 엔트리(step=49)의 rod_position_after = 14 + 49 = 63.0
        expected_rod_position = 14.0 + 49.0
        config = LLMControllerConfig(
            log_dir=log_dir,
            resume_from_log=True,
        )
        controller = LLMController(config)

        assert controller.rod_position == expected_rod_position, (
            f"복원된 rod_position={controller.rod_position} "
            f"(기대값={expected_rod_position})"
        )


class TestNoLogFile:
    """log.json이 없는 깨끗한 환경에서의 초기화 테스트."""

    def test_step_zero_with_no_log(self, tmp_path: Path) -> None:
        """log.json이 없으면 step=0으로 시작한다."""
        empty_dir = tmp_path / "empty_logs"
        config = LLMControllerConfig(
            log_dir=empty_dir,
            resume_from_log=False,
        )
        controller = LLMController(config)

        assert controller.step == 0

    def test_step_zero_with_resume_true_and_no_log(self, tmp_path: Path) -> None:
        """resume_from_log=True이더라도 log.json이 없으면 step=0으로 시작한다."""
        empty_dir = tmp_path / "empty_logs"
        config = LLMControllerConfig(
            log_dir=empty_dir,
            resume_from_log=True,
        )
        controller = LLMController(config)

        assert controller.step == 0
