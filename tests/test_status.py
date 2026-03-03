"""케이스 실행 상태 관리 테스트.

meta/status.json의 저장/조회/업데이트와
run_case 통합 함수를 테스트한다.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.armi_layer.models import RunConfig, RunStatus, StatusType
from src.run_manager.runner import RunResult
from src.run_manager.status import (
    STATUS_FILENAME,
    get_status,
    run_case,
    save_status,
    update_status,
)


def _make_case_dir(tmp_path: Path) -> Path:
    """테스트용 케이스 디렉토리를 생성한다."""
    case_dir = tmp_path / "case_0001"
    case_dir.mkdir()
    (case_dir / "input").mkdir()
    (case_dir / "output").mkdir()
    (case_dir / "meta").mkdir()
    return case_dir


def _init_status(case_dir: Path, status: RunStatus | None = None) -> RunStatus:
    """테스트용 초기 status.json을 저장한다."""
    if status is None:
        status = RunStatus()
    save_status(case_dir, status)
    return status


class TestSaveStatus:
    """save_status 함수 테스트."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        status = RunStatus()
        path = save_status(case_dir, status)
        assert path.exists()
        assert path.name == STATUS_FILENAME

    def test_save_content_is_valid_json(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        status = RunStatus(status=StatusType.RUNNING, attempt=1)
        save_status(case_dir, status)
        restored = RunStatus.model_validate_json(
            (case_dir / "meta" / STATUS_FILENAME).read_text(encoding="utf-8")
        )
        assert restored.status == StatusType.RUNNING
        assert restored.attempt == 1

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        save_status(case_dir, RunStatus(status=StatusType.QUEUED))
        save_status(case_dir, RunStatus(status=StatusType.RUNNING))
        restored = get_status(case_dir)
        assert restored.status == StatusType.RUNNING

    def test_save_raises_if_no_meta_dir(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        # meta/ 디렉토리가 없음
        with pytest.raises(FileNotFoundError, match="meta 디렉토리"):
            save_status(case_dir, RunStatus())


class TestGetStatus:
    """get_status 함수 테스트."""

    def test_get_returns_saved_status(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir, RunStatus(status=StatusType.DONE, attempt=3))
        result = get_status(case_dir)
        assert result.status == StatusType.DONE
        assert result.attempt == 3

    def test_get_raises_if_no_file(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        with pytest.raises(FileNotFoundError, match="status.json"):
            get_status(case_dir)

    def test_get_preserves_timestamps(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        now = datetime.now(UTC)
        _init_status(
            case_dir,
            RunStatus(status=StatusType.RUNNING, started_at=now),
        )
        result = get_status(case_dir)
        assert result.started_at is not None


class TestUpdateStatus:
    """update_status 함수 테스트."""

    def test_queued_to_running(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir)
        updated = update_status(case_dir, StatusType.RUNNING, increment_attempt=True)
        assert updated.status == StatusType.RUNNING
        assert updated.started_at is not None
        assert updated.attempt == 1

    def test_running_to_done(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir, RunStatus(status=StatusType.RUNNING, attempt=1))
        updated = update_status(case_dir, StatusType.DONE)
        assert updated.status == StatusType.DONE
        assert updated.completed_at is not None

    def test_running_to_failed_with_message(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir, RunStatus(status=StatusType.RUNNING, attempt=1))
        updated = update_status(
            case_dir,
            StatusType.FAILED,
            error_message="exit_code=1",
        )
        assert updated.status == StatusType.FAILED
        assert updated.completed_at is not None
        assert updated.error_message == "exit_code=1"

    def test_increment_attempt(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir, RunStatus(attempt=2))
        updated = update_status(
            case_dir, StatusType.RUNNING, increment_attempt=True
        )
        assert updated.attempt == 3

    def test_persists_to_file(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir)
        update_status(case_dir, StatusType.RUNNING)
        # 파일에서 다시 읽어도 동일
        from_file = get_status(case_dir)
        assert from_file.status == StatusType.RUNNING

    def test_preserves_created_at(self, tmp_path: Path) -> None:
        case_dir = _make_case_dir(tmp_path)
        original = _init_status(case_dir)
        updated = update_status(case_dir, StatusType.RUNNING)
        assert updated.created_at == original.created_at


class TestRunCase:
    """run_case 통합 함수 테스트."""

    def test_successful_run_transitions(self, tmp_path: Path) -> None:
        """성공 실행 시 queued → running → done 전이."""
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir)
        rc = RunConfig(omp_threads=4)

        with patch("src.run_manager.status.run_openmc") as mock_run:
            mock_run.return_value = RunResult(
                exit_code=0,
                runtime=10.0,
                log_path=case_dir / "output" / "run.log",
                success=True,
            )
            result = run_case(rc, case_dir)

        assert result.success is True
        final = get_status(case_dir)
        assert final.status == StatusType.DONE
        assert final.attempt == 1
        assert final.started_at is not None
        assert final.completed_at is not None

    def test_failed_run_transitions(self, tmp_path: Path) -> None:
        """실패 실행 시 queued → running → failed 전이."""
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir)
        rc = RunConfig()

        with patch("src.run_manager.status.run_openmc") as mock_run:
            mock_run.return_value = RunResult(
                exit_code=1,
                runtime=5.0,
                log_path=case_dir / "output" / "run.log",
                success=False,
            )
            result = run_case(rc, case_dir)

        assert result.success is False
        final = get_status(case_dir)
        assert final.status == StatusType.FAILED
        assert final.error_message == "exit_code=1"

    def test_attempt_increments_on_retry(self, tmp_path: Path) -> None:
        """재시도 시 attempt가 증가."""
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir, RunStatus(status=StatusType.FAILED, attempt=1))
        rc = RunConfig()

        with patch("src.run_manager.status.run_openmc") as mock_run:
            mock_run.return_value = RunResult(
                exit_code=0,
                runtime=8.0,
                log_path=case_dir / "output" / "run.log",
                success=True,
            )
            run_case(rc, case_dir)

        final = get_status(case_dir)
        assert final.attempt == 2
        assert final.status == StatusType.DONE

    def test_passes_openmc_command(self, tmp_path: Path) -> None:
        """openmc_command가 run_openmc에 전달되는지 확인."""
        case_dir = _make_case_dir(tmp_path)
        _init_status(case_dir)
        rc = RunConfig()

        with patch("src.run_manager.status.run_openmc") as mock_run:
            mock_run.return_value = RunResult(
                exit_code=0,
                runtime=1.0,
                log_path=case_dir / "output" / "run.log",
                success=True,
            )
            run_case(rc, case_dir, openmc_command="/custom/openmc")

        mock_run.assert_called_once_with(
            rc, case_dir, openmc_command="/custom/openmc"
        )
