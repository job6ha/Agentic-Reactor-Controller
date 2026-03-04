"""OpenMC 실행기 테스트.

실제 OpenMC 없이 subprocess를 mock하여 테스트한다.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.armi_layer.models import RunConfig
from src.run_manager.runner import RunResult, _build_env, run_openmc


class TestBuildEnv:
    """_build_env 함수 테스트."""

    def test_default_config(self) -> None:
        rc = RunConfig()
        env = _build_env(rc)
        # 기존 환경변수 상속
        assert "PATH" in env
        # omp_threads=None이면 자동 감지값이 설정됨
        assert "OMP_NUM_THREADS" in env
        assert int(env["OMP_NUM_THREADS"]) > 0

    def test_omp_threads_override(self) -> None:
        rc = RunConfig(omp_threads=8)
        env = _build_env(rc)
        assert env["OMP_NUM_THREADS"] == "8"

    def test_cross_sections_override(self) -> None:
        rc = RunConfig(cross_sections_path=Path("/data/xs/cross_sections.xml"))
        env = _build_env(rc)
        assert env["OPENMC_CROSS_SECTIONS"] == "/data/xs/cross_sections.xml"

    def test_both_overrides(self) -> None:
        rc = RunConfig(
            omp_threads=4,
            cross_sections_path=Path("/nucdata/cross_sections.xml"),
        )
        env = _build_env(rc)
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["OPENMC_CROSS_SECTIONS"] == "/nucdata/cross_sections.xml"


class TestRunOpenmc:
    """run_openmc 함수 테스트."""

    def test_successful_run(self, tmp_path: Path) -> None:
        """정상 실행 시 exit_code=0, success=True."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        (case_dir / "input").mkdir()

        rc = RunConfig(omp_threads=4)

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_openmc(rc, case_dir)

        assert result.success is True
        assert result.exit_code == 0
        assert result.runtime >= 0
        assert result.log_path == case_dir / "output" / "run.log"
        assert result.log_path.parent.exists()

    def test_failed_run(self, tmp_path: Path) -> None:
        """비정상 종료 시 exit_code != 0, success=False."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig()

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = run_openmc(rc, case_dir)

        assert result.success is False
        assert result.exit_code == 1

    def test_timeout(self, tmp_path: Path) -> None:
        """타임아웃 시 exit_code=-1, 로그에 TIMEOUT 기록."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig(timeout=1.0)

        import subprocess

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="openmc", timeout=1.0)
            result = run_openmc(rc, case_dir)

        assert result.success is False
        assert result.exit_code == -1
        # 로그 파일에 TIMEOUT 메시지가 기록됨
        log_content = result.log_path.read_text()
        assert "[TIMEOUT]" in log_content

    def test_env_passed_to_subprocess(self, tmp_path: Path) -> None:
        """환경변수가 subprocess에 전달되는지 확인."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig(
            omp_threads=16,
            cross_sections_path=Path("/data/cross_sections.xml"),
        )

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_openmc(rc, case_dir)

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs["env"]
        assert env["OMP_NUM_THREADS"] == "16"
        assert env["OPENMC_CROSS_SECTIONS"] == "/data/cross_sections.xml"

    def test_cwd_is_output_dir(self, tmp_path: Path) -> None:
        """기본 작업 디렉토리가 case_path/output (statepoint 출력 위치)."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig()

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_openmc(rc, case_dir)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["cwd"] == str(case_dir / "output")

    def test_custom_working_dir(self, tmp_path: Path) -> None:
        """working_dir 설정 시 그 디렉토리를 cwd로 사용."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        work_dir = tmp_path / "custom_work"
        work_dir.mkdir()

        rc = RunConfig(working_dir=work_dir)

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_openmc(rc, case_dir)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["cwd"] == str(work_dir)

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """output 디렉토리가 없으면 자동 생성."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        # output/ 디렉토리를 만들지 않음

        rc = RunConfig()

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_openmc(rc, case_dir)

        assert (case_dir / "output").is_dir()
        assert result.log_path.parent.exists()

    def test_custom_openmc_command(self, tmp_path: Path) -> None:
        """커스텀 OpenMC 명령어 사용."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig()

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_openmc(rc, case_dir, openmc_command="/usr/local/bin/openmc")

        call_args = mock_run.call_args
        assert call_args.args[0][0] == "/usr/local/bin/openmc"

    def test_attempt_based_log_filename(self, tmp_path: Path) -> None:
        """attempt 지정 시 run_{attempt}.log 파일명 사용."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig()

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_openmc(rc, case_dir, attempt=3)

        assert result.log_path == case_dir / "output" / "run_3.log"

    def test_no_attempt_uses_default_log(self, tmp_path: Path) -> None:
        """attempt 미지정 시 run.log 파일명 사용."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()

        rc = RunConfig()

        with patch("src.run_manager.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_openmc(rc, case_dir)

        assert result.log_path == case_dir / "output" / "run.log"


class TestRunResult:
    """RunResult 데이터클래스 테스트."""

    def test_frozen(self) -> None:
        result = RunResult(
            exit_code=0,
            runtime=120.5,
            log_path=Path("/tmp/run.log"),
            success=True,
        )
        with pytest.raises(AttributeError):
            result.exit_code = 1  # type: ignore[misc]

    def test_fields(self) -> None:
        result = RunResult(
            exit_code=0,
            runtime=60.0,
            log_path=Path("/work/runs/case_0001/output/run.log"),
            success=True,
        )
        assert result.exit_code == 0
        assert result.runtime == 60.0
        assert result.success is True
