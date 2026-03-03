"""CPU 코어 수 자동 감지 테스트.

플랫폼별 감지 로직을 mock하여 테스트한다.
"""

from unittest.mock import MagicMock, patch

from src.run_manager.cpu_detect import (
    DEFAULT_THREAD_COUNT,
    _detect_macos_pcores,
    _sysctl_int,
    detect_physical_cores,
)


class TestSysctlInt:
    """_sysctl_int 함수 테스트."""

    def test_successful_query(self) -> None:
        with patch("src.run_manager.cpu_detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="8\n")
            result = _sysctl_int("hw.perflevel0.physicalcpu")
        assert result == 8

    def test_command_failure(self) -> None:
        with patch("src.run_manager.cpu_detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = _sysctl_int("hw.nonexistent")
        assert result is None

    def test_non_numeric_output(self) -> None:
        with patch("src.run_manager.cpu_detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not_a_number\n")
            result = _sysctl_int("hw.something")
        assert result is None

    def test_zero_value_returns_none(self) -> None:
        with patch("src.run_manager.cpu_detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0\n")
            result = _sysctl_int("hw.something")
        assert result is None

    def test_subprocess_error(self) -> None:
        with patch("src.run_manager.cpu_detect.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("command not found")
            result = _sysctl_int("hw.something")
        assert result is None


class TestDetectMacosPcores:
    """_detect_macos_pcores 함수 테스트."""

    def test_apple_silicon_pcore(self) -> None:
        """Apple Silicon에서 perflevel0 감지."""
        with patch("src.run_manager.cpu_detect._sysctl_int") as mock_sysctl:
            mock_sysctl.return_value = 6
            result = _detect_macos_pcores()
        assert result == 6
        mock_sysctl.assert_called_once_with("hw.perflevel0.physicalcpu")

    def test_intel_mac_fallback(self) -> None:
        """Intel Mac에서 hw.physicalcpu fallback."""
        with patch("src.run_manager.cpu_detect._sysctl_int") as mock_sysctl:
            mock_sysctl.side_effect = [None, 4]
            result = _detect_macos_pcores()
        assert result == 4

    def test_all_sysctl_fail(self) -> None:
        """모든 sysctl 실패 시 None 반환."""
        with patch("src.run_manager.cpu_detect._sysctl_int") as mock_sysctl:
            mock_sysctl.return_value = None
            result = _detect_macos_pcores()
        assert result is None


class TestDetectPhysicalCores:
    """detect_physical_cores 함수 테스트."""

    def test_macos_uses_pcore_detection(self) -> None:
        """macOS에서 P-core 감지를 사용."""
        with (
            patch("src.run_manager.cpu_detect.platform.system", return_value="Darwin"),
            patch("src.run_manager.cpu_detect._detect_macos_pcores", return_value=8),
        ):
            result = detect_physical_cores()
        assert result == 8

    def test_linux_uses_cpu_count(self) -> None:
        """Linux에서 os.cpu_count 사용."""
        with (
            patch("src.run_manager.cpu_detect.platform.system", return_value="Linux"),
            patch("src.run_manager.cpu_detect.os.cpu_count", return_value=16),
        ):
            result = detect_physical_cores()
        assert result == 16

    def test_macos_pcore_fail_falls_back_to_cpu_count(self) -> None:
        """macOS P-core 감지 실패 시 cpu_count fallback."""
        with (
            patch("src.run_manager.cpu_detect.platform.system", return_value="Darwin"),
            patch("src.run_manager.cpu_detect._detect_macos_pcores", return_value=None),
            patch("src.run_manager.cpu_detect.os.cpu_count", return_value=10),
        ):
            result = detect_physical_cores()
        assert result == 10

    def test_all_detection_fails(self) -> None:
        """모든 감지 실패 시 기본값 반환."""
        with (
            patch("src.run_manager.cpu_detect.platform.system", return_value="Linux"),
            patch("src.run_manager.cpu_detect.os.cpu_count", return_value=None),
        ):
            result = detect_physical_cores()
        assert result == DEFAULT_THREAD_COUNT


class TestBuildEnvAutoDetect:
    """_build_env에서 자동 감지 통합 테스트."""

    def test_auto_detect_when_omp_threads_none(self) -> None:
        """omp_threads=None이면 자동 감지값 사용."""
        from src.armi_layer.models import RunConfig
        from src.run_manager.runner import _build_env

        rc = RunConfig()
        with patch("src.run_manager.runner.detect_physical_cores", return_value=6):
            env = _build_env(rc)
        assert env["OMP_NUM_THREADS"] == "6"

    def test_explicit_threads_overrides_auto(self) -> None:
        """omp_threads 명시 시 자동 감지 무시."""
        from src.armi_layer.models import RunConfig
        from src.run_manager.runner import _build_env

        rc = RunConfig(omp_threads=4)
        with patch("src.run_manager.runner.detect_physical_cores") as mock_detect:
            env = _build_env(rc)
        assert env["OMP_NUM_THREADS"] == "4"
        mock_detect.assert_not_called()
