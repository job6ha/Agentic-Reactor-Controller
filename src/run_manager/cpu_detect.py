"""CPU 코어 수 자동 감지.

OMP_NUM_THREADS 기본값을 결정하기 위해 시스템의 물리 코어 수를 감지한다.
macOS에서는 P-core(성능 코어) 수를, Linux에서는 물리 CPU 수를 반환한다.
"""

import logging
import os
import platform
import subprocess

logger = logging.getLogger(__name__)

# 감지 실패 시 기본값
DEFAULT_THREAD_COUNT = 1


def detect_physical_cores() -> int:
    """시스템의 물리 코어 수를 감지한다.

    macOS Apple Silicon에서는 P-core 수를 반환하고,
    그 외 플랫폼에서는 os.cpu_count()를 사용한다.

    Returns:
        감지된 물리 코어 수. 감지 실패 시 1.
    """
    system = platform.system()

    if system == "Darwin":
        count = _detect_macos_pcores()
        if count is not None:
            logger.info("macOS P-core 수 감지: %d", count)
            return count

    # Linux 또는 macOS fallback
    count = os.cpu_count()
    if count is not None and count > 0:
        logger.info("CPU 코어 수 감지 (os.cpu_count): %d", count)
        return count

    logger.warning("CPU 코어 수 감지 실패, 기본값 %d 사용", DEFAULT_THREAD_COUNT)
    return DEFAULT_THREAD_COUNT


def _detect_macos_pcores() -> int | None:
    """macOS에서 P-core(성능 코어) 수를 감지한다.

    Apple Silicon (M1/M2/M3/M4)에서는 hw.perflevel0.physicalcpu로
    P-core 수를 조회한다. Intel Mac에서는 hw.physicalcpu를 사용한다.

    Returns:
        P-core 수. 감지 실패 시 None.
    """
    # Apple Silicon: P-core 전용 sysctl
    count = _sysctl_int("hw.perflevel0.physicalcpu")
    if count is not None:
        return count

    # Intel Mac fallback: 전체 물리 코어
    count = _sysctl_int("hw.physicalcpu")
    if count is not None:
        return count

    return None


def _sysctl_int(key: str) -> int | None:
    """sysctl로 정수 값을 조회한다.

    Args:
        key: sysctl 키 (예: hw.perflevel0.physicalcpu).

    Returns:
        정수 값. 실패 시 None.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            value = int(result.stdout.strip())
            if value > 0:
                return value
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return None
