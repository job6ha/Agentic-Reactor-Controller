"""Agentic-Reactor-Controller 엔트리포인트.

``python -m src`` 또는 ``python -m src --config path/to/config.json``
으로 시뮬레이션 파이프라인을 실행한다.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import SimulationPipeline

DEFAULT_CONFIG = Path("configs/default.json")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="원자로 시뮬레이션 파이프라인 실행",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"설정 파일 경로 (기본: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨 (기본: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """메인 함수.

    Args:
        argv: CLI 인자. None이면 sys.argv 사용.

    Returns:
        종료 코드 (0=성공, 1=에러).
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    logger = logging.getLogger(__name__)

    try:
        pipeline = SimulationPipeline.from_config(args.config)
        result = pipeline.run()

        logger.info(
            "실행 완료: %d회 반복, 성공 %d, 실패 %d, 종료 사유: %s",
            result.total_iterations,
            result.successful_cases,
            result.failed_cases,
            result.stop_reason,
        )
        return 0

    except FileNotFoundError as e:
        logger.error("파일을 찾을 수 없습니다: %s", e)
        return 1
    except Exception:
        logger.error("파이프라인 실행 중 오류 발생", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
