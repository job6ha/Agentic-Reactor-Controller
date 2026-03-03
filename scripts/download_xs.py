#!/usr/bin/env python3
"""핵데이터(Cross Section) 다운로드 스크립트.

OpenMC 공식 핵데이터 라이브러리를 다운로드하고
nucdata/ 디렉토리에 설치한다.

사용법:
    uv run python scripts/download_xs.py
    uv run python scripts/download_xs.py --library endfb-viii.0
    uv run python scripts/download_xs.py --dest /path/to/nucdata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 지원하는 핵데이터 라이브러리 목록
# URL 출처: https://openmc.org/official-data-libraries/
LIBRARIES: dict[str, dict[str, str]] = {
    "endfb-viii.0": {
        "url": "https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz",
        "filename": "endfb-viii.0-hdf5.tar.xz",
        "description": "ENDF/B-VIII.0 (recommended)",
        "sha256": "",
    },
    "endfb-vii.1": {
        "url": "https://anl.box.com/shared/static/9igk353zpy8fn9ttvtrqgzvw1vtejoz6.xz",
        "filename": "endfb-vii.1-hdf5.tar.xz",
        "description": "ENDF/B-VII.1",
        "sha256": "",
    },
}

DEFAULT_LIBRARY = "endfb-viii.0"

# 프로젝트 루트 기준 기본 설치 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "nucdata"

# 버전 메타데이터 파일명
VERSION_FILENAME = "version.json"


def _compute_sha256(path: Path) -> str:
    """파일의 SHA-256 해시를 계산한다."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


DOWNLOAD_TIMEOUT = 3600  # 1시간


def _download_file(url: str, dest: Path) -> None:
    """URL에서 파일을 다운로드한다. 진행률을 표시한다.

    Raises:
        RuntimeError: 네트워크 오류 또는 HTTP 에러 발생 시.
    """
    logger.info("다운로드 시작: %s", url)

    try:
        req = urllib.request.Request(url)  # noqa: S310
        with urllib.request.urlopen(  # noqa: S310
            req, timeout=DOWNLOAD_TIMEOUT
        ) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB

            with dest.open("wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        if int(pct) % 10 == 0 and int(pct) > 0:
                            logger.info(
                                "  진행: %.0f%% (%d / %d MB)",
                                pct,
                                downloaded // (1024 * 1024),
                                total // (1024 * 1024),
                            )

    except urllib.error.HTTPError as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"HTTP 오류 {e.code}: {url}") from e
    except urllib.error.URLError as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"네트워크 오류: {e.reason}") from e
    except Exception:
        if dest.exists():
            dest.unlink()
            logger.warning("불완전한 다운로드 파일 삭제: %s", dest)
        raise

    logger.info("다운로드 완료: %s", dest)


def _extract_tarball(tarball: Path, dest: Path) -> Path:
    """tar.xz 파일을 추출한다.

    Returns:
        추출된 디렉토리 경로.
    """
    logger.info("압축 해제 중: %s", tarball)

    with tarfile.open(tarball, "r:xz") as tar:
        # 최상위 디렉토리명 확인
        members = tar.getnames()
        top_dirs = {m.split("/")[0] for m in members if "/" in m}

        if sys.version_info >= (3, 12):
            tar.extractall(path=dest, filter="data")  # noqa: S202
        else:
            tar.extractall(path=dest)  # noqa: S202

    extracted = top_dirs.pop() if len(top_dirs) == 1 else ""
    extracted_path = dest / extracted if extracted else dest

    logger.info("압축 해제 완료: %s", extracted_path)
    return extracted_path


def _find_cross_sections_xml(search_dir: Path) -> Path | None:
    """cross_sections.xml 파일을 재귀적으로 탐색한다."""
    for p in search_dir.rglob("cross_sections.xml"):
        return p
    return None


def _write_version_metadata(
    dest: Path,
    *,
    library: str,
    cross_sections_xml: Path,
    archive_sha256: str,
) -> Path:
    """버전 메타데이터를 JSON으로 저장한다."""
    lib_info = LIBRARIES[library]

    metadata = {
        "library": library,
        "description": lib_info["description"],
        "source_url": lib_info["url"],
        "archive_sha256": archive_sha256,
        "cross_sections_xml": str(cross_sections_xml.relative_to(dest)),
        "installed_at": datetime.now(UTC).isoformat(),
    }

    version_path = dest / VERSION_FILENAME
    version_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("버전 메타데이터 저장: %s", version_path)
    return version_path


def download_nuclear_data(
    library: str = DEFAULT_LIBRARY,
    dest: Path = DEFAULT_DEST,
    *,
    keep_archive: bool = False,
) -> Path:
    """핵데이터를 다운로드하고 설치한다.

    Args:
        library: 핵데이터 라이브러리 이름.
        dest: 설치 대상 디렉토리.
        keep_archive: True면 다운로드된 아카이브를 보존.

    Returns:
        cross_sections.xml 파일 경로.

    Raises:
        ValueError: 지원하지 않는 라이브러리 이름.
        FileNotFoundError: cross_sections.xml을 찾을 수 없을 때.
    """
    if library not in LIBRARIES:
        available = ", ".join(LIBRARIES.keys())
        raise ValueError(f"지원하지 않는 라이브러리: {library} (가능: {available})")

    lib_info = LIBRARIES[library]
    dest.mkdir(parents=True, exist_ok=True)

    # 이미 설치된 경우 확인
    existing_xml = _find_cross_sections_xml(dest)
    if existing_xml is not None:
        logger.info("이미 설치된 핵데이터 발견: %s", existing_xml)
        return existing_xml

    # 다운로드
    archive_path = dest / lib_info["filename"]
    _download_file(lib_info["url"], archive_path)

    # 체크섬 계산
    archive_sha256 = _compute_sha256(archive_path)
    logger.info("아카이브 SHA-256: %s", archive_sha256)

    # 체크섬 검증
    if lib_info["sha256"]:
        if archive_sha256 != lib_info["sha256"]:
            raise ValueError(
                f"체크섬 불일치! expected={lib_info['sha256']}, got={archive_sha256}"
            )
    else:
        logger.warning(
            "SHA-256 체크섬이 등록되지 않았습니다. 무결성 검증을 건너뜁니다."
        )

    # 압축 해제
    _extract_tarball(archive_path, dest)

    # 아카이브 정리
    if not keep_archive and archive_path.exists():
        archive_path.unlink()
        logger.info("아카이브 삭제: %s", archive_path)

    # cross_sections.xml 탐색
    xs_xml = _find_cross_sections_xml(dest)
    if xs_xml is None:
        raise FileNotFoundError(f"cross_sections.xml을 찾을 수 없습니다: {dest}")

    # 버전 메타데이터 기록
    _write_version_metadata(
        dest,
        library=library,
        cross_sections_xml=xs_xml,
        archive_sha256=archive_sha256,
    )

    logger.info("핵데이터 설치 완료: %s", xs_xml)
    return xs_xml


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="OpenMC 핵데이터 다운로드 스크립트",
    )
    parser.add_argument(
        "--library",
        choices=list(LIBRARIES.keys()),
        default=DEFAULT_LIBRARY,
        help=f"핵데이터 라이브러리 (기본: {DEFAULT_LIBRARY})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"설치 경로 (기본: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="다운로드된 아카이브 파일 보존",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_libs",
        help="사용 가능한 라이브러리 목록 출력",
    )

    args = parser.parse_args()

    if args.list_libs:
        print("사용 가능한 핵데이터 라이브러리:")
        for name, info in LIBRARIES.items():
            marker = " (기본)" if name == DEFAULT_LIBRARY else ""
            print(f"  {name}: {info['description']}{marker}")
        return

    try:
        xs_path = download_nuclear_data(
            library=args.library,
            dest=args.dest,
            keep_archive=args.keep_archive,
        )
        print("\n핵데이터 설치 완료!")
        print(f"  경로: {xs_path}")
        print("\n.env에 다음을 추가하세요:")
        print(f"  OPENMC_CROSS_SECTIONS={xs_path}")
    except Exception as e:
        logger.error("핵데이터 설치 실패: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
