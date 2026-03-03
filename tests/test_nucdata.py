"""핵데이터(Cross Section) 경로 관리 테스트.

nucdata 모듈의 경로 해석, 버전 정보 조회, 설치 확인 기능을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.run_manager.nucdata import (
    CROSS_SECTIONS_FILENAME,
    VERSION_FILENAME,
    NucdataNotFoundError,
    get_version_info,
    is_installed,
    resolve_cross_sections_path,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nucdata_dir(tmp_path: Path) -> Path:
    """cross_sections.xml이 있는 임시 nucdata 디렉토리."""
    nd = tmp_path / "nucdata"
    nd.mkdir()
    xs_xml = nd / CROSS_SECTIONS_FILENAME
    xs_xml.write_text('<?xml version="1.0"?>\n<cross_sections/>\n')
    return nd


@pytest.fixture()
def nucdata_nested(tmp_path: Path) -> Path:
    """중첩 디렉토리에 cross_sections.xml이 있는 경우."""
    nd = tmp_path / "nucdata"
    inner = nd / "endfb-viii.0-hdf5"
    inner.mkdir(parents=True)
    xs_xml = inner / CROSS_SECTIONS_FILENAME
    xs_xml.write_text('<?xml version="1.0"?>\n<cross_sections/>\n')
    return nd


@pytest.fixture()
def version_file(nucdata_dir: Path) -> Path:
    """version.json이 있는 nucdata 디렉토리."""
    metadata = {
        "library": "endfb-viii.0",
        "description": "ENDF/B-VIII.0 (recommended)",
        "source_url": "https://example.com/data.tar.xz",
        "archive_sha256": "abc123",
        "cross_sections_xml": "cross_sections.xml",
        "installed_at": "2026-03-04T00:00:00+00:00",
    }
    version_path = nucdata_dir / VERSION_FILENAME
    version_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return version_path


# ---------------------------------------------------------------------------
# resolve_cross_sections_path 테스트
# ---------------------------------------------------------------------------


class TestResolveCrossSectionsPath:
    """resolve_cross_sections_path 함수 테스트."""

    def test_explicit_path(self, nucdata_dir: Path) -> None:
        """명시적 경로가 주어지면 해당 경로를 반환한다."""
        xs_path = nucdata_dir / CROSS_SECTIONS_FILENAME
        result = resolve_cross_sections_path(explicit_path=xs_path)
        assert result == xs_path.resolve()

    def test_explicit_path_not_found(self, tmp_path: Path) -> None:
        """명시적 경로가 존재하지 않으면 NucdataNotFoundError."""
        fake = tmp_path / "nonexistent" / CROSS_SECTIONS_FILENAME
        with pytest.raises(NucdataNotFoundError, match="명시된"):
            resolve_cross_sections_path(explicit_path=fake)

    def test_env_var(self, nucdata_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OPENMC_CROSS_SECTIONS 환경변수로 경로를 해석한다."""
        xs_path = nucdata_dir / CROSS_SECTIONS_FILENAME
        monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(xs_path))
        result = resolve_cross_sections_path()
        assert result == xs_path.resolve()

    def test_env_var_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """환경변수 경로가 존재하지 않으면 NucdataNotFoundError."""
        fake = tmp_path / "nonexistent.xml"
        monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(fake))
        with pytest.raises(NucdataNotFoundError, match="OPENMC_CROSS_SECTIONS"):
            resolve_cross_sections_path()

    def test_explicit_overrides_env(
        self, nucdata_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """명시적 경로가 환경변수보다 우선한다."""
        xs_path = nucdata_dir / CROSS_SECTIONS_FILENAME
        monkeypatch.setenv("OPENMC_CROSS_SECTIONS", "/wrong/path.xml")
        result = resolve_cross_sections_path(explicit_path=xs_path)
        assert result == xs_path.resolve()

    def test_no_path_no_env_no_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """경로도 환경변수도 프로젝트도 없으면 NucdataNotFoundError."""
        monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
        # _find_project_root가 실패하도록 모킹
        monkeypatch.setattr(
            "src.run_manager.nucdata._find_project_root",
            lambda: (_ for _ in ()).throw(FileNotFoundError("mock")),
        )
        with pytest.raises(NucdataNotFoundError, match="찾을 수 없습니다"):
            resolve_cross_sections_path()

    def test_auto_discover_in_nucdata_dir(
        self,
        nucdata_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """프로젝트 nucdata/ 디렉토리에서 자동 탐색한다."""
        monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
        project_root = nucdata_dir.parent
        monkeypatch.setattr(
            "src.run_manager.nucdata._find_project_root",
            lambda: project_root,
        )
        result = resolve_cross_sections_path()
        expected = nucdata_dir / CROSS_SECTIONS_FILENAME
        assert result == expected

    def test_auto_discover_nested(
        self,
        nucdata_nested: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """중첩 디렉토리에서도 cross_sections.xml을 탐색한다."""
        monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
        project_root = nucdata_nested.parent
        monkeypatch.setattr(
            "src.run_manager.nucdata._find_project_root",
            lambda: project_root,
        )
        result = resolve_cross_sections_path()
        assert result.name == CROSS_SECTIONS_FILENAME
        assert result.exists()

    def test_nucdata_dir_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nucdata/ 디렉토리가 비어있으면 NucdataNotFoundError."""
        monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
        empty_nucdata = tmp_path / "nucdata"
        empty_nucdata.mkdir()
        monkeypatch.setattr(
            "src.run_manager.nucdata._find_project_root",
            lambda: tmp_path,
        )
        with pytest.raises(NucdataNotFoundError, match="cross_sections.xml이 없습니다"):
            resolve_cross_sections_path()

    def test_nucdata_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nucdata/ 디렉토리 자체가 없으면 NucdataNotFoundError."""
        monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
        monkeypatch.setattr(
            "src.run_manager.nucdata._find_project_root",
            lambda: tmp_path,
        )
        with pytest.raises(NucdataNotFoundError, match="nucdata/ 디렉토리가 없습니다"):
            resolve_cross_sections_path()


# ---------------------------------------------------------------------------
# get_version_info 테스트
# ---------------------------------------------------------------------------


class TestGetVersionInfo:
    """get_version_info 함수 테스트."""

    def test_reads_version_metadata(
        self, nucdata_dir: Path, version_file: Path
    ) -> None:
        """version.json에서 메타데이터를 읽는다."""
        info = get_version_info(nucdata_dir)
        assert info["library"] == "endfb-viii.0"
        assert info["archive_sha256"] == "abc123"

    def test_all_values_are_strings(
        self, nucdata_dir: Path, version_file: Path
    ) -> None:
        """모든 값이 문자열로 변환된다."""
        info = get_version_info(nucdata_dir)
        for value in info.values():
            assert isinstance(value, str)

    def test_missing_version_file(self, tmp_path: Path) -> None:
        """version.json이 없으면 FileNotFoundError."""
        fake_dir = tmp_path / "nucdata"
        fake_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="버전 정보"):
            get_version_info(fake_dir)


# ---------------------------------------------------------------------------
# is_installed 테스트
# ---------------------------------------------------------------------------


class TestIsInstalled:
    """is_installed 함수 테스트."""

    def test_installed(
        self,
        nucdata_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """핵데이터가 설치되어 있으면 True."""
        xs_path = nucdata_dir / CROSS_SECTIONS_FILENAME
        monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(xs_path))
        assert is_installed() is True

    def test_not_installed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """핵데이터가 없으면 False."""
        monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
        monkeypatch.setattr(
            "src.run_manager.nucdata._find_project_root",
            lambda: (_ for _ in ()).throw(FileNotFoundError("mock")),
        )
        assert is_installed() is False


# ---------------------------------------------------------------------------
# download_xs.py 유틸 함수 테스트
# ---------------------------------------------------------------------------


class TestDownloadXsUtils:
    """download_xs.py의 유틸리티 함수 테스트."""

    def test_compute_sha256(self, tmp_path: Path) -> None:
        """SHA-256 해시를 정확히 계산한다."""
        from scripts.download_xs import _compute_sha256

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")
        result = _compute_sha256(test_file)
        # sha256("hello world\n")
        assert len(result) == 64
        assert isinstance(result, str)

    def test_find_cross_sections_xml(self, nucdata_dir: Path) -> None:
        """cross_sections.xml을 재귀적으로 찾는다."""
        from scripts.download_xs import _find_cross_sections_xml

        result = _find_cross_sections_xml(nucdata_dir)
        assert result is not None
        assert result.name == "cross_sections.xml"

    def test_find_cross_sections_xml_not_found(self, tmp_path: Path) -> None:
        """cross_sections.xml이 없으면 None."""
        from scripts.download_xs import _find_cross_sections_xml

        result = _find_cross_sections_xml(tmp_path)
        assert result is None

    def test_write_version_metadata(self, nucdata_dir: Path) -> None:
        """버전 메타데이터를 올바르게 기록한다."""
        from scripts.download_xs import _write_version_metadata

        xs_xml = nucdata_dir / CROSS_SECTIONS_FILENAME
        path = _write_version_metadata(
            nucdata_dir,
            library="endfb-viii.0",
            cross_sections_xml=xs_xml,
            archive_sha256="test_hash",
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["library"] == "endfb-viii.0"
        assert data["archive_sha256"] == "test_hash"
        assert "installed_at" in data

    def test_libraries_dict_structure(self) -> None:
        """LIBRARIES 딕셔너리 구조가 올바르다."""
        from scripts.download_xs import LIBRARIES

        for name, info in LIBRARIES.items():
            assert "url" in info
            assert "filename" in info
            assert "description" in info
            assert "sha256" in info

    def test_download_nuclear_data_already_installed(self, nucdata_dir: Path) -> None:
        """이미 설치된 경우 재다운로드 없이 경로를 반환한다."""
        from scripts.download_xs import download_nuclear_data

        result = download_nuclear_data(dest=nucdata_dir)
        expected = nucdata_dir / CROSS_SECTIONS_FILENAME
        assert result == expected

    def test_download_nuclear_data_invalid_library(self, tmp_path: Path) -> None:
        """지원하지 않는 라이브러리면 ValueError."""
        from scripts.download_xs import download_nuclear_data

        with pytest.raises(ValueError, match="지원하지 않는"):
            download_nuclear_data(library="invalid-lib", dest=tmp_path)


# ---------------------------------------------------------------------------
# _find_project_root 테스트
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    """_find_project_root 함수 테스트."""

    def test_finds_root(self) -> None:
        """프로젝트 루트를 찾는다 (실제 프로젝트에서 실행)."""
        from src.run_manager.nucdata import _find_project_root

        root = _find_project_root()
        assert (root / "pyproject.toml").exists()
