"""OpenMC 입력 생성기 테스트.

CaseConfig → XML 변환 결과를 검증한다.
2D pin cell (하위 호환) 및 3D 제어봉 모델 모두 테스트한다.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    SimulationSettings,
)
from src.openmc_layer.input_generator import (
    _build_geometry_xml,
    _build_materials_xml,
    _build_settings_xml,
    _compute_rod_tip_z,
    _has_control_rod,
    generate_input,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def geo_2d() -> GeometryParams:
    """2D pin cell (제어봉 없음)."""
    return GeometryParams()


@pytest.fixture()
def geo_3d() -> GeometryParams:
    """3D pin cell (제어봉 포함, 완전 인출)."""
    return GeometryParams(
        absorber_outer_radius=0.52,
        extra_params={"rod_position": 228},
    )


@pytest.fixture()
def geo_3d_inserted() -> GeometryParams:
    """3D pin cell (제어봉 완전 삽입)."""
    return GeometryParams(
        absorber_outer_radius=0.52,
        extra_params={"rod_position": 0},
    )


@pytest.fixture()
def geo_3d_half() -> GeometryParams:
    """3D pin cell (제어봉 절반 삽입)."""
    return GeometryParams(
        absorber_outer_radius=0.52,
        extra_params={"rod_position": 114},
    )


# ---------------------------------------------------------------------------
# 2D Materials
# ---------------------------------------------------------------------------

class TestBuildMaterialsXml:
    """_build_materials_xml 함수 테스트."""

    def test_three_materials_created(self) -> None:
        mat = MaterialParams()
        root = _build_materials_xml(mat)
        materials = root.findall("material")
        assert len(materials) == 3

    def test_fuel_material(self) -> None:
        mat = MaterialParams(fuel_density=10.5)
        root = _build_materials_xml(mat)
        fuel = root.find("material[@name='UO2']")
        assert fuel is not None
        density = fuel.find("density")
        assert density is not None
        assert density.get("value") == "10.5"

    def test_fuel_enrichment_affects_composition(self) -> None:
        mat_low = MaterialParams(fuel_enrichment=2.0)
        mat_high = MaterialParams(fuel_enrichment=5.0)
        root_low = _build_materials_xml(mat_low)
        root_high = _build_materials_xml(mat_high)

        u235_low = root_low.find(".//nuclide[@name='U235']")
        u235_high = root_high.find(".//nuclide[@name='U235']")
        assert u235_low is not None
        assert u235_high is not None
        assert float(u235_high.get("wo")) > float(u235_low.get("wo"))

    def test_clad_material(self) -> None:
        mat = MaterialParams(clad_density=6.6)
        root = _build_materials_xml(mat)
        clad = root.find("material[@name='Zircaloy-4']")
        assert clad is not None
        density = clad.find("density")
        assert density.get("value") == "6.6"

    def test_water_material(self) -> None:
        mat = MaterialParams(coolant_density=0.75)
        root = _build_materials_xml(mat)
        water = root.find("material[@name='H2O']")
        assert water is not None
        density = water.find("density")
        assert density.get("value") == "0.75"
        sab = water.find("sab")
        assert sab is not None
        assert sab.get("name") == "c_H_in_H2O"


class TestBuildMaterialsXmlAbsorber:
    """B4C 흡수체 재료 테스트."""

    def test_no_absorber_by_default(self) -> None:
        mat = MaterialParams()
        root = _build_materials_xml(mat, include_absorber=False)
        assert root.find("material[@name='B4C']") is None

    def test_absorber_included_when_requested(self) -> None:
        mat = MaterialParams()
        root = _build_materials_xml(mat, include_absorber=True)
        b4c = root.find("material[@name='B4C']")
        assert b4c is not None
        assert b4c.get("id") == "4"

    def test_absorber_has_boron_and_carbon(self) -> None:
        mat = MaterialParams()
        root = _build_materials_xml(mat, include_absorber=True)
        b4c = root.find("material[@name='B4C']")
        nuclides = {n.get("name") for n in b4c.findall("nuclide")}
        assert nuclides == {"B10", "B11", "C12"}

    def test_absorber_density(self) -> None:
        mat = MaterialParams(absorber_density=2.6)
        root = _build_materials_xml(mat, include_absorber=True)
        b4c = root.find("material[@name='B4C']")
        density = b4c.find("density")
        assert density.get("value") == "2.6"

    def test_four_materials_with_absorber(self) -> None:
        mat = MaterialParams()
        root = _build_materials_xml(mat, include_absorber=True)
        assert len(root.findall("material")) == 4


# ---------------------------------------------------------------------------
# 2D Geometry (하위 호환)
# ---------------------------------------------------------------------------

class TestBuildGeometryXml2D:
    """2D _build_geometry_xml 함수 테스트."""

    def test_surfaces_created(self, geo_2d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_2d)
        surfaces = root.findall(".//surface")
        assert len(surfaces) == 7  # 3 cylinders + 4 planes

    def test_fuel_cylinder_radius(self, geo_2d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_2d)
        fuel_surf = root.find(".//surface[@id='1']")
        assert fuel_surf is not None
        assert str(geo_2d.fuel_radius) in fuel_surf.get("coeffs")

    def test_boundary_conditions(self, geo_2d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_2d)
        planes = [
            s for s in root.findall(".//surface")
            if "plane" in s.get("type", "")
        ]
        for plane in planes:
            assert plane.get("boundary") == "reflective"

    def test_four_cells_created(self, geo_2d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_2d)
        cells = root.findall(".//cell")
        assert len(cells) == 4

    def test_cell_names(self, geo_2d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_2d)
        cell_names = {c.get("name") for c in root.findall(".//cell")}
        assert cell_names == {"fuel", "gap", "clad", "water"}

    def test_custom_geometry(self) -> None:
        geo = GeometryParams(
            fuel_radius=0.35,
            clad_inner_radius=0.38,
            clad_outer_radius=0.42,
            pitch=1.5,
        )
        root = _build_geometry_xml(geo)
        fuel_surf = root.find(".//surface[@id='1']")
        assert "0.35" in fuel_surf.get("coeffs")
        x_plane = root.find(".//surface[@id='5']")
        assert "0.75" in x_plane.get("coeffs")


# ---------------------------------------------------------------------------
# 3D Geometry (제어봉 모델)
# ---------------------------------------------------------------------------

class TestHasControlRod:
    """_has_control_rod 판단 테스트."""

    def test_no_rod_by_default(self, geo_2d: GeometryParams) -> None:
        assert _has_control_rod(geo_2d) is False

    def test_rod_when_absorber_set(self, geo_3d: GeometryParams) -> None:
        assert _has_control_rod(geo_3d) is True


class TestComputeRodTipZ:
    """_compute_rod_tip_z 계산 테스트."""

    def test_fully_withdrawn(self, geo_3d: GeometryParams) -> None:
        """완전 인출: z_tip ≈ active_height."""
        z_tip = _compute_rod_tip_z(geo_3d)
        assert z_tip == pytest.approx(geo_3d.fuel_height - 0.01, abs=0.02)

    def test_fully_inserted(self, geo_3d_inserted: GeometryParams) -> None:
        """완전 삽입: z_tip ≈ 0."""
        z_tip = _compute_rod_tip_z(geo_3d_inserted)
        assert z_tip == pytest.approx(0.01, abs=0.02)

    def test_half_inserted(self, geo_3d_half: GeometryParams) -> None:
        """절반 삽입: z_tip ≈ active_height/2."""
        z_tip = _compute_rod_tip_z(geo_3d_half)
        expected = geo_3d_half.fuel_height / 2.0
        assert z_tip == pytest.approx(expected, abs=1.0)

    def test_no_rod_position_defaults_to_withdrawn(self) -> None:
        """rod_position 미설정 시 완전 인출."""
        geo = GeometryParams(absorber_outer_radius=0.52)
        z_tip = _compute_rod_tip_z(geo)
        assert z_tip == pytest.approx(geo.fuel_height - 0.01, abs=0.02)


class TestBuildGeometryXml3D:
    """3D 제어봉 기하 구조 테스트."""

    def test_surfaces_count(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        surfaces = root.findall(".//surface")
        # 3 cyl + 4 xy-planes + 1 absorber cyl + 2 z-planes + 1 rod tip = 11
        assert len(surfaces) == 11

    def test_six_cells_created(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        cells = root.findall(".//cell")
        assert len(cells) == 6

    def test_cell_names_3d(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        cell_names = {c.get("name") for c in root.findall(".//cell")}
        expected = {"fuel", "gap", "clad", "absorber", "guide_coolant", "water"}
        assert cell_names == expected

    def test_absorber_cell_material(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        absorber = root.find(".//cell[@name='absorber']")
        assert absorber.get("material") == "4"

    def test_z_boundary_reflective(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        z_bottom = root.find(".//surface[@id='9']")
        z_top = root.find(".//surface[@id='10']")
        assert z_bottom.get("boundary") == "reflective"
        assert z_top.get("boundary") == "reflective"

    def test_rod_tip_plane_exists(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        rod_tip = root.find(".//surface[@id='11']")
        assert rod_tip is not None
        assert rod_tip.get("type") == "z-plane"

    def test_rod_tip_z_changes_with_position(self) -> None:
        """rod_position이 다르면 rod tip Z도 다르다."""
        geo_a = GeometryParams(
            absorber_outer_radius=0.52,
            extra_params={"rod_position": 50},
        )
        geo_b = GeometryParams(
            absorber_outer_radius=0.52,
            extra_params={"rod_position": 200},
        )
        root_a = _build_geometry_xml(geo_a)
        root_b = _build_geometry_xml(geo_b)
        tip_a = float(root_a.find(".//surface[@id='11']").get("coeffs"))
        tip_b = float(root_b.find(".//surface[@id='11']").get("coeffs"))
        assert tip_a < tip_b  # 더 많이 삽입 → z_tip 더 낮음

    def test_absorber_cylinder_radius(self, geo_3d: GeometryParams) -> None:
        root = _build_geometry_xml(geo_3d)
        absorber_cyl = root.find(".//surface[@id='8']")
        assert "0.52" in absorber_cyl.get("coeffs")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestBuildSettingsXml:
    """_build_settings_xml 함수 테스트."""

    def test_eigenvalue_mode(self, geo_2d: GeometryParams) -> None:
        settings = SimulationSettings()
        root = _build_settings_xml(settings, geo_2d)
        run_mode = root.find("run_mode")
        assert run_mode is not None
        assert run_mode.text == "eigenvalue"

    def test_batch_settings(self, geo_2d: GeometryParams) -> None:
        settings = SimulationSettings(batches=200, inactive=20, particles=5000)
        root = _build_settings_xml(settings, geo_2d)
        assert root.find("batches").text == "200"
        assert root.find("inactive").text == "20"
        assert root.find("particles").text == "5000"

    def test_source_defined(self, geo_2d: GeometryParams) -> None:
        settings = SimulationSettings()
        root = _build_settings_xml(settings, geo_2d)
        source = root.find("source")
        assert source is not None

    def test_point_source_for_2d(self, geo_2d: GeometryParams) -> None:
        settings = SimulationSettings(source_type="point")
        root = _build_settings_xml(settings, geo_2d)
        params = root.find(".//parameters")
        assert params is not None
        assert params.text == "0.0 0.0 0.0"

    def test_box_source_for_3d(self, geo_3d: GeometryParams) -> None:
        """3D 모델은 box 소스를 사용한다."""
        settings = SimulationSettings()
        root = _build_settings_xml(settings, geo_3d)
        space = root.find(".//space")
        assert space.get("type") == "box"

    def test_box_source_spans_fuel(self, geo_3d: GeometryParams) -> None:
        """box 소스가 연료 영역을 커버한다."""
        settings = SimulationSettings()
        root = _build_settings_xml(settings, geo_3d)
        params = root.find(".//parameters")
        values = [float(v) for v in params.text.split()]
        assert len(values) == 6
        # z_max = active_height
        assert values[5] == pytest.approx(geo_3d.fuel_height)


# ---------------------------------------------------------------------------
# GeometryParams Validation
# ---------------------------------------------------------------------------

class TestGeometryParamsValidation:
    """GeometryParams 제어봉 검증 테스트."""

    def test_absorber_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="absorber_outer_radius"):
            GeometryParams(absorber_outer_radius=0.40)

    def test_absorber_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="absorber_outer_radius"):
            GeometryParams(absorber_outer_radius=0.63)

    def test_absorber_valid(self) -> None:
        geo = GeometryParams(absorber_outer_radius=0.52)
        assert geo.absorber_outer_radius == 0.52


# ---------------------------------------------------------------------------
# Integration: generate_input
# ---------------------------------------------------------------------------

class TestGenerateInput:
    """generate_input 통합 함수 테스트."""

    def test_generates_three_files(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        files = generate_input(config, case_dir)
        assert len(files) == 3

    def test_files_in_input_dir(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        files = generate_input(config, case_dir)
        for f in files:
            assert f.parent == case_dir / "input"

    def test_file_names(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        files = generate_input(config, case_dir)
        names = {f.name for f in files}
        assert names == {"materials.xml", "geometry.xml", "settings.xml"}

    def test_files_are_valid_xml(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        files = generate_input(config, case_dir)
        for f in files:
            tree = ET.parse(f)
            assert tree.getroot() is not None

    def test_creates_input_dir(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        generate_input(config, case_dir)
        assert (case_dir / "input").is_dir()

    def test_custom_config_reflected(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig(
            materials=MaterialParams(fuel_enrichment=4.5, fuel_density=10.0),
            settings=SimulationSettings(batches=300, particles=10000),
        )
        generate_input(config, case_dir)

        mat_tree = ET.parse(case_dir / "input" / "materials.xml")
        fuel_density = mat_tree.find(".//material[@name='UO2']/density")
        assert fuel_density.get("value") == "10.0"

        set_tree = ET.parse(case_dir / "input" / "settings.xml")
        assert set_tree.find("batches").text == "300"
        assert set_tree.find("particles").text == "10000"

    def test_3d_generates_absorber_material(self, tmp_path: Path) -> None:
        """3D 모델에서 B4C 재료가 포함된다."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig(
            geometry=GeometryParams(
                absorber_outer_radius=0.52,
                extra_params={"rod_position": 114},
            ),
        )
        generate_input(config, case_dir)

        mat_tree = ET.parse(case_dir / "input" / "materials.xml")
        assert mat_tree.find(".//material[@name='B4C']") is not None

        geo_tree = ET.parse(case_dir / "input" / "geometry.xml")
        cells = geo_tree.findall(".//cell")
        assert len(cells) == 6

    def test_2d_no_absorber_material(self, tmp_path: Path) -> None:
        """2D 모델에서 B4C 재료가 없다."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        generate_input(config, case_dir)

        mat_tree = ET.parse(case_dir / "input" / "materials.xml")
        assert mat_tree.find(".//material[@name='B4C']") is None
