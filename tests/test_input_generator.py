"""OpenMC 입력 생성기 테스트.

CaseConfig → XML 변환 결과를 검증한다.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

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
    generate_input,
)


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
        # S(a,b) 테이블 포함
        sab = water.find("sab")
        assert sab is not None
        assert sab.get("name") == "c_H_in_H2O"


class TestBuildGeometryXml:
    """_build_geometry_xml 함수 테스트."""

    def test_surfaces_created(self) -> None:
        geo = GeometryParams()
        root = _build_geometry_xml(geo)
        surfaces = root.findall(".//surface")
        # 3 cylinders + 4 planes = 7
        assert len(surfaces) == 7

    def test_fuel_cylinder_radius(self) -> None:
        geo = GeometryParams()
        root = _build_geometry_xml(geo)
        fuel_surf = root.find(".//surface[@id='1']")
        assert fuel_surf is not None
        assert str(geo.fuel_radius) in fuel_surf.get("coeffs")

    def test_boundary_conditions(self) -> None:
        geo = GeometryParams()
        root = _build_geometry_xml(geo)
        planes = [s for s in root.findall(".//surface") if "plane" in s.get("type", "")]
        for plane in planes:
            assert plane.get("boundary") == "reflective"

    def test_four_cells_created(self) -> None:
        geo = GeometryParams()
        root = _build_geometry_xml(geo)
        cells = root.findall(".//cell")
        # fuel, gap, clad, water
        assert len(cells) == 4

    def test_cell_names(self) -> None:
        geo = GeometryParams()
        root = _build_geometry_xml(geo)
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
        # 반 피치 경계
        x_plane = root.find(".//surface[@id='5']")
        assert "0.75" in x_plane.get("coeffs")


class TestBuildSettingsXml:
    """_build_settings_xml 함수 테스트."""

    def test_eigenvalue_mode(self) -> None:
        settings = SimulationSettings()
        root = _build_settings_xml(settings)
        run_mode = root.find("run_mode")
        assert run_mode is not None
        assert run_mode.text == "eigenvalue"

    def test_batch_settings(self) -> None:
        settings = SimulationSettings(batches=200, inactive=20, particles=5000)
        root = _build_settings_xml(settings)
        assert root.find("batches").text == "200"
        assert root.find("inactive").text == "20"
        assert root.find("particles").text == "5000"

    def test_source_defined(self) -> None:
        settings = SimulationSettings()
        root = _build_settings_xml(settings)
        source = root.find("source")
        assert source is not None

    def test_point_source_parameters(self) -> None:
        settings = SimulationSettings(source_type="point")
        root = _build_settings_xml(settings)
        params = root.find(".//parameters")
        assert params is not None
        assert params.text == "0.0 0.0 0.0"


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
            # XML 파싱이 성공하면 유효
            tree = ET.parse(f)
            assert tree.getroot() is not None

    def test_creates_input_dir(self, tmp_path: Path) -> None:
        """input/ 디렉토리가 없으면 자동 생성."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig()
        generate_input(config, case_dir)
        assert (case_dir / "input").is_dir()

    def test_custom_config_reflected(self, tmp_path: Path) -> None:
        """커스텀 설정이 XML에 반영되는지 확인."""
        case_dir = tmp_path / "case_0001"
        case_dir.mkdir()
        config = CaseConfig(
            materials=MaterialParams(fuel_enrichment=4.5, fuel_density=10.0),
            settings=SimulationSettings(batches=300, particles=10000),
        )
        generate_input(config, case_dir)

        # materials.xml 검증
        mat_tree = ET.parse(case_dir / "input" / "materials.xml")
        fuel_density = mat_tree.find(".//material[@name='UO2']/density")
        assert fuel_density.get("value") == "10.0"

        # settings.xml 검증
        set_tree = ET.parse(case_dir / "input" / "settings.xml")
        assert set_tree.find("batches").text == "300"
        assert set_tree.find("particles").text == "10000"
