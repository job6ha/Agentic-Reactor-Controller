"""OpenMC XML 입력 파일 생성기.

CaseConfig의 geometry/material/settings 파라미터를
OpenMC 형식의 XML 파일(geometry.xml, materials.xml, settings.xml)로
변환하여 케이스 폴더의 input/ 디렉토리에 출력한다.

현재 PWR pin cell 기하 구조를 지원한다.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    SimulationSettings,
)

logger = logging.getLogger(__name__)


def _pretty_xml(element: ET.Element) -> str:
    """ElementTree를 보기 좋게 포매팅한 XML 문자열로 변환한다.

    Args:
        element: XML 루트 엘리먼트.

    Returns:
        들여쓰기된 XML 문자열.
    """
    rough = ET.tostring(element, encoding="unicode")
    parsed = minidom.parseString(rough)
    # toprettyxml의 첫 줄 (xml declaration)을 제거하고 반환
    lines = parsed.toprettyxml(indent="  ").split("\n")
    return "\n".join(lines[1:]).strip() + "\n"


def _build_materials_xml(materials: MaterialParams) -> ET.Element:
    """MaterialParams를 materials.xml 엘리먼트로 변환한다.

    PWR pin cell 기준: UO2 연료, Zircaloy-4 피복관, 경수 냉각재.

    Args:
        materials: 재료 파라미터.

    Returns:
        materials XML 루트 엘리먼트.
    """
    root = ET.Element("materials")

    # 연료: UO2
    fuel = ET.SubElement(root, "material", id="1", name="UO2")
    ET.SubElement(fuel, "density", value=str(materials.fuel_density), units="g/cc")
    fuel_nuclides = ET.SubElement(fuel, "nuclides")
    # U-235, U-238, O-16 조성 (wt% 기반 단순화)
    enrichment = materials.fuel_enrichment / 100.0
    u235_wo = enrichment * 238.0 / (238.0 + 2 * 16.0)  # UO2 내 U-235 질량분율 근사
    u238_wo = (1 - enrichment) * 238.0 / (238.0 + 2 * 16.0)
    o16_wo = 2 * 16.0 / (238.0 + 2 * 16.0)

    ET.SubElement(fuel_nuclides, "nuclide", name="U235", wo=f"{u235_wo:.6f}")
    ET.SubElement(fuel_nuclides, "nuclide", name="U238", wo=f"{u238_wo:.6f}")
    ET.SubElement(fuel_nuclides, "nuclide", name="O16", wo=f"{o16_wo:.6f}")

    # 피복관: Zircaloy-4
    clad = ET.SubElement(root, "material", id="2", name="Zircaloy-4")
    ET.SubElement(clad, "density", value=str(materials.clad_density), units="g/cc")
    clad_nuclides = ET.SubElement(clad, "nuclides")
    ET.SubElement(clad_nuclides, "nuclide", name="Zr90", wo="0.5145")
    ET.SubElement(clad_nuclides, "nuclide", name="Zr91", wo="0.1122")
    ET.SubElement(clad_nuclides, "nuclide", name="Zr92", wo="0.1715")
    ET.SubElement(clad_nuclides, "nuclide", name="Zr94", wo="0.1738")
    ET.SubElement(clad_nuclides, "nuclide", name="Zr96", wo="0.0280")

    # 냉각재: H2O
    water = ET.SubElement(root, "material", id="3", name="H2O")
    ET.SubElement(water, "density", value=str(materials.coolant_density), units="g/cc")
    water_nuclides = ET.SubElement(water, "nuclides")
    ET.SubElement(water_nuclides, "nuclide", name="H1", wo="0.111894")
    ET.SubElement(water_nuclides, "nuclide", name="O16", wo="0.888106")
    ET.SubElement(water, "sab", name="c_H_in_H2O")

    return root


def _build_geometry_xml(geometry: GeometryParams) -> ET.Element:
    """GeometryParams를 geometry.xml 엘리먼트로 변환한다.

    PWR pin cell: 연료봉 → 피복관 → 냉각재 (사각 격자).

    Args:
        geometry: 기하 구조 파라미터.

    Returns:
        geometry XML 루트 엘리먼트.
    """
    root = ET.Element("geometry")

    # 표면 정의
    surfaces = ET.SubElement(root, "surfaces")
    half_pitch = geometry.pitch / 2.0

    ET.SubElement(
        surfaces,
        "surface",
        id="1",
        type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.fuel_radius}",
    )
    ET.SubElement(
        surfaces,
        "surface",
        id="2",
        type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.clad_inner_radius}",
    )
    ET.SubElement(
        surfaces,
        "surface",
        id="3",
        type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.clad_outer_radius}",
    )
    ET.SubElement(
        surfaces,
        "surface",
        id="4",
        type="x-plane",
        coeffs=f"{-half_pitch}",
        boundary="reflective",
    )
    ET.SubElement(
        surfaces,
        "surface",
        id="5",
        type="x-plane",
        coeffs=f"{half_pitch}",
        boundary="reflective",
    )
    ET.SubElement(
        surfaces,
        "surface",
        id="6",
        type="y-plane",
        coeffs=f"{-half_pitch}",
        boundary="reflective",
    )
    ET.SubElement(
        surfaces,
        "surface",
        id="7",
        type="y-plane",
        coeffs=f"{half_pitch}",
        boundary="reflective",
    )

    # 셀 정의
    cells = ET.SubElement(root, "cells")

    # 연료 셀: surface 1 내부
    ET.SubElement(
        cells,
        "cell",
        id="1",
        material="1",
        region="-1",
        name="fuel",
    )
    # 갭 셀: surface 1~2 사이 (진공)
    ET.SubElement(
        cells,
        "cell",
        id="2",
        material="void",
        region="1 -2",
        name="gap",
    )
    # 피복관 셀: surface 2~3 사이
    ET.SubElement(
        cells,
        "cell",
        id="3",
        material="2",
        region="2 -3",
        name="clad",
    )
    # 냉각재 셀: surface 3 바깥, 사각 경계 내부
    ET.SubElement(
        cells,
        "cell",
        id="4",
        material="3",
        region="3 4 -5 6 -7",
        name="water",
    )

    return root


def _build_settings_xml(settings: SimulationSettings) -> ET.Element:
    """SimulationSettings를 settings.xml 엘리먼트로 변환한다.

    Args:
        settings: 시뮬레이션 설정.

    Returns:
        settings XML 루트 엘리먼트.
    """
    root = ET.Element("settings")

    # 실행 모드
    ET.SubElement(root, "run_mode").text = "eigenvalue"

    # 소스 정의
    source = ET.SubElement(root, "source", strength="1.0")
    space = ET.SubElement(source, "space", type=settings.source_type)
    if settings.source_type == "point":
        ET.SubElement(space, "parameters").text = "0.0 0.0 0.0"

    # 배치 설정
    ET.SubElement(root, "batches").text = str(settings.batches)
    ET.SubElement(root, "inactive").text = str(settings.inactive)
    ET.SubElement(root, "particles").text = str(settings.particles)

    return root


def generate_input(config: CaseConfig, case_path: Path) -> list[Path]:
    """CaseConfig를 OpenMC XML 입력 파일로 변환한다.

    case_path/input/ 디렉토리에 geometry.xml, materials.xml,
    settings.xml을 생성한다.

    Args:
        config: 케이스 설정.
        case_path: 케이스 폴더 경로.

    Returns:
        생성된 XML 파일 경로 목록.
    """
    input_dir = case_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # materials.xml
    materials_xml = _build_materials_xml(config.materials)
    materials_path = input_dir / "materials.xml"
    materials_path.write_text(_pretty_xml(materials_xml), encoding="utf-8")
    generated.append(materials_path)

    # geometry.xml
    geometry_xml = _build_geometry_xml(config.geometry)
    geometry_path = input_dir / "geometry.xml"
    geometry_path.write_text(_pretty_xml(geometry_xml), encoding="utf-8")
    generated.append(geometry_path)

    # settings.xml
    settings_xml = _build_settings_xml(config.settings)
    settings_path = input_dir / "settings.xml"
    settings_path.write_text(_pretty_xml(settings_xml), encoding="utf-8")
    generated.append(settings_path)

    logger.info(
        "OpenMC 입력 파일 생성 완료: case=%s, files=%d",
        case_path.name,
        len(generated),
    )

    return generated
