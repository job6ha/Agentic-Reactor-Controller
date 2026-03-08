"""OpenMC XML 입력 파일 생성기.

CaseConfig의 geometry/material/settings 파라미터를
OpenMC 형식의 XML 파일(geometry.xml, materials.xml, settings.xml)로
변환하여 케이스 폴더의 input/ 디렉토리에 출력한다.

제어봉 모델링:
- ``absorber_outer_radius``가 설정되면 3D 모델 (B4C 제어봉 포함)
- 미설정 시 기존 2D pin cell 모델 (하위 호환)
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

# 제어봉 Z축 최소 오프셋 (cm) — 완전 삽입/인출 시 영체적 셀 방지
_Z_EPSILON = 0.01


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


def _has_control_rod(geometry: GeometryParams) -> bool:
    """제어봉 모델링 여부를 판단한다."""
    return geometry.absorber_outer_radius is not None


def _compute_rod_tip_z(geometry: GeometryParams) -> float:
    """제어봉 삽입 깊이로부터 rod tip Z 좌표를 계산한다.

    rod_position 규약:
    - 0 = 완전 삽입 (흡수체가 전체 활성 구간 차지)
    - rod_steps_max = 완전 인출 (흡수체 없음)

    rod tip = 흡수체 하단 Z 좌표.
    - 완전 삽입: z_tip ≈ 0 (epsilon)
    - 완전 인출: z_tip ≈ active_height (epsilon)

    Args:
        geometry: 기하 파라미터 (extra_params에 rod_position 포함).

    Returns:
        rod tip Z 좌표 (cm).
    """
    rod_position = geometry.extra_params.get("rod_position", geometry.rod_steps_max)
    rod_max = geometry.rod_steps_max
    active_height = geometry.fuel_height

    # rod_position 범위 검증
    if rod_position < 0 or rod_position > rod_max:
        logger.warning(
            "rod_position(%.1f)이 유효 범위 [0, %d]를 벗어남. 클램핑 적용.",
            rod_position, rod_max,
        )

    # rod_position=0 → z_tip=0 (전체 삽입), rod_position=max → z_tip=H (전체 인출)
    fraction = max(0.0, min(1.0, rod_position / rod_max))
    z_tip = fraction * active_height

    # 영체적 셀 방지: 경계에서 epsilon만큼 오프셋
    z_tip = max(_Z_EPSILON, min(active_height - _Z_EPSILON, z_tip))

    return z_tip


def _build_materials_xml(
    materials: MaterialParams,
    *,
    include_absorber: bool = False,
) -> ET.Element:
    """MaterialParams를 materials.xml 엘리먼트로 변환한다.

    PWR pin cell 기준: UO2 연료, Zircaloy-4 피복관, 경수 냉각재.
    제어봉 모델 시 B4C 흡수체 추가.

    Args:
        materials: 재료 파라미터.
        include_absorber: B4C 흡수체 재료 포함 여부.

    Returns:
        materials XML 루트 엘리먼트.
    """
    root = ET.Element("materials")

    # 연료: UO2
    fuel = ET.SubElement(root, "material", id="1", name="UO2")
    ET.SubElement(fuel, "density", value=str(materials.fuel_density), units="g/cc")
    enrichment = materials.fuel_enrichment / 100.0
    # 동위원소 원자량 기반 UO2 질량분율 계산
    m_u235 = 235.044
    m_u238 = 238.051
    m_o16 = 15.999
    m_uo2 = enrichment * m_u235 + (1 - enrichment) * m_u238 + 2 * m_o16
    u235_wo = enrichment * m_u235 / m_uo2
    u238_wo = (1 - enrichment) * m_u238 / m_uo2
    o16_wo = 2 * m_o16 / m_uo2

    ET.SubElement(fuel, "nuclide", name="U235", wo=f"{u235_wo:.6f}")
    ET.SubElement(fuel, "nuclide", name="U238", wo=f"{u238_wo:.6f}")
    ET.SubElement(fuel, "nuclide", name="O16", wo=f"{o16_wo:.6f}")

    # 피복관: Zircaloy-4
    clad = ET.SubElement(root, "material", id="2", name="Zircaloy-4")
    ET.SubElement(clad, "density", value=str(materials.clad_density), units="g/cc")
    ET.SubElement(clad, "nuclide", name="Zr90", wo="0.5145")
    ET.SubElement(clad, "nuclide", name="Zr91", wo="0.1122")
    ET.SubElement(clad, "nuclide", name="Zr92", wo="0.1715")
    ET.SubElement(clad, "nuclide", name="Zr94", wo="0.1738")
    ET.SubElement(clad, "nuclide", name="Zr96", wo="0.0280")

    # 냉각재: H2O
    water = ET.SubElement(root, "material", id="3", name="H2O")
    ET.SubElement(water, "density", value=str(materials.coolant_density), units="g/cc")
    ET.SubElement(water, "nuclide", name="H1", wo="0.111894")
    ET.SubElement(water, "nuclide", name="O16", wo="0.888106")
    ET.SubElement(water, "sab", name="c_H_in_H2O")

    # 흡수체: B4C (boron carbide)
    if include_absorber:
        absorber = ET.SubElement(root, "material", id="4", name="B4C")
        ET.SubElement(
            absorber, "density",
            value=str(materials.absorber_density), units="g/cc",
        )
        # 자연 붕소 동위원소비: B-10 ~19.9%, B-11 ~80.1%
        # B4C 화학량론: 4B + 1C → 분자량 = 4*10.81 + 12.011 = 55.251
        b_mass_frac = 4 * 10.81 / 55.251  # ~0.7826
        c_mass_frac = 12.011 / 55.251  # ~0.2174
        b10_wo = b_mass_frac * 0.199
        b11_wo = b_mass_frac * 0.801
        ET.SubElement(absorber, "nuclide", name="B10", wo=f"{b10_wo:.6f}")
        ET.SubElement(absorber, "nuclide", name="B11", wo=f"{b11_wo:.6f}")
        ET.SubElement(absorber, "nuclide", name="C12", wo=f"{c_mass_frac:.6f}")

    return root


def _build_geometry_xml_2d(geometry: GeometryParams) -> ET.Element:
    """2D pin cell 기하 구조 (제어봉 없음, 하위 호환).

    Args:
        geometry: 기하 구조 파라미터.

    Returns:
        geometry XML 루트 엘리먼트.
    """
    root = ET.Element("geometry")
    half_pitch = geometry.pitch / 2.0

    ET.SubElement(
        root, "surface", id="1", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.fuel_radius}",
    )
    ET.SubElement(
        root, "surface", id="2", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.clad_inner_radius}",
    )
    ET.SubElement(
        root, "surface", id="3", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.clad_outer_radius}",
    )
    ET.SubElement(
        root, "surface", id="4", type="x-plane",
        coeffs=f"{-half_pitch}", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="5", type="x-plane",
        coeffs=f"{half_pitch}", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="6", type="y-plane",
        coeffs=f"{-half_pitch}", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="7", type="y-plane",
        coeffs=f"{half_pitch}", boundary="reflective",
    )

    ET.SubElement(root, "cell", id="1", material="1", region="-1", name="fuel")
    ET.SubElement(root, "cell", id="2", material="void", region="1 -2", name="gap")
    ET.SubElement(root, "cell", id="3", material="2", region="2 -3", name="clad")
    ET.SubElement(
        root, "cell", id="4", material="3",
        region="3 4 -5 6 -7", name="water",
    )

    return root


def _build_geometry_xml_3d(geometry: GeometryParams) -> ET.Element:
    """3D pin cell 기하 구조 (B4C 제어봉 포함).

    표면 구성:
        S1: 연료 실린더
        S2: 피복관 내경 실린더
        S3: 피복관 외경 실린더
        S4-S7: XY 경계면 (반사)
        S8: 흡수체 외경 실린더
        S9: Z 하한 (z=0)
        S10: Z 상한 (z=active_height)
        S11: 제어봉 tip Z 평면 (z=z_tip)

    셀 구성:
        C1: 연료 (S1 내부, Z 범위)
        C2: 갭 (S1~S2, Z 범위)
        C3: 피복관 (S2~S3, Z 범위)
        C4: 흡수체 (S3~S8, z_tip 위) — B4C
        C5: 가이드튜브 냉각재 (S3~S8, z_tip 아래) — H2O
        C6: 외부 냉각재 (S8~경계, Z 범위)

    Args:
        geometry: 기하 구조 파라미터.

    Returns:
        geometry XML 루트 엘리먼트.
    """
    root = ET.Element("geometry")
    half_pitch = geometry.pitch / 2.0
    active_height = geometry.fuel_height
    z_tip = _compute_rod_tip_z(geometry)

    # --- 실린더 표면 ---
    ET.SubElement(
        root, "surface", id="1", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.fuel_radius}",
    )
    ET.SubElement(
        root, "surface", id="2", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.clad_inner_radius}",
    )
    ET.SubElement(
        root, "surface", id="3", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.clad_outer_radius}",
    )

    # --- XY 경계면 (반사) ---
    ET.SubElement(
        root, "surface", id="4", type="x-plane",
        coeffs=f"{-half_pitch}", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="5", type="x-plane",
        coeffs=f"{half_pitch}", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="6", type="y-plane",
        coeffs=f"{-half_pitch}", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="7", type="y-plane",
        coeffs=f"{half_pitch}", boundary="reflective",
    )

    # --- 흡수체 실린더 ---
    ET.SubElement(
        root, "surface", id="8", type="z-cylinder",
        coeffs=f"0.0 0.0 {geometry.absorber_outer_radius}",
    )

    # --- Z축 경계면 ---
    ET.SubElement(
        root, "surface", id="9", type="z-plane",
        coeffs="0.0", boundary="reflective",
    )
    ET.SubElement(
        root, "surface", id="10", type="z-plane",
        coeffs=f"{active_height}", boundary="reflective",
    )

    # --- 제어봉 tip Z 평면 ---
    ET.SubElement(
        root, "surface", id="11", type="z-plane",
        coeffs=f"{z_tip}",
    )

    # --- 셀 정의 ---
    # Z 범위 공통: 9 -10 (z>0 AND z<H)
    z_range = "9 -10"

    # C1: 연료
    ET.SubElement(
        root, "cell", id="1", material="1",
        region=f"-1 {z_range}", name="fuel",
    )
    # C2: 갭 (진공)
    ET.SubElement(
        root, "cell", id="2", material="void",
        region=f"1 -2 {z_range}", name="gap",
    )
    # C3: 피복관
    ET.SubElement(
        root, "cell", id="3", material="2",
        region=f"2 -3 {z_range}", name="clad",
    )
    # C4: 흡수체 (z_tip 위 = 제어봉 존재 영역)
    # region: S3 밖, S8 안, z > z_tip, z < H
    ET.SubElement(
        root, "cell", id="4", material="4",
        region="3 -8 11 -10", name="absorber",
    )
    # C5: 가이드튜브 냉각재 (z_tip 아래 = 제어봉 없는 영역)
    # region: S3 밖, S8 안, z > 0, z < z_tip
    ET.SubElement(
        root, "cell", id="5", material="3",
        region="3 -8 9 -11", name="guide_coolant",
    )
    # C6: 외부 냉각재 (S8 밖, 사각 경계 안)
    ET.SubElement(
        root, "cell", id="6", material="3",
        region=f"8 4 -5 6 -7 {z_range}", name="water",
    )

    logger.info(
        "3D 제어봉 기하 생성: z_tip=%.2f cm (rod_position=%.0f/%d)",
        z_tip,
        geometry.extra_params.get("rod_position", geometry.rod_steps_max),
        geometry.rod_steps_max,
    )

    return root


def _build_geometry_xml(geometry: GeometryParams) -> ET.Element:
    """GeometryParams를 geometry.xml 엘리먼트로 변환한다.

    제어봉 모델링 여부에 따라 2D 또는 3D 모델을 선택한다.

    Args:
        geometry: 기하 구조 파라미터.

    Returns:
        geometry XML 루트 엘리먼트.
    """
    if _has_control_rod(geometry):
        return _build_geometry_xml_3d(geometry)
    return _build_geometry_xml_2d(geometry)


def _build_settings_xml(
    settings: SimulationSettings,
    geometry: GeometryParams,
) -> ET.Element:
    """SimulationSettings를 settings.xml 엘리먼트로 변환한다.

    3D 모델인 경우 box 소스로 자동 전환한다.

    Args:
        settings: 시뮬레이션 설정.
        geometry: 기하 구조 파라미터 (3D 판단용).

    Returns:
        settings XML 루트 엘리먼트.
    """
    root = ET.Element("settings")

    ET.SubElement(root, "run_mode").text = "eigenvalue"

    # 소스 정의
    source = ET.SubElement(root, "source", strength="1.0")

    if _has_control_rod(geometry):
        # 3D 모델: box 소스 (연료 영역 전체)
        half_pitch = geometry.pitch / 2.0
        space = ET.SubElement(source, "space", type="box")
        ET.SubElement(space, "parameters").text = (
            f"{-half_pitch} {-half_pitch} 0.0 "
            f"{half_pitch} {half_pitch} {geometry.fuel_height}"
        )
    else:
        # 2D 모델: 기존 소스 유형
        space = ET.SubElement(source, "space", type=settings.source_type)
        if settings.source_type == "point":
            ET.SubElement(space, "parameters").text = "0.0 0.0 0.0"

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

    include_absorber = _has_control_rod(config.geometry)
    generated: list[Path] = []

    # materials.xml
    materials_xml = _build_materials_xml(
        config.materials, include_absorber=include_absorber,
    )
    materials_path = input_dir / "materials.xml"
    materials_path.write_text(_pretty_xml(materials_xml), encoding="utf-8")
    generated.append(materials_path)

    # geometry.xml
    geometry_xml = _build_geometry_xml(config.geometry)
    geometry_path = input_dir / "geometry.xml"
    geometry_path.write_text(_pretty_xml(geometry_xml), encoding="utf-8")
    generated.append(geometry_path)

    # settings.xml
    settings_xml = _build_settings_xml(config.settings, config.geometry)
    settings_path = input_dir / "settings.xml"
    settings_path.write_text(_pretty_xml(settings_xml), encoding="utf-8")
    generated.append(settings_path)

    logger.info(
        "OpenMC 입력 파일 생성 완료: case=%s, files=%d, control_rod=%s",
        case_path.name,
        len(generated),
        include_absorber,
    )

    return generated
