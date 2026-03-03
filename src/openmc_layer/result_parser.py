"""OpenMC statepoint 결과 파서.

OpenMC 시뮬레이션 결과 파일(statepoint.h5)을 읽어
SimulationResult 모델로 변환한다.

h5py를 사용하여 HDF5 파일을 직접 읽으며,
openmc Python 패키지에 의존하지 않는다.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from src.armi_layer.models import SimulationResult, TallyResult

logger = logging.getLogger(__name__)

# statepoint 파일명 패턴
STATEPOINT_GLOB = "statepoint.*.h5"


class StatepointNotFoundError(FileNotFoundError):
    """statepoint 파일을 찾을 수 없을 때 발생하는 예외."""


class StatepointParseError(ValueError):
    """statepoint 파일 파싱 중 오류가 발생했을 때 발생하는 예외."""


def _find_statepoint(case_path: Path) -> Path:
    """케이스 output 디렉토리에서 statepoint 파일을 찾는다.

    가장 큰 배치 번호의 statepoint 파일을 반환한다.
    (예: statepoint.100.h5 > statepoint.50.h5)

    Args:
        case_path: 케이스 폴더 경로.

    Returns:
        statepoint 파일 경로.

    Raises:
        StatepointNotFoundError: 파일을 찾을 수 없을 때.
    """
    output_dir = case_path / "output"

    if not output_dir.is_dir():
        raise StatepointNotFoundError(
            f"output 디렉토리가 존재하지 않습니다: {output_dir}"
        )

    statepoints = sorted(output_dir.glob(STATEPOINT_GLOB))

    if not statepoints:
        raise StatepointNotFoundError(
            f"statepoint 파일을 찾을 수 없습니다: {output_dir}"
        )

    # 배치 번호 기준으로 가장 마지막 파일 선택
    return statepoints[-1]


def _extract_keff(h5file: h5py.File) -> tuple[float, float]:
    """statepoint에서 keff와 표준편차를 추출한다.

    OpenMC statepoint의 k_combined 데이터셋에서
    유효증배계수와 표준편차를 읽는다.

    Args:
        h5file: 열린 HDF5 파일.

    Returns:
        (keff, keff_std) 튜플.

    Raises:
        StatepointParseError: k_combined 데이터가 없을 때.
    """
    if "k_combined" not in h5file:
        raise StatepointParseError(
            "k_combined 데이터셋을 찾을 수 없습니다"
        )

    k_combined = h5file["k_combined"][()]
    keff = float(k_combined[0])
    keff_std = float(k_combined[1])

    return keff, keff_std


def _extract_batches(h5file: h5py.File) -> int:
    """statepoint에서 완료된 배치 수를 추출한다.

    Args:
        h5file: 열린 HDF5 파일.

    Returns:
        완료된 배치 수.
    """
    if "n_batches" in h5file.attrs:
        return int(h5file.attrs["n_batches"])

    if "n_batches" in h5file:
        return int(h5file["n_batches"][()])

    # statepoint 파일명에서 추출 시도 (statepoint.100.h5 → 100)
    filename = Path(h5file.filename).name
    parts = filename.split(".")
    if len(parts) >= 3:
        try:
            return int(parts[1])
        except ValueError:
            pass

    return 0


def _extract_tallies(h5file: h5py.File) -> list[TallyResult]:
    """statepoint에서 탈리 데이터를 추출한다.

    각 탈리의 이름, 스코어 유형, 평균값, 표준편차를 파싱한다.

    Args:
        h5file: 열린 HDF5 파일.

    Returns:
        TallyResult 목록.
    """
    tallies: list[TallyResult] = []

    if "tallies" not in h5file:
        logger.debug("tallies 그룹이 없습니다")
        return tallies

    tallies_group = h5file["tallies"]

    for tally_key in tallies_group:
        tally_group = tallies_group[tally_key]

        try:
            tally = _parse_single_tally(tally_group, tally_key)
            tallies.append(tally)
        except Exception:
            logger.warning("탈리 파싱 실패: %s", tally_key, exc_info=True)

    return tallies


def _parse_single_tally(
    tally_group: h5py.Group,
    tally_key: str,
) -> TallyResult:
    """단일 탈리 그룹을 TallyResult로 변환한다.

    Args:
        tally_group: 탈리 HDF5 그룹.
        tally_key: 탈리 키 이름.

    Returns:
        파싱된 TallyResult.
    """
    # 탈리 이름
    if "name" in tally_group.attrs:
        name = tally_group.attrs["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
    else:
        name = tally_key

    # 스코어 유형
    scores: list[str] = []
    if "score_bins" in tally_group:
        score_data = tally_group["score_bins"][()]
        for s in score_data:
            if isinstance(s, bytes):
                scores.append(s.decode("utf-8"))
            else:
                scores.append(str(s))

    # 결과 데이터: shape (n_bins, n_scores, 3)
    # [:, :, 1] = sum, [:, :, 2] = sum_sq
    if "results" not in tally_group:
        return TallyResult(
            name=str(name),
            scores=scores,
            mean=[],
            std_dev=[],
        )

    results = tally_group["results"][()]
    n_realizations = int(tally_group.attrs.get("n_realizations", 1))

    # 모든 빈을 평탄화하여 1D로
    # shape: (n_total_bins, n_scores, 3)
    result_sum = results[:, :, 1]  # shape: (n_bins, n_scores)
    result_sum_sq = results[:, :, 2]  # shape: (n_bins, n_scores)

    # 빈×스코어를 1D로 평탄화
    flat_sum = result_sum.flatten()
    flat_sum_sq = result_sum_sq.flatten()

    mean = flat_sum / max(n_realizations, 1)
    if n_realizations > 1:
        variance = np.abs(flat_sum_sq / n_realizations - mean**2) / (
            n_realizations - 1
        )
        std_dev = np.sqrt(variance)
    else:
        std_dev = np.zeros_like(mean)

    return TallyResult(
        name=str(name),
        scores=scores,
        mean=mean.tolist(),
        std_dev=std_dev.tolist(),
    )


def _extract_runtime(h5file: h5py.File) -> float:
    """statepoint에서 시뮬레이션 실행 시간을 추출한다.

    Args:
        h5file: 열린 HDF5 파일.

    Returns:
        실행 시간 (초). 정보가 없으면 0.0.
    """
    for key in ("runtime", "total_time"):
        if key in h5file.attrs:
            return float(h5file.attrs[key])
        if key in h5file:
            return float(h5file[key][()])

    return 0.0


def parse_results(case_path: Path) -> SimulationResult:
    """케이스 결과를 파싱하여 SimulationResult로 반환한다.

    case_path/output/ 디렉토리에서 statepoint 파일을 찾아
    keff, 탈리, 실행 시간 등을 추출한다.

    Args:
        case_path: 케이스 폴더 경로.

    Returns:
        파싱된 SimulationResult.

    Raises:
        StatepointNotFoundError: statepoint 파일이 없을 때.
        StatepointParseError: 파싱 오류 시.
    """
    statepoint_path = _find_statepoint(case_path)

    logger.info("statepoint 파싱 시작: %s", statepoint_path)

    try:
        with h5py.File(statepoint_path, "r") as h5file:
            keff, keff_std = _extract_keff(h5file)
            batches = _extract_batches(h5file)
            tallies = _extract_tallies(h5file)
            runtime = _extract_runtime(h5file)
    except (OSError, KeyError) as e:
        raise StatepointParseError(
            f"statepoint 파싱 실패: {statepoint_path}, {e}"
        ) from e

    result = SimulationResult(
        keff=keff,
        keff_std=keff_std,
        tallies=tallies,
        runtime=runtime,
        statepoint_path=statepoint_path,
        batches_completed=batches,
    )

    logger.info(
        "statepoint 파싱 완료: keff=%.5f±%.5f, tallies=%d, batches=%d",
        keff,
        keff_std,
        len(tallies),
        batches,
    )

    return result
