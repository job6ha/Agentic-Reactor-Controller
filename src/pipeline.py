"""시뮬레이션 오케스트레이션 파이프라인.

설정 파일을 로드하여 Controller 기반 시뮬레이션 루프를 자동 실행한다.
propose → execute → evaluate → update 사이클을 반복하고,
수렴 또는 종료 조건 충족 시 결과를 수집하여 보고서를 생성한다.

컨트롤러 팩토리를 통해 SimpleController, LLMController 등
다양한 컨트롤러를 설정 기반으로 동적 생성한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.armi_layer.case_folder import create_case
from src.armi_layer.case_manager import CaseManager
from src.armi_layer.models import (
    CaseConfig,
    ReactorState,
    SimulationResult,
)
from src.armi_layer.result_collector import (
    collect_results,
    export_csv,
    export_json,
)
from src.controller.base import ActionType, BaseController
from src.controller.factory import create_controller
from src.openmc_layer.input_generator import generate_input
from src.openmc_layer.kpi_calculator import calculate_kpi, save_kpi
from src.openmc_layer.result_parser import parse_results
from src.pipeline_config import PipelineConfig, load_config
from src.run_manager.status import run_case

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과 요약.

    Attributes:
        total_iterations: 전체 반복 횟수.
        successful_cases: 성공한 케이스 수.
        failed_cases: 실패한 케이스 수.
        skipped_cases: 스킵된 케이스 수.
        stop_reason: 종료 사유.
        case_ids: 생성된 케이스 ID 목록.
    """

    total_iterations: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    skipped_cases: int = 0
    stop_reason: str = ""
    case_ids: list[str] = field(default_factory=list)


class SimulationPipeline:
    """오케스트레이션 파이프라인.

    Controller의 제안에 따라 시뮬레이션을 반복 실행하고,
    수렴 시 결과를 수집하여 보고서를 생성한다.

    컨트롤러는 설정의 ``controller_type``에 따라 팩토리에서 자동 생성된다.

    Args:
        config: 파이프라인 설정.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._controller: BaseController = create_controller(config.controller)
        self._run_config = config.run
        self._runs_dir = config.output.runs_dir
        self._case_manager = CaseManager(self._runs_dir)

    @classmethod
    def from_config(cls, config_path: Path) -> SimulationPipeline:
        """설정 파일에서 파이프라인을 생성한다.

        Args:
            config_path: JSON 설정 파일 경로.

        Returns:
            초기화된 SimulationPipeline.
        """
        config = load_config(config_path)
        return cls(config)

    def run(self) -> PipelineResult:
        """메인 시뮬레이션 루프를 실행한다.

        Returns:
            파이프라인 실행 결과 요약.
        """
        # 설정 파일의 simulation 설정을 초기 CaseConfig에 반영
        initial_config = CaseConfig(settings=self._config.simulation)
        state = ReactorState(current_config=initial_config)
        result = PipelineResult()

        ctrl_name = type(self._controller).__name__
        logger.info("파이프라인 시작: controller_type=%s", ctrl_name)

        while True:
            # 1. Controller에게 다음 행동 제안 요청
            actions = self._controller.propose_actions(state)

            # 2. STOP 액션이 있으면 종료
            stop_actions = [a for a in actions if a.action_type == ActionType.STOP]
            if stop_actions:
                result.stop_reason = stop_actions[0].reason
                logger.info("루프 종료: %s", result.stop_reason)
                break

            # 3. 액션을 적용하여 새 CaseConfig 생성
            case_config = self._controller.apply_actions_to_case(actions, state)
            result.total_iterations += 1

            # 4. 단일 케이스 실행
            sim_result = self._run_single_case(case_config, result)

            if sim_result is None:
                # 실패한 케이스: iteration만 증가시키고 다음으로
                state = state.model_copy(
                    update={"iteration": state.iteration + 1},
                )
                continue

            # 5. 결과 평가 및 상태 업데이트
            metrics = self._controller.evaluate_results(sim_result)
            state = self._controller.update_state(state, metrics)
            state = state.model_copy(
                update={
                    "current_config": case_config,
                    "history": [*state.history, sim_result],
                },
            )

        # 6. 결과 수집 및 보고서 생성
        self._finalize(result)

        logger.info(
            "파이프라인 완료: iterations=%d, success=%d, failed=%d, skipped=%d",
            result.total_iterations,
            result.successful_cases,
            result.failed_cases,
            result.skipped_cases,
        )

        return result

    def _run_single_case(
        self,
        case_config: CaseConfig,
        result: PipelineResult,
    ) -> SimulationResult | None:
        """단일 케이스를 실행한다.

        케이스 폴더 생성 → XML 입력 생성 → OpenMC 실행 → 결과 파싱 → KPI 계산.
        실패 시 None을 반환하고 다음 케이스로 넘어간다.

        Args:
            case_config: 케이스 설정.
            result: 파이프라인 결과 (카운터 업데이트용).

        Returns:
            성공 시 SimulationResult, 실패 시 None.
        """
        case_dir: Path | None = None

        try:
            # 케이스 폴더 생성
            case_dir = create_case(case_config, self._runs_dir)
            case_id = case_dir.name
            result.case_ids.append(case_id)

            logger.info("케이스 실행 시작: %s", case_id)

            # OpenMC XML 입력 생성
            generate_input(case_config, case_dir)

            # OpenMC 실행 (재시도 포함)
            run_result = run_case(self._run_config, case_dir)

            if not run_result.success:
                logger.warning("케이스 실행 실패, 스킵: %s", case_id)
                result.failed_cases += 1
                result.skipped_cases += 1
                return None

            # 결과 파싱
            sim_result = parse_results(case_dir)

            # KPI 계산 및 저장
            kpi = calculate_kpi(sim_result)
            save_kpi(kpi, case_dir)

            result.successful_cases += 1
            logger.info(
                "케이스 완료: %s, keff=%.5f±%.5f",
                case_id,
                sim_result.keff,
                sim_result.keff_std,
            )

            return sim_result

        except Exception:
            case_name = case_dir.name if case_dir else "unknown"
            logger.error("케이스 실행 중 예외 발생, 스킵: %s", case_name, exc_info=True)
            result.failed_cases += 1
            result.skipped_cases += 1
            return None

    def _finalize(self, result: PipelineResult) -> None:
        """결과를 수집하고 보고서를 생성한다.

        Args:
            result: 파이프라인 결과.
        """
        if not result.case_ids:
            logger.warning("실행된 케이스가 없어 보고서를 생성하지 않습니다")
            return

        df = collect_results(self._case_manager, result.case_ids)

        if df.empty:
            logger.warning("수집된 결과가 없습니다")
            return

        # CSV 내보내기
        if self._config.output.export_csv:
            csv_path = self._runs_dir / "report.csv"
            export_csv(df, csv_path)

        # JSON 내보내기
        if self._config.output.export_json:
            json_path = self._runs_dir / "report.json"
            export_json(df, json_path)
