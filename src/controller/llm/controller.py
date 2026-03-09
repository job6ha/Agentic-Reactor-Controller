"""LLM + 베이지안 옵티마이제이션 컨트롤러.

LLM 플래너가 제어봉 이동 후보를 제안하고,
GP 기반 BO가 안전 점수를 부여하여 최적 후보를 선택한다.
매 스텝 결과는 JSON 로그에 기록되고 GP 모델 재피팅에 활용된다.
"""

from __future__ import annotations

import logging

from src.armi_layer.models import CaseConfig, ReactorState, SimulationResult
from src.controller.base import Action, ActionType, BaseController
from src.controller.llm.log_store import LogStore
from src.controller.llm.models import (
    CandidateLog,
    LLMControllerConfig,
    LogEntry,
    ScoredCandidate,
)
from src.controller.llm.optimizer import BayesianOptimizer
from src.controller.llm.planner import LLMPlanner
from src.controller.utils import set_nested

logger = logging.getLogger(__name__)


class LLMController(BaseController):
    """LLM + BO 기반 원자로 컨트롤러.

    제어 루프:
    1. LLM 플래너: 제어봉 이동값 후보 N개 생성
    2. BO: 각 후보에 안전 점수 부여 (GP → keff 예측 → 점수)
    3. 최고 점수 후보 선택
    4. 실행 결과로 GP 재피팅 + 로그 저장

    Args:
        config: LLMController 설정.
    """

    def __init__(self, config: LLMControllerConfig) -> None:
        self._config = config
        self._planner = LLMPlanner(
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
        self._optimizer = BayesianOptimizer(target_keff=config.target_keff)
        self._log_store = LogStore(config.log_dir)
        self._rod_position = float(config.initial_rod_position)
        self._step = 0
        self._last_scored: list[ScoredCandidate] = []
        self._last_best: ScoredCandidate | None = None

        # 기존 로그 처리: resume_from_log=True일 때만 복원
        if config.resume_from_log:
            observations = self._log_store.get_observations()
            if observations:
                self._optimizer.refit(observations)
                self._step = len(observations)
                self._rod_position = observations[-1][0]
                logger.info(
                    "기존 로그 복원: step=%d, rod_position=%.1f",
                    self._step,
                    self._rod_position,
                )
        else:
            # 기존 로그가 있으면 삭제하고 초기 상태로 시작
            if self._log_store.log_path.exists():
                self._log_store.log_path.unlink()
                logger.info("기존 로그 초기화 (resume_from_log=False)")

    @property
    def config(self) -> LLMControllerConfig:
        """컨트롤러 설정."""
        return self._config

    @property
    def rod_position(self) -> int:
        """현재 제어봉 위치."""
        return self._rod_position

    @property
    def step(self) -> int:
        """현재 스텝 번호."""
        return self._step

    def propose_actions(self, state: ReactorState) -> list[Action]:
        """LLM + BO로 최적 제어봉 이동을 제안한다.

        Args:
            state: 현재 원자로 상태.

        Returns:
            Action 목록 (MODIFY_PARAM 1개 또는 STOP 1개).
        """
        # 종료 조건 1: 최대 스텝 도달
        if self._step >= self._config.max_iterations:
            logger.info("최대 스텝 도달: %d", self._step)
            return [
                Action(
                    action_type=ActionType.STOP,
                    reason=f"최대 스텝 {self._config.max_iterations}회 도달",
                )
            ]

        # 종료 조건 2: keff 이탈
        if state.kpi and "keff" in state.kpi:
            keff = state.kpi["keff"]
            deviation = abs(keff - self._config.target_keff)
            if deviation > self._config.keff_critical_deviation:
                logger.warning("keff 이탈: keff=%.5f, deviation=%.5f", keff, deviation)
                return [
                    Action(
                        action_type=ActionType.STOP,
                        reason=(
                            f"keff 이탈 (|{keff:.5f} - {self._config.target_keff}|"
                            f" = {deviation:.5f}"
                            f" > {self._config.keff_critical_deviation})"
                        ),
                    )
                ]

        # LLM으로 목표 위치 후보 생성 (과거 로그를 이력으로 전달)
        log_entries = self._log_store.load()
        history_log = [e.model_dump(mode="json") for e in log_entries]
        positions = self._planner.generate(
            state,
            self._rod_position,
            n=self._config.n_candidates,
            max_movement=self._config.max_single_movement,
            max_retries=self._config.llm_max_retries,
            history_log=history_log,
        )

        # BO로 후보 점수 부여
        self._last_scored = self._optimizer.score(
            positions,
            self._rod_position,
            position_min=self._config.rod_position_min,
            position_max=self._config.rod_position_max,
        )

        # 최고 점수 후보 선택 (동점 시 현재 위치에서 가장 큰 변화 우선)
        self._last_best = max(
            self._last_scored,
            key=lambda c: (c.safety_score, -abs(c.rod_movement) if c.rod_movement == 0 else abs(c.rod_movement)),
        )

        logger.info(
            "후보 선택: movement=%d, position=%d, safety=%.3f",
            self._last_best.rod_movement,
            self._last_best.rod_position_after,
            self._last_best.safety_score,
        )

        return [self._last_best.to_action()]

    def apply_actions_to_case(
        self,
        actions: list[Action],
        state: ReactorState,
    ) -> CaseConfig:
        """액션을 적용하여 새 CaseConfig를 생성한다.

        제어봉 위치를 geometry.extra_params에 반영한다.

        Args:
            actions: 적용할 액션 목록.
            state: 현재 원자로 상태.

        Returns:
            수정된 CaseConfig.
        """
        data = state.current_config.model_dump(mode="json")

        for action in actions:
            if action.action_type == ActionType.MODIFY_PARAM and action.field_path:
                set_nested(data, action.field_path, action.value)

        return CaseConfig.model_validate(data)

    def evaluate_results(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """시뮬레이션 결과를 평가하고 로그를 기록한다.

        GP 모델을 재피팅하고, 로그 엔트리를 저장한다.

        Args:
            result: 시뮬레이션 결과.

        Returns:
            메트릭 딕셔너리.
        """
        keff = result.keff
        keff_std = result.keff_std
        deviation = abs(keff - self._config.target_keff)

        # 제어봉 위치 업데이트
        if self._last_best is not None:
            self._rod_position = self._last_best.rod_position_after

        # 안전 점수 (실측 기반 재계산)
        safety_score = self._last_best.safety_score if self._last_best else 0.5

        # 로그 저장
        candidate_logs = [
            CandidateLog(
                rod_movement=c.rod_movement,
                safety_score=c.safety_score,
            )
            for c in self._last_scored
        ]
        entry = LogEntry(
            step=self._step,
            rod_movement=self._last_best.rod_movement if self._last_best else 0,
            rod_position_after=self._rod_position,
            keff=keff,
            keff_std=keff_std,
            safety_score=safety_score,
            candidates=candidate_logs,
        )
        self._log_store.append(entry)

        # GP 재피팅
        observations = self._log_store.get_observations()
        self._optimizer.refit(observations)

        self._step += 1

        converged = 1.0 if deviation <= self._config.keff_tolerance else 0.0

        return {
            "keff": keff,
            "keff_std": keff_std,
            "keff_deviation": deviation,
            "safety_score": safety_score,
            "rod_position": float(self._rod_position),
            "converged": converged,
        }

    def update_state(
        self,
        state: ReactorState,
        metrics: dict[str, float],
    ) -> ReactorState:
        """메트릭으로 상태를 업데이트한다.

        Args:
            state: 현재 상태.
            metrics: 평가 메트릭.

        Returns:
            업데이트된 ReactorState.
        """
        return state.model_copy(
            update={
                "kpi": metrics,
                "iteration": state.iteration + 1,
            },
        )


__all__ = ["LLMController"]
