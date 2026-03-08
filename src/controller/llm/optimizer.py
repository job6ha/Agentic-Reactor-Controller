"""베이지안 옵티마이제이션 모듈.

Gaussian Process 서로게이트 모델을 사용하여
제어봉 위치별 keff를 예측하고 안전 점수를 산출한다.
과거 실행 로그로 GP 모델을 재피팅하여 예측 정확도를 높인다.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from src.controller.llm.models import ScoredCandidate

logger = logging.getLogger(__name__)

# GP 데이터가 이 수 이상이어야 피팅 수행
MIN_OBSERVATIONS_FOR_FIT = 2

# 안전 점수 계산용 상수
SUPERCRITICAL_SIGMA = 0.005  # 초임계 방향 페널티 (좁음 → 엄격)
SUBCRITICAL_SIGMA = 0.02  # 아임계 방향 페널티 (넓음 → 관대)
UNCERTAINTY_WEIGHT = 2.0  # 불확실성 페널티 가중치
DEFAULT_SAFETY_SCORE = 0.5  # GP 미피팅 시 기본 안전 점수


class BayesianOptimizer:
    """GP 기반 베이지안 옵티마이저.

    제어봉 위치 → keff 관계를 GP로 모델링하고,
    각 후보의 예상 keff로부터 안전 점수를 산출한다.

    안전 점수 설계:
    - keff ≈ 1.0 일수록 높은 점수 (최대 1.0)
    - 초임계 방향(keff > 1.0)은 엄격한 페널티
    - 아임계 방향(keff < 1.0)은 상대적으로 관대
    - GP 불확실성이 클수록 점수 감소

    Args:
        target_keff: 목표 keff 값.
    """

    def __init__(self, target_keff: float = 1.0) -> None:
        self._target_keff = target_keff
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=50.0)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=3,
            normalize_y=True,
        )
        self._is_fitted = False
        self._X: np.ndarray = np.empty((0, 1))
        self._y: np.ndarray = np.empty(0)

    @property
    def is_fitted(self) -> bool:
        """GP 모델이 피팅되었는지 여부."""
        return self._is_fitted

    @property
    def n_observations(self) -> int:
        """학습 데이터 수."""
        return len(self._y)

    def score(
        self,
        positions: list[int],
        current_position: int,
        position_min: int = 0,
        position_max: int = 228,
    ) -> list[ScoredCandidate]:
        """목표 위치 후보에 안전 점수를 부여한다.

        Args:
            positions: 제어봉 목표 위치 후보 리스트.
            current_position: 현재 제어봉 위치.
            position_min: 제어봉 최소 위치.
            position_max: 제어봉 최대 위치.

        Returns:
            안전 점수가 부여된 ScoredCandidate 목록.
        """
        results: list[ScoredCandidate] = []

        for target_position in positions:
            position = max(position_min, min(position_max, target_position))
            movement = position - current_position

            if self._is_fitted:
                x_pred = np.array([[position]])
                pred_keff, pred_std = self._gp.predict(x_pred, return_std=True)
                keff = float(pred_keff[0])
                std = float(pred_std[0])
                safety = self._compute_safety(keff, std)
            else:
                keff = self._target_keff
                safety = DEFAULT_SAFETY_SCORE

            results.append(
                ScoredCandidate(
                    rod_movement=movement,
                    rod_position_after=position,
                    predicted_keff=keff,
                    safety_score=safety,
                )
            )

        return results

    def refit(self, observations: list[tuple[int, float]]) -> None:
        """과거 관측 데이터로 GP 모델을 재피팅한다.

        Args:
            observations: (rod_position, keff) 쌍 목록.
        """
        if len(observations) < MIN_OBSERVATIONS_FOR_FIT:
            logger.debug(
                "관측 데이터 부족 (%d < %d), 피팅 건너뜀",
                len(observations),
                MIN_OBSERVATIONS_FOR_FIT,
            )
            return

        self._X = np.array([[pos] for pos, _ in observations])
        self._y = np.array([keff for _, keff in observations])

        self._gp.fit(self._X, self._y)
        self._is_fitted = True

        logger.info("GP 모델 재피팅 완료: n_observations=%d", len(observations))

    def _compute_safety(
        self,
        predicted_keff: float,
        uncertainty: float,
    ) -> float:
        """예측 keff와 불확실성으로 안전 점수를 계산한다.

        비대칭 Gaussian 페널티:
        - 초임계(keff > target): sigma = 0.005 → 급격한 감소
        - 아임계(keff < target): sigma = 0.02 → 완만한 감소

        Args:
            predicted_keff: GP 예측 keff.
            uncertainty: GP 예측 표준편차.

        Returns:
            안전 점수 (0.0 ~ 1.0).
        """
        deviation = predicted_keff - self._target_keff

        if deviation > 0:
            sigma = SUPERCRITICAL_SIGMA
        else:
            sigma = SUBCRITICAL_SIGMA

        proximity_score = math.exp(-(deviation**2) / (2 * sigma**2))

        uncertainty_penalty = max(0.0, 1.0 - uncertainty * UNCERTAINTY_WEIGHT)
        raw_score = proximity_score * uncertainty_penalty

        return max(0.0, min(1.0, raw_score))
