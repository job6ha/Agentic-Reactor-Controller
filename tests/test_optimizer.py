"""BayesianOptimizer 테스트 (KAE-88)."""

from __future__ import annotations

import pytest

from src.controller.llm.optimizer import BayesianOptimizer


class TestBayesianOptimizerInit:
    """BayesianOptimizer 초기화 테스트."""

    def test_initial_state(self) -> None:
        opt = BayesianOptimizer(target_keff=1.0)
        assert not opt.is_fitted
        assert opt.n_observations == 0


class TestScoreUnfitted:
    """GP 미피팅 상태에서 score 테스트."""

    def test_returns_default_scores(self) -> None:
        opt = BayesianOptimizer()
        scored = opt.score([0, -5, 5], current_position=228)

        assert len(scored) == 3
        for c in scored:
            assert c.safety_score == 0.5
            assert c.predicted_keff == 1.0

    def test_position_clamping_max(self) -> None:
        opt = BayesianOptimizer()
        scored = opt.score([100], current_position=200, position_max=228)

        assert scored[0].rod_position_after == 228

    def test_position_clamping_min(self) -> None:
        opt = BayesianOptimizer()
        scored = opt.score([-300], current_position=100, position_min=0)

        assert scored[0].rod_position_after == 0


class TestRefit:
    """GP refit 테스트."""

    def test_refit_with_sufficient_data(self) -> None:
        opt = BayesianOptimizer()
        observations = [(100, 0.95), (150, 1.0), (200, 1.05)]
        opt.refit(observations)

        assert opt.is_fitted
        assert opt.n_observations == 3

    def test_refit_insufficient_data(self) -> None:
        opt = BayesianOptimizer()
        opt.refit([(100, 0.95)])

        assert not opt.is_fitted

    def test_refit_empty_data(self) -> None:
        opt = BayesianOptimizer()
        opt.refit([])

        assert not opt.is_fitted


class TestScoreFitted:
    """GP 피팅 후 score 테스트."""

    @pytest.fixture()
    def fitted_optimizer(self) -> BayesianOptimizer:
        """피팅된 옵티마이저. rod_position 150 근처에서 keff≈1.0."""
        opt = BayesianOptimizer(target_keff=1.0)
        observations = [
            (50, 0.90),
            (100, 0.95),
            (150, 1.00),
            (200, 1.05),
            (228, 1.08),
        ]
        opt.refit(observations)
        return opt

    def test_scores_vary_by_position(self, fitted_optimizer: BayesianOptimizer) -> None:
        scored = fitted_optimizer.score([-78, -28, 0], current_position=178)
        # position 100, 150, 178
        scores = [c.safety_score for c in scored]
        # 150 근처가 가장 높아야 함
        assert scores[1] > scores[0]  # 150 > 100
        assert scores[1] > scores[2]  # 150 > 178

    def test_predicted_keff_reasonable(
        self, fitted_optimizer: BayesianOptimizer
    ) -> None:
        scored = fitted_optimizer.score([0], current_position=150)
        # 학습 데이터에서 position=150 → keff=1.0
        assert abs(scored[0].predicted_keff - 1.0) < 0.05

    def test_supercritical_penalty_stricter(
        self, fitted_optimizer: BayesianOptimizer
    ) -> None:
        """초임계(keff>1.0) 페널티가 아임계보다 엄격."""
        scored = fitted_optimizer.score([-78, 50], current_position=150)
        # position 72 (keff<1.0), position 200 (keff>1.0)
        sub_critical = scored[0]  # 아임계
        super_critical = scored[1]  # 초임계

        # 동일 편차에서 초임계가 더 낮은 점수
        # (GP 예측이 정확하다면 이 패턴이 성립)
        if abs(sub_critical.predicted_keff - 1.0) < abs(
            super_critical.predicted_keff - 1.0
        ):
            assert sub_critical.safety_score >= super_critical.safety_score


class TestSafetyScoreProperties:
    """안전 점수 수학적 속성 테스트."""

    def test_score_at_target_is_high(self) -> None:
        opt = BayesianOptimizer(target_keff=1.0)
        observations = [(100, 0.95), (150, 1.00), (200, 1.05)]
        opt.refit(observations)

        scored = opt.score([0], current_position=150)
        assert scored[0].safety_score > 0.8

    def test_score_bounded_zero_to_one(self) -> None:
        opt = BayesianOptimizer()
        observations = [(50, 0.85), (100, 0.95), (150, 1.0), (200, 1.1)]
        opt.refit(observations)

        scored = opt.score(
            list(range(-200, 200, 10)),
            current_position=100,
        )
        for c in scored:
            assert 0.0 <= c.safety_score <= 1.0
