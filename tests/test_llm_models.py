"""LLM 컨트롤러 데이터 모델 테스트 (KAE-86)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.controller.base import Action, ActionType
from src.controller.llm.models import (
    CandidateLog,
    LLMControllerConfig,
    LogEntry,
    ScoredCandidate,
)


class TestLLMControllerConfig:
    """LLMControllerConfig 설정 모델 테스트."""

    def test_defaults(self) -> None:
        config = LLMControllerConfig()
        assert config.controller_type == "llm"
        assert config.llm_base_url == "http://localhost:8001/v1"
        assert config.n_candidates == 10
        assert config.initial_rod_position == 228
        assert config.rod_position_min == 0
        assert config.rod_position_max == 228
        assert config.target_keff == 1.0
        assert config.max_iterations == 12

    def test_custom_values(self) -> None:
        config = LLMControllerConfig(
            llm_model="qwen2.5",
            n_candidates=5,
            initial_rod_position=100,
            max_iterations=24,
        )
        assert config.llm_model == "qwen2.5"
        assert config.n_candidates == 5
        assert config.initial_rod_position == 100
        assert config.max_iterations == 24

    def test_n_candidates_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LLMControllerConfig(n_candidates=0)

    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LLMControllerConfig(max_iterations=0)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            LLMControllerConfig(unknown_field="value")

    def test_controller_type_literal(self) -> None:
        with pytest.raises(ValidationError):
            LLMControllerConfig(controller_type="simple")


class TestScoredCandidate:
    """ScoredCandidate 모델 테스트."""

    def test_creation(self) -> None:
        candidate = ScoredCandidate(
            rod_movement=-5,
            rod_position_after=223,
            predicted_keff=1.00032,
            safety_score=0.91,
        )
        assert candidate.rod_movement == -5
        assert candidate.rod_position_after == 223
        assert candidate.predicted_keff == 1.00032
        assert candidate.safety_score == 0.91

    def test_frozen(self) -> None:
        candidate = ScoredCandidate(
            rod_movement=-5,
            rod_position_after=223,
            predicted_keff=1.0,
            safety_score=0.9,
        )
        with pytest.raises(ValidationError):
            candidate.rod_movement = 10

    def test_safety_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ScoredCandidate(
                rod_movement=0,
                rod_position_after=228,
                predicted_keff=1.0,
                safety_score=1.5,
            )

    def test_to_action(self) -> None:
        candidate = ScoredCandidate(
            rod_movement=-5,
            rod_position_after=223,
            predicted_keff=1.00032,
            safety_score=0.91,
        )
        action = candidate.to_action()
        assert isinstance(action, Action)
        assert action.action_type == ActionType.MODIFY_PARAM
        assert action.field_path == "geometry.extra_params.rod_position"
        assert action.value == 223.0
        assert "이동=-5" in action.reason
        assert "안전점수=0.910" in action.reason


class TestCandidateLog:
    """CandidateLog 모델 테스트."""

    def test_creation(self) -> None:
        log = CandidateLog(rod_movement=-5, safety_score=0.91)
        assert log.rod_movement == -5
        assert log.safety_score == 0.91

    def test_frozen(self) -> None:
        log = CandidateLog(rod_movement=0, safety_score=0.5)
        with pytest.raises(ValidationError):
            log.rod_movement = 1


class TestLogEntry:
    """LogEntry 모델 테스트."""

    def test_creation(self) -> None:
        entry = LogEntry(
            step=0,
            rod_movement=-5,
            rod_position_after=223,
            keff=1.00032,
            keff_std=0.00012,
            safety_score=0.91,
        )
        assert entry.step == 0
        assert entry.rod_movement == -5
        assert entry.timestamp is not None

    def test_with_candidates(self) -> None:
        entry = LogEntry(
            step=1,
            rod_movement=-3,
            rod_position_after=220,
            keff=0.999,
            keff_std=0.001,
            safety_score=0.85,
            candidates=[
                CandidateLog(rod_movement=-3, safety_score=0.85),
                CandidateLog(rod_movement=0, safety_score=0.80),
            ],
        )
        assert len(entry.candidates) == 2

    def test_step_must_be_nonnegative(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry(
                step=-1,
                rod_movement=0,
                rod_position_after=228,
                keff=1.0,
                keff_std=0.0,
                safety_score=0.5,
            )

    def test_serialization_roundtrip(self) -> None:
        entry = LogEntry(
            step=0,
            rod_movement=-5,
            rod_position_after=223,
            keff=1.00032,
            keff_std=0.00012,
            safety_score=0.91,
        )
        data = entry.model_dump(mode="json")
        restored = LogEntry.model_validate(data)
        assert restored.step == entry.step
        assert restored.rod_movement == entry.rod_movement
        assert restored.keff == entry.keff
