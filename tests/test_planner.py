"""LLMPlanner 테스트 (KAE-89)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.armi_layer.models import ReactorState, SimulationResult
from src.controller.llm.planner import LLMPlanner


@pytest.fixture()
def planner() -> LLMPlanner:
    """테스트용 LLMPlanner."""
    return LLMPlanner(base_url="http://localhost:8001/v1", model="qwen3.5")


class TestParseResponse:
    """_parse_response 테스트."""

    def test_json_array(self, planner: LLMPlanner) -> None:
        result = planner._parse_response("[-5, -3, 0, 3, 5]", n=5)
        assert result == [-5, -3, 0, 3, 5]

    def test_json_array_with_text(self, planner: LLMPlanner) -> None:
        response = "제안합니다:\n[-5, -3, 0, 3, 5]\n이상입니다."
        result = planner._parse_response(response, n=5)
        assert result == [-5, -3, 0, 3, 5]

    def test_truncates_to_n(self, planner: LLMPlanner) -> None:
        result = planner._parse_response("[1, 2, 3, 4, 5, 6, 7]", n=3)
        assert len(result) == 3

    def test_regex_fallback(self, planner: LLMPlanner) -> None:
        response = "이동값: -5, -3, 0, 3, 5 steps"
        result = planner._parse_response(response, n=5)
        assert result == [-5, -3, 0, 3, 5]

    def test_no_integers_raises(self, planner: LLMPlanner) -> None:
        with pytest.raises(ValueError, match="정수를 추출할 수 없음"):
            planner._parse_response("이동값이 없습니다.", n=5)

    def test_float_values_converted_to_int(self, planner: LLMPlanner) -> None:
        result = planner._parse_response("[-5.0, 0.0, 5.0]", n=3)
        assert result == [-5, 0, 5]
        assert all(isinstance(v, int) for v in result)


class TestFallbackCandidates:
    """_fallback_candidates 테스트."""

    def test_returns_n_candidates(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(n=10)
        assert len(candidates) == 10

    def test_includes_zero(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(n=10)
        assert 0 in candidates

    def test_all_integers(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(n=10)
        assert all(isinstance(v, int) for v in candidates)


class TestBuildUserPrompt:
    """_build_user_prompt 테스트."""

    def test_includes_rod_position(self, planner: LLMPlanner) -> None:
        state = ReactorState()
        prompt = planner._build_user_prompt(state, rod_position=200, n=10)
        assert "200" in prompt

    def test_includes_keff_info(self, planner: LLMPlanner) -> None:
        state = ReactorState(kpi={"keff": 1.005, "keff_std": 0.001})
        prompt = planner._build_user_prompt(state, rod_position=200, n=10)
        assert "1.005" in prompt

    def test_includes_history(self, planner: LLMPlanner) -> None:
        state = ReactorState(
            history=[
                SimulationResult(keff=1.01, keff_std=0.001, runtime=10.0),
                SimulationResult(keff=1.005, keff_std=0.001, runtime=10.0),
            ]
        )
        prompt = planner._build_user_prompt(state, rod_position=200, n=10)
        assert "이력" in prompt


class TestGenerate:
    """generate 통합 테스트."""

    @patch.object(LLMPlanner, "_call_llm")
    def test_success(self, mock_call: MagicMock, planner: LLMPlanner) -> None:
        mock_call.return_value = "[-5, -3, -1, 0, 1, 3, 5, 7, 10, -10]"
        state = ReactorState()

        result = planner.generate(state, current_rod_position=200, n=10)

        assert len(result) == 10
        assert result[0] == -5

    @patch.object(LLMPlanner, "_call_llm")
    def test_llm_failure_uses_fallback(
        self, mock_call: MagicMock, planner: LLMPlanner
    ) -> None:
        mock_call.side_effect = ConnectionError("서버 연결 실패")
        state = ReactorState()

        result = planner.generate(state, current_rod_position=200, n=10)

        assert len(result) == 10
        assert 0 in result  # 폴백은 항상 0 포함
