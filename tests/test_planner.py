"""LLMPlanner 테스트 (KAE-91).

절대 위치 기반 프롬프트 + 리트라이 로직 테스트.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.armi_layer.models import ReactorState, SimulationResult
from src.controller.llm.planner import LLMPlanner


@pytest.fixture()
def planner() -> LLMPlanner:
    """테스트용 LLMPlanner."""
    return LLMPlanner(base_url="http://localhost:8001/v1", model="qwen3.5")


class TestPlannerInit:
    """LLMPlanner 초기화 테스트."""

    def test_valid_localhost(self) -> None:
        planner = LLMPlanner(base_url="http://localhost:8001/v1", model="test")
        assert planner._base_url == "http://localhost:8001/v1"

    def test_rejects_external_host(self) -> None:
        with pytest.raises(ValueError, match="허용되지 않는 LLM 호스트"):
            LLMPlanner(base_url="http://evil.com/v1", model="test")

    def test_rejects_metadata_endpoint(self) -> None:
        with pytest.raises(ValueError, match="허용되지 않는 LLM 호스트"):
            LLMPlanner(base_url="http://169.254.169.254/latest", model="test")

    def test_rejects_ftp_scheme(self) -> None:
        with pytest.raises(ValueError, match="허용되지 않는 URL 스킴"):
            LLMPlanner(base_url="ftp://localhost:8001/v1", model="test")


class TestParseResponse:
    """_parse_response 테스트 — 절대 위치 기반."""

    def test_json_array(self, planner: LLMPlanner) -> None:
        result = planner._parse_response("[100, 110, 120, 130, 140]", n=5)
        assert result == [100, 110, 120, 130, 140]

    def test_json_array_with_text(self, planner: LLMPlanner) -> None:
        response = "제안합니다:\n[100, 120, 140, 160, 180]\n이상입니다."
        result = planner._parse_response(response, n=5)
        assert result == [100, 120, 140, 160, 180]

    def test_truncates_to_n(self, planner: LLMPlanner) -> None:
        result = planner._parse_response("[100, 110, 120, 130, 140, 150, 160]", n=3)
        assert len(result) == 3

    def test_regex_fallback(self, planner: LLMPlanner) -> None:
        response = "목표 위치: 100, 120, 140, 160, 180 steps"
        result = planner._parse_response(response, n=5)
        assert result == [100, 120, 140, 160, 180]

    def test_no_valid_positions_raises(self, planner: LLMPlanner) -> None:
        with pytest.raises(ValueError, match="유효한 위치를 추출할 수 없음"):
            planner._parse_response("위치가 없습니다.", n=5)

    def test_float_values_preserved(self, planner: LLMPlanner) -> None:
        result = planner._parse_response("[100.0, 150.5, 200.3]", n=3)
        assert result == [100.0, 150.5, 200.3]
        assert all(isinstance(v, float) for v in result)

    def test_thinking_tag_stripped(self, planner: LLMPlanner) -> None:
        response = (
            "<think>keff가 1.3이니까 삽입해야...</think>\n"
            "[100, 120, 140, 160, 180]"
        )
        result = planner._parse_response(response, n=5)
        assert result == [100, 120, 140, 160, 180]

    def test_thinking_only_raises(self, planner: LLMPlanner) -> None:
        response = "<think>생각 중...</think>"
        with pytest.raises(ValueError, match="수치를 추출할 수 없음"):
            planner._parse_response(response, n=5)

    def test_thinking_with_numbers_ignored(self, planner: LLMPlanner) -> None:
        response = (
            "<think>step 1에서 keff가 1.005라면 위치 100</think>\n"
            "[110, 120, 130, 140, 150]"
        )
        result = planner._parse_response(response, n=5)
        assert result == [110, 120, 130, 140, 150]

    def test_filters_out_of_range_in_regex(self, planner: LLMPlanner) -> None:
        """정규식 폴백 시 0~228 범위 밖 값은 필터링."""
        response = "위치 후보: 500, 100, 150, 300"
        result = planner._parse_response(response, n=5)
        assert result == [100, 150]


class TestFallbackCandidates:
    """_fallback_candidates 테스트."""

    def test_returns_n_candidates(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(
            current_position=150, n=10, max_movement=50,
        )
        assert len(candidates) == 10

    def test_includes_current_position(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(
            current_position=150, n=10, max_movement=50,
        )
        assert 150 in candidates

    def test_all_floats(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(
            current_position=150, n=10, max_movement=50,
        )
        assert all(isinstance(v, float | int) for v in candidates)

    def test_within_range(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(
            current_position=10, n=10, max_movement=50,
        )
        assert all(0 <= v <= 60 for v in candidates)

    def test_clamped_at_boundary(self, planner: LLMPlanner) -> None:
        candidates = planner._fallback_candidates(
            current_position=220, n=5, max_movement=50,
        )
        assert all(0 <= v <= 228 for v in candidates)


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

    def test_includes_absolute_position_instruction(self, planner: LLMPlanner) -> None:
        state = ReactorState()
        prompt = planner._build_user_prompt(state, rod_position=200, n=10)
        assert "목표 위치" in prompt

    def test_includes_history(self, planner: LLMPlanner) -> None:
        state = ReactorState()
        history_log = [
            {"step": 0, "rod_position_after": 200, "keff": 1.05, "keff_std": 0.001},
            {"step": 1, "rod_position_after": 180, "keff": 1.02, "keff_std": 0.001},
        ]
        prompt = planner._build_user_prompt(
            state, rod_position=180, n=10, history_log=history_log,
        )
        assert "이력" in prompt
        assert "위치=200" in prompt


class TestGenerate:
    """generate 통합 테스트."""

    @patch.object(LLMPlanner, "_call_llm")
    def test_success(self, mock_call: MagicMock, planner: LLMPlanner) -> None:
        mock_call.return_value = "[180, 185, 190, 195, 200, 205, 210, 215, 220, 225]"
        state = ReactorState()

        result = planner.generate(state, current_rod_position=200, n=10)

        assert len(result) == 10
        # 모든 후보는 max_movement(50) 범위 내: 150~228
        assert all(150 <= p <= 228 for p in result)

    @patch.object(LLMPlanner, "_call_llm")
    def test_retry_on_parse_failure(
        self, mock_call: MagicMock, planner: LLMPlanner
    ) -> None:
        """파싱 실패 시 리트라이한다."""
        mock_call.side_effect = [
            "<think>생각만...</think>",  # 1차: 파싱 실패
            "[150, 160, 170, 180, 190]",  # 2차: 성공
        ]
        state = ReactorState()

        result = planner.generate(
            state, current_rod_position=200, n=5, max_retries=3,
        )

        assert len(result) == 5
        assert mock_call.call_count == 2

    @patch.object(LLMPlanner, "_call_llm")
    def test_all_retries_fail_uses_fallback(
        self, mock_call: MagicMock, planner: LLMPlanner
    ) -> None:
        """모든 리트라이 실패 시 폴백 사용."""
        mock_call.side_effect = ConnectionError("서버 연결 실패")
        state = ReactorState()

        result = planner.generate(
            state, current_rod_position=200, n=10, max_retries=2,
        )

        assert len(result) == 10
        assert 200 in result  # 폴백은 현재 위치 포함
        assert mock_call.call_count == 3  # 1 + 2 retries

    @patch.object(LLMPlanner, "_call_llm")
    def test_clamped_to_max_movement(
        self, mock_call: MagicMock, planner: LLMPlanner
    ) -> None:
        """max_movement 범위로 클램핑."""
        # 일부는 범위 안, 일부는 밖 → 범위 안 값 존재하므로 통과, 밖 값은 클램핑
        mock_call.return_value = "[130, 150, 170, 228]"
        state = ReactorState()

        result = planner.generate(
            state, current_rod_position=150, n=4, max_movement=30,
        )

        # 150 ± 30 = [120, 180] 범위로 클램핑
        assert all(120 <= p <= 180 for p in result)

    @patch.object(LLMPlanner, "_call_llm")
    def test_out_of_range_positions_trigger_retry(
        self, mock_call: MagicMock, planner: LLMPlanner
    ) -> None:
        """추출된 위치가 전부 이동 범위 밖이면 리트라이한다."""
        mock_call.side_effect = [
            "[100, 110, 120, 130, 140, 150, 160, 170, 180, 190]",  # 전부 범위 밖
            "[15, 16, 17, 18, 19, 20]",  # 범위 안
        ]
        state = ReactorState()

        result = planner.generate(
            state, current_rod_position=17, n=6, max_movement=7,
        )

        assert mock_call.call_count == 2
        assert all(10 <= p <= 24 for p in result)

    @patch.object(LLMPlanner, "_call_llm")
    def test_all_out_of_range_uses_fallback(
        self, mock_call: MagicMock, planner: LLMPlanner
    ) -> None:
        """범위 밖 위치가 모든 리트라이에서 반복되면 폴백 사용."""
        mock_call.return_value = "[100, 110, 120, 130, 140]"
        state = ReactorState()

        result = planner.generate(
            state, current_rod_position=17, n=5, max_movement=7, max_retries=2,
        )

        # 폴백: 현재 위치(17) 포함, 10~24 범위
        assert 17 in result
        assert all(10 <= p <= 24 for p in result)
        assert mock_call.call_count == 3  # 1 + 2 retries
