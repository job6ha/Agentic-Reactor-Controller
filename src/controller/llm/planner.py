"""LLM 플래너 모듈.

Qwen 로컬 서버(mlx-lm, OpenAI 호환 REST API)를 호출하여
제어봉 이동값 후보를 생성한다.
LLM 응답 파싱 실패 시 랜덤 폴백 후보를 생성한다.
"""

from __future__ import annotations

import json
import logging
import random
import re

import httpx

from src.armi_layer.models import ReactorState

logger = logging.getLogger(__name__)

# LLM 호출 타임아웃 (초)
LLM_TIMEOUT = 30.0

# 폴백 후보 생성 범위
FALLBACK_MOVEMENT_RANGE = (-20, 20)

SYSTEM_PROMPT = """\
당신은 원자로 제어봉 조작 전문가입니다.
현재 원자로 상태를 분석하고, 제어봉 이동값 후보를 제안해야 합니다.

규칙:
- 양수 = 제어봉 삽입 (반응도 감소), 음수 = 제어봉 인출 (반응도 증가)
- 목표: keff를 1.0에 가깝게 유지
- 안전 우선: 급격한 이동보다 점진적 조정 선호
- 정수값만 사용 (단위: steps)

응답 형식: 반드시 JSON 배열로 정수만 반환하세요.
예: [-5, -3, -1, 0, 1, 3, 5, 7, 10, -10]
"""


class LLMPlanner:
    """LLM 기반 제어봉 이동값 후보 생성기.

    OpenAI 호환 REST API를 통해 로컬 LLM을 호출한다.

    Args:
        base_url: LLM 서버 기본 URL.
        model: 사용할 LLM 모델명.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate(
        self,
        state: ReactorState,
        current_rod_position: int,
        n: int = 10,
    ) -> list[int]:
        """LLM을 호출하여 제어봉 이동값 후보를 생성한다.

        LLM 호출 실패 시 랜덤 폴백 후보를 반환한다.

        Args:
            state: 현재 원자로 상태.
            current_rod_position: 현재 제어봉 위치.
            n: 생성할 후보 수.

        Returns:
            정수 이동값 리스트.
        """
        user_prompt = self._build_user_prompt(state, current_rod_position, n)

        try:
            response_text = self._call_llm(user_prompt)
            movements = self._parse_response(response_text, n)
            logger.info("LLM 후보 생성 성공: %s", movements)
            return movements
        except Exception:
            logger.warning("LLM 호출/파싱 실패, 폴백 후보 사용", exc_info=True)
            return self._fallback_candidates(n)

    def _build_user_prompt(
        self,
        state: ReactorState,
        rod_position: int,
        n: int,
    ) -> str:
        """LLM에 전달할 사용자 프롬프트를 구성한다."""
        keff_info = ""
        if state.kpi and "keff" in state.kpi:
            keff_info = f"- 현재 keff: {state.kpi['keff']:.5f}"
            if "keff_std" in state.kpi:
                keff_info += f" ± {state.kpi['keff_std']:.5f}"

        history_info = ""
        if state.history:
            recent = state.history[-3:]
            history_lines = [
                f"  step {i}: keff={r.keff:.5f}" for i, r in enumerate(recent)
            ]
            history_info = "- 최근 이력:\n" + "\n".join(history_lines)

        return (
            f"현재 원자로 상태:\n"
            f"- 제어봉 위치: {rod_position} steps\n"
            f"- 반복 횟수: {state.iteration}\n"
            f"{keff_info}\n"
            f"{history_info}\n\n"
            f"제어봉 이동값 후보 {n}개를 JSON 배열로 제안하세요."
        )

    def _call_llm(self, user_prompt: str) -> str:
        """OpenAI 호환 API로 LLM을 호출한다.

        Args:
            user_prompt: 사용자 프롬프트.

        Returns:
            LLM 응답 텍스트.

        Raises:
            httpx.HTTPStatusError: HTTP 오류 시.
            httpx.ConnectError: 연결 실패 시.
        """
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 256,
        }

        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_response(self, response_text: str, n: int) -> list[int]:
        """LLM 응답에서 정수 리스트를 추출한다.

        JSON 배열 파싱을 시도하고, 실패 시 정규식으로 정수를 추출한다.

        Args:
            response_text: LLM 응답 텍스트.
            n: 필요한 후보 수.

        Returns:
            정수 이동값 리스트.

        Raises:
            ValueError: 유효한 정수를 추출할 수 없을 때.
        """
        # JSON 배열 파싱 시도
        json_match = re.search(r"\[[\s\S]*?\]", response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and all(
                    isinstance(v, int | float) for v in parsed
                ):
                    movements = [int(v) for v in parsed]
                    if movements:
                        return movements[:n]
            except json.JSONDecodeError:
                pass

        # 정규식으로 정수 추출 시도
        integers = [int(m) for m in re.findall(r"-?\d+", response_text)]
        if integers:
            return integers[:n]

        raise ValueError(f"LLM 응답에서 정수를 추출할 수 없음: {response_text[:100]}")

    def _fallback_candidates(self, n: int) -> list[int]:
        """랜덤 폴백 후보를 생성한다.

        0을 포함하고, 대칭적인 범위에서 균등하게 샘플링한다.

        Args:
            n: 생성할 후보 수.

        Returns:
            정수 이동값 리스트.
        """
        low, high = FALLBACK_MOVEMENT_RANGE
        candidates = [0]  # 현 위치 유지 항상 포함
        candidates.extend(random.sample(range(low, high + 1), min(n - 1, high - low)))
        return candidates[:n]
