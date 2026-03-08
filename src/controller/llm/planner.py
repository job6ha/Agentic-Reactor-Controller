"""LLM 플래너 모듈.

Qwen 로컬 서버(mlx-lm, OpenAI 호환 REST API)를 호출하여
제어봉 **목표 위치** 후보를 생성한다.
LLM 응답 파싱 실패 시 최대 N회 리트라이 후 폴백 후보를 생성한다.
"""

from __future__ import annotations

import json
import logging
import random
import re
import urllib.parse

import httpx

from src.armi_layer.models import ReactorState

logger = logging.getLogger(__name__)

# LLM 호출 타임아웃 (초) — thinking 모드 LLM은 응답이 오래 걸림
LLM_TIMEOUT = 300.0

# SSRF 방지: 허용된 LLM 서버 호스트
ALLOWED_LLM_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})
ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

SYSTEM_PROMPT = """\
당신은 원자로 제어봉 조작 전문가입니다.
현재 원자로 상태를 분석하고, 제어봉 **목표 위치** 후보를 제안해야 합니다.

규칙:
- 제어봉 위치 범위: 0 (완전 삽입, 반응도 최소) ~ 228 (완전 인출, 반응도 최대)
- 위치가 낮을수록 제어봉이 깊이 삽입되어 중성자 흡수 증가 → keff 감소
- 위치가 높을수록 제어봉이 인출되어 중성자 흡수 감소 → keff 증가
- 목표: keff를 1.0에 가깝게 유지
- 안전 우선: 급격한 이동보다 점진적 조정 선호
- 실수값 사용 가능 (소수점 1자리, 단위: steps, 0.0~228.0 범위)
- 현재 위치에서 크게 벗어나지 않는 값을 제안하세요

응답 형식: 반드시 JSON 배열만 출력하세요. 다른 텍스트 없이 배열만 반환합니다.
"""


class LLMPlanner:
    """LLM 기반 제어봉 목표 위치 후보 생성기.

    OpenAI 호환 REST API를 통해 로컬 LLM을 호출한다.

    Args:
        base_url: LLM 서버 기본 URL.
        model: 사용할 LLM 모델명.
    """

    def __init__(self, base_url: str, model: str) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            raise ValueError(f"허용되지 않는 URL 스킴: {parsed.scheme}")
        if parsed.hostname not in ALLOWED_LLM_HOSTS:
            raise ValueError(
                f"허용되지 않는 LLM 호스트: {parsed.hostname}. "
                f"허용 목록: {ALLOWED_LLM_HOSTS}"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate(
        self,
        state: ReactorState,
        current_rod_position: float,
        n: int = 10,
        max_movement: float = 50.0,
        max_retries: int = 3,
        history_log: list[dict] | None = None,
    ) -> list[float]:
        """LLM을 호출하여 제어봉 목표 위치 후보를 생성한다.

        파싱 실패 시 최대 ``max_retries``회 재호출하고,
        모두 실패하면 폴백 후보를 반환한다.
        모든 후보는 ``max_movement`` 범위와 0~228 범위로 클램핑된다.

        Args:
            state: 현재 원자로 상태.
            current_rod_position: 현재 제어봉 위치.
            n: 생성할 후보 수.
            max_movement: 단일 스텝 최대 이동량.
            max_retries: LLM 파싱 실패 시 최대 재시도 횟수.
            history_log: 과거 스텝 로그 (LogEntry.model_dump() 목록).

        Returns:
            목표 위치 실수 리스트 (0.0~228.0).
        """
        user_prompt = self._build_user_prompt(
            state, current_rod_position, n, history_log=history_log,
        )

        positions: list[int] | None = None
        last_error: Exception | None = None

        pos_min = max(0.0, current_rod_position - max_movement)
        pos_max = min(228.0, current_rod_position + max_movement)

        for attempt in range(1 + max_retries):
            try:
                response_text = self._call_llm(user_prompt)
                raw_positions = self._parse_response(response_text, n)

                # 검증: 추출 값 중 이동 범위 안에 있는 것이 하나라도 있는지
                in_range = [p for p in raw_positions if pos_min <= p <= pos_max]
                if not in_range:
                    raise ValueError(
                        f"추출 위치 전부 이동 범위 밖: {raw_positions} "
                        f"(허용: {pos_min}~{pos_max})"
                    )
                positions = raw_positions
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "LLM 호출/파싱 실패 (시도 %d/%d): %s",
                        attempt + 1,
                        1 + max_retries,
                        e,
                    )
                else:
                    logger.warning(
                        "LLM 최대 재시도 초과 (%d회), 폴백 사용: %s",
                        1 + max_retries,
                        last_error,
                    )

        if positions is None:
            positions = self._fallback_candidates(current_rod_position, n, max_movement)

        # 안전 제한: max_movement 범위 및 0~228 클램핑
        positions = [max(pos_min, min(pos_max, p)) for p in positions]

        logger.info("목표 위치 후보: %s (현재: %d)", positions, current_rod_position)
        return positions

    def _build_user_prompt(
        self,
        state: ReactorState,
        rod_position: float,
        n: int,
        history_log: list[dict] | None = None,
    ) -> str:
        """LLM에 전달할 사용자 프롬프트를 구성한다."""
        keff_info = ""
        if state.kpi and "keff" in state.kpi:
            keff_info = f"- 현재 keff: {state.kpi['keff']:.5f}"
            if "keff_std" in state.kpi:
                keff_info += f" ± {state.kpi['keff_std']:.5f}"
            deviation = abs(state.kpi["keff"] - 1.0)
            keff_info += f" (목표 1.0과의 편차: {deviation:.5f})"

        # 상세 제어 이력
        history_info = ""
        if history_log:
            recent = history_log[-5:]  # 최근 5스텝
            lines = []
            for entry in recent:
                step = entry["step"]
                pos_after = entry["rod_position_after"]
                keff = entry["keff"]
                keff_std = entry["keff_std"]
                lines.append(
                    f"  step {step}: "
                    f"제어봉 위치={pos_after}, "
                    f"keff={keff:.5f}±{keff_std:.5f}"
                )
            history_info = "- 제어 이력 (위치→결과):\n" + "\n".join(lines)

        return (
            f"현재 원자로 상태:\n"
            f"- 제어봉 위치: {rod_position} steps "
            f"(범위: 0=완전삽입/keff최소, 228=완전인출/keff최대)\n"
            f"- 반복 횟수: {state.iteration}\n"
            f"{keff_info}\n"
            f"{history_info}\n\n"
            f"keff를 1.0에 가깝게 만들 제어봉 **목표 위치** 후보 {n}개를 "
            f"JSON 배열(0.0~228.0 실수)로 제안하세요."
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
            "max_tokens": 2048,
        }

        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content", "") or ""

        # mlx-lm 등 일부 서버는 thinking을 reasoning 필드로 분리하고
        # content를 비워두는 경우가 있음 → reasoning에서 답변 추출
        if not content.strip() and "reasoning" in message:
            reasoning = message["reasoning"] or ""
            # reasoning에서 모든 JSON 배열을 찾고 마지막 것을 사용
            # (LLM은 reasoning 끝에 최종 답변을 배치하는 경향이 있음)
            all_matches = re.findall(r"\[[\d\s,\.]+\]", reasoning)
            if all_matches:
                content = all_matches[-1]
                logger.info(
                    "content 비어있어 reasoning에서 추출 (마지막 배열, %d개 중): %s",
                    len(all_matches),
                    content[:80],
                )

        return content

    def _parse_response(self, response_text: str, n: int) -> list[float]:
        """LLM 응답에서 목표 위치 실수 리스트를 추출한다.

        thinking 태그를 제거한 후, JSON 배열 파싱을 시도하고,
        실패 시 정규식으로 수치를 추출한다.

        Args:
            response_text: LLM 응답 텍스트.
            n: 필요한 후보 수.

        Returns:
            목표 위치 실수 리스트.

        Raises:
            ValueError: 유효한 수치를 추출할 수 없을 때.
        """
        # thinking 태그 제거 (Qwen3.5 등 thinking 모드 LLM 대응)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()
        # markdown 코드 블록 제거 (```json ... ```)
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
        if not cleaned:
            raise ValueError(
                f"LLM 응답에서 수치를 추출할 수 없음: {response_text[:100]}"
            )

        # JSON 배열 파싱 시도
        json_match = re.search(r"\[[\s\S]*?\]", cleaned)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and all(
                    isinstance(v, int | float) for v in parsed
                ):
                    positions = [float(v) for v in parsed]
                    if positions:
                        return positions[:n]
            except json.JSONDecodeError:
                pass

        # 정규식으로 수치 추출 시도 (정수 및 소수점 포함)
        numbers = [float(m) for m in re.findall(r"\d+\.?\d*", cleaned)]
        # 0~228 범위인 값만 필터링
        valid = [v for v in numbers if 0.0 <= v <= 228.0]
        if valid:
            return valid[:n]

        raise ValueError(f"LLM 응답에서 유효한 위치를 추출할 수 없음: {cleaned[:100]}")

    def _fallback_candidates(
        self,
        current_position: float,
        n: int,
        max_movement: float,
    ) -> list[float]:
        """현재 위치 주변의 폴백 후보를 생성한다.

        현재 위치를 포함하고, max_movement 범위에서 균등 샘플링한다.

        Args:
            current_position: 현재 제어봉 위치.
            n: 생성할 후보 수.
            max_movement: 최대 이동량.

        Returns:
            목표 위치 실수 리스트.
        """
        pos_min = max(0.0, current_position - max_movement)
        pos_max = min(228.0, current_position + max_movement)

        candidates: list[float] = [current_position]
        for _ in range(n - 1):
            candidates.append(round(random.uniform(pos_min, pos_max), 1))
        return candidates[:n]
