"""LLM 컨트롤러 전용 데이터 모델.

제어봉 이동값, 안전 점수 후보, 로그 엔트리 등
LLM 컨트롤러 내부에서만 사용하는 Pydantic 모델을 정의한다.
기존 ReactorState, CaseConfig 등은 수정하지 않는다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.controller.base import Action, ActionType


class LLMControllerConfig(BaseModel):
    """LLMController 설정.

    Attributes:
        controller_type: 컨트롤러 유형 식별자.
        llm_base_url: LLM 서버 기본 URL.
        llm_model: 사용할 LLM 모델명.
        n_candidates: LLM이 생성할 후보 수.
        initial_rod_position: 초기 제어봉 위치 (steps).
        rod_position_min: 제어봉 최소 위치 (완전 삽입).
        rod_position_max: 제어봉 최대 위치 (완전 인출).
        target_keff: 목표 keff 값.
        keff_tolerance: keff 수렴 허용 오차.
        keff_critical_deviation: keff 이탈 판정 기준 (이 이상이면 종료).
        max_iterations: 최대 반복 횟수 (= depletion step 수).
        log_dir: 로그 저장 디렉토리.
    """

    model_config = ConfigDict(extra="forbid")

    controller_type: Literal["llm"] = "llm"
    llm_base_url: str = Field(
        default="http://localhost:8001/v1",
        description="LLM 서버 기본 URL",
    )
    llm_model: str = Field(
        default="qwen3.5",
        description="사용할 LLM 모델명",
    )
    n_candidates: int = Field(default=10, gt=0, description="LLM이 생성할 후보 수")
    initial_rod_position: int = Field(
        default=228,
        ge=0,
        description="초기 제어봉 위치 (steps, 228=완전 인출)",
    )
    rod_position_min: int = Field(
        default=0,
        ge=0,
        description="제어봉 최소 위치 (완전 삽입)",
    )
    rod_position_max: int = Field(
        default=228,
        gt=0,
        description="제어봉 최대 위치 (완전 인출)",
    )
    target_keff: float = Field(default=1.0, description="목표 keff 값")
    keff_tolerance: float = Field(
        default=0.01,
        gt=0,
        description="keff 수렴 허용 오차",
    )
    keff_critical_deviation: float = Field(
        default=0.05,
        gt=0,
        description="keff 이탈 판정 기준",
    )
    max_iterations: int = Field(
        default=12,
        gt=0,
        description="최대 반복 횟수 (= depletion step 수)",
    )
    max_single_movement: int = Field(
        default=50,
        gt=0,
        description="단일 스텝 최대 이동량 (안전 제한)",
    )
    log_dir: Path = Field(
        default=Path("runs/llm_logs"),
        description="로그 저장 디렉토리",
    )

    @model_validator(mode="after")
    def validate_rod_position_range(self) -> LLMControllerConfig:
        """제어봉 위치 범위와 초기 위치의 유효성을 검증한다."""
        if self.rod_position_min >= self.rod_position_max:
            raise ValueError(
                f"rod_position_min({self.rod_position_min})은 "
                f"rod_position_max({self.rod_position_max})보다 작아야 합니다"
            )
        in_range = (
            self.rod_position_min
            <= self.initial_rod_position
            <= self.rod_position_max
        )
        if not in_range:
            raise ValueError(
                f"initial_rod_position({self.initial_rod_position})은 "
                f"[{self.rod_position_min}, {self.rod_position_max}] 범위여야 합니다"
            )
        return self


class ScoredCandidate(BaseModel):
    """안전 점수가 부여된 제어봉 이동 후보.

    Attributes:
        rod_movement: 제어봉 이동 스텝 수 (양수=삽입, 음수=인출).
        rod_position_after: 이동 후 제어봉 위치.
        predicted_keff: GP 모델이 예측한 keff.
        safety_score: 안전 점수 (0~1, 1이 가장 안전).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rod_movement: int = Field(description="제어봉 이동 스텝 수")
    rod_position_after: int = Field(description="이동 후 제어봉 위치")
    predicted_keff: float = Field(description="GP 예측 keff")
    safety_score: float = Field(ge=0.0, le=1.0, description="안전 점수")

    def to_action(self) -> Action:
        """Action 객체로 변환한다."""
        return Action(
            action_type=ActionType.MODIFY_PARAM,
            field_path="geometry.extra_params.rod_position",
            value=float(self.rod_position_after),
            reason=(
                f"LLM+BO 선택: 이동={self.rod_movement}, "
                f"위치={self.rod_position_after}, "
                f"예측keff={self.predicted_keff:.5f}, "
                f"안전점수={self.safety_score:.3f}"
            ),
        )


class CandidateLog(BaseModel):
    """로그 내 후보 기록.

    Attributes:
        rod_movement: 제어봉 이동 스텝 수.
        safety_score: 안전 점수.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rod_movement: int = Field(description="제어봉 이동 스텝 수")
    safety_score: float = Field(description="안전 점수")


class LogEntry(BaseModel):
    """시뮬레이션 스텝 로그 엔트리.

    Attributes:
        step: 스텝 번호 (0-based).
        timestamp: 기록 시각 (UTC).
        rod_movement: 선택된 제어봉 이동값.
        rod_position_after: 이동 후 제어봉 위치.
        keff: 실측 keff.
        keff_std: keff 표준편차.
        safety_score: 선택된 후보의 안전 점수.
        candidates: 해당 스텝의 전체 후보 목록.
    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0, description="스텝 번호")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="기록 시각 (UTC)",
    )
    rod_movement: int = Field(description="선택된 제어봉 이동값")
    rod_position_after: int = Field(description="이동 후 제어봉 위치")
    keff: float = Field(description="실측 keff")
    keff_std: float = Field(ge=0, description="keff 표준편차")
    safety_score: float = Field(ge=0.0, le=1.0, description="안전 점수")
    candidates: list[CandidateLog] = Field(
        default_factory=list,
        description="전체 후보 목록",
    )
