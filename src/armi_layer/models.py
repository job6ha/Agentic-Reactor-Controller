"""시스템 전체에서 사용하는 핵심 Pydantic 데이터 모델.

원자로 시뮬레이션의 케이스 설정, 실행 관리, 결과 수집에 필요한
데이터 구조를 정의한다. 모든 모듈 간 데이터 교환은 이 모델을 통해
타입 안전하게 수행된다.
"""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StatusType(str, Enum):
    """시뮬레이션 실행 상태."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class GeometryParams(BaseModel):
    """원자로 기하 구조 파라미터.

    PWR pin cell 기본값을 제공하며, extra_params로 추가 파라미터를
    유연하게 확장할 수 있다.

    Attributes:
        fuel_radius: 연료봉 반지름 (cm).
        clad_inner_radius: 피복관 내경 (cm).
        clad_outer_radius: 피복관 외경 (cm).
        pitch: 격자 피치 (cm).
        fuel_height: 연료 활성 높이 (cm).
        extra_params: 추가 기하 파라미터.
    """

    model_config = ConfigDict(extra="forbid")

    fuel_radius: float = Field(default=0.39218, gt=0, description="연료봉 반지름 (cm)")
    clad_inner_radius: float = Field(
        default=0.40005, gt=0, description="피복관 내경 (cm)"
    )
    clad_outer_radius: float = Field(
        default=0.45720, gt=0, description="피복관 외경 (cm)"
    )
    pitch: float = Field(default=1.25984, gt=0, description="격자 피치 (cm)")
    fuel_height: float = Field(default=200.0, gt=0, description="연료 활성 높이 (cm)")
    extra_params: dict[str, float] = Field(
        default_factory=dict, description="추가 기하 파라미터"
    )

    @model_validator(mode="after")
    def validate_geometry_ordering(self) -> "GeometryParams":
        """fuel_radius < clad_inner < clad_outer < pitch/2 순서를 검증한다."""
        if not (self.fuel_radius < self.clad_inner_radius < self.clad_outer_radius):
            raise ValueError(
                "fuel_radius < clad_inner_radius < clad_outer_radius 순서여야 합니다"
            )
        if self.clad_outer_radius >= self.pitch / 2:
            raise ValueError("clad_outer_radius는 pitch/2보다 작아야 합니다")
        return self


class MaterialParams(BaseModel):
    """원자로 재료 파라미터.

    PWR 표준 UO2 연료 + Zircaloy 피복관 + 경수 냉각재 기본값을 제공한다.

    Attributes:
        fuel_enrichment: U-235 농축도 (wt%).
        fuel_density: 연료 밀도 (g/cm3).
        coolant_temperature: 냉각재 온도 (K).
        coolant_density: 냉각재 밀도 (g/cm3).
        clad_density: 피복관 밀도 (g/cm3).
        extra_params: 추가 재료 파라미터.
    """

    model_config = ConfigDict(extra="forbid")

    fuel_enrichment: float = Field(
        default=3.0, ge=0.0, le=100.0, description="U-235 농축도 (wt%)"
    )
    fuel_density: float = Field(
        default=10.29769, gt=0, description="연료 밀도 (g/cm3)"
    )
    coolant_temperature: float = Field(
        default=600.0, gt=0, description="냉각재 온도 (K)"
    )
    coolant_density: float = Field(
        default=0.7, gt=0, description="냉각재 밀도 (g/cm3)"
    )
    clad_density: float = Field(
        default=6.55, gt=0, description="피복관 밀도 (g/cm3)"
    )
    extra_params: dict[str, float] = Field(
        default_factory=dict, description="추가 재료 파라미터"
    )


class SimulationSettings(BaseModel):
    """OpenMC 시뮬레이션 설정.

    Attributes:
        batches: 총 배치 수.
        inactive: 비활성 배치 수 (소스 수렴용).
        particles: 배치당 입자 수.
        source_type: 소스 유형 (point, box, fission 등).
        extra_settings: 추가 시뮬레이션 설정.
    """

    model_config = ConfigDict(extra="forbid")

    batches: int = Field(default=110, gt=0, description="총 배치 수")
    inactive: int = Field(default=10, ge=0, description="비활성 배치 수")
    particles: int = Field(default=1000, gt=0, description="배치당 입자 수")
    source_type: str = Field(default="point", description="소스 유형")
    extra_settings: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="추가 시뮬레이션 설정"
    )

    @model_validator(mode="after")
    def validate_batches(self) -> "SimulationSettings":
        """inactive < batches 조건을 검증한다."""
        if self.inactive >= self.batches:
            raise ValueError(
                f"inactive({self.inactive})는 batches({self.batches})보다 작아야 합니다"
            )
        return self


class CaseConfig(BaseModel):
    """시뮬레이션 케이스 설정.

    하나의 OpenMC 시뮬레이션 실행에 필요한 모든 입력 파라미터를
    포함한다. geometry, material, settings 세 영역으로 구성된다.

    Attributes:
        name: 케이스 이름.
        description: 케이스 설명.
        geometry: 기하 구조 파라미터.
        materials: 재료 파라미터.
        settings: 시뮬레이션 설정.
        tags: 분류/검색용 태그.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="케이스 이름")
    description: str = Field(default="", description="케이스 설명")
    geometry: GeometryParams = Field(
        default_factory=GeometryParams, description="기하 구조 파라미터"
    )
    materials: MaterialParams = Field(
        default_factory=MaterialParams, description="재료 파라미터"
    )
    settings: SimulationSettings = Field(
        default_factory=SimulationSettings, description="시뮬레이션 설정"
    )
    tags: list[str] = Field(default_factory=list, description="분류/검색용 태그")


class RunConfig(BaseModel):
    """OpenMC 실행 설정.

    프로세스 실행에 필요한 환경변수, 타임아웃, 재시도 등을 설정한다.

    Attributes:
        omp_threads: OpenMP 스레드 수. None이면 자동 감지.
        cross_sections_path: 핵데이터 경로. None이면 환경변수 사용.
        timeout: 실행 타임아웃 (초). None이면 무제한.
        max_retries: 최대 재시도 횟수.
        working_dir: 실행 작업 디렉토리.
    """

    model_config = ConfigDict(extra="forbid")

    omp_threads: int | None = Field(
        default=None, gt=0, description="OpenMP 스레드 수 (None=자동 감지)"
    )
    cross_sections_path: Path | None = Field(
        default=None, description="핵데이터 경로 (None=환경변수 사용)"
    )
    timeout: float | None = Field(
        default=None, gt=0, description="실행 타임아웃 초 (None=무제한)"
    )
    max_retries: int = Field(default=0, ge=0, description="최대 재시도 횟수")
    working_dir: Path | None = Field(
        default=None, description="실행 작업 디렉토리"
    )


class RunStatus(BaseModel):
    """시뮬레이션 실행 상태.

    케이스의 실행 상태와 타임스탬프를 추적한다.
    meta/status.json으로 직렬화되어 케이스 폴더에 저장된다.

    Attributes:
        status: 현재 실행 상태.
        created_at: 케이스 생성 시각.
        started_at: 실행 시작 시각.
        completed_at: 실행 완료 시각.
        error_message: 실패 시 에러 메시지.
        attempt: 현재 시도 횟수.
    """

    model_config = ConfigDict(extra="forbid")

    status: StatusType = Field(
        default=StatusType.QUEUED, description="현재 실행 상태"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="케이스 생성 시각 (UTC)"
    )
    started_at: datetime | None = Field(
        default=None, description="실행 시작 시각"
    )
    completed_at: datetime | None = Field(
        default=None, description="실행 완료 시각"
    )
    error_message: str | None = Field(
        default=None, description="실패 시 에러 메시지"
    )
    attempt: int = Field(default=0, ge=0, description="현재 시도 횟수")


class TallyResult(BaseModel):
    """단일 탈리 결과.

    OpenMC tally 데이터의 요약 정보를 담는다.

    Attributes:
        name: 탈리 이름.
        scores: 스코어 유형 목록 (예: flux, fission).
        mean: 평균값 리스트.
        std_dev: 표준편차 리스트.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="탈리 이름")
    scores: list[str] = Field(description="스코어 유형 목록")
    mean: list[float] = Field(description="평균값 리스트")
    std_dev: list[float] = Field(description="표준편차 리스트")

    @model_validator(mode="after")
    def validate_lengths(self) -> "TallyResult":
        """mean과 std_dev 길이가 일치하는지 검증한다."""
        if len(self.mean) != len(self.std_dev):
            raise ValueError(
                f"mean({len(self.mean)})과 std_dev({len(self.std_dev)}) 길이가 일치해야 합니다"
            )
        return self


class SimulationResult(BaseModel):
    """시뮬레이션 결과.

    OpenMC statepoint에서 파싱된 핵심 결과 데이터를 담는다.

    Attributes:
        keff: 유효증배계수 (k-effective).
        keff_std: keff 표준편차.
        tallies: 탈리 결과 목록.
        runtime: 시뮬레이션 실행 시간 (초).
        statepoint_path: statepoint 파일 경로.
        batches_completed: 완료된 배치 수.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keff: float = Field(description="유효증배계수 (k-effective)")
    keff_std: float = Field(ge=0, description="keff 표준편차")
    tallies: list[TallyResult] = Field(
        default_factory=list, description="탈리 결과 목록"
    )
    runtime: float = Field(ge=0, description="시뮬레이션 실행 시간 (초)")
    statepoint_path: Path | None = Field(
        default=None, description="statepoint 파일 경로"
    )
    batches_completed: int = Field(default=0, ge=0, description="완료된 배치 수")


class ReactorState(BaseModel):
    """원자로 상태.

    현재 설계 파라미터, 시뮬레이션 이력, 핵심 성능 지표(KPI)를
    통합하여 관리한다. Controller가 다음 행동을 결정할 때 참조한다.

    Attributes:
        current_config: 현재 케이스 설정.
        history: 이전 시뮬레이션 결과 이력.
        kpi: 핵심 성능 지표.
        iteration: 현재 반복 횟수.
    """

    model_config = ConfigDict(extra="forbid")

    current_config: CaseConfig = Field(
        default_factory=CaseConfig, description="현재 케이스 설정"
    )
    history: list[SimulationResult] = Field(
        default_factory=list, description="이전 시뮬레이션 결과 이력"
    )
    kpi: dict[str, float] = Field(
        default_factory=dict, description="핵심 성능 지표"
    )
    iteration: int = Field(default=0, description="현재 반복 횟수")
