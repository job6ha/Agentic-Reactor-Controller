"""LLM 컨트롤러 벤치마크 스크립트.

10개 블루프린트에 대해 LLM+BO 컨트롤러와 SimpleController를 비교 실행한다.
Docker 컨테이너 내부에서 실행되며, 실제 OpenMC 시뮬레이션을 수행한다.
LLM 서버는 호스트(localhost:8001)에서 실행 중이어야 한다.

Usage (Docker):
    docker compose run --rm openmc uv run python scripts/benchmark.py

Usage (로컬, OpenMC 설치 시):
    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    ReactorState,
    SimulationResult,
    SimulationSettings,
)
from src.armi_layer.case_folder import create_case
from src.controller.factory import create_controller
from src.controller.llm.models import LLMControllerConfig
from src.controller.simple import SimpleControllerConfig
from src.controller.base import ActionType
from src.openmc_layer.input_generator import generate_input
from src.openmc_layer.result_parser import parse_results
from src.run_manager.status import run_case
from src.armi_layer.models import RunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")

# 시뮬레이션 설정 (빠른 벤치마크용)
BENCHMARK_SETTINGS = SimulationSettings(
    batches=120,
    inactive=20,
    particles=5000,
)

RUN_CONFIG = RunConfig(
    omp_threads=4,
    timeout=600.0,
    max_retries=1,
)

MAX_STEPS = 20  # 벤치마크용 최대 스텝 (현실적 제어봉 이동 반영)
RUNS_DIR = Path("runs/benchmark")


@dataclass
class StepResult:
    """단일 스텝 결과."""

    step: int
    rod_position: float
    keff: float
    keff_std: float
    keff_deviation: float
    safety_score: float
    converged: bool
    runtime: float


@dataclass
class BenchmarkResult:
    """단일 블루프린트 벤치마크 결과."""

    blueprint_name: str
    controller_type: str
    steps: list[StepResult] = field(default_factory=list)
    total_runtime: float = 0.0
    final_keff: float = 0.0
    final_deviation: float = 0.0
    converged: bool = False
    converged_at_step: int = -1
    stop_reason: str = ""
    error: str = ""


def load_blueprints(path: Path) -> list[dict]:
    """블루프린트 목록을 로드한다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["blueprints"]


def build_case_config(bp: dict) -> CaseConfig:
    """블루프린트에서 CaseConfig를 생성한다."""
    geo_params = bp.get("geometry", {})
    mat_params = bp.get("materials", {})

    return CaseConfig(
        name=bp["name"],
        description=bp.get("description", ""),
        geometry=GeometryParams(**geo_params),
        materials=MaterialParams(**mat_params),
        settings=BENCHMARK_SETTINGS,
        tags=bp.get("tags", []),
    )


def run_blueprint_llm(bp: dict, runs_dir: Path) -> BenchmarkResult:
    """LLM+BO 컨트롤러로 블루프린트를 실행한다."""
    name = bp["name"]
    result = BenchmarkResult(blueprint_name=name, controller_type="llm")
    log_dir = runs_dir / f"llm_logs_{name}"

    config = LLMControllerConfig(
        llm_base_url="http://host.docker.internal:8001/v1",
        llm_model="mlx-community/Qwen3.5-9B-4bit",
        n_candidates=10,
        initial_rod_position=bp.get("initial_rod_position", 228),
        max_iterations=MAX_STEPS,
        target_keff=1.0,
        keff_tolerance=0.05,
        keff_critical_deviation=1.0,
        log_dir=log_dir,
    )

    controller = create_controller(config)
    case_config = build_case_config(bp)
    state = ReactorState(current_config=case_config)
    t_start = time.time()

    try:
        for step in range(MAX_STEPS):
            # 1. propose
            actions = controller.propose_actions(state)
            stop_actions = [a for a in actions if a.action_type == ActionType.STOP]
            if stop_actions:
                result.stop_reason = stop_actions[0].reason
                logger.info("[%s/llm] STOP at step %d: %s", name, step, result.stop_reason)
                break

            # 2. apply
            case_config = controller.apply_actions_to_case(actions, state)

            # 3. run OpenMC
            sim_result = _run_openmc(case_config, runs_dir, f"llm_{name}_step{step}")
            if sim_result is None:
                result.error = f"OpenMC 실행 실패 at step {step}"
                logger.error("[%s/llm] %s", name, result.error)
                break

            # 4. evaluate
            metrics = controller.evaluate_results(sim_result)
            state = controller.update_state(state, metrics)
            state = state.model_copy(
                update={
                    "current_config": case_config,
                    "history": [*state.history, sim_result],
                },
            )

            sr = StepResult(
                step=step,
                rod_position=metrics["rod_position"],
                keff=metrics["keff"],
                keff_std=metrics["keff_std"],
                keff_deviation=metrics["keff_deviation"],
                safety_score=metrics["safety_score"],
                converged=metrics["converged"] > 0.5,
                runtime=sim_result.runtime,
            )
            result.steps.append(sr)

            logger.info(
                "[%s/llm] step=%d rod=%d keff=%.5f±%.5f dev=%.5f safety=%.3f",
                name, step, int(sr.rod_position), sr.keff, sr.keff_std,
                sr.keff_deviation, sr.safety_score,
            )

            if sr.converged and result.converged_at_step < 0:
                result.converged = True
                result.converged_at_step = step

    except Exception as e:
        result.error = str(e)
        logger.error("[%s/llm] 에러: %s", name, e, exc_info=True)

    result.total_runtime = time.time() - t_start
    if result.steps:
        result.final_keff = result.steps[-1].keff
        result.final_deviation = result.steps[-1].keff_deviation

    return result


def run_blueprint_simple(bp: dict, runs_dir: Path) -> BenchmarkResult:
    """SimpleController로 블루프린트를 실행한다 (비교 대조군)."""
    name = bp["name"]
    result = BenchmarkResult(blueprint_name=name, controller_type="simple")

    # SimpleController는 파라미터 스윕 — fuel_enrichment 기준으로 비교
    enrichment = bp.get("materials", {}).get("fuel_enrichment", 3.0)
    config = SimpleControllerConfig(
        sweep_field="materials.fuel_enrichment",
        sweep_values=[enrichment - 0.3, enrichment - 0.1, enrichment, enrichment + 0.1, enrichment + 0.3],
        max_iterations=min(5, MAX_STEPS),
    )

    controller = create_controller(config)
    case_config = build_case_config(bp)
    state = ReactorState(current_config=case_config)
    t_start = time.time()

    try:
        for step in range(config.max_iterations):
            actions = controller.propose_actions(state)
            stop_actions = [a for a in actions if a.action_type == ActionType.STOP]
            if stop_actions:
                result.stop_reason = stop_actions[0].reason
                break

            case_config = controller.apply_actions_to_case(actions, state)

            sim_result = _run_openmc(case_config, runs_dir, f"simple_{name}_step{step}")
            if sim_result is None:
                result.error = f"OpenMC 실행 실패 at step {step}"
                break

            metrics = controller.evaluate_results(sim_result)
            state = controller.update_state(state, metrics)
            state = state.model_copy(
                update={
                    "current_config": case_config,
                    "history": [*state.history, sim_result],
                },
            )

            keff = sim_result.keff
            deviation = abs(keff - 1.0)
            sr = StepResult(
                step=step,
                rod_position=0.0,  # simple은 rod 미사용
                keff=keff,
                keff_std=sim_result.keff_std,
                keff_deviation=deviation,
                safety_score=0.0,
                converged=deviation <= 0.01,
                runtime=sim_result.runtime,
            )
            result.steps.append(sr)

            logger.info(
                "[%s/simple] step=%d keff=%.5f±%.5f dev=%.5f",
                name, step, keff, sim_result.keff_std, deviation,
            )

            if sr.converged and result.converged_at_step < 0:
                result.converged = True
                result.converged_at_step = step

    except Exception as e:
        result.error = str(e)
        logger.error("[%s/simple] 에러: %s", name, e, exc_info=True)

    result.total_runtime = time.time() - t_start
    if result.steps:
        result.final_keff = result.steps[-1].keff
        result.final_deviation = result.steps[-1].keff_deviation

    return result


def _run_openmc(
    case_config: CaseConfig,
    runs_dir: Path,
    case_name: str,
) -> SimulationResult | None:
    """단일 OpenMC 시뮬레이션을 실행한다."""
    try:
        case_dir = create_case(case_config, runs_dir)
        generate_input(case_config, case_dir)

        run_result = run_case(RUN_CONFIG, case_dir)
        if not run_result.success:
            logger.warning("OpenMC 실행 실패: %s", case_name)
            return None

        return parse_results(case_dir)
    except Exception:
        logger.error("OpenMC 실행 에러: %s", case_name, exc_info=True)
        return None


def generate_report(
    llm_results: list[BenchmarkResult],
    simple_results: list[BenchmarkResult],
    output_path: Path,
) -> None:
    """벤치마크 보고서를 JSON으로 저장한다."""
    report = {
        "metadata": {
            "max_steps": MAX_STEPS,
            "sim_batches": BENCHMARK_SETTINGS.batches,
            "sim_particles": BENCHMARK_SETTINGS.particles,
            "n_blueprints": len(llm_results),
        },
        "summary": _build_summary(llm_results, simple_results),
        "llm_results": [_result_to_dict(r) for r in llm_results],
        "simple_results": [_result_to_dict(r) for r in simple_results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("보고서 저장: %s", output_path)


def _build_summary(
    llm_results: list[BenchmarkResult],
    simple_results: list[BenchmarkResult],
) -> dict:
    """요약 통계를 생성한다."""
    llm_ok = [r for r in llm_results if not r.error]
    simple_ok = [r for r in simple_results if not r.error]

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "llm": {
            "total_runs": len(llm_results),
            "successful": len(llm_ok),
            "converged": sum(1 for r in llm_ok if r.converged),
            "avg_final_deviation": avg([r.final_deviation for r in llm_ok]),
            "avg_runtime": avg([r.total_runtime for r in llm_ok]),
            "avg_converge_step": avg(
                [r.converged_at_step for r in llm_ok if r.converged]
            ),
        },
        "simple": {
            "total_runs": len(simple_results),
            "successful": len(simple_ok),
            "converged": sum(1 for r in simple_ok if r.converged),
            "avg_final_deviation": avg([r.final_deviation for r in simple_ok]),
            "avg_runtime": avg([r.total_runtime for r in simple_ok]),
            "avg_converge_step": avg(
                [r.converged_at_step for r in simple_ok if r.converged]
            ),
        },
    }


def _result_to_dict(r: BenchmarkResult) -> dict:
    """BenchmarkResult를 직렬화한다."""
    return {
        "blueprint_name": r.blueprint_name,
        "controller_type": r.controller_type,
        "total_runtime": round(r.total_runtime, 2),
        "final_keff": round(r.final_keff, 5),
        "final_deviation": round(r.final_deviation, 5),
        "converged": r.converged,
        "converged_at_step": r.converged_at_step,
        "stop_reason": r.stop_reason,
        "error": r.error,
        "steps": [
            {
                "step": s.step,
                "rod_position": s.rod_position,
                "keff": round(s.keff, 5),
                "keff_std": round(s.keff_std, 5),
                "keff_deviation": round(s.keff_deviation, 5),
                "safety_score": round(s.safety_score, 3),
                "converged": s.converged,
                "runtime": round(s.runtime, 2),
            }
            for s in r.steps
        ],
    }


def main() -> None:
    """벤치마크 메인 함수."""
    bp_path = Path("configs/blueprints.json")
    if not bp_path.exists():
        logger.error("블루프린트 파일 없음: %s", bp_path)
        sys.exit(1)

    blueprints = load_blueprints(bp_path)
    logger.info("블루프린트 %d개 로드", len(blueprints))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    llm_results: list[BenchmarkResult] = []
    simple_results: list[BenchmarkResult] = []

    for i, bp in enumerate(blueprints):
        name = bp["name"]
        logger.info("=" * 60)
        logger.info("[%d/%d] 블루프린트: %s — %s", i + 1, len(blueprints), name, bp.get("description", ""))
        logger.info("=" * 60)

        # LLM+BO 컨트롤러
        logger.info("--- LLM+BO 컨트롤러 실행 ---")
        llm_r = run_blueprint_llm(bp, RUNS_DIR)
        llm_results.append(llm_r)
        logger.info(
            "[%s/llm] 완료: keff=%.5f dev=%.5f converged=%s time=%.1fs",
            name, llm_r.final_keff, llm_r.final_deviation,
            llm_r.converged, llm_r.total_runtime,
        )

        # Simple 컨트롤러 (비교군)
        logger.info("--- Simple 컨트롤러 실행 ---")
        simple_r = run_blueprint_simple(bp, RUNS_DIR)
        simple_results.append(simple_r)
        logger.info(
            "[%s/simple] 완료: keff=%.5f dev=%.5f converged=%s time=%.1fs",
            name, simple_r.final_keff, simple_r.final_deviation,
            simple_r.converged, simple_r.total_runtime,
        )

    # 보고서 생성
    report_path = RUNS_DIR / "benchmark_report.json"
    generate_report(llm_results, simple_results, report_path)

    # 콘솔 요약
    print("\n" + "=" * 70)
    print("벤치마크 결과 요약")
    print("=" * 70)
    print(f"{'Blueprint':<12} {'LLM keff':>10} {'LLM dev':>10} {'LLM conv':>10} "
          f"{'Simple keff':>12} {'Simple dev':>12}")
    print("-" * 70)
    for llm_r, simple_r in zip(llm_results, simple_results):
        print(
            f"{llm_r.blueprint_name:<12} "
            f"{llm_r.final_keff:>10.5f} {llm_r.final_deviation:>10.5f} "
            f"{'✓' if llm_r.converged else '✗':>10} "
            f"{simple_r.final_keff:>12.5f} {simple_r.final_deviation:>12.5f}"
        )
    print("=" * 70)
    print(f"보고서: {report_path}")


if __name__ == "__main__":
    main()
