"""LLM 컨트롤러 벤치마크 스크립트.

10개 블루프린트에 대해 LLM+BO 컨트롤러와 SimpleController를 비교 실행한다.
Docker 컨테이너 내부에서 실행되며, 실제 OpenMC 시뮬레이션을 수행한다.
LLM 서버는 호스트(localhost:8001)에서 실행 중이어야 한다.

병렬 실행: LLM 서버의 동시 처리 능력을 활용하여 블루프린트를 병렬로 실행한다.
N_WORKERS로 동시 실행 수를 조절한다.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.controller.pid import PIDControllerConfig, ProportionalControllerConfig
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

MAX_STEPS = 50  # 벤치마크용 최대 스텝 (아임계 시작 → 임계 수렴)
RUNS_DIR = Path("runs/benchmark")
N_WORKERS = 2  # 병렬 블루프린트 실행 수 (안정성 위해 감소, 4→2)


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
        llm_model="mlx-community/Qwen3.5-9B-bf16",
        n_candidates=10,
        initial_rod_position=bp.get("initial_rod_position", 228),
        max_iterations=MAX_STEPS,
        target_keff=1.0,
        keff_tolerance=0.05,
        keff_critical_deviation=1.0,
        log_dir=log_dir,
        resume_from_log=False,
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


def _run_blueprint_rod_controller(
    bp: dict,
    runs_dir: Path,
    controller_type: str,
    config: PIDControllerConfig | ProportionalControllerConfig,
) -> BenchmarkResult:
    """PID 또는 P-only 제어봉 컨트롤러로 블루프린트를 실행한다."""
    name = bp["name"]
    result = BenchmarkResult(blueprint_name=name, controller_type=controller_type)

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
                logger.info("[%s/%s] STOP at step %d: %s", name, controller_type, step, result.stop_reason)
                break

            case_config = controller.apply_actions_to_case(actions, state)

            sim_result = _run_openmc(case_config, runs_dir, f"{controller_type}_{name}_step{step}")
            if sim_result is None:
                result.error = f"OpenMC 실행 실패 at step {step}"
                logger.error("[%s/%s] %s", name, controller_type, result.error)
                break

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
                rod_position=metrics.get("rod_position", 0.0),
                keff=metrics["keff"],
                keff_std=metrics["keff_std"],
                keff_deviation=metrics["keff_deviation"],
                safety_score=0.0,
                converged=metrics["converged"] > 0.5,
                runtime=sim_result.runtime,
            )
            result.steps.append(sr)

            logger.info(
                "[%s/%s] step=%d rod=%d keff=%.5f±%.5f dev=%.5f",
                name, controller_type, step, int(sr.rod_position),
                sr.keff, sr.keff_std, sr.keff_deviation,
            )

            if sr.converged and result.converged_at_step < 0:
                result.converged = True
                result.converged_at_step = step

    except Exception as e:
        result.error = str(e)
        logger.error("[%s/%s] 에러: %s", name, controller_type, e, exc_info=True)

    result.total_runtime = time.time() - t_start
    if result.steps:
        result.final_keff = result.steps[-1].keff
        result.final_deviation = result.steps[-1].keff_deviation

    return result


def run_blueprint_pid(bp: dict, runs_dir: Path) -> BenchmarkResult:
    """PID 컨트롤러로 블루프린트를 실행한다."""
    config = PIDControllerConfig(
        initial_rod_position=bp.get("initial_rod_position", 14),
        max_iterations=MAX_STEPS,
        keff_tolerance=0.05,
        keff_critical_deviation=1.0,
    )
    return _run_blueprint_rod_controller(bp, runs_dir, "pid", config)


def run_blueprint_proportional(bp: dict, runs_dir: Path) -> BenchmarkResult:
    """P-only 컨트롤러로 블루프린트를 실행한다."""
    config = ProportionalControllerConfig(
        initial_rod_position=bp.get("initial_rod_position", 14),
        max_iterations=MAX_STEPS,
        keff_tolerance=0.05,
        keff_critical_deviation=1.0,
    )
    return _run_blueprint_rod_controller(bp, runs_dir, "proportional", config)


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
    pid_results: list[BenchmarkResult],
    prop_results: list[BenchmarkResult],
    output_path: Path,
) -> None:
    """벤치마크 보고서를 JSON으로 저장한다."""
    report = {
        "metadata": {
            "max_steps": MAX_STEPS,
            "sim_batches": BENCHMARK_SETTINGS.batches,
            "sim_particles": BENCHMARK_SETTINGS.particles,
            "n_blueprints": len(llm_results),
            "controllers": ["llm", "pid", "proportional"],
        },
        "summary": _build_summary(llm_results, pid_results, prop_results),
        "llm_results": [_result_to_dict(r) for r in llm_results],
        "pid_results": [_result_to_dict(r) for r in pid_results],
        "proportional_results": [_result_to_dict(r) for r in prop_results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("보고서 저장: %s", output_path)


def _summarize_group(results: list[BenchmarkResult]) -> dict:
    """단일 컨트롤러 그룹의 요약 통계."""
    ok = [r for r in results if not r.error]

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "total_runs": len(results),
        "successful": len(ok),
        "converged": sum(1 for r in ok if r.converged),
        "avg_final_deviation": avg([r.final_deviation for r in ok]),
        "avg_runtime": avg([r.total_runtime for r in ok]),
        "avg_converge_step": avg(
            [r.converged_at_step for r in ok if r.converged]
        ),
    }


def _build_summary(
    llm_results: list[BenchmarkResult],
    pid_results: list[BenchmarkResult],
    prop_results: list[BenchmarkResult],
) -> dict:
    """요약 통계를 생성한다."""
    return {
        "llm": _summarize_group(llm_results),
        "pid": _summarize_group(pid_results),
        "proportional": _summarize_group(prop_results),
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


def _run_one_blueprint(
    idx: int,
    total: int,
    bp: dict,
    runs_dir: Path,
) -> tuple[BenchmarkResult, BenchmarkResult, BenchmarkResult]:
    """단일 블루프린트를 LLM + PID + P-only 컨트롤러로 실행한다.

    ThreadPoolExecutor에서 호출된다.
    """
    name = bp["name"]
    logger.info(
        "[%d/%d] 블루프린트 시작: %s — %s",
        idx + 1, total, name, bp.get("description", ""),
    )

    llm_r = run_blueprint_llm(bp, runs_dir)
    logger.info(
        "[%s/llm] 완료: keff=%.5f dev=%.5f converged=%s time=%.1fs",
        name, llm_r.final_keff, llm_r.final_deviation,
        llm_r.converged, llm_r.total_runtime,
    )

    pid_r = run_blueprint_pid(bp, runs_dir)
    logger.info(
        "[%s/pid] 완료: keff=%.5f dev=%.5f converged=%s time=%.1fs",
        name, pid_r.final_keff, pid_r.final_deviation,
        pid_r.converged, pid_r.total_runtime,
    )

    prop_r = run_blueprint_proportional(bp, runs_dir)
    logger.info(
        "[%s/proportional] 완료: keff=%.5f dev=%.5f converged=%s time=%.1fs",
        name, prop_r.final_keff, prop_r.final_deviation,
        prop_r.converged, prop_r.total_runtime,
    )

    return llm_r, pid_r, prop_r


def main() -> None:
    """벤치마크 메인 함수."""
    bp_path = Path("configs/blueprints.json")
    if not bp_path.exists():
        logger.error("블루프린트 파일 없음: %s", bp_path)
        sys.exit(1)

    blueprints = load_blueprints(bp_path)
    logger.info("블루프린트 %d개 로드, 병렬 워커 %d개", len(blueprints), N_WORKERS)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # LLM 로그 초기화 (이전 실행의 stale 로그 방지, 방어적 2차 보호)
    # 1차 보호: LLMController가 resume_from_log=False일 때 자체적으로 로그를 삭제
    for bp in blueprints:
        log_file = RUNS_DIR / f"llm_logs_{bp['name']}" / "log.json"
        if log_file.exists():
            log_file.unlink()
            logger.info("이전 LLM 로그 삭제: %s", log_file)

    # 블루프린트 이름 순서 보존용 (결과 정렬)
    bp_names = [bp["name"] for bp in blueprints]
    llm_map: dict[str, BenchmarkResult] = {}
    pid_map: dict[str, BenchmarkResult] = {}
    prop_map: dict[str, BenchmarkResult] = {}

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                _run_one_blueprint, i, len(blueprints), bp, RUNS_DIR,
            ): bp["name"]
            for i, bp in enumerate(blueprints)
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                llm_r, pid_r, prop_r = future.result()
                llm_map[name] = llm_r
                pid_map[name] = pid_r
                prop_map[name] = prop_r
            except Exception as e:
                logger.error("[%s] 치명적 에러: %s", name, e, exc_info=True)
                for m, ct in [(llm_map, "llm"), (pid_map, "pid"), (prop_map, "proportional")]:
                    m[name] = BenchmarkResult(
                        blueprint_name=name, controller_type=ct, error=str(e),
                    )

    # 원래 블루프린트 순서로 정렬
    llm_results = [llm_map[n] for n in bp_names]
    pid_results = [pid_map[n] for n in bp_names]
    prop_results = [prop_map[n] for n in bp_names]

    # 보고서 생성
    report_path = RUNS_DIR / "benchmark_report.json"
    generate_report(llm_results, pid_results, prop_results, report_path)

    # 콘솔 요약
    print("\n" + "=" * 90)
    print("벤치마크 결과 요약: LLM+BO vs PID vs P-only")
    print("=" * 90)
    print(
        f"{'Blueprint':<12} "
        f"{'LLM keff':>10} {'dev':>8} {'conv':>5}  "
        f"{'PID keff':>10} {'dev':>8} {'conv':>5}  "
        f"{'P-only keff':>12} {'dev':>8} {'conv':>5}"
    )
    print("-" * 90)
    for llm_r, pid_r, prop_r in zip(llm_results, pid_results, prop_results):
        print(
            f"{llm_r.blueprint_name:<12} "
            f"{llm_r.final_keff:>10.5f} {llm_r.final_deviation:>8.5f} "
            f"{'Y' if llm_r.converged else 'N':>5}  "
            f"{pid_r.final_keff:>10.5f} {pid_r.final_deviation:>8.5f} "
            f"{'Y' if pid_r.converged else 'N':>5}  "
            f"{prop_r.final_keff:>12.5f} {prop_r.final_deviation:>8.5f} "
            f"{'Y' if prop_r.converged else 'N':>5}"
        )
    print("=" * 90)
    print(f"보고서: {report_path}")


if __name__ == "__main__":
    main()
