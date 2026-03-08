"""단일 블루프린트 스모크 테스트.

PWR-STD 블루프린트로 LLM 컨트롤러 3스텝만 실행하여
제어봉 방향 수정이 제대로 작동하는지 확인한다.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.armi_layer.case_folder import create_case
from src.armi_layer.models import (
    CaseConfig,
    GeometryParams,
    MaterialParams,
    ReactorState,
    RunConfig,
    SimulationSettings,
)
from src.controller.base import ActionType
from src.controller.factory import create_controller
from src.controller.llm.models import LLMControllerConfig
from src.openmc_layer.input_generator import generate_input
from src.openmc_layer.result_parser import parse_results
from src.run_manager.status import run_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

RUNS_DIR = Path("runs/smoke_test")
MAX_STEPS = 3

RUN_CONFIG = RunConfig(omp_threads=4, timeout=600.0, max_retries=1)
SIM_SETTINGS = SimulationSettings(batches=50, inactive=10, particles=2000)


def main() -> None:
    """스모크 테스트 실행."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    config = LLMControllerConfig(
        llm_base_url="http://host.docker.internal:8001/v1",
        llm_model="mlx-community/Qwen3.5-9B-4bit",
        n_candidates=10,
        initial_rod_position=14,
        max_iterations=MAX_STEPS,
        target_keff=1.0,
        keff_tolerance=0.05,
        keff_critical_deviation=1.0,
    )

    controller = create_controller(config)
    case_config = CaseConfig(
        name="PWR-STD",
        geometry=GeometryParams(absorber_outer_radius=0.52),
        materials=MaterialParams(),
        settings=SIM_SETTINGS,
    )
    state = ReactorState(current_config=case_config)

    print("\n" + "=" * 60)
    print("스모크 테스트: PWR-STD, LLM+BO, 3 steps")
    print("=" * 60)

    t_start = time.time()

    for step in range(MAX_STEPS):
        actions = controller.propose_actions(state)
        stop_actions = [a for a in actions if a.action_type == ActionType.STOP]
        if stop_actions:
            print(f"\n[STOP] {stop_actions[0].reason}")
            break

        case_config = controller.apply_actions_to_case(actions, state)

        # rod_position 확인
        rod_pos = case_config.geometry.extra_params.get("rod_position", "N/A")
        print(f"\n--- Step {step} | rod_position={rod_pos} ---")

        try:
            case_dir = create_case(case_config, RUNS_DIR)
            generate_input(case_config, case_dir)
            run_result = run_case(RUN_CONFIG, case_dir)

            if not run_result.success:
                print(f"  OpenMC 실행 실패!")
                continue

            sim_result = parse_results(case_dir)
            metrics = controller.evaluate_results(sim_result)
            state = controller.update_state(state, metrics)
            state = state.model_copy(
                update={
                    "current_config": case_config,
                    "history": [*state.history, sim_result],
                },
            )

            print(f"  keff={sim_result.keff:.5f} ± {sim_result.keff_std:.5f}")
            print(f"  deviation={metrics['keff_deviation']:.5f}")
            print(f"  safety_score={metrics['safety_score']:.3f}")
            print(f"  converged={'YES' if metrics['converged'] > 0.5 else 'NO'}")

        except Exception as e:
            print(f"  에러: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"스모크 테스트 완료 ({elapsed:.1f}s)")

    # 핵심 검증: rod_position이 실제로 변했는지
    if state.history:
        positions = []
        for h in state.history:
            positions.append(state.kpi.get("rod_position", "?"))
        keffs = [h.keff for h in state.history]
        print(f"keff 이력: {[f'{k:.5f}' for k in keffs]}")
        print(f"최종 rod_position: {state.kpi.get('rod_position', 'N/A')}")

        # 방향 검증: 초기 228에서 keff>1이면 위치가 감소해야 함
        final_pos = state.kpi.get("rod_position", 228)
        if final_pos < 228:
            print("✓ 제어봉이 삽입 방향으로 이동함 (방향 버그 수정 확인)")
        else:
            print("✗ 제어봉이 이동하지 않음 (문제 가능성)")
    print("=" * 60)


if __name__ == "__main__":
    main()
