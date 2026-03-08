"""벤치마크 결과 분석 스크립트.

benchmark.py가 생성한 benchmark_report.json을 읽어
콘솔 보고서 + Markdown 보고서를 생성한다.

Usage:
    uv run python scripts/analyze_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT_PATH = Path("runs/benchmark/benchmark_report.json")
OUTPUT_MD = Path("runs/benchmark/BENCHMARK_REPORT.md")


def load_report(path: Path) -> dict:
    """벤치마크 보고서를 로드한다."""
    return json.loads(path.read_text(encoding="utf-8"))


def print_console_report(report: dict) -> None:
    """콘솔 요약을 출력한다."""
    meta = report["metadata"]
    summary = report["summary"]

    print("=" * 80)
    print("  LLM+BO 컨트롤러 벤치마크 결과 보고서")
    print("=" * 80)
    print(f"\n실험 조건:")
    print(f"  블루프린트 수: {meta['n_blueprints']}")
    print(f"  최대 스텝: {meta['max_steps']}")
    print(f"  배치/입자: {meta['sim_batches']}/{meta['sim_particles']}")

    print(f"\n{'=' * 80}")
    print("  요약 비교: LLM+BO vs Simple")
    print(f"{'=' * 80}")

    for ctrl_type in ["llm", "simple"]:
        s = summary[ctrl_type]
        print(f"\n  [{ctrl_type.upper()}]")
        print(f"    실행: {s['successful']}/{s['total_runs']}")
        print(f"    수렴: {s['converged']}")
        print(f"    평균 최종 keff 편차: {s['avg_final_deviation']:.5f}")
        print(f"    평균 실행 시간: {s['avg_runtime']:.1f}s")
        if s["converged"] > 0:
            print(f"    평균 수렴 스텝: {s['avg_converge_step']:.1f}")

    print(f"\n{'=' * 80}")
    print("  블루프린트별 상세 결과")
    print(f"{'=' * 80}")

    llm_results = report["llm_results"]
    simple_results = report["simple_results"]

    header = (
        f"{'Blueprint':<12} │ {'LLM keff':>10} {'dev':>8} {'conv':>5} {'time':>7}"
        f" │ {'Simple keff':>12} {'dev':>8} {'conv':>5} {'time':>7}"
    )
    print(f"\n{header}")
    print("─" * len(header))

    for llm_r, simple_r in zip(llm_results, simple_results):
        llm_conv = "✓" if llm_r["converged"] else "✗"
        sim_conv = "✓" if simple_r["converged"] else "✗"
        print(
            f"{llm_r['blueprint_name']:<12} │ "
            f"{llm_r['final_keff']:>10.5f} {llm_r['final_deviation']:>8.5f} "
            f"{llm_conv:>5} {llm_r['total_runtime']:>6.1f}s"
            f" │ {simple_r['final_keff']:>12.5f} {simple_r['final_deviation']:>8.5f} "
            f"{sim_conv:>5} {simple_r['total_runtime']:>6.1f}s"
        )

    # LLM 스텝별 상세
    print(f"\n{'=' * 80}")
    print("  LLM 컨트롤러 스텝별 상세")
    print(f"{'=' * 80}")

    for r in llm_results:
        if r["error"]:
            print(f"\n  [{r['blueprint_name']}] ERROR: {r['error']}")
            continue
        print(f"\n  [{r['blueprint_name']}] stop={r['stop_reason'] or 'max_steps'}")
        for s in r["steps"]:
            print(
                f"    step={s['step']} rod={s['rod_position']:>5.0f} "
                f"keff={s['keff']:.5f}±{s['keff_std']:.5f} "
                f"safety={s['safety_score']:.3f} "
                f"{'✓' if s['converged'] else '✗'}"
            )


def generate_markdown_report(report: dict, output: Path) -> None:
    """Markdown 보고서를 생성한다."""
    meta = report["metadata"]
    summary = report["summary"]
    llm_results = report["llm_results"]
    simple_results = report["simple_results"]

    lines: list[str] = []
    lines.append("# LLM+BO 컨트롤러 벤치마크 보고서\n")
    lines.append(f"## 실험 조건\n")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|---|---|")
    lines.append(f"| 블루프린트 수 | {meta['n_blueprints']} |")
    lines.append(f"| 최대 스텝 | {meta['max_steps']} |")
    lines.append(f"| OpenMC 배치 | {meta['sim_batches']} |")
    lines.append(f"| OpenMC 입자수 | {meta['sim_particles']} |")

    lines.append(f"\n## 요약 비교\n")
    lines.append(f"| 지표 | LLM+BO | Simple |")
    lines.append(f"|---|---|---|")
    lines.append(
        f"| 성공 | {summary['llm']['successful']}/{summary['llm']['total_runs']} "
        f"| {summary['simple']['successful']}/{summary['simple']['total_runs']} |"
    )
    lines.append(
        f"| 수렴 | {summary['llm']['converged']} "
        f"| {summary['simple']['converged']} |"
    )
    lines.append(
        f"| 평균 keff 편차 | {summary['llm']['avg_final_deviation']:.5f} "
        f"| {summary['simple']['avg_final_deviation']:.5f} |"
    )
    lines.append(
        f"| 평균 실행 시간 | {summary['llm']['avg_runtime']:.1f}s "
        f"| {summary['simple']['avg_runtime']:.1f}s |"
    )

    lines.append(f"\n## 블루프린트별 결과\n")
    lines.append(
        f"| Blueprint | LLM keff | LLM dev | LLM conv | Simple keff | Simple dev | Simple conv |"
    )
    lines.append(f"|---|---|---|---|---|---|---|")

    for llm_r, simple_r in zip(llm_results, simple_results):
        llm_conv = "O" if llm_r["converged"] else "X"
        sim_conv = "O" if simple_r["converged"] else "X"
        lines.append(
            f"| {llm_r['blueprint_name']} "
            f"| {llm_r['final_keff']:.5f} | {llm_r['final_deviation']:.5f} | {llm_conv} "
            f"| {simple_r['final_keff']:.5f} | {simple_r['final_deviation']:.5f} | {sim_conv} |"
        )

    lines.append(f"\n## LLM 컨트롤러 스텝별 상세\n")
    for r in llm_results:
        lines.append(f"### {r['blueprint_name']}\n")
        if r["error"]:
            lines.append(f"**ERROR**: {r['error']}\n")
            continue
        if r["stop_reason"]:
            lines.append(f"종료 사유: {r['stop_reason']}\n")

        if r["steps"]:
            lines.append(f"| Step | Rod Position | keff | keff_std | Safety | Converged |")
            lines.append(f"|---|---|---|---|---|---|")
            for s in r["steps"]:
                conv = "O" if s["converged"] else "X"
                lines.append(
                    f"| {s['step']} | {s['rod_position']:.0f} "
                    f"| {s['keff']:.5f} | {s['keff_std']:.5f} "
                    f"| {s['safety_score']:.3f} | {conv} |"
                )
        lines.append("")

    lines.append(f"\n## 분석\n")
    lines.append("### 관찰 사항\n")
    lines.append(
        "1. **Pin cell 한계**: 현재 OpenMC pin cell 모델에서 `rod_position`은 "
        "`geometry.extra_params`에 메타데이터로만 저장되며, 실제 제어봉 기하 구조에 "
        "반영되지 않는다. 따라서 rod_position 변경이 keff에 영향을 주지 않는다.\n"
    )
    lines.append(
        "2. **블루프린트 간 keff 차이**: 농축도, 격자 피치, 냉각재 밀도 등 "
        "재료/기하 파라미터 변경은 keff에 유의미한 차이를 만든다.\n"
    )
    lines.append(
        "3. **LLM+BO 파이프라인**: Qwen3.5 thinking 모드로 인해 응답 시간이 길며, "
        "타임아웃 시 랜덤 폴백 후보가 사용된다. BO의 GP 모델은 "
        "동일 rod_position에서 동일 keff를 관측하므로 유의미한 서로게이트 모델을 "
        "구축하지 못한다.\n"
    )
    lines.append("### 개선 방향\n")
    lines.append(
        "1. **제어봉 기하 모델링**: `input_generator.py`에서 rod_position을 "
        "실제 흡수체(B4C) 셀 높이로 변환하여 OpenMC 기하에 반영\n"
    )
    lines.append(
        "2. **LLM 응답 최적화**: thinking 모드 비활성화 또는 "
        "max_tokens 감소로 응답 시간 단축\n"
    )
    lines.append(
        "3. **다중 pin cell → assembly 모델**: 제어봉 효과가 명확히 나타나는 "
        "집합체 수준 모델로 확장\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown 보고서 저장: {output}")


def main() -> None:
    """메인."""
    if not REPORT_PATH.exists():
        print(f"보고서 파일 없음: {REPORT_PATH}", file=sys.stderr)
        sys.exit(1)

    report = load_report(REPORT_PATH)
    print_console_report(report)
    generate_markdown_report(report, OUTPUT_MD)


if __name__ == "__main__":
    main()
