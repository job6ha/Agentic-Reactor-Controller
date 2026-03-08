"""Benchmark results visualization script.

Reads runs/benchmark/benchmark_report.json and generates publication-quality
plots saved to runs/benchmark/plots/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "runs" / "benchmark" / "benchmark_report.json"
PLOTS_DIR = REPO_ROOT / "runs" / "benchmark" / "plots"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
STYLE = "seaborn-v0_8-paper"

COLORS: dict[str, str] = {
    "llm": "#1f77b4",         # blue
    "pid": "#d62728",         # red
    "proportional": "#2ca02c", # green
}

LABELS: dict[str, str] = {
    "llm": "LLM+BO",
    "pid": "PID",
    "proportional": "P-only",
}

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
DPI = 300

SKIP_BLUEPRINTS: set[str] = set()  # 10개 전부 포함 (병렬 안정화 후 실패 없음 기대)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_report(path: Path) -> dict:
    """Load benchmark report JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_data_map(report: dict) -> dict[str, dict[str, dict]]:
    """Build mapping: blueprint -> controller -> result dict.

    Only includes blueprints where ALL 3 controllers have >0 steps.
    """
    result_keys = ["llm_results", "pid_results", "proportional_results"]
    controller_names = ["llm", "pid", "proportional"]

    # blueprint -> controller -> result
    raw: dict[str, dict[str, dict]] = {}

    for key, ctrl in zip(result_keys, controller_names):
        for entry in report.get(key, []):
            bp = entry["blueprint_name"]
            if bp in SKIP_BLUEPRINTS:
                continue
            raw.setdefault(bp, {})[ctrl] = entry

    # Filter: all 3 controllers must have >0 steps
    valid: dict[str, dict[str, dict]] = {}
    for bp, ctrl_map in raw.items():
        if len(ctrl_map) == 3 and all(
            len(ctrl_map[c].get("steps", [])) > 0 for c in controller_names
        ):
            valid[bp] = ctrl_map

    return valid


# ---------------------------------------------------------------------------
# Plot 1: keff convergence per blueprint
# ---------------------------------------------------------------------------

def plot_keff_convergence(data: dict[str, dict[str, dict]], out_dir: Path) -> None:
    """Per-blueprint subplot: keff vs step for all 3 controllers."""
    blueprints = list(data.keys())
    n = len(blueprints)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, bp in enumerate(blueprints):
        ax = axes_flat[idx]
        ctrl_map = data[bp]

        for ctrl in ["llm", "pid", "proportional"]:
            steps_data = ctrl_map[ctrl]["steps"]
            xs = [s["step"] for s in steps_data]
            ys = [s["keff"] for s in steps_data]
            ax.plot(
                xs, ys,
                color=COLORS[ctrl],
                label=LABELS[ctrl],
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        # Target line
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Target (k=1.0)")
        # Tolerance band ±0.05
        ax.axhspan(0.95, 1.05, alpha=0.1, color="gray", label="±0.05 Tolerance")

        # Annotation for PWR-HT LLM single step
        if bp == "PWR-HT":
            ax.annotate(
                "LLM: 1 step\n(OpenMC fail)",
                xy=(0, ctrl_map["llm"]["steps"][0]["keff"]),
                xytext=(1, ctrl_map["llm"]["steps"][0]["keff"] - 0.05),
                fontsize=7,
                color=COLORS["llm"],
                arrowprops=dict(arrowstyle="->", color=COLORS["llm"], lw=0.8),
            )

        title = bp
        ax.set_title(title, fontsize=TITLE_SIZE - 2)
        ax.set_xlabel("Step", fontsize=LABEL_SIZE - 1)
        ax.set_ylabel("k$_{eff}$", fontsize=LABEL_SIZE - 1)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "k$_{eff}$ Convergence per Blueprint — Controller Comparison",
        fontsize=TITLE_SIZE,
        y=1.01,
    )
    plt.tight_layout()
    out_path = out_dir / "keff_convergence_per_blueprint.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: rod position per blueprint
# ---------------------------------------------------------------------------

def plot_rod_position(data: dict[str, dict[str, dict]], out_dir: Path) -> None:
    """Per-blueprint subplot: rod_position vs step for all 3 controllers."""
    blueprints = list(data.keys())
    n = len(blueprints)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, bp in enumerate(blueprints):
        ax = axes_flat[idx]
        ctrl_map = data[bp]

        for ctrl in ["llm", "pid", "proportional"]:
            steps_data = ctrl_map[ctrl]["steps"]
            xs = [s["step"] for s in steps_data]
            ys = [s["rod_position"] for s in steps_data]
            ax.plot(
                xs, ys,
                color=COLORS[ctrl],
                label=LABELS[ctrl],
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        ax.set_title(bp, fontsize=TITLE_SIZE - 2)
        ax.set_xlabel("Step", fontsize=LABEL_SIZE - 1)
        ax.set_ylabel("Rod Position (cm)", fontsize=LABEL_SIZE - 1)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Control Rod Position per Blueprint — Controller Comparison",
        fontsize=TITLE_SIZE,
        y=1.01,
    )
    plt.tight_layout()
    out_path = out_dir / "rod_position_per_blueprint.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: final deviation comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_final_deviation_comparison(
    data: dict[str, dict[str, dict]], out_dir: Path
) -> None:
    """Grouped bar chart: final keff deviation per blueprint x controller."""
    blueprints = list(data.keys())
    controllers = ["llm", "pid", "proportional"]

    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(10, 6))

    n_bp = len(blueprints)
    n_ctrl = len(controllers)
    bar_width = 0.25
    x = np.arange(n_bp)

    for ci, ctrl in enumerate(controllers):
        deviations = [data[bp][ctrl]["final_deviation"] for bp in blueprints]
        offset = (ci - 1) * bar_width
        bars = ax.bar(
            x + offset,
            deviations,
            width=bar_width,
            color=COLORS[ctrl],
            label=LABELS[ctrl],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="Tolerance (0.05)")
    ax.set_xticks(x)
    ax.set_xticklabels(blueprints, fontsize=TICK_SIZE, rotation=15, ha="right")
    ax.set_ylabel("Final |k$_{eff}$ - 1.0|", fontsize=LABEL_SIZE)
    ax.set_xlabel("Blueprint", fontsize=LABEL_SIZE)
    ax.set_title(
        "Final k$_{eff}$ Deviation by Blueprint and Controller",
        fontsize=TITLE_SIZE,
    )
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = out_dir / "final_deviation_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: convergence summary bar chart
# ---------------------------------------------------------------------------

def plot_convergence_summary(
    data: dict[str, dict[str, dict]], report: dict, out_dir: Path
) -> None:
    """Summary bar chart: convergence rate, avg final deviation, avg convergence step."""
    controllers = ["llm", "pid", "proportional"]
    summary = report["summary"]

    # Compute from valid blueprints subset (data) for accurate numbers
    n_valid = len(data)

    conv_rates: list[float] = []
    avg_devs: list[float] = []
    avg_conv_steps: list[float] = []

    for ctrl in controllers:
        converged_count = sum(
            1 for bp in data if data[bp][ctrl]["converged"]
        )
        conv_rate = (converged_count / n_valid) * 100 if n_valid > 0 else 0.0

        final_devs = [data[bp][ctrl]["final_deviation"] for bp in data]
        avg_dev = float(np.mean(final_devs)) if final_devs else 0.0

        conv_steps = [
            data[bp][ctrl]["converged_at_step"]
            for bp in data
            if data[bp][ctrl]["converged"] and data[bp][ctrl]["converged_at_step"] >= 0
        ]
        avg_step = float(np.mean(conv_steps)) if conv_steps else float("nan")

        conv_rates.append(conv_rate)
        avg_devs.append(avg_dev)
        avg_conv_steps.append(avg_step)

    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(controllers))
    ctrl_labels = [LABELS[c] for c in controllers]
    colors = [COLORS[c] for c in controllers]

    # Convergence rate
    ax = axes[0]
    bars = ax.bar(x, conv_rates, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Convergence Rate (%)", fontsize=LABEL_SIZE)
    ax.set_title("Convergence Rate\n(valid blueprints)", fontsize=TITLE_SIZE - 1)
    ax.set_ylim(0, 110)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, conv_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE,
        )

    # Avg final deviation
    ax = axes[1]
    bars = ax.bar(x, avg_devs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="Tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Avg Final |k$_{eff}$ - 1.0|", fontsize=LABEL_SIZE)
    ax.set_title("Average Final k$_{eff}$ Deviation\n(valid blueprints)", fontsize=TITLE_SIZE - 1)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, avg_devs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE,
        )

    # Avg convergence step
    ax = axes[2]
    step_vals = [s if not np.isnan(s) else 0.0 for s in avg_conv_steps]
    bars = ax.bar(x, step_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Avg Convergence Step", fontsize=LABEL_SIZE)
    ax.set_title("Average Step at First Convergence\n(converged runs only)", fontsize=TITLE_SIZE - 1)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val, raw in zip(bars, step_vals, avg_conv_steps):
        label = f"{val:.1f}" if not np.isnan(raw) else "N/A"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE,
        )

    fig.suptitle(
        "Controller Performance Summary (7 Valid PWR Blueprints)",
        fontsize=TITLE_SIZE,
    )
    plt.tight_layout()
    out_path = out_dir / "convergence_summary.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: keff deviation boxplot
# ---------------------------------------------------------------------------

def plot_keff_deviation_boxplot(
    data: dict[str, dict[str, dict]], out_dir: Path
) -> None:
    """Box plot of final keff deviations per controller."""
    controllers = ["llm", "pid", "proportional"]

    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [
        [data[bp][ctrl]["final_deviation"] for bp in data]
        for ctrl in controllers
    ]
    ctrl_labels = [LABELS[c] for c in controllers]

    bp_plot = ax.boxplot(
        box_data,
        patch_artist=True,
        notch=False,
        widths=0.4,
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, ctrl in zip(bp_plot["boxes"], controllers):
        patch.set_facecolor(COLORS[ctrl])
        patch.set_alpha(0.75)

    for whisker in bp_plot["whiskers"]:
        whisker.set(linewidth=1.2)
    for cap in bp_plot["caps"]:
        cap.set(linewidth=1.2)
    for flier in bp_plot["fliers"]:
        flier.set(marker="o", markersize=5, alpha=0.6)

    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="Tolerance (0.05)")

    ax.set_xticks(range(1, len(controllers) + 1))
    ax.set_xticklabels(ctrl_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Final |k$_{eff}$ - 1.0|", fontsize=LABEL_SIZE)
    ax.set_xlabel("Controller", fontsize=LABEL_SIZE)
    ax.set_title(
        "Distribution of Final k$_{eff}$ Deviations by Controller\n(7 Valid PWR Blueprints)",
        fontsize=TITLE_SIZE,
    )
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    # Manual legend patches
    handles = [
        mpatches.Patch(facecolor=COLORS[c], alpha=0.75, label=LABELS[c])
        for c in controllers
    ]
    handles.append(
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Tolerance")
    )
    ax.legend(handles=handles, fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "keff_deviation_boxplot.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all benchmark plots."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Benchmark report not found: {REPORT_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading report: {REPORT_PATH}")
    report = load_report(REPORT_PATH)

    data = build_data_map(report)
    blueprints = list(data.keys())
    print(f"Valid blueprints ({len(blueprints)}): {blueprints}")

    print("Generating plots...")
    plot_keff_convergence(data, PLOTS_DIR)
    plot_rod_position(data, PLOTS_DIR)
    plot_final_deviation_comparison(data, PLOTS_DIR)
    plot_convergence_summary(data, report, PLOTS_DIR)
    plot_keff_deviation_boxplot(data, PLOTS_DIR)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
