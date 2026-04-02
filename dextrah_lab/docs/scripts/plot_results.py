"""
Thesis result plots — generates all figures for Report/figures/plots/.

Usage:
    python scripts/plot_results.py            # generate all plots
    python scripts/plot_results.py --show     # also display interactively

Each plot function is self-contained and can be called independently.
Data is embedded directly (from experiment logs) to keep this script
standalone — no external data files needed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_style import apply_style, COLORS, save_fig

EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports"

# ============================================================================
# Data from experiment logs (exp_02 through exp_06)
# ============================================================================

# --- Teacher evaluations (480 episodes each) ---
TEACHERS = {
    "Teacher 10\n(ADR-19)":       {"gsr": 96.25, "uer": 53.75},
    "Teacher 11\n(Sim2Real)":     {"gsr": 85.83, "uer": 23.13},
}

# --- All standalone student evaluations (480 episodes unless noted) ---
ALL_RUNS = {
    "labels": ["1c\nSafeDag\nT10", "2a\nDAgger\nT10", "3a\nSafeDag\nT11", "3b\nDAgger\nT11"],
    "gsr":    [36.3,  50.0,  59.0,  57.7],
    "uer":    [67.5,  75.0,  72.7,  70.6],
    "method": ["SafeDAgger", "DAgger", "SafeDAgger", "DAgger"],
    "teacher": ["T10", "T10", "T11", "T11"],
}

# --- Failure breakdown for key runs ---
FAILURE_BREAKDOWN = {
    "categories": ["Physics\ninstab.", "Harmful\ncollision", "Object\nout of bound", "Palm\nflipped"],
    "Teacher 10":    [55.8, 11.6, 32.6, 0.0],
    "Teacher 11":    [47.7,  7.2, 40.5, 2.7],
    "Run 1c (T10)":  [ 0.0, 40.7, 33.3, 1.9],
    "Run 2a (T10)":  [43.3, 45.0, 11.7, 0.0],
    "Run 3a (T11)":  [62.5, 19.5, 17.8, 0.3],
    "Run 3b (T11)":  [65.2, 18.0, 16.8, 0.0],
}

# --- Teacher training progression (exp_02, run1p + extended) ---
# Approximate data points from experiment logs
TEACHER_TRAINING = {
    "epochs":  [0,  2000, 4000, 6000, 8000, 8850, 10000, 11124, 15000, 20000],
    "reward":  [0,  1500, 4000, 7000, 9500, 10500, 11500, 12104, 8500,  5317],
    "adr":     [0,     0,    0,    0,    0,     1,     3,     5,    12,    19],
}

# --- ADR parameter ranges (from env_cfg.py / exp_03) ---
ADR_PARAMS = {
    "Object mass":           (1.0, 0.5, 2.0),  # (start, min, max)
    "Object friction":       (1.0, 0.5, 1.5),
    "Arm stiffness":         (1.0, 0.5, 2.0),
    "Finger MCP stiffness":  (1.0, 0.1, 1.0),
    "Finger PIP stiffness":  (1.0, 0.1, 1.0),
    "Thumb vel. limit\n(rad/s)": (20.0, 0.21, 20.0),
}


# ============================================================================
# Plot functions
# ============================================================================

def plot_all_runs_gsr():
    """Bar chart: standalone GSR for all distillation runs, with teacher ceilings."""
    fig, ax = plt.subplots()

    labels = ALL_RUNS["labels"]
    gsr = ALL_RUNS["gsr"]
    x = np.arange(len(labels))

    colors = [COLORS["blue"] if m == "SafeDAgger" else COLORS["orange"]
              for m in ALL_RUNS["method"]]

    bars = ax.bar(x, gsr, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, gsr):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Teacher ceilings
    ax.axhline(y=96.25, color=COLORS["gray"], linestyle="--", linewidth=0.8,
               alpha=0.6, label="Teacher 10 ceiling (96.3%)")
    ax.axhline(y=85.83, color=COLORS["green"], linestyle="--", linewidth=0.8,
               alpha=0.6, label="Teacher 11 ceiling (85.8%)")

    ax.set_ylabel("Standalone GSR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 105)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["blue"], label="SafeDAgger (L2)"),
        Patch(facecolor=COLORS["orange"], label="DAgger (KL)"),
    ]
    ax.legend(handles=legend_elements + ax.get_lines(), loc="upper left", fontsize=7)
    ax.set_title("Standalone Student Performance Across Runs")

    save_fig(fig, "all_runs_gsr")


def plot_failure_breakdown():
    """Grouped bar chart: failure mode breakdown for teacher 11 vs best students."""
    fig, ax = plt.subplots()

    categories = FAILURE_BREAKDOWN["categories"]
    x = np.arange(len(categories))
    w = 0.2

    keys = ["Teacher 11", "Run 3a (T11)", "Run 3b (T11)", "Run 2a (T10)"]
    colors_list = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["gray"]]
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    for key, color, offset in zip(keys, colors_list, offsets):
        vals = FAILURE_BREAKDOWN[key]
        ax.bar(x + offset, vals, w, label=key, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_ylabel("% of unsafe episodes")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Failure Mode Breakdown")

    save_fig(fig, "failure_breakdown")


def plot_teacher_comparison():
    """Side-by-side bar chart comparing teacher 10 and teacher 11."""
    fig, ax = plt.subplots()

    categories = ["GSR (%)", "UER (%)"]
    t10 = [96.25, 53.75]
    t11 = [85.83, 23.13]

    x = np.arange(len(categories))
    w = 0.3

    bars1 = ax.bar(x - w/2, t10, w, label="Teacher 10 (ADR-19)",
                   color=COLORS["blue"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, t11, w, label="Teacher 11 (Sim2Real, ADR-14)",
                   color=COLORS["green"], edgecolor="white", linewidth=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.set_title("Teacher Policy Comparison")

    save_fig(fig, "teacher_comparison")


def plot_teacher_training_curve():
    """Dual-axis plot: episode reward and ADR level over training epochs."""
    fig, ax1 = plt.subplots()

    epochs = TEACHER_TRAINING["epochs"]
    reward = TEACHER_TRAINING["reward"]
    adr = TEACHER_TRAINING["adr"]

    # Reward on left axis
    ln1 = ax1.plot(epochs, reward, color=COLORS["blue"], label="Mean episode reward")
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Mean episode reward", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax1.set_xlim(0, 21000)

    # ADR on right axis
    ax2 = ax1.twinx()
    ln2 = ax2.plot(epochs, adr, color=COLORS["orange"], linestyle="--", label="ADR level")
    ax2.set_ylabel("ADR level", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.set_ylim(0, 22)

    # Mark key events
    ax1.axvline(x=8850, color=COLORS["gray"], linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.text(9200, max(reward)*0.95, "First ADR\nstep", fontsize=7,
             color=COLORS["gray"], va="top")

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", fontsize=8)

    ax1.set_title("Teacher Policy Training")
    fig.tight_layout()

    save_fig(fig, "teacher_training_curve")


def plot_adr_ranges():
    """Horizontal bar chart showing ADR parameter start vs max ranges."""
    fig, ax = plt.subplots(figsize=(14/2.54, 7/2.54))

    params = list(ADR_PARAMS.keys())
    starts = [v[0] for v in ADR_PARAMS.values()]
    mins = [v[1] for v in ADR_PARAMS.values()]
    maxs = [v[2] for v in ADR_PARAMS.values()]

    y = np.arange(len(params))

    # Draw range bars
    for i in range(len(params)):
        ax.barh(y[i], maxs[i] - mins[i], left=mins[i], height=0.5,
                color=COLORS["blue"], alpha=0.3, edgecolor=COLORS["blue"],
                linewidth=0.8)
        # Mark start value
        ax.plot(starts[i], y[i], "D", color=COLORS["red"], markersize=5,
                zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=7)
    ax.set_xlabel("Parameter value (or scale factor)")
    ax.set_title("ADR Parameter Ranges (start $\\to$ max)")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["blue"], alpha=0.3, edgecolor=COLORS["blue"],
              label="Max ADR range"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["red"],
               markersize=5, label="Start value"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "adr_ranges")


def plot_beta_decay():
    """Line plot showing SafeDAgger beta decay over distillation iterations."""
    fig, ax = plt.subplots()

    # Data points from runs 1a (100k), 1b (100k), 1c (350k)
    iters = [0, 100_000, 100_000, 350_000]
    beta = [1.0, 1.0, 0.875, 0.75]

    # Run 1a (ADR-5)
    ax.plot([0, 100_000], [1.0, 1.0], "o-", color=COLORS["gray"],
            label="Run 1a (ADR-5 teacher)", markersize=5)

    # Run 1b (ADR-19, 100k)
    ax.plot([0, 100_000], [1.0, 0.875], "s-", color=COLORS["blue"],
            label="Run 1b (ADR-19, 100k)", markersize=5)

    # Run 1c (ADR-19, 350k)
    ax.plot([0, 350_000], [1.0, 0.75], "^-", color=COLORS["orange"],
            label="Run 1c (ADR-19, 350k)", markersize=5)

    ax.axhline(y=0.5, color=COLORS["red"], linestyle=":", linewidth=0.8,
               alpha=0.7, label="Student-majority threshold")

    ax.set_xlabel("Distillation iterations")
    ax.set_ylabel(r"$\beta_{\mathrm{safe}}$ (teacher intervention rate)")
    ax.set_xlim(-10_000, 400_000)
    ax.set_ylim(0, 1.1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("SafeDAgger Intervention Rate Decay")

    save_fig(fig, "beta_decay")


def plot_physics_challenges():
    """Timeline/waterfall showing key physics fixes and their impact."""
    fig, ax = plt.subplots(figsize=(14/2.54, 5/2.54))

    fixes = [
        ("Thumb init\nfix", "1a-1b"),
        ("Effort limit\n10→2 Nm", "1l"),
        ("Damping\n3→6", "1l"),
        ("Early term.\npenalty", "1j"),
        ("512 envs\n(from 16)", "1p"),
    ]

    x = np.arange(len(fixes))
    colors_bar = [COLORS["red"], COLORS["red"], COLORS["orange"],
                  COLORS["orange"], COLORS["green"]]

    ax.bar(x, [1]*len(fixes), color=colors_bar, alpha=0.7,
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in fixes], fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel("")

    # Add run labels below
    for i, (_, run) in enumerate(fixes):
        ax.text(i, -0.15, f"Run {run}", ha="center", fontsize=6,
                color=COLORS["gray"])

    ax.set_title("Key Physics Stability Fixes (chronological)")
    ax.set_ylim(-0.3, 1.3)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, "physics_challenges_timeline")


def _load_distillation_metric(run_csv: str, metric: str, num_envs: int,
                               max_iter: int = None, smooth_window: int = 500):
    """Load a metric from an exported distillation CSV, convert steps→iterations, smooth."""
    csv_path = EXPORTS_DIR / "distillation" / run_csv
    df = pd.read_csv(csv_path)
    df = df[df["metric"] == metric].copy()
    df["iteration"] = df["step"] / num_envs
    df = df.sort_values("iteration")
    if max_iter is not None:
        df = df[df["iteration"] <= max_iter]
    # Rolling mean for smoothing (raw data is per-batch, very noisy)
    df["value_smooth"] = df["value"].rolling(smooth_window, min_periods=1, center=True).mean()
    return df


def plot_distillation_lift_success(max_iter: int = 100_000, smooth: int = 500):
    """Lift success (in_success_region) for all distillation runs, 0 to max_iter.

    NOTE: in_success_region during SafeDagger training includes teacher actions
    in unsafe envs. DAgger (vanilla) runs are student-only throughout.
    """
    fig, ax = plt.subplots()

    runs = [
        ("student_run1c_safedagger_l2_teacher10.csv", 16,
         "Run 1c — SafeDAgger (T10)", COLORS["blue"], "-"),
        ("student_run2a_vanilla_kl_teacher10.csv", 16,
         "Run 2a — DAgger (T10)", COLORS["orange"], "-"),
        ("student_run3a_safedagger_l2_teacher11.csv", 24,
         "Run 3a — SafeDAgger (T11)", COLORS["green"], "-"),
        ("student_run3b_vanilla_kl_teacher11.csv", 24,
         "Run 3b — DAgger (T11)", COLORS["red"], "-"),
    ]

    for csv_name, n_envs, label, color, ls in runs:
        df = _load_distillation_metric(csv_name, "in_success_region", n_envs,
                                       max_iter=max_iter, smooth_window=smooth)
        # Convert to percentage
        ax.plot(df["iteration"], df["value_smooth"] * 100,
                color=color, linestyle=ls, label=label, alpha=0.9)

    ax.set_xlabel("Distillation iteration")
    ax.set_ylabel("Lift success (%)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Student Lift Success During Distillation (0–100k)")

    save_fig(fig, "distillation_lift_success_100k")


def plot_distillation_unsafe_rate(max_iter: int = 100_000, smooth: int = 500):
    """Unsafe rate (beta) for all distillation runs, 0 to max_iter.

    beta = fraction of envs where student L2 loss exceeds unsafe_l2_threshold (0.5).
    For SafeDagger, teacher overrides in those envs; for vanilla DAgger, no override.
    """
    fig, ax = plt.subplots()

    runs = [
        ("student_run1c_safedagger_l2_teacher10.csv", 16,
         "Run 1c — SafeDAgger (T10)", COLORS["blue"], "-"),
        ("student_run2a_vanilla_kl_teacher10.csv", 16,
         "Run 2a — DAgger (T10)", COLORS["orange"], "-"),
        ("student_run3a_safedagger_l2_teacher11.csv", 24,
         "Run 3a — SafeDAgger (T11)", COLORS["green"], "-"),
        ("student_run3b_vanilla_kl_teacher11.csv", 24,
         "Run 3b — DAgger (T11)", COLORS["red"], "-"),
    ]

    for csv_name, n_envs, label, color, ls in runs:
        df = _load_distillation_metric(csv_name, "beta", n_envs,
                                       max_iter=max_iter, smooth_window=smooth)
        ax.plot(df["iteration"], df["value_smooth"] * 100,
                color=color, linestyle=ls, label=label, alpha=0.9)

    ax.set_xlabel("Distillation iteration")
    ax.set_ylabel("Unsafe rate (%)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Student Unsafe Rate During Distillation (0–100k)")

    save_fig(fig, "distillation_unsafe_rate_100k")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate thesis result plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    apply_style()

    print("Generating plots...")
    plot_all_runs_gsr()
    plot_failure_breakdown()
    plot_teacher_comparison()
    plot_teacher_training_curve()
    plot_adr_ranges()
    plot_beta_decay()
    plot_physics_challenges()
    plot_distillation_lift_success()
    plot_distillation_unsafe_rate()
    print("Done.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
