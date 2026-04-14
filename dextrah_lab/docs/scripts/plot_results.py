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
        ("student_run4a_safedagger_l2_teacher11.csv", 24,
         "Run 4a — SafeDAgger (T11)", COLORS["cyan"], "--"),
        ("student_run4b_vanilla_kl_teacher11.csv", 24,
         "Run 4b — DAgger (T11)", COLORS["purple"], "--"),
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
        ("student_run4a_safedagger_l2_teacher11.csv", 24,
         "Run 4a — SafeDAgger (T11)", COLORS["cyan"], "--"),
        ("student_run4b_vanilla_kl_teacher11.csv", 24,
         "Run 4b — DAgger (T11)", COLORS["purple"], "--"),
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


def plot_per_object_unsafe_comparison(smooth: int = 2000):
    """Per-object unsafe rate comparison: SafeDagger (run4a) vs DAgger (run4b).

    Uses the final 20k iterations (80k-100k) averaged to get a stable per-object value.
    """
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))

    objects = [
        "basketball_shoe", "chicken_head_in_car", "closed_fist", "elephant_toy",
        "homer", "mario", "milk_pot", "plane", "teddy_bear", "toy_bagger",
        "toy_cow", "train", "tutle_candle_holder",
    ]

    safedagger_vals = []
    dagger_vals = []

    for obj in objects:
        metric = f"per_object_unsafe/{obj}"
        df_sd = _load_distillation_metric(
            "student_run4a_safedagger_l2_teacher11.csv", metric, 24, smooth_window=smooth)
        df_da = _load_distillation_metric(
            "student_run4b_vanilla_kl_teacher11.csv", metric, 24, smooth_window=smooth)
        # Average over last 20k iterations for stable value
        safedagger_vals.append(df_sd[df_sd["iteration"] >= 80000]["value_smooth"].mean() * 100)
        dagger_vals.append(df_da[df_da["iteration"] >= 80000]["value_smooth"].mean() * 100)

    x = np.arange(len(objects))
    w = 0.35
    short_names = [o.replace("_", "\n") for o in objects]

    ax.bar(x - w/2, safedagger_vals, w, label="Run 4a — SafeDAgger", color=COLORS["blue"],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, dagger_vals, w, label="Run 4b — DAgger", color=COLORS["orange"],
           edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Unsafe rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=5, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Per-Object Unsafe Rate (avg 80k–100k iters)")
    fig.tight_layout()

    save_fig(fig, "per_object_unsafe_comparison")


OBJECTS = [
    "basketball_shoe", "chicken_head_in_car", "closed_fist", "elephant_toy",
    "homer", "mario", "milk_pot", "plane", "teddy_bear", "toy_bagger",
    "toy_cow", "train", "tutle_candle_holder",
]


def plot_per_object_lift_comparison(smooth: int = 2000):
    """Per-object lift success comparison: SafeDagger (run4a) vs DAgger (run4b).

    Uses the final 20k iterations (80k-100k) averaged to get a stable per-object value.
    """
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))

    safedagger_vals = []
    dagger_vals = []

    for obj in OBJECTS:
        metric = f"per_object_lift/{obj}"
        df_sd = _load_distillation_metric(
            "student_run4a_safedagger_l2_teacher11.csv", metric, 24, smooth_window=smooth)
        df_da = _load_distillation_metric(
            "student_run4b_vanilla_kl_teacher11.csv", metric, 24, smooth_window=smooth)
        safedagger_vals.append(df_sd[df_sd["iteration"] >= 80000]["value_smooth"].mean() * 100)
        dagger_vals.append(df_da[df_da["iteration"] >= 80000]["value_smooth"].mean() * 100)

    x = np.arange(len(OBJECTS))
    w = 0.35
    short_names = [o.replace("_", "\n") for o in OBJECTS]

    ax.bar(x - w/2, safedagger_vals, w, label="Run 4a — SafeDAgger", color=COLORS["blue"],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, dagger_vals, w, label="Run 4b — DAgger", color=COLORS["orange"],
           edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Lift success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=5, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Per-Object Lift Success (avg 80k–100k iters)")
    fig.tight_layout()

    save_fig(fig, "per_object_lift_comparison")


def plot_per_object_unsafe_over_time(smooth: int = 2000, max_iter: int = 100_000):
    """Per-object unsafe rate over training iterations for run4a (SafeDagger) and run4b (DAgger)."""
    fig, axes = plt.subplots(1, 2, figsize=(14/2.54, 7/2.54), sharey=True)

    runs = [
        ("student_run4a_safedagger_l2_teacher11.csv", 24, "SafeDAgger (Run 4a)", axes[0]),
        ("student_run4b_vanilla_kl_teacher11.csv", 24, "DAgger (Run 4b)", axes[1]),
    ]

    for csv_name, n_envs, title, ax in runs:
        for i, obj in enumerate(OBJECTS):
            color = plt.cm.tab20(i / len(OBJECTS))
            df = _load_distillation_metric(
                csv_name, f"per_object_unsafe/{obj}", n_envs,
                max_iter=max_iter, smooth_window=smooth)
            ax.plot(df["iteration"], df["value_smooth"] * 100,
                    color=color, linewidth=0.8, alpha=0.8,
                    label=obj.replace("_", " "))
        ax.set_xlabel("Iteration")
        ax.set_xlim(0, max_iter)
        ax.set_ylim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Unsafe rate (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=5)
    fig.suptitle("Per-Object Unsafe Rate During Distillation", fontsize=9)
    fig.tight_layout()

    save_fig(fig, "per_object_unsafe_over_time")


def plot_per_object_unsafe_real(eval_dir: Path = None):
    """Per-object unsafe episode rate from standalone eval (real terminations).

    Uses eval JSON files from run4a (SafeDagger) and run4b (DAgger).
    Unsafe = early termination due to: hand too far, object OOB, palm flipped,
    harmful collision, physics instability.
    """
    import json
    if eval_dir is None:
        eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    # run4a = SafeDagger (02-14-30-48), run4b = DAgger (02-14-28-35)
    eval_files = {
        "SafeDAgger (run4a)": eval_dir / "eval_metrics_20260402_181427.json",
        "DAgger (run4b)": eval_dir / "eval_metrics_20260402_181413.json",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14/2.54, 7/2.54))

    for ax_idx, (metric_key, ylabel, title) in enumerate([
        ("unsafe_episode_rate", "Unsafe rate (%)", "Per-Object Unsafe Episode Rate (Eval)"),
        ("lift_success", "Lift success (%)", "Per-Object Lift Success (Eval)"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(OBJECTS))
        w = 0.35
        colors_list = [COLORS["blue"], COLORS["orange"]]

        for i, (label, path) in enumerate(eval_files.items()):
            with open(path) as fh:
                data = json.load(fh)
            vals = [data["per_object_metrics"].get(obj, {}).get(metric_key, 0) * 100
                    for obj in OBJECTS]
            ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=colors_list[i],
                   edgecolor="white", linewidth=0.5)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", "\n") for o in OBJECTS],
                           fontsize=4, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=5, loc="upper right")
        ax.set_title(title, fontsize=7)

    fig.tight_layout()
    save_fig(fig, "per_object_unsafe_real_comparison")


def plot_per_object_unsafe_real_per_run(eval_dir: Path = None):
    """Separate per-object bar charts for run4a and run4b showing lift + unsafe side by side."""
    import json
    if eval_dir is None:
        eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    eval_runs = [
        ("SafeDAgger (run4a)", eval_dir / "eval_metrics_20260402_181427.json",
         COLORS["blue"], COLORS["cyan"]),
        ("DAgger (run4b)", eval_dir / "eval_metrics_20260402_181413.json",
         COLORS["orange"], COLORS["brown"]),
    ]

    n_cols = 4
    n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4.5/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Load all eval data
    eval_data = {}
    for label, path, _, _ in eval_runs:
        with open(path) as fh:
            eval_data[label] = json.load(fh)

    x = np.arange(2)  # two bars per subplot: SafeDagger, DAgger
    w = 0.35

    for idx, obj in enumerate(OBJECTS):
        ax = axes_flat[idx]
        unsafe_vals = []
        lift_vals = []
        for label, _, _, _ in eval_runs:
            m = eval_data[label]["per_object_metrics"].get(obj, {})
            unsafe_vals.append(m.get("unsafe_episode_rate", 0) * 100)
            lift_vals.append(m.get("lift_success", 0) * 100)

        ax.bar(x - w/2, unsafe_vals, w, color=COLORS["red"],
               alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.bar(x + w/2, lift_vals, w, color=COLORS["green"],
               alpha=0.7, edgecolor="white", linewidth=0.5)

        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["SafeD", "DAgger"], fontsize=5)
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=5)

    for idx in range(len(OBJECTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("%", fontsize=7)

    # Legend from first subplot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["red"], alpha=0.6, label="Unsafe rate"),
        Patch(facecolor=COLORS["green"], alpha=0.6, label="Lift success"),
    ]
    axes_flat[0].legend(handles=legend_elements, fontsize=4, loc="upper right")
    fig.suptitle("Per-Object Eval: Unsafe Rate & Lift Success", fontsize=9)
    fig.tight_layout()

    save_fig(fig, "per_object_unsafe_real_per_run")


def _ema(values, alpha: float = 0.99):
    """Exponential moving average (matching TensorBoard smoothing)."""
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * result[i - 1] + (1 - alpha) * values[i]
    return result


def plot_per_object_lift_over_time(smooth: int = 2000, max_iter: int = 100_000):
    """Per-object lift success over training iterations for run4a (SafeDagger) and run4b (DAgger)."""
    fig, axes = plt.subplots(1, 2, figsize=(14/2.54, 7/2.54), sharey=True)

    runs = [
        ("student_run4a_safedagger_l2_teacher11.csv", 24, "SafeDAgger (Run 4a)", axes[0]),
        ("student_run4b_vanilla_kl_teacher11.csv", 24, "DAgger (Run 4b)", axes[1]),
    ]

    for csv_name, n_envs, title, ax in runs:
        for i, obj in enumerate(OBJECTS):
            color = plt.cm.tab20(i / len(OBJECTS))
            df = _load_distillation_metric(
                csv_name, f"per_object_lift/{obj}", n_envs,
                max_iter=max_iter, smooth_window=smooth)
            ax.plot(df["iteration"], df["value_smooth"] * 100,
                    color=color, linewidth=0.8, alpha=0.8,
                    label=obj.replace("_", " "))
        ax.set_xlabel("Iteration")
        ax.set_xlim(0, max_iter)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Lift success (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=5)
    fig.suptitle("Per-Object Lift Success During Distillation", fontsize=9)
    fig.tight_layout()

    save_fig(fig, "per_object_lift_over_time")


def plot_per_object_lift_head_to_head(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """One subplot per object: SafeDagger vs DAgger lift success over time.

    Heavy EMA smoothing (alpha=0.99, matching TensorBoard), with raw data
    shown as a light shaded curve in the background.
    """
    n_cols = 4
    n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    runs = [
        ("student_run4a_safedagger_l2_teacher11.csv", 24,
         "SafeDAgger", COLORS["blue"]),
        ("student_run4b_vanilla_kl_teacher11.csv", 24,
         "DAgger", COLORS["orange"]),
    ]

    for idx, obj in enumerate(OBJECTS):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in runs:
            df = _load_distillation_metric(
                csv_name, f"per_object_lift/{obj}", n_envs,
                max_iter=max_iter, smooth_window=1)  # no rolling, use EMA
            raw = df["value"].values * 100
            smoothed = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values

            # Downsample smoothed curve to ~500 points for clean rendering
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)

        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)

    # Hide unused subplots
    for idx in range(len(OBJECTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared labels
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Lift (%)", fontsize=7)

    # Single legend from first subplot
    axes_flat[0].legend(fontsize=5, loc="lower right")
    fig.suptitle("Per-Object Lift Success: SafeDAgger vs DAgger (0–100k)", fontsize=9)
    fig.tight_layout()

    save_fig(fig, "per_object_lift_head_to_head")


TEACHER_11_LIFT = 0.8583  # Teacher 11 standalone eval lift success (480 episodes)

DISTILLATION_RUNS_4 = [
    ("student_run4a_safedagger_l2_teacher11.csv", 24,
     "SafeDAgger", COLORS["blue"]),
    ("student_run4b_vanilla_kl_teacher11.csv", 24,
     "DAgger", COLORS["orange"]),
]


def _plot_distillation_single(metric, title, ylabel, ylim, runs, max_iter,
                               ema_alpha, normalize_by=None, beta_dagger_zero=False):
    """Helper: single distillation comparison plot with EMA + shaded band."""
    fig, ax = plt.subplots()

    for csv_name, n_envs, label, color in runs:
        is_dagger = "vanilla" in csv_name
        if beta_dagger_zero and is_dagger:
            ax.axhline(y=0, color=color, linewidth=1.5, alpha=0.9, label=label)
            continue
        df = _load_distillation_metric(csv_name, metric, n_envs,
                                       max_iter=max_iter, smooth_window=1)
        if df.empty:
            continue
        raw = df["value"].values
        if normalize_by is not None:
            raw = raw / normalize_by
        smoothed = _ema(raw, alpha=ema_alpha)
        iters = df["iteration"].values

        raw_s = pd.Series(raw)
        lo = raw_s.rolling(2000, min_periods=100, center=True).quantile(0.1).values
        hi = raw_s.rolling(2000, min_periods=100, center=True).quantile(0.9).values
        step = max(1, len(iters) // 500)
        if not beta_dagger_zero:
            ax.fill_between(iters[::step], lo[::step], hi[::step],
                            color=color, alpha=0.12, linewidth=0)
        ax.plot(iters[::step], smoothed[::step], color=color,
                linewidth=1.5, alpha=0.95, label=label)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"Training Iteration ($\times 10^4$)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_distillation_comparison(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Three separate plots (for LaTeX): lift success (normed to teacher),
    intervention rate beta, unsafe episode rate."""
    runs = DISTILLATION_RUNS_4

    # 1. Lift success normalized to teacher ceiling
    fig = _plot_distillation_single(
        "in_success_region", "Student Policy Lift Success",
        "Lift Success (normed to teacher)", (0, 1.05), runs, max_iter, ema_alpha,
        normalize_by=TEACHER_11_LIFT)
    save_fig(fig, "distillation_lift_success_normed")

    # 2. Intervention rate beta
    fig = _plot_distillation_single(
        "beta", "Intervention Rate Beta",
        r"$\beta$ (teacher intervention rate)", (0, 1.05), runs, max_iter, ema_alpha,
        beta_dagger_zero=True)
    save_fig(fig, "distillation_intervention_rate")

    # 3. Unsafe episode rate
    fig = _plot_distillation_single(
        "termination/real_unsafe", "Unsafe Episode Rate",
        "Unsafe rate", (0, 0.15), runs, max_iter, ema_alpha)
    save_fig(fig, "distillation_unsafe_episode_rate")


DISTILLATION_RUNS_5 = [
    ("student_run5a_safedagger_l2_teacher11.csv", 24,
     "SafeDAgger", COLORS["blue"]),
    ("student_run5b_vanilla_kl_teacher11.csv", 24,
     "DAgger", COLORS["orange"]),
]

DISTILLATION_RUNS_6 = [
    ("student_run6a_safedagger_l2_teacher11.csv", 24,
     "SafeDAgger", COLORS["blue"]),
    ("student_run6b_dagger_l2_teacher11.csv", 24,
     "DAgger", COLORS["orange"]),
]

DISTILLATION_RUNS_7 = [
    ("student_run7a_safedagger_l2_teacher11.csv", 24,
     "SafeDAgger", COLORS["blue"]),
    ("student_run7b_dagger_l2_teacher11.csv", 24,
     "DAgger", COLORS["orange"]),
]

OBJECTS_TOP8 = [
    "basketball_shoe", "closed_fist", "elephant_toy", "mario",
    "milk_pot", "teddy_bear", "toy_bagger", "tutle_candle_holder",
]


def plot_distillation_comparison_run6(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Three separate plots for run6a/6b: scaled L2 threshold + episode-level metrics."""
    runs = DISTILLATION_RUNS_6

    # 1. Lift success normalized to teacher ceiling
    fig = _plot_distillation_single(
        "in_success_region", "Student Policy Lift Success (Run 6)",
        "Lift Success (normed to teacher)", (0, 1.05), runs, max_iter, ema_alpha,
        normalize_by=TEACHER_11_LIFT)
    save_fig(fig, "run6_lift_success_normed")

    # 2. Intervention rate beta
    fig = _plot_distillation_single(
        "beta", "Intervention Rate Beta (Run 6)",
        r"$\beta$ (teacher intervention rate)", (0, 1.05), runs, max_iter, ema_alpha,
        beta_dagger_zero=True)
    save_fig(fig, "run6_intervention_rate")

    # 3. Episode-level unsafe rate (AverageMeter)
    fig = _plot_distillation_single(
        "train/avg/unsafe_episode_rate", "Unsafe Episode Rate (Run 6)",
        "Unsafe episode rate", (0, 1.0), runs, max_iter, ema_alpha)
    save_fig(fig, "run6_unsafe_episode_rate")


def plot_per_object_lifted_head_to_head_run6(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object lifted (above table) head-to-head for run6."""
    n_cols = 4
    n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, obj in enumerate(OBJECTS):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in DISTILLATION_RUNS_6:
            df = _load_distillation_metric(
                csv_name, f"per_object_lifted/{obj}", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df.empty:
                continue
            raw = df["value"].values * 100
            smoothed = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)

        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)

    for idx in range(len(OBJECTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Lifted (%)", fontsize=7)
    axes_flat[0].legend(fontsize=5, loc="lower right")
    fig.suptitle("Per-Object Lift (above table): SafeDAgger vs DAgger (Run 6)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "run6_per_object_lifted_head_to_head")


def plot_per_object_unsafe_episode_head_to_head_run6(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object episode-level unsafe rate head-to-head for run6."""
    n_cols = 4
    n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, obj in enumerate(OBJECTS):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in DISTILLATION_RUNS_6:
            df = _load_distillation_metric(
                csv_name, f"train/{obj}/unsafe_episode_rate", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df.empty:
                continue
            raw = df["value"].values * 100
            smoothed = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)

        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)

    for idx in range(len(OBJECTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Unsafe (%)", fontsize=7)
    axes_flat[0].legend(fontsize=5, loc="upper right")
    fig.suptitle("Per-Object Unsafe Episode Rate: SafeDAgger vs DAgger (Run 6)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "run6_per_object_unsafe_episode_head_to_head")


UNSAFE_REASONS = ["object_out_of_bound", "harmful_collision", "palm_flipped", "physics_instability"]
REASON_COLORS = {
    "object_out_of_bound": COLORS["orange"],
    "harmful_collision": COLORS["red"],
    "palm_flipped": COLORS["purple"],
    "physics_instability": COLORS["gray"],
}
REASON_LABELS = {
    "object_out_of_bound": "Object OOB",
    "harmful_collision": "Collision",
    "palm_flipped": "Palm flip",
    "physics_instability": "Physics",
}


def plot_per_object_failure_mode_run6(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object failure mode breakdown over training for run6a (SafeDagger) and run6b (DAgger).
    One subplot per object, stacked reason lines.
    """
    for csv_name, n_envs, method, _ in DISTILLATION_RUNS_6:
        n_cols = 4
        n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                                 sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for idx, obj in enumerate(OBJECTS):
            ax = axes_flat[idx]
            for reason in UNSAFE_REASONS:
                metric = f"train/{obj}/unsafe_reason_prop/{reason}"
                df = _load_distillation_metric(
                    csv_name, metric, n_envs,
                    max_iter=max_iter, smooth_window=1)
                if df.empty:
                    continue
                raw = df["value"].values * 100
                smoothed = _ema(raw, alpha=ema_alpha)
                iters = df["iteration"].values
                step = max(1, len(iters) // 500)
                ax.plot(iters[::step], smoothed[::step],
                        color=REASON_COLORS[reason], linewidth=1.0, alpha=0.9,
                        label=REASON_LABELS[reason])

            ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
            ax.set_ylim(0, 30)
            ax.set_xlim(0, max_iter)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            ax.tick_params(labelsize=5)

        for idx in range(len(OBJECTS), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Iteration", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("Rate (%)", fontsize=7)
        axes_flat[0].legend(fontsize=4, loc="upper right")
        tag = "safedagger" if "SafeD" in method else "dagger"
        fig.suptitle(f"Per-Object Failure Modes — {method} (Run 6)", fontsize=9)
        fig.tight_layout()
        save_fig(fig, f"run6_per_object_failure_modes_{tag}")


def _compute_global_lifted(run_csv: str, n_envs: int, objects: list,
                            max_iter: int = None, ema_alpha: float = 0.999):
    """Compute global lifted rate by averaging per-object lifted metrics."""
    dfs = []
    for obj in objects:
        df = _load_distillation_metric(run_csv, f"per_object_lifted/{obj}", n_envs,
                                       max_iter=max_iter, smooth_window=1)
        if not df.empty:
            dfs.append(df[["iteration", "value"]].rename(columns={"value": obj}))
    if not dfs:
        return None, None
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge_asof(merged, d, on="iteration", tolerance=2)
    obj_cols = [c for c in merged.columns if c != "iteration"]
    merged["avg_lifted"] = merged[obj_cols].mean(axis=1)
    smoothed = _ema(merged["avg_lifted"].values, alpha=ema_alpha)
    return merged["iteration"].values, smoothed


def plot_distillation_comparison_run7(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Three separate plots for run7a/7b: corrected threshold + visdex_top8."""
    runs = DISTILLATION_RUNS_7

    # 1. Lift success using per_object_lifted (above table, less strict)
    fig, ax = plt.subplots()
    for csv_name, n_envs, label, color in runs:
        iters, smoothed = _compute_global_lifted(
            csv_name, n_envs, OBJECTS_TOP8, max_iter=max_iter, ema_alpha=ema_alpha)
        if iters is not None:
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step] / TEACHER_11_LIFT,
                    color=color, linewidth=1.5, alpha=0.95, label=label)
    ax.set_title("Student Policy Lift Success (Run 7)")
    ax.set_ylabel("Lift Success (normed to teacher)")
    ax.set_xlabel(r"Training Iteration ($\times 10^4$)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}"))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "run7_lift_success_normed")

    fig = _plot_distillation_single(
        "beta", "Intervention Rate Beta (Run 7)",
        r"$\beta$ (teacher intervention rate)", (0, 1.05), runs, max_iter, ema_alpha,
        beta_dagger_zero=True)
    save_fig(fig, "run7_intervention_rate")

    fig = _plot_distillation_single(
        "train/avg/unsafe_episode_rate", "Unsafe Episode Rate (Run 7)",
        "Unsafe episode rate", (0, 1.0), runs, max_iter, ema_alpha)
    save_fig(fig, "run7_unsafe_episode_rate")


def plot_per_object_lifted_head_to_head_run7(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object lifted head-to-head for run7 (visdex_top8)."""
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, obj in enumerate(OBJECTS_TOP8):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in DISTILLATION_RUNS_7:
            df = _load_distillation_metric(
                csv_name, f"per_object_lifted/{obj}", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df.empty:
                continue
            raw = df["value"].values * 100
            smoothed = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)
        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Lifted (%)", fontsize=7)
    axes_flat[0].legend(fontsize=5, loc="lower right")
    fig.suptitle("Per-Object Lift (above table): SafeDAgger vs DAgger (Run 7, top8)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "run7_per_object_lifted_head_to_head")


def plot_per_object_failure_mode_run7(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object failure modes for run7 (visdex_top8)."""
    for csv_name, n_envs, method, _ in DISTILLATION_RUNS_7:
        n_cols = 4
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                                 sharex=True, sharey=True)
        axes_flat = axes.flatten()
        for idx, obj in enumerate(OBJECTS_TOP8):
            ax = axes_flat[idx]
            for reason in UNSAFE_REASONS:
                metric = f"train/{obj}/unsafe_reason_prop/{reason}"
                df = _load_distillation_metric(
                    csv_name, metric, n_envs,
                    max_iter=max_iter, smooth_window=1)
                if df.empty:
                    continue
                raw = df["value"].values * 100
                smoothed = _ema(raw, alpha=ema_alpha)
                iters = df["iteration"].values
                step = max(1, len(iters) // 500)
                ax.plot(iters[::step], smoothed[::step],
                        color=REASON_COLORS[reason], linewidth=1.0, alpha=0.9,
                        label=REASON_LABELS[reason])
            ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
            ax.set_ylim(0, 30)
            ax.set_xlim(0, max_iter)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            ax.tick_params(labelsize=5)
        for ax in axes[-1, :]:
            ax.set_xlabel("Iteration", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("Rate (%)", fontsize=7)
        axes_flat[0].legend(fontsize=4, loc="upper right")
        tag = "safedagger" if "SafeD" in method else "dagger"
        fig.suptitle(f"Per-Object Failure Modes — {method} (Run 7, top8)", fontsize=9)
        fig.tight_layout()
        save_fig(fig, f"run7_per_object_failure_modes_{tag}")


DISTILLATION_RUNS_8 = [
    ("student_run8a_safedagger_l2_teacher11.csv", 24,
     "SafeDAgger", COLORS["blue"]),
    ("student_run8b_dagger_l2_teacher11.csv", 24,
     "DAgger", COLORS["orange"]),
]

RUN8_EVAL_FILES = {
    "SafeDAgger (run8a)": "eval_metrics_20260404_130619.json",
    "DAgger (run8b)": "eval_metrics_20260404_130532.json",
}

DISTILLATION_RUNS_9 = [
    ("student_run9a_safedagger_l2_teacher11.csv", 24,
     "SafeDAgger", COLORS["blue"]),
    ("student_run9b_dagger_l2_teacher11.csv", 24,
     "DAgger", COLORS["orange"]),
    ("student_run9c_bc_teacher11.csv", 32,
     "BC", COLORS["green"]),
]

RUN9_EVAL_FILES = {
    "SafeDAgger (run9a)": "eval_metrics_20260404_201637.json",
    "DAgger (run9b)": "eval_metrics_20260404_201628.json",
    "BC (run9c)": "eval_metrics_20260413_233515.json",
}

DISTILLATION_RUNS_10 = [
    ("student_run10a_safedagger_arm_rand.csv", 24,
     "SafeD + arm rand", COLORS["blue"]),
    ("student_run10b_safedagger_32envs.csv", 32,
     "SafeD + 32 envs", COLORS["green"]),
    ("student_run9a_safedagger_l2_teacher11.csv", 24,
     "SafeD baseline (run9a)", COLORS["gray"]),
]

RUN10_EVAL_FILES = {
    "arm rand (run10a)": "eval_metrics_20260405_224028.json",
    "32 envs (run10b)": "eval_metrics_20260405_224012.json",
}

DISTILLATION_RUNS_11 = [
    ("student_run11a_safedagger_adr5.csv", 32,
     "SafeDAgger + ADR5", COLORS["blue"]),
    ("student_run11b_dagger_adr5.csv", 32,
     "DAgger + ADR5", COLORS["orange"]),
]

RUN11_EVAL_FILES = {
    "SafeDAgger (run11a)": "eval_metrics_20260407_084953.json",
    "DAgger (run11b)": "eval_metrics_20260407_084926.json",
}


def plot_run(run_name, runs, objects, eval_files=None,
             max_iter=100_000, ema_alpha=0.999, failure_mode_ylim=30):
    """Generate all plots for a given run into a subdirectory."""
    subdir = run_name
    eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    # 1. Lift success normed (using per_object_lifted)
    fig, ax = plt.subplots()
    for csv_name, n_envs, label, color in runs:
        iters, smoothed = _compute_global_lifted(
            csv_name, n_envs, objects, max_iter=max_iter, ema_alpha=ema_alpha)
        if iters is not None:
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step] / TEACHER_11_LIFT,
                    color=color, linewidth=1.5, alpha=0.95, label=label)
    ax.set_title("Student Policy Lift Success")
    ax.set_ylabel("Lift Success (normed to teacher)")
    ax.set_xlabel(r"Training Iteration ($\times 10^4$)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}"))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "lift_success_normed", subdir=subdir)

    # 2. Intervention rate beta
    fig = _plot_distillation_single(
        "beta", "Intervention Rate Beta",
        r"$\beta$ (teacher intervention rate)", (0, 1.05), runs, max_iter, ema_alpha,
        beta_dagger_zero=True)
    save_fig(fig, "intervention_rate", subdir=subdir)

    # 3. Unsafe episode rate (episode-level, excluding physics instabilities)
    fig, ax = plt.subplots()
    for csv_name, n_envs, label, color in runs:
        is_dagger = "vanilla" in csv_name
        df_total = _load_distillation_metric(csv_name, "train/avg/unsafe_episode_rate", n_envs,
                                              max_iter=max_iter, smooth_window=1)
        df_phys = _load_distillation_metric(csv_name, "train/avg/unsafe_reason_prop/physics_instability", n_envs,
                                             max_iter=max_iter, smooth_window=1)
        if df_total.empty:
            continue
        total_raw = df_total["value"].values
        if not df_phys.empty and len(df_phys) == len(df_total):
            phys_raw = df_phys["value"].values
            real_raw = np.clip(total_raw - phys_raw, 0, 1)
        else:
            real_raw = total_raw
        smoothed = _ema(real_raw, alpha=ema_alpha)
        iters = df_total["iteration"].values
        step = max(1, len(iters) // 500)
        ax.plot(iters[::step], smoothed[::step], color=color,
                linewidth=1.5, alpha=0.95, label=label)
    ax.set_title("Real Unsafe Episode Rate (excl. physics)")
    ax.set_ylabel("Unsafe episode rate")
    ax.set_xlabel(r"Training Iteration ($\times 10^4$)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "unsafe_episode_rate", subdir=subdir)

    # 4. Per-object lifted head-to-head
    n_cols = 4
    n_rows = (len(objects) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for idx, obj in enumerate(objects):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in runs:
            df = _load_distillation_metric(
                csv_name, f"per_object_lifted/{obj}", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df.empty:
                continue
            raw = df["value"].values * 100
            smoothed_v = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed_v[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)
        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)
    for idx in range(len(objects), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Lifted (%)", fontsize=7)
    axes_flat[0].legend(fontsize=5, loc="lower right")
    method_list = " vs ".join(m for _, _, m, _ in runs)
    fig.suptitle(f"Per-Object Lift (above table): {method_list}", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "per_object_lifted_head_to_head", subdir=subdir)

    # 5. Per-object unsafe episode rate head-to-head (excluding physics instabilities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for idx, obj in enumerate(objects):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in runs:
            df_total = _load_distillation_metric(
                csv_name, f"train/{obj}/unsafe_episode_rate", n_envs,
                max_iter=max_iter, smooth_window=1)
            df_phys = _load_distillation_metric(
                csv_name, f"train/{obj}/unsafe_reason_prop/physics_instability", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df_total.empty:
                continue
            total_raw = df_total["value"].values * 100
            if not df_phys.empty and len(df_phys) == len(df_total):
                phys_raw = df_phys["value"].values * 100
                real_raw = np.clip(total_raw - phys_raw, 0, 100)
            else:
                real_raw = total_raw
            smoothed_v = _ema(real_raw, alpha=ema_alpha)
            iters = df_total["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed_v[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)
        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)
    for idx in range(len(objects), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Real unsafe (%)", fontsize=7)
    axes_flat[0].legend(fontsize=5, loc="upper right")
    method_list = " vs ".join(m for _, _, m, _ in runs)
    fig.suptitle(f"Per-Object Real Unsafe Rate (excl. physics): {method_list}", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "per_object_unsafe_episode_head_to_head", subdir=subdir)

    # 6. Per-object failure modes (one plot per method)
    for csv_name, n_envs, method, _ in runs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                                 sharex=True, sharey=True)
        axes_flat = axes.flatten()
        for idx, obj in enumerate(objects):
            ax = axes_flat[idx]
            for reason in UNSAFE_REASONS:
                metric = f"train/{obj}/unsafe_reason_prop/{reason}"
                df = _load_distillation_metric(
                    csv_name, metric, n_envs,
                    max_iter=max_iter, smooth_window=1)
                if df.empty:
                    continue
                raw = df["value"].values * 100
                smoothed_v = _ema(raw, alpha=ema_alpha)
                iters = df["iteration"].values
                step = max(1, len(iters) // 500)
                ax.plot(iters[::step], smoothed_v[::step],
                        color=REASON_COLORS[reason], linewidth=1.0, alpha=0.9,
                        label=REASON_LABELS[reason])
            ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
            ax.set_ylim(0, failure_mode_ylim)
            ax.set_xlim(0, max_iter)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            ax.tick_params(labelsize=5)
        for idx in range(len(objects), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Iteration", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("Rate (%)", fontsize=7)
        axes_flat[0].legend(fontsize=4, loc="upper right")
        if "SafeD" in method:
            tag = "safedagger"
        elif "BC" in method:
            tag = "bc"
        else:
            tag = "dagger"
        fig.suptitle(f"Per-Object Failure Modes — {method}", fontsize=9)
        fig.tight_layout()
        save_fig(fig, f"per_object_failure_modes_{tag}", subdir=subdir)

    # 7. Eval per-object bar charts (if eval files provided)
    if eval_files:
        import json

        # Helper to get per-object real unsafe (excluding physics instability)
        def _get_real_unsafe(per_obj_data, obj):
            m = per_obj_data.get(obj, {})
            total = m.get("unsafe_episode_rate", 0)
            reasons = m.get("unsafe_reason_prop", {})
            phys = reasons.get("physics_instability", 0)
            return total - phys  # reason_prop is fraction of all episodes

        for metric_key, ylabel, title, fname, use_real_unsafe in [
            ("lift_success", "Lift success (%)", "Per-Object Lift Success — Eval", "per_object_lift_eval", False),
            ("unsafe_episode_rate", "Real unsafe rate (%)", "Per-Object Real Unsafe Rate — Eval (excl. physics)", "per_object_unsafe_eval", True),
        ]:
            fig, ax = plt.subplots(figsize=(14/2.54, 7/2.54))
            x = np.arange(len(objects))
            n_methods = len(eval_files)
            w = 0.8 / n_methods
            colors_list = [COLORS["blue"], COLORS["orange"], COLORS["green"],
                           COLORS["red"], COLORS["purple"], COLORS["gray"]]
            offset_start = -(n_methods - 1) / 2
            for i, (label, eval_fname) in enumerate(eval_files.items()):
                path = eval_dir / eval_fname
                if not path.exists():
                    continue
                with open(path) as fh:
                    data = json.load(fh)
                per_obj = data.get("per_object_metrics", {})
                if use_real_unsafe:
                    vals = [_get_real_unsafe(per_obj, obj) * 100 for obj in objects]
                else:
                    vals = [per_obj.get(obj, {}).get(metric_key, 0) * 100 for obj in objects]
                bars = ax.bar(x + (offset_start + i) * w, vals, w, label=label,
                              color=colors_list[i % len(colors_list)], edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, vals):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                                f"{val:.0f}", ha="center", va="bottom", fontsize=4)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([o.replace("_", " ") for o in objects],
                               fontsize=5, rotation=45, ha="right")
            ax.set_ylim(0, 105)
            ax.legend(fontsize=7)
            ax.set_title(title, fontsize=8)
            fig.tight_layout()
            save_fig(fig, fname, subdir=subdir)


def plot_run9_single_object_single_reason(
    obj: str = "basketball_shoe",
    reason: str = "object_out_of_bound",
    max_iter: int = 100_000,
    ema_alpha: float = 0.999,
    ylim: float = 60,
):
    """Single-object, single-reason head-to-head comparison for run9 (SafeDAgger vs DAgger)."""
    runs = DISTILLATION_RUNS_9
    subdir = "run9/single_object_reason"

    fig, ax = plt.subplots()
    for csv_name, n_envs, method, color in runs:
        metric = f"train/{obj}/unsafe_reason_prop/{reason}"
        df = _load_distillation_metric(
            csv_name, metric, n_envs,
            max_iter=max_iter, smooth_window=1)
        if df.empty:
            continue
        raw = df["value"].values * 100
        smoothed = _ema(raw, alpha=ema_alpha)
        iters = df["iteration"].values
        step = max(1, len(iters) // 500)
        ax.plot(iters[::step], smoothed[::step], color=color,
                linewidth=1.5, alpha=0.95, label=method)

    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, ylim)
    ax.set_xlabel(r"Training Iteration ($\times 10^4$)")
    ax.set_ylabel(f"{REASON_LABELS.get(reason, reason)} rate (%)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}"))
    obj_pretty = obj.replace("_", " ")
    reason_pretty = REASON_LABELS.get(reason, reason).lower()
    ax.set_title(f"{obj_pretty}: {reason_pretty} — SafeDAgger vs DAgger", fontsize=8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_fig(fig, f"single_{obj}_{reason}", subdir=subdir)


def plot_run11_unsafe(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object UER and per-object failure mode plots for run 11 (SafeDAgger vs DAgger)."""
    runs = DISTILLATION_RUNS_11
    objects = OBJECTS_TOP8
    subdir = "run11"
    n_cols = 4
    n_rows = (len(objects) + n_cols - 1) // n_cols

    # 1. Per-object unsafe episode rate head-to-head (excluding physics instabilities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for idx, obj in enumerate(objects):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in runs:
            df_total = _load_distillation_metric(
                csv_name, f"train/{obj}/unsafe_episode_rate", n_envs,
                max_iter=max_iter, smooth_window=1)
            df_phys = _load_distillation_metric(
                csv_name, f"train/{obj}/unsafe_reason_prop/physics_instability", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df_total.empty:
                continue
            total_raw = df_total["value"].values * 100
            if not df_phys.empty and len(df_phys) == len(df_total):
                phys_raw = df_phys["value"].values * 100
                real_raw = np.clip(total_raw - phys_raw, 0, 100)
            else:
                real_raw = total_raw
            smoothed_v = _ema(real_raw, alpha=ema_alpha)
            iters = df_total["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed_v[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)
        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)
    for idx in range(len(objects), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Real unsafe (%)", fontsize=7)
    axes_flat[0].legend(fontsize=5, loc="upper right")
    fig.suptitle("Per-Object Real Unsafe Rate (excl. physics): SafeDAgger vs DAgger", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "per_object_unsafe_episode_head_to_head", subdir=subdir)

    # 2. Per-object failure modes (one plot per method)
    for csv_name, n_envs, method, _ in runs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                                 sharex=True, sharey=True)
        axes_flat = axes.flatten()
        for idx, obj in enumerate(objects):
            ax = axes_flat[idx]
            for reason in UNSAFE_REASONS:
                metric = f"train/{obj}/unsafe_reason_prop/{reason}"
                df = _load_distillation_metric(
                    csv_name, metric, n_envs,
                    max_iter=max_iter, smooth_window=1)
                if df.empty:
                    continue
                raw = df["value"].values * 100
                smoothed_v = _ema(raw, alpha=ema_alpha)
                iters = df["iteration"].values
                step = max(1, len(iters) // 500)
                ax.plot(iters[::step], smoothed_v[::step],
                        color=REASON_COLORS[reason], linewidth=1.0, alpha=0.9,
                        label=REASON_LABELS[reason])
            ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
            ax.set_ylim(0, 60)
            ax.set_xlim(0, max_iter)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            ax.tick_params(labelsize=5)
        for idx in range(len(objects), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Iteration", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("Rate (%)", fontsize=7)
        axes_flat[0].legend(fontsize=4, loc="upper right")
        if "SafeD" in method:
            tag = "safedagger"
        elif "BC" in method:
            tag = "bc"
        else:
            tag = "dagger"
        fig.suptitle(f"Per-Object Failure Modes — {method}", fontsize=9)
        fig.tight_layout()
        save_fig(fig, f"per_object_failure_modes_{tag}", subdir=subdir)


def plot_run10b_vs_run11_comparison(eval_dir: Path = None):
    """Side-by-side per-object lift comparison: run10b (no ADR) vs run11a (SafeD ADR5) vs run11b (DAgger ADR5)."""
    import json
    if eval_dir is None:
        eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    runs = [
        ("run10b SafeD (no ADR)", "eval_metrics_20260405_224012.json", COLORS["green"]),
        ("run11a SafeD + ADR5", "eval_metrics_20260407_084953.json", COLORS["blue"]),
        ("run11b DAgger + ADR5", "eval_metrics_20260407_084926.json", COLORS["orange"]),
    ]

    fig, ax = plt.subplots(figsize=(14/2.54, 7/2.54))
    x = np.arange(len(OBJECTS_TOP8))
    w = 0.27

    for i, (label, fname, color) in enumerate(runs):
        with open(eval_dir / fname) as fh:
            d = json.load(fh)
        per_obj = d.get("per_object_metrics", {})
        vals = [per_obj.get(obj, {}).get("lift_success", 0) * 100 for obj in OBJECTS_TOP8]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color,
                      edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=4)

    ax.set_ylabel("Lift success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", " ") for o in OBJECTS_TOP8],
                       fontsize=5, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Per-Object Lift Success: Best (run10b) vs ADR 5 ablation (run11a/b)", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "run10b_vs_run11_per_object_lift", subdir="comparisons")


def plot_physics_vs_real_unsafe_breakdown(eval_dir: Path = None):
    """Stacked bar showing physics vs real unsafe contribution for run10b/11a/11b."""
    import json
    if eval_dir is None:
        eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    runs = [
        ("run10b SafeD\n(no ADR)", "eval_metrics_20260405_224012.json"),
        ("run11a SafeD\n+ ADR5", "eval_metrics_20260407_084953.json"),
        ("run11b DAgger\n+ ADR5", "eval_metrics_20260407_084926.json"),
    ]

    labels, real_vals, phys_vals, lift_vals = [], [], [], []
    for label, fname in runs:
        with open(eval_dir / fname) as fh:
            d = json.load(fh)
        m = d["metrics"]
        uer = m["eval/unsafe_episode_rate"]
        reasons = m.get("eval/out_of_reach_reason_pct", {})
        phys_pct_of_unsafe = reasons.get("physics_instability", 0) / 100
        phys = uer * phys_pct_of_unsafe
        real = uer - phys
        labels.append(label)
        real_vals.append(real * 100)
        phys_vals.append(phys * 100)
        lift_vals.append(m["eval/lift_success"] * 100)

    fig, ax = plt.subplots(figsize=(12/2.54, 7/2.54))
    x = np.arange(len(labels))
    w = 0.35

    # Stacked unsafe bars
    b1 = ax.bar(x - w/2, real_vals, w, label="Real unsafe", color=COLORS["red"], alpha=0.85)
    b2 = ax.bar(x - w/2, phys_vals, w, bottom=real_vals, label="Physics instab.",
                color=COLORS["gray"], alpha=0.7)
    # Lift bars
    b3 = ax.bar(x + w/2, lift_vals, w, label="Lift success", color=COLORS["green"], alpha=0.85)

    for i, (r, p, l) in enumerate(zip(real_vals, phys_vals, lift_vals)):
        ax.text(x[i] - w/2, r + p + 1, f"{r+p:.0f}", ha="center", fontsize=6)
        ax.text(x[i] + w/2, l + 1, f"{l:.0f}", ha="center", fontsize=6)

    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Unsafe Breakdown: Physics vs Real (eval)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "physics_vs_real_unsafe_breakdown", subdir="comparisons")


def plot_object_stability_across_runs(eval_dir: Path = None):
    """Per-object lift across all major runs to show object-level stability/sensitivity."""
    import json
    if eval_dir is None:
        eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    runs = [
        ("run9a SafeD", "eval_metrics_20260404_201637.json"),
        ("run9b DAgger", "eval_metrics_20260404_201628.json"),
        ("run10b SafeD 32env", "eval_metrics_20260405_224012.json"),
        ("run11a SafeD ADR5", "eval_metrics_20260407_084953.json"),
        ("run11b DAgger ADR5", "eval_metrics_20260407_084926.json"),
    ]

    obj_data = {obj: [] for obj in OBJECTS_TOP8}
    for label, fname in runs:
        with open(eval_dir / fname) as fh:
            d = json.load(fh)
        per_obj = d.get("per_object_metrics", {})
        for obj in OBJECTS_TOP8:
            obj_data[obj].append(per_obj.get(obj, {}).get("lift_success", 0) * 100)

    # Compute stability score = std deviation across runs (lower = more stable)
    obj_stability = {obj: (np.mean(vals), np.std(vals)) for obj, vals in obj_data.items()}
    sorted_objs = sorted(obj_stability.items(), key=lambda x: x[1][1])  # sort by std

    fig, ax = plt.subplots(figsize=(14/2.54, 7/2.54))
    x = np.arange(len(sorted_objs))
    means = [s[1][0] for s in sorted_objs]
    stds = [s[1][1] for s in sorted_objs]
    names = [s[0].replace("_", " ") for s in sorted_objs]

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=COLORS["blue"], alpha=0.7,
                  edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 1.5,
                f"σ={s:.0f}", ha="center", fontsize=5)

    ax.set_ylabel("Lift success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    ax.set_ylim(0, 110)
    ax.set_title("Per-Object Stability Across Runs (sorted by std, lowest=most stable)", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "object_stability_across_runs", subdir="comparisons")


def plot_failure_mode_detail(objects=("basketball_shoe", "teddy_bear"),
                            max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Detailed failure mode plot for selected objects — SafeDagger vs DAgger side by side.
    Each row is one object, columns are SafeDagger and DAgger.
    """
    n_rows = len(objects)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14/2.54, n_rows * 5/2.54),
                             sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, obj in enumerate(objects):
        for col, (csv_name, n_envs, method, _) in enumerate(DISTILLATION_RUNS_6):
            ax = axes[row, col]
            for reason in UNSAFE_REASONS:
                metric = f"train/{obj}/unsafe_reason_prop/{reason}"
                df = _load_distillation_metric(
                    csv_name, metric, n_envs,
                    max_iter=max_iter, smooth_window=1)
                if df.empty:
                    continue
                raw = df["value"].values * 100
                smoothed = _ema(raw, alpha=ema_alpha)
                iters = df["iteration"].values
                step = max(1, len(iters) // 500)
                ax.fill_between(iters[::step], 0, smoothed[::step],
                                color=REASON_COLORS[reason], alpha=0.15)
                ax.plot(iters[::step], smoothed[::step],
                        color=REASON_COLORS[reason], linewidth=1.2, alpha=0.9,
                        label=REASON_LABELS[reason] if row == 0 else None)

            ax.set_xlim(0, max_iter)
            ax.set_ylim(0, 30)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(method, fontsize=9)
            if col == 0:
                ax.set_ylabel(obj.replace("_", " ") + "\nRate (%)", fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel("Iteration", fontsize=7)

    axes[0, 0].legend(fontsize=5, loc="upper right")
    fig.suptitle("Failure Mode Analysis: SafeDAgger vs DAgger (Run 6)", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, "run6_failure_mode_detail")


def plot_distillation_comparison_run5(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Three separate plots for run5a/5b with real termination data."""
    runs = DISTILLATION_RUNS_5

    # 1. Lift success normalized to teacher ceiling
    fig = _plot_distillation_single(
        "in_success_region", "Student Policy Lift Success (Run 5)",
        "Lift Success (normed to teacher)", (0, 1.05), runs, max_iter, ema_alpha,
        normalize_by=TEACHER_11_LIFT)
    save_fig(fig, "run5_lift_success_normed")

    # 2. Intervention rate beta
    fig = _plot_distillation_single(
        "beta", "Intervention Rate Beta (Run 5)",
        r"$\beta$ (teacher intervention rate)", (0, 1.05), runs, max_iter, ema_alpha,
        beta_dagger_zero=True)
    save_fig(fig, "run5_intervention_rate")

    # 3. Real unsafe episode rate (per-step termination rate)
    fig = _plot_distillation_single(
        "termination/real_unsafe", "Real Unsafe Termination Rate (Run 5)",
        "Per-step termination rate", (0, 0.01), runs, max_iter, ema_alpha)
    save_fig(fig, "run5_unsafe_episode_rate")


def plot_per_object_eval_run5(eval_dir: Path = None):
    """Per-object lift success and unsafe rate from run5 standalone evals."""
    import json
    if eval_dir is None:
        eval_dir = EXPORTS_DIR.parent.parent / "distillation_new" / "eval_results"

    eval_files = {
        "SafeDAgger (run5a)": eval_dir / "eval_metrics_20260403_105154.json",
        "DAgger (run5b)": eval_dir / "eval_metrics_20260403_105805.json",
    }

    for metric_key, ylabel, title, fname in [
        ("lift_success", "Lift success (%)", "Per-Object Lift Success — Standalone Eval (Run 5)", "run5_per_object_lift_eval"),
        ("unsafe_episode_rate", "Unsafe rate (%)", "Per-Object Unsafe Rate — Standalone Eval (Run 5)", "run5_per_object_unsafe_eval"),
    ]:
        fig, ax = plt.subplots(figsize=(14/2.54, 7/2.54))
        x = np.arange(len(OBJECTS))
        w = 0.35
        colors_list = [COLORS["blue"], COLORS["orange"]]

        for i, (label, path) in enumerate(eval_files.items()):
            with open(path) as fh:
                data = json.load(fh)
            vals = [data["per_object_metrics"].get(obj, {}).get(metric_key, 0) * 100
                    for obj in OBJECTS]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=label,
                          color=colors_list[i], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                            f"{val:.0f}", ha="center", va="bottom", fontsize=4)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", " ") for o in OBJECTS],
                           fontsize=5, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=8)
        fig.tight_layout()
        save_fig(fig, fname)


def plot_per_object_term_real_head_to_head(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object REAL unsafe termination rate: SafeDagger vs DAgger (run5a/5b).
    One subplot per object, EMA smoothed.
    """
    n_cols = 4
    n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    runs = DISTILLATION_RUNS_5

    for idx, obj in enumerate(OBJECTS):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in runs:
            df = _load_distillation_metric(
                csv_name, f"per_object_term_real/{obj}", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df.empty:
                continue
            raw = df["value"].values * 100
            smoothed = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)

        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 2)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)

    for idx in range(len(OBJECTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Unsafe (%)", fontsize=7)

    axes_flat[0].legend(fontsize=5, loc="upper right")
    fig.suptitle("Per-Object Real Unsafe Rate (per-step): SafeDAgger vs DAgger (Run 5)", fontsize=9)
    fig.tight_layout()

    save_fig(fig, "per_object_term_real_head_to_head")


def plot_per_object_lift_head_to_head_run5(max_iter: int = 100_000, ema_alpha: float = 0.999):
    """Per-object lift success: SafeDagger vs DAgger (run5a/5b)."""
    n_cols = 4
    n_rows = (len(OBJECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18/2.54, n_rows * 4/2.54),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    runs = DISTILLATION_RUNS_5

    for idx, obj in enumerate(OBJECTS):
        ax = axes_flat[idx]
        for csv_name, n_envs, method, color in runs:
            df = _load_distillation_metric(
                csv_name, f"per_object_lift/{obj}", n_envs,
                max_iter=max_iter, smooth_window=1)
            if df.empty:
                continue
            raw = df["value"].values * 100
            smoothed = _ema(raw, alpha=ema_alpha)
            iters = df["iteration"].values
            step = max(1, len(iters) // 500)
            ax.plot(iters[::step], smoothed[::step], color=color,
                    linewidth=1.2, alpha=0.9, label=method)

        ax.set_title(obj.replace("_", " "), fontsize=6, pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_iter)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(labelsize=5)

    for idx in range(len(OBJECTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Lift (%)", fontsize=7)

    axes_flat[0].legend(fontsize=5, loc="lower right")
    fig.suptitle("Per-Object Lift Success: SafeDAgger vs DAgger (Run 5)", fontsize=9)
    fig.tight_layout()

    save_fig(fig, "per_object_lift_head_to_head_run5")


def plot_termination_breakdown(max_iter: int = 100_000, smooth: int = 1000):
    """Termination breakdown: real_unsafe vs physics_instability for run4a and run4b."""
    fig, ax = plt.subplots()

    runs = [
        ("student_run4a_safedagger_l2_teacher11.csv", 24, "SafeDAgger"),
        ("student_run4b_vanilla_kl_teacher11.csv", 24, "DAgger"),
    ]

    styles = {
        "SafeDAgger": {"real": COLORS["blue"], "phys": COLORS["cyan"]},
        "DAgger":     {"real": COLORS["orange"], "phys": COLORS["brown"]},
    }

    for csv_name, n_envs, method in runs:
        df_real = _load_distillation_metric(
            csv_name, "termination/real_unsafe", n_envs,
            max_iter=max_iter, smooth_window=smooth)
        ax.plot(df_real["iteration"], df_real["value_smooth"] * 100,
                color=styles[method]["real"], linestyle="-",
                label=f"{method} — real unsafe", alpha=0.9)

        df_phys = _load_distillation_metric(
            csv_name, "termination/physics_instability", n_envs,
            max_iter=max_iter, smooth_window=smooth)
        if not df_phys.empty:
            ax.plot(df_phys["iteration"], df_phys["value_smooth"] * 100,
                    color=styles[method]["phys"], linestyle="--",
                    label=f"{method} — physics instab.", alpha=0.9)

    ax.set_xlabel("Distillation iteration")
    ax.set_ylabel("Termination rate (%)")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 50)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Termination Breakdown: Real Unsafe vs Physics Instability")

    save_fig(fig, "termination_breakdown")


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
    plot_per_object_unsafe_comparison()
    plot_per_object_lift_comparison()
    plot_per_object_unsafe_over_time()
    plot_per_object_lift_over_time()
    plot_per_object_lift_head_to_head()
    plot_per_object_unsafe_real()
    plot_per_object_unsafe_real_per_run()
    plot_distillation_comparison()
    plot_distillation_comparison_run5()
    plot_distillation_comparison_run6()
    plot_per_object_lifted_head_to_head_run6()
    plot_per_object_unsafe_episode_head_to_head_run6()
    plot_per_object_failure_mode_run6()
    plot_per_object_eval_run5()
    plot_per_object_term_real_head_to_head()
    plot_per_object_lift_head_to_head_run5()
    plot_termination_breakdown()
    # Per-run organized plots
    plot_run("run6", DISTILLATION_RUNS_6, OBJECTS)
    plot_run("run7", DISTILLATION_RUNS_7, OBJECTS_TOP8)
    plot_run("run8", DISTILLATION_RUNS_8, OBJECTS_TOP8, eval_files=RUN8_EVAL_FILES)
    plot_run("run9", DISTILLATION_RUNS_9, OBJECTS_TOP8, eval_files=RUN9_EVAL_FILES, failure_mode_ylim=60)
    # Run 9 single-object, single-reason head-to-heads
    for _obj in ["basketball_shoe", "teddy_bear", "closed_fist", "elephant_toy",
                 "milk_pot", "toy_bagger", "tutle_candle_holder", "mario"]:
        for _reason in UNSAFE_REASONS:
            plot_run9_single_object_single_reason(obj=_obj, reason=_reason)
    plot_run("run10", DISTILLATION_RUNS_10, OBJECTS_TOP8, eval_files=RUN10_EVAL_FILES)
    plot_run11_unsafe()
    print("Done.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
