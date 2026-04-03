"""
Per-object REAL unsafe rate (excluding physics instabilities) — SafeDagger
(run3a) vs Vanilla DAgger (run3b), both with teacher 11 (ADR 14), ~100k iters.

Physics instability is a simulation artifact — subtracting it gives
the sim2real-relevant unsafe rate.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style, COLORS, save_fig

apply_style(fig_width_cm=18.0)

# Load eval JSONs
eval_dir = Path(__file__).resolve().parent.parent.parent / "distillation_new" / "eval_results"
with open(eval_dir / "eval_metrics_20260331_182256.json") as f:
    data_sd = json.load(f)  # run3a SafeDagger ~120k
with open(eval_dir / "eval_metrics_20260331_231008.json") as f:
    data_vd = json.load(f)  # run3b Vanilla DAgger ~105k

real_reason_keys = ["harmful_collision", "object_out_of_bound", "palm_flipped"]
real_reason_labels = ["Harmful collision", "Object OOB", "Palm flipped"]
real_reason_colors = [COLORS["red"], COLORS["orange"], COLORS["purple"]]

runs = [
    ("SafeDagger (run3a, ~120k)", data_sd, COLORS["blue"]),
    ("Vanilla DAgger (run3b, ~105k)", data_vd, COLORS["cyan"]),
]


def real_unsafe_rate(per_obj, obj):
    return sum(per_obj[obj]["unsafe_reason_prop"][k] for k in real_reason_keys) * 100


# Sort by SafeDagger real unsafe rate descending
all_objects = sorted(
    data_sd["per_object_metrics"].keys(),
    key=lambda o: real_unsafe_rate(data_sd["per_object_metrics"], o),
    reverse=True,
)
display_names = [o.replace("_", " ") for o in all_objects]

y = np.arange(len(all_objects))
bar_h = 0.35

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.2, 1.2]})

# --- Left: real unsafe rate (grouped bars) ---
for i, (label, data, color) in enumerate(runs):
    per_obj = data["per_object_metrics"]
    rates = [real_unsafe_rate(per_obj, o) for o in all_objects]
    offset = -bar_h / 2 + i * bar_h
    bars = ax1.barh(y + offset, rates, height=bar_h, color=color,
                    edgecolor="white", linewidth=0.3, label=label)
    for j, v in enumerate(rates):
        ax1.text(v + 0.5, y[j] + offset, f"{v:.0f}", va="center", fontsize=6)

ax1.set_xlabel("Real unsafe rate (%)")
ax1.set_yticks(y)
ax1.set_yticklabels(display_names, fontsize=7.5)
ax1.set_xlim(0, 55)
ax1.invert_yaxis()
ax1.set_title("Real unsafe rate per object")
ax1.legend(fontsize=6.5, loc="lower right")

# --- Middle: SafeDagger real reason breakdown ---
per_obj_sd = data_sd["per_object_metrics"]
left = np.zeros(len(all_objects))
for k, label, color in zip(real_reason_keys, real_reason_labels, real_reason_colors):
    vals = np.array([per_obj_sd[o]["unsafe_reason_prop"][k] * 100 for o in all_objects])
    ax2.barh(y, vals, left=left, height=0.65, label=label, color=color,
             edgecolor="white", linewidth=0.3)
    left += vals
ax2.set_xlabel("Proportion of total episodes (%)")
ax2.set_xlim(0, 55)
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_title("SafeDagger — real reason breakdown")
ax2.legend(fontsize=6, loc="lower right")

# --- Right: Vanilla DAgger real reason breakdown ---
per_obj_vd = data_vd["per_object_metrics"]
left = np.zeros(len(all_objects))
for k, label, color in zip(real_reason_keys, real_reason_labels, real_reason_colors):
    vals = np.array([per_obj_vd[o]["unsafe_reason_prop"][k] * 100 for o in all_objects])
    ax3.barh(y, vals, left=left, height=0.65, label=label, color=color,
             edgecolor="white", linewidth=0.3)
    left += vals
ax3.set_xlabel("Proportion of total episodes (%)")
ax3.set_xlim(0, 55)
ax3.set_yticks(y)
ax3.set_yticklabels([])
ax3.invert_yaxis()
ax3.set_title("Vanilla DAgger — real reason breakdown")

# Aggregate real unsafe rates
agg_sd = data_sd["metrics"]["eval/unsafe_episode_rate"] * 100 - \
    data_sd["metrics"]["eval/unsafe_reason_prop"]["physics_instability"] * 100
agg_vd = data_vd["metrics"]["eval/unsafe_episode_rate"] * 100 - \
    data_vd["metrics"]["eval/unsafe_reason_prop"]["physics_instability"] * 100

fig.suptitle(
    f"Real unsafe rate (excl. physics instability) — Teacher 11 (ADR 14)\n"
    f"SafeDagger: {agg_sd:.1f}% | Vanilla DAgger: {agg_vd:.1f}% | 480 episodes each",
    fontsize=9, y=1.05,
)
fig.tight_layout()
save_fig(fig, "per_object_unsafe_real_comparison")
print("Done.")
