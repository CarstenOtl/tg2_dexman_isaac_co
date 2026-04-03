"""
Per-object unsafe rate and unsafe reason breakdown — SafeDagger (run3a) vs
Vanilla DAgger (run3b), both with teacher 11 (ADR 14), ~100k iters.
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

runs = [
    ("SafeDagger (run3a, ~120k)", data_sd, COLORS["blue"]),
    ("Vanilla DAgger (run3b, ~105k)", data_vd, COLORS["orange"]),
]

# Use union of objects, sorted by SafeDagger unsafe rate descending
all_objects = sorted(
    data_sd["per_object_metrics"].keys(),
    key=lambda o: data_sd["per_object_metrics"][o]["unsafe_episode_rate"],
    reverse=True,
)
display_names = [o.replace("_", " ") for o in all_objects]

reason_keys = ["harmful_collision", "object_out_of_bound", "palm_flipped", "physics_instability"]
reason_labels = ["Harmful collision", "Object OOB", "Palm flipped", "Physics instab. (sim)"]
reason_colors = [COLORS["red"], COLORS["orange"], COLORS["purple"], COLORS["gray"]]

y = np.arange(len(all_objects))
bar_h = 0.35

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.2, 1.2]})

# --- Left: per-object unsafe rate (grouped bars) ---
for i, (label, data, color) in enumerate(runs):
    per_obj = data["per_object_metrics"]
    rates = [per_obj[o]["unsafe_episode_rate"] * 100 for o in all_objects]
    offset = -bar_h / 2 + i * bar_h
    bars = ax1.barh(y + offset, rates, height=bar_h, color=color,
                    edgecolor="white", linewidth=0.3, label=label)
    for j, v in enumerate(rates):
        ax1.text(v + 1, y[j] + offset, f"{v:.0f}", va="center", fontsize=6)

ax1.set_xlabel("Unsafe episode rate (%)")
ax1.set_yticks(y)
ax1.set_yticklabels(display_names, fontsize=7.5)
ax1.set_xlim(0, 105)
ax1.invert_yaxis()
ax1.set_title("Unsafe rate per object")
ax1.legend(fontsize=6.5, loc="lower right")

# --- Middle: SafeDagger reason breakdown ---
per_obj_sd = data_sd["per_object_metrics"]
left = np.zeros(len(all_objects))
for k, label, color in zip(reason_keys, reason_labels, reason_colors):
    vals = np.array([per_obj_sd[o]["unsafe_reason_prop"][k] * 100 for o in all_objects])
    ax2.barh(y, vals, left=left, height=0.65, label=label, color=color,
             edgecolor="white", linewidth=0.3)
    left += vals
ax2.set_xlabel("Proportion of total episodes (%)")
ax2.set_xlim(0, 105)
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_title("SafeDagger — reason breakdown")
ax2.legend(fontsize=6, loc="lower right")

# --- Right: Vanilla DAgger reason breakdown ---
per_obj_vd = data_vd["per_object_metrics"]
left = np.zeros(len(all_objects))
for k, label, color in zip(reason_keys, reason_labels, reason_colors):
    vals = np.array([per_obj_vd[o]["unsafe_reason_prop"][k] * 100 for o in all_objects])
    ax3.barh(y, vals, left=left, height=0.65, label=label, color=color,
             edgecolor="white", linewidth=0.3)
    left += vals
ax3.set_xlabel("Proportion of total episodes (%)")
ax3.set_xlim(0, 105)
ax3.set_yticks(y)
ax3.set_yticklabels([])
ax3.invert_yaxis()
ax3.set_title("Vanilla DAgger — reason breakdown")

fig.suptitle(
    "Per-object unsafe analysis — Teacher 11 (ADR 14), 480 episodes each",
    fontsize=10, y=1.02,
)
fig.tight_layout()
save_fig(fig, "per_object_unsafe_comparison")
print("Done.")
