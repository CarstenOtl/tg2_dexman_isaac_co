---
name: checkpoint-inspect
description: Inspect a .pth checkpoint file — print network architecture, obs/action sizes it was trained with, training metadata, and whether it's compatible with the current environment config. Use when the user wants to know what a checkpoint contains or whether it's compatible before running distillation or evaluation.
tools: Bash, Grep, Read
---

# Checkpoint Inspect

Load and summarize a `.pth` checkpoint without running the full Isaac Sim environment.

## Workflow

### Step 1: Locate the checkpoint

If the user gave a relative path, check under:
- `dextrah_lab/stored_policies/<task>/`
- `dextrah_lab/rl_games/logs/`
- `dextrah_lab/distillation_new/runs/`

### Step 2: Load and inspect with Python

```bash
python3 - <<'EOF'
import torch, sys, json
path = "<CHECKPOINT_PATH>"
ckpt = torch.load(path, map_location="cpu")
print("Top-level keys:", list(ckpt.keys()))

# rl-games teacher checkpoint
if "model" in ckpt:
    model = ckpt["model"]
    print("\nModel keys:", list(model.keys())[:20])
    for k, v in model.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")

# metadata
for key in ["epoch", "frame", "last_mean_rewards", "config", "env_config"]:
    if key in ckpt:
        val = ckpt[key]
        if isinstance(val, dict):
            print(f"\n{key}:", json.dumps(val, indent=2, default=str)[:500])
        else:
            print(f"\n{key}:", val)
EOF
```

### Step 3: Extract obs/action sizes

From the model weights, infer input/output sizes:
- First linear layer weight shape → obs size (input dim)
- Last policy head weight shape → action size (output dim)
- Look for `a2c_network.actor_mlp.0.weight` or similar

### Step 4: Compare to current env config

If a task was specified, read `_setup_policy_params()` from the env and compare:
- Checkpoint obs dim vs `num_teacher_observations` or `num_student_observations`
- Checkpoint action dim vs `num_actions`

### Step 5: Report

```
Checkpoint: <path>
Type: teacher / student (inferred from obs size)
Obs size: X  →  matches env: ✓/✗ (env expects Y)
Action size: X  →  matches env: ✓/✗
Epoch: N
Last mean reward: X
Network: <architecture summary>
```

Flag incompatibilities clearly with what would need to change to use this checkpoint.
