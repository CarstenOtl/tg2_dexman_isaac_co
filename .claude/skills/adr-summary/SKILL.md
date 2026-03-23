---
name: adr-summary
description: Parse ADR (Adaptive Domain Randomization) configuration from a dextrah task env_cfg and print a readable table of all parameters — starting values, ADR ranges, and what they control. Use when the user asks about ADR, randomization ranges, curriculum parameters, or domain randomization settings.
tools: Read, Grep
---

# ADR Summary

Parse and display all ADR parameters from a dextrah task config in a human-readable format.

## Workflow

### Step 1: Find the task config

Locate `dextrah_*_env_cfg.py` for the relevant task in `dextrah_lab/tasks/<task_name>/`.

### Step 2: Extract starting values

Find the `EventCfg` class (usually named `DextrahEventCfg`). For each randomization term, extract the **starting** parameter ranges:
- `robot_physics_material`: static/dynamic friction, restitution
- `fingertip_physics_material`: static/dynamic friction, restitution
- `object_physics_material`: static/dynamic friction, restitution
- `robot_joint_stiffness_and_damping`: stiffness/damping scale
- `robot_joint_friction`: friction distribution
- `object_scale_mass`: mass distribution

### Step 3: Extract ADR max ranges

Find `adr_cfg_dict` and `adr_custom_cfg_dict` in the main env config class. These define the **maximum ranges** ADR can expand to.

### Step 4: Extract ADR curriculum settings

Find:
- `num_adr_increments` — how many steps to reach max range
- `success_for_adr` — success rate threshold to trigger ADR step
- Any `adr_custom_cfg_dict` entries (reward weights, fabric params, etc.)

### Step 5: Print summary tables

**Physics Randomization:**

| Parameter | Start (min, max) | ADR Max (min, max) | Notes |
|---|---|---|---|
| fingertip static friction | ... | ... | |
| fingertip dynamic friction | ... | ... | |
| object static friction | ... | ... | |
| ... | ... | ... | |

**Custom ADR Parameters** (reward weights, gains, etc.):

| Parameter | Start | ADR Range | Notes |
|---|---|---|---|
| ... | ... | ... | |

**Curriculum Settings:**
- Increments: N
- Success threshold: X
- Approx envs needed per increment: ...

Flag any parameters that start at their max (no curriculum effect) and any that are disabled (`0,0` range).
