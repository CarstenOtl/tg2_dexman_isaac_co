---
name: physics-diff
description: Compare physics material settings side by side between two dextrah task configs — friction, restitution, combine modes, joint gains, contact sensor thresholds, PhysX settings. Use when the user wants to compare physics between tasks or robot configurations.
tools: Read, Grep
---

# Physics Diff

Compare physics configuration between two dextrah tasks or configs side by side.

## Workflow

### Step 1: Identify configs to compare

Default: compare `fr3_agilehand` vs another task specified by the user.
If only one task given, compare against `dextrah_kuka_allegro` as baseline.

### Step 2: Extract from each config

For each `dextrah_*_env_cfg.py`, extract:

**Global simulation material** (`SimulationCfg.physics_material`):
- `static_friction`, `dynamic_friction`
- `friction_combine_mode`, `restitution_combine_mode`

**Per-body starting values** (from `EventCfg`):
- Robot body: static/dynamic friction, restitution
- Fingertip bodies: static/dynamic friction, restitution
- Object: static/dynamic friction, restitution

**ADR max ranges** (from `adr_cfg_dict`):
- Same categories as above

**Joint properties:**
- `stiffness`, `damping` per joint group
- `friction_distribution_params`

**PhysX settings** (`PhysxCfg`):
- `bounce_threshold_velocity`
- `gpu_max_rigid_patch_count`
- any solver settings

**Contact sensor thresholds:**
- `filter_prim_paths_expr`
- `force_threshold`, `torque_threshold` if present

### Step 3: Print side-by-side diff

```
Parameter                    | Task A          | Task B          | Diff?
-----------------------------|-----------------|-----------------|------
sim static_friction          | 1.0             | 1.0             | same
sim friction_combine_mode    | max             | average         | ✗
fingertip static friction    | 2.0             | 1.5             | ✗
...
```

Highlight differences with ✗ and flag anything that looks like it could cause grasping issues (low friction, average combine mode, high restitution).
