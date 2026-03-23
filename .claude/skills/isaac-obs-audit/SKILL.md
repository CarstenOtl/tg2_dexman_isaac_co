---
name: isaac-obs-audit
description: Audit observation tensor sizes in an Isaac Lab task — compute actual dims from obs composition and verify they match the declared sizes in _setup_policy_params(). Use when the user asks to check, verify, or debug observation sizes for any dextrah task.
tools: Read, Grep, Bash
---

# Isaac Obs Audit

Verify that declared observation sizes in `_setup_policy_params()` match the actual tensors concatenated in `compute_*_observations()`.

## Workflow

### Step 1: Find the task

If the user didn't specify a task, check the current branch or ask. Tasks live in `dextrah_lab/tasks/<task_name>/`.

### Step 2: Read declared sizes

In `dextrah_*_env.py`, find `_setup_policy_params()` and extract:
- `num_teacher_observations`
- `num_student_observations`
- `num_states` (critic)

Note the formula used — e.g. `teacher_base + 1` or `teacher_base + num_unique_objects`.

### Step 3: Parse each obs function

For each of `compute_policy_observations()`, `compute_student_policy_observations()`, `compute_critic_observations()`:

List every tensor concatenated and its expected shape. Use these rules:
- `robot_dof_pos/vel` → `num_actuated` dims each (count from `actuated_joint_names` in cfg)
- `hand_pos/vel_noisy` → `num_hand_bodies * 3` each (count from `hand_body_names` in cfg)
- `hand_vel` (not noisy) → `num_hand_bodies * 6` if using full vel (pos+rot)
- `object_pos` → 3, `object_rot` → 4, `object_vel` → 6, `object_goal` → 3, `object_scale` → 1
- `actions` → `num_actuated`
- `multi_object_idx_onehot` → `num_unique_objects`
- `teacher_onehot` → 1
- `hand_forces[:, :3]` → 3
- `measured_joint_torque` → `num_dofs` (from `starting_robot_dof_friction_coefficients` in cfg)
- `fabric_q/qd/qdd` → `num_actuated` each

### Step 4: Sum and compare

Build a table:

| Obs type | Declared | Computed | Match? |
|---|---|---|---|
| Teacher | X | Y | ✓/✗ |
| Student | X | Y | ✓/✗ |
| Critic | X | Y | ✓/✗ |

If any mismatch, show which tensor is wrong and the correct value.

### Step 5: Report

Show the full breakdown with per-tensor dim counts, then the summary table. Flag any mismatches clearly with the fix needed.
