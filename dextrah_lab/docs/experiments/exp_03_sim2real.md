---
name: exp_03 sim2real curriculum
description: Experiment 03 — sim2real focused training for FR3+AgileHand, adding hardware-realistic constraints via ADR curriculum
type: project
---

# Experiment 03 — Sim2Real Curriculum

**Goal:** Take the working lift policy from exp_02 (run1p, first ADR step at epoch 8850) and add hardware-realistic constraints via ADR to close the sim2real gap.

**Baseline:** exp_02 run1p checkpoint (`logs/rl_games/dextrah_tekken_lstm/03-25_19-01-51/`)

## Key sim2real targets

1. **Thumb rotation velocity** — real hardware limit is 12–8 deg/s (0.14–0.21 rad/s). Current sim limit: 20 rad/s. Need ADR to ramp this down.
2. **Actuator stiffness/damping** — randomize to match real hardware ballpark (values TBD)
3. **Action delay/latency** — not yet implemented (identified gap from exp_02 ADR audit)
4. **Object CoM offset** — not yet implemented

## ADR mechanics (from dextrah_adr.py)
- Single shared `increment_counter` — ALL parameters progress linearly together
- Each step interpolates from start → max by `1/num_increments`
- Custom params read via `get_custom_param_value(group, name)`
- No per-parameter scheduling — everything moves in lockstep

## Implementation

### Finger stiffness/damping — via per-group EventTerms (EventCfg + adr_cfg_dict)

Split the old unified `robot_joint_stiffness_and_damping` EventTerm into per-group terms, each with their own ADR scale ranges. Asset defaults stay sim-stable; ADR widens the scale range to cover hardware values.

| EventTerm | Stiffness ADR max (scale) | Damping ADR max (scale) | Notes |
|---|---|---|---|
| `arm_joint_stiffness_and_damping` | (0.5, 2.0) | (0.5, 2.0) | Same as before |
| `finger_mcp_pitch_gains` | (0.1, 1.0) | (0.1, 1.0) | Relaxed from (0.17, 0.083) — hardware-exact was too aggressive |
| `finger_mcp_yaw_gains` | (0.1, 1.0) | (0.1, 1.0) | Relaxed from (0.025, 0.033) |
| `finger_pip_gains` | (0.1, 1.0) | (0.1, 1.0) | Relaxed from (0.02, 0.033) |
| `thumb_rot_gains` | (0.5, 2.0) | (0.5, 2.0) | No hardware target yet |

All start at (1.0, 1.0) — no randomization. ADR linearly widens to max range over 50 increments.

Hardware values from Isaac gain tuner (commits from 2026-03-21–23, restored to sim-stable in 4c9beb5):
- mcp_pitch: stiffness=1.78, damping=0.5
- mcp_yaw: stiffness=0.28, damping=0.2
- pip: stiffness=0.24, damping=0.2

### Effort limits + thumb velocity — via adr_custom_cfg_dict + _apply_actuator_curriculum()

No EventTerm API for effort/velocity limits, so these use custom ADR params applied in `_apply_actuator_curriculum()` (called on every reset + after ADR steps).

| Parameter | Start | ADR terminal | Source |
|---|---|---|---|
| `thumb_rot_vel_limit` | 20.0 rad/s | 0.21 rad/s (≈12 deg/s) | Hardware spec |
| `arm_14_effort_limit` | 100 Nm | 87 Nm | FR3 factory spec |
| `arm_57_effort_limit` | 50 Nm | 12 Nm | FR3 factory spec |

Uses Isaac Lab APIs: `write_joint_effort_limit_to_sim`, `write_joint_velocity_limit_to_sim`.

## Runs

### run1a — 2026-03-27
**Checkpoint:** `stored_policies/fr3_agilehand/10_multi_object_adr19_1024envs_03-26_00-21-45/nn/last_..._ep_20000.pth`
**Config:** 1024 envs, starting_adr=15, min_steps_for_dr_change=30k, max_epochs=50000
**Finger gain ADR:** mcp_pitch (0.17, 0.083), mcp_yaw (0.025, 0.033), pip (0.02, 0.033) — hardware-exact targets

**Result:** Policy degraded — lift_success dropped 0.6→0.15 over training. Only reached ADR 16 (one step from 15). 30 vel_explode terminations per cycle. Finger stiffness ADR ranges too aggressive — at ADR 16, fingers were too soft/unpredictable to maintain grasps. Policy was forgetting how to lift.

**Lesson:** Hardware-exact finger gain targets are too extreme for in-lockstep ADR. Sim PD stiffness/damping doesn't directly map to real hardware (real hand has its own firmware PID). What matters for sim2real is effort limits and velocity limits (achievable positions per step), not the sim actuator tracking dynamics.

### run1b — 2026-03-27
**Checkpoint:** same ADR 19 checkpoint
**Config:** 1024 envs, starting_adr=0, min_steps_for_dr_change=30k, max_epochs=50000
**Finger gain ADR:** all groups relaxed to (0.1, 1.0) for both stiffness and damping

**Result:** ADR too slow — only 2 ADR steps (0→2) in ~7k training epochs. `min_steps_for_dr_change=30k` was 10x too conservative; counter ticks once per `_reset_idx` call (multiple times per epoch with 1024 envs), not once per epoch. Policy maintained ~80% lift_success and ~45% in_success_region but ADR barely progressed. Would need ~100k epochs for full ADR at this rate.

**Lesson:** `min_steps_for_dr_change` counts reset-batch calls, not epochs. With 1024 envs, 3,000 ticks ≈ 300-600 epochs per ADR step. The 30k value was way too slow.

### run1c — 2026-03-27
**Checkpoint:** same ADR 19 checkpoint
**Config:** 1024 envs, starting_adr=10, min_steps_for_dr_change=3k (reverted to original), max_epochs=50000
**Finger gain ADR:** all groups (0.1, 1.0) for both stiffness and damping

**Code changes:**
- Reverted `min_steps_for_dr_change` from `50 * episode_steps` (30k) back to `5 * episode_steps` (3k)
- Fixed `successes` debug metric: now counts episodes where object held in goal >= `success_timeout` (2s), instead of checking `in_success_region` at exact reset frame (was always near 0% due to timing)
- Added `episode_succeeded` flag, set when `time_in_success_region >= success_timeout`, reset on episode end

**Result:** Same ADR 18 wall. Ramped 10→18 quickly, then lift_success dropped 0.7→0.4, in_success_region dropped 0.45→0.1. 41 vel_explode/cycle. Identical pattern to run1a.

### Key finding: checkpoint resume doesn't work for LSTM policies

All three runs (1a, 1b, 1c) resumed from the ADR 19 checkpoint and all degraded at ADR 17-18. Investigation of rl_games `set_full_state_weights()` confirmed:
- **Saved:** model weights, optimizer state (Adam), running mean/std, epoch, central value net
- **NOT saved:** LSTM hidden states (reset to zero on load)

The LSTM needs to rebuild temporal representations from scratch. At ADR 0-17 it can recover (easy enough), but at 18+ the difficulty margin is too thin for a cold LSTM to re-learn while also handling randomization. The original training succeeded because the LSTM built context *continuously* through ADR 0→19.

### run1d — 2026-03-27
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000
**Finger gain ADR:** all groups (0.1, 1.0) for both stiffness and damping
**Sim2real curriculum:** thumb_rot_vel (20→0.21), arm_14_effort (100→87), arm_57_effort (50→12)

**Why:** Checkpoint resume fails due to LSTM cold-start. Train from scratch with all sim2real terms from the beginning so the LSTM builds temporal context continuously through the full ADR range.
