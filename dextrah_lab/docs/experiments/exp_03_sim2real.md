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
| `finger_mcp_pitch_gains` | (0.17, 1.0) | (0.083, 1.0) | Hardware stiffness=1.78 (default 10.0) |
| `finger_mcp_yaw_gains` | (0.025, 1.0) | (0.033, 1.0) | Hardware stiffness=0.28 (default 10.0) |
| `finger_pip_gains` | (0.02, 1.0) | (0.033, 1.0) | Hardware stiffness=0.24 (default 10.0) |
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
(to be filled as runs proceed)
