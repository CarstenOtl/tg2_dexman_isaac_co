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

**Result:** Reached ADR 12/50 after 45k epochs, then plateaued. `in_success_region` hovered at 0.25–0.35, barely crossing the 0.4 threshold. Policy lifts well (~54%) but doesn't bring objects to goal consistently. Root cause: reward imbalance — `contact` (24) and `lift` (16) dominate, while `obj_to_goal` (4.7) is too weak. Policy optimizes for holding objects rather than moving them to the goal.

**Lesson:** `object_to_goal_weight=20` is too low relative to contact/lift rewards. Need stronger goal-reaching incentive.

### run1e — 2026-03-28
**Checkpoint:** none — fresh start from random init
**Config:** 2048 envs across 2 GPUs (`--distributed`, `nproc_per_node=2`), starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000
**Finger gain ADR:** all groups (0.1, 1.0) for both stiffness and damping
**Sim2real curriculum:** thumb_rot_vel (20→0.21), arm_14_effort (100→87), arm_57_effort (50→12)

**Reward changes:**
- `object_to_goal_weight`: 20 → 40 (double the goal-reaching gradient)
- Added `success_bonus_weight=10.0` — flat per-step bonus when object is within `object_goal_tol` (0.1m) of goal
- Both changes aim to make goal-reaching competitive with contact/lift rewards

**Why:** Fresh start with rebalanced rewards and multi-GPU for faster training.

**Result:** Reached ADR 19/50 after 21k epochs (~27 hours). `in_goal_now: 35%`, `successes: 34%`, `lifted_now: 41%`. `vel_explode=65` per cycle still high. Multi-GPU was 2x slower per epoch than single GPU (790 vs 1667 epochs/hr) — gradient sync overhead via PCIe (GPUs 0 and 3, no NVLink). 1024 envs on single GPU is sufficient and faster.

**Lesson:** `torch.distributed` not worth it for this workload — single GPU reaches same ADR in half the wall-clock time. Multi-GPU only useful when single GPU can't fit enough envs in VRAM.

### run1f — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, single GPU, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000

**Physics changes (from Isaac gain tuner testing):**
- mcp_pitch: stiffness 10→20, damping 6→2
- mcp_yaw: stiffness 10→20, damping 6→2
- pip: stiffness 10→30, damping 6→2
- Better joint tracking performance verified in gain tuner

**ADR changes:**
- All finger gain ranges: (0.1, 1.0) → (0.5, 2.0) — matched to proven kuka_allegro/tg2_inspirehand sim2real
- Thumb rot velocity: start 10.0 rad/s → end 0.1396 rad/s (8 deg/s)

**Reward:** object_to_goal_weight=40, success_bonus_weight=10.0 (from run1e)

**Why:** Better finger tracking + proven ADR ranges should reduce vel_explode and allow ADR to progress past 19.

**Result:** Policy did not reach ADR. vel_explode at 1-2% of envs. Main issue: thumb consistently curls inward between the object and other fingers, blocking any successful grasp. The new finger gains (stiffness 20/30, damping 2) make the thumb much more responsive — it now actually reaches where the policy commands, but the policy drives it into the wrong position.

### run1g — 2026-03-30
**Config:** 32 envs, livestream mode, short diagnostic run
**Purpose:** Visual confirmation of thumb behavior with new finger gains

**Command:**
```bash
python train.py --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 32 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=False \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Result:** Confirmed thumb curls inward during episode due to stiffer gains (mcp_pitch stiffness 20, damping 2). Thumb tracks commands accurately but policy drives it between object and fingers. Need stronger finger_curl penalty to discourage this.

### run1h — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 64 envs, livestream mode, starting_adr=0, min_steps_for_dr_change=3k
Same physics/ADR as run1f.

**Command:**
```bash
python train.py --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 64 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=False \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Changes vs run1f:**
- `finger_curl_reg` ADR start: -0.1 → -0.3 (stronger initial curl penalty to prevent thumb curling inward)

**Why:** Stiffer finger gains cause thumb to curl more aggressively. Stronger initial finger_curl penalty should discourage the policy from driving the thumb between object and fingers.

**Result:** Thumb still curling inward, same trend as run1f/1g. Curl penalty alone not enough.

### run1i — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 64 envs, livestream mode, starting_adr=0, min_steps_for_dr_change=3k

**Command:**
```bash
python train.py --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 64 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=False \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Changes vs run1h:**
- Added `thumb_rot_init` EventTerm: randomizes thumb rotation between -20° and 0° on reset via `mdp.reset_joints_by_offset`

**Why:** Thumb always starts at -20° (joint min), which biases the policy toward curling inward. Randomizing the init forces the policy to learn grasping with thumb in various positions.

**Result:** Significant improvement! Thumb init randomization allowed the policy to discover that keeping thumb_mcp_pitch near 0 (open) gives much better grasping. Policy no longer drives thumb between object and fingers.

### run1j — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, single GPU, headless, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000

**Command:**
```bash
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 1024 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Same config as run1i** but scaled to 1024 envs for full training. Testing if thumb init randomization + updated gains can push through ADR.

**Result:** Thumb still finds local optimum between fingers and object. Contact reward dominates — policy maximizes finger contact by wedging thumb in. Curl penalty too weak to prevent it.

### run1k — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, single GPU, headless, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000

**Command:**
```bash
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 1024 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Changes vs run1j:**
- `finger_curl_reg` ADR: (-0.3, -1.0) → (-0.5, -1.2) — stronger curl penalty to prevent thumb wedging

**Why:** Thumb still curling inward despite init randomization. Need stronger curl penalty to make the contact-via-wedging strategy unprofitable.

**Result:** Thumb behavior improved but policy not lifting. Contact reward (28.5) dominates lift (3.6) — policy maximizes finger contact without lifting.

### run1l — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, single GPU, headless, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000

**Command:**
```bash
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 1024 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Changes vs run1k:**
- `hand_object_contact_weight`: 8.0 → 4.0 (halved)
- `good_grasp_weight`: 6.0 → 3.0 (halved)
- `lift_weight`: unchanged (40, 20)

**Why:** Contact reward dominated at 28.5 vs lift at 3.6. Halving contact/grasp should make lift competitive once discovered.

**Result:** Contact down to 12.9, still 4x lift (3.4). Policy not lifting — hand approaches sideways, curls fingers around object, maximizes contact but can't lift from that configuration.

### run1m — 2026-03-30
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, single GPU, headless, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000

**Command:**
```bash
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 1024 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Changes vs run1l:**
- `hand_object_contact_weight`: 4.0 → 3.0
- `thumb_mcp_pitch` init: 0.05 → 0.0 (fully open)
- `lift_sharpness`: 5.0 → 2.0 (much flatter gradient — lift reward ~18 at table height vs ~5.4 before)

**Why:** Policy not attempting to lift at all. Flatter lift gradient gives signal even with object on table, should incentivize upward arm motion.

**Result:** Success! Policy lifts and reaches ADR 14/50. Rewards well balanced: lift (15.4) > obj_to_goal (10.8) > contact (4.8). 33% in_goal, 29% success, 39% lifted. Plateaued at ADR 14 after ~33k epochs (~3 hrs stuck). vel_explode=20/cycle. Thumb velocity at 7.2 rad/s at ADR 14 — approaching real hardware limits.

**Stored policy:** `stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/`
**Pretrained ckpt:** `pretrained_ckpts/best_dextrah_tekken_lstm_adr14.pth`

#### Per-object standalone eval (200 eps/object, 2026-04-14)

Eval JSON: `rl_games/logs/eval_tb_20260414_024531/eval_metrics_20260414_031655.json`

Run command:
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=3 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python eval_teacher.py \
  --task dextrah_fr3_agilehand --headless --num_envs 26 --eval_episodes 100 \
  --checkpoint /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --objects_dir multi_objects/visdex_selected
```

Aggregate: **82.7% lift, 22.3% unsafe** (2600 eps total). Matches the historical 480-ep aggregate (85.8% / 23.1%) within sample variance.

| Object | Lift | Unsafe | Coll. | OOB | Palm | Phys. |
|---|---:|---:|---:|---:|---:|---:|
| basketball_shoe       |  99.0% | 32.0% |  7.0% | 21.0% |  0.0% |  4.0% |
| closed_fist           | 100.0% | 19.5% |  0.0% |  9.0% |  0.0% | 10.5% |
| elephant_toy          | 100.0% | 22.0% |  2.0% |  0.5% |  0.5% | 19.0% |
| homer                 |  99.0% | 18.5% |  2.0% | 10.5% |  0.0% |  6.0% |
| mario                 |  98.0% |  8.5% |  0.0% |  1.0% |  0.0% |  6.5% |
| milk_pot              |  96.5% | 15.0% |  0.5% |  5.0% |  0.0% |  9.5% |
| plane                 |  83.5% | 32.5% |  1.5% |  2.5% |  8.0% | 20.5% |
| teddy_bear            |  99.5% | 21.5% |  0.5% | 14.5% |  0.0% |  6.5% |
| toy_bagger            |  99.0% | 25.5% |  2.0% |  7.5% |  1.0% | 15.0% |
| toy_cow               | 100.0% | 23.5% |  0.0% | 12.0% |  0.5% | 11.0% |
| tutle_candle_holder   | 100.0% | 35.0% |  0.0% | 24.0% |  0.0% | 10.5% |
| chicken_head_in_car   |   0.0% |  4.5% |  0.0% |  0.0% |  0.0% |  4.5% |
| train                 |   0.0% | 31.5% |  2.0% |  7.5% |  0.5% | 20.5% |

(Reason columns are % of all episodes, summing to the total Unsafe column.)

**Findings:**
- 11 of 13 objects achieve ≥83% lift; **chicken_head_in_car and train fail completely (0% lift)** — likely shape/scale outliers the teacher never learned.
- `basketball_shoe`, `tutle_candle_holder`, `train` show high `object_out_of_bound` (21-24%) — the lift trajectory throws the object out of the workspace.
- `plane`'s 8% palm_flipped is unusually high — only object where palm pose collapses systematically.
- Physics instability dominates the unsafe budget for 6/13 objects (sim artifact, not policy failure).

### run2a — Teacher v2: hardware-realistic starting limits (2026-04-03)

**Branch:** `fr3_agilehand_teacher_v2` (commit `32a8924`)
**Checkpoint:** none — fresh start from random init
**Config:** 1024 envs, single GPU, headless, starting_adr=0, min_steps_for_dr_change=3k, max_epochs=100000

**Motivation:** Teacher 11 (run1m) plateaued at ADR 14/50 with large curriculum gaps — the policy learned with unrealistically permissive starting limits (100 Nm arm torque, 10 rad/s thumb velocity, 20° thumb stiffness) and then struggled when ADR tightened them toward hardware values. Teacher v2 starts near hardware from the beginning, reducing the curriculum gap and forcing the policy to learn under realistic constraints from step 0.

**Changes vs run1m (Teacher 11):**

| Parameter | Teacher 11 (run1m) | Teacher v2 (run2a) | Why |
|---|---|---|---|
| `soft_joint_pos_limit_factor` | 0.9 | 0.8 | More margin for mimic joint physics stability |
| `franka_arm effort_limit_sim` (j1-4) | 100 Nm | 90 Nm | FR3 hardware spec |
| `franka_joints_ee effort_limit_sim` (j5-7) | 50 Nm | 20 Nm | FR3 hardware spec |
| `thumb_rot stiffness` | 20 | 60 | 3× stiffer for better position tracking |
| `thumb_rot velocity_limit_sim` | 20.0 rad/s | 0.2618 rad/s (~15 deg/s) | Near hardware (~8-12 deg/s) |
| `thumb_rot_vel_limit` ADR start | 10.0 rad/s | 0.2618 rad/s | Start near hardware, tiny ramp to 0.1396 |
| `arm_14_effort_limit` ADR start | 100 Nm | 90 Nm | Start at hardware, ramp to 87 |
| `arm_57_effort_limit` ADR start | 50 Nm | 20 Nm | Start at hardware, ramp to 12 |
| Arm joint init randomization | None (only ADR `robot_spawn`) | ±0.2 rad EventTerm from step 0 | Diverse approach trajectories from the start |
| `robot_spawn` finger noise | Applied to all joints | Arm joints only | Fingers start at fixed init positions |

**Key design principle:** Minimal curriculum. Instead of starting easy (high torque, fast thumb) and ramping to hard (hardware limits), start at hardware limits and let the policy learn under realistic constraints. The remaining ADR ramps are tiny:
- arm j1-4 effort: 90→87 Nm (3% reduction)
- arm j5-7 effort: 20→12 Nm (40% reduction — most significant remaining curriculum)
- thumb velocity: 0.2618→0.1396 rad/s (15→8 deg/s)

**Expected outcome:** Slower early learning (harder constraints from start), but higher ADR ceiling (no cliff when curriculum tightens). If the policy can learn to lift under these constraints, it should transfer better to hardware.

**Result:** *running*
