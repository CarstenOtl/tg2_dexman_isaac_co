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

**Result:** Rapid ADR progression to 13/50, then plateau. Policy couldn't push past ADR 13 — analysis showed lift reward decaying (40→~35) while object-to-goal sharpness increasing (making goal reward harder to earn), creating a reward gap mid-curriculum.

**Eval (ep 6500, best reward, ADR ~13):** 640 episodes, 32 envs
- **lift_success: 78.3%** | **unsafe_rate: 21.6%**
- Failure breakdown: physics_instability 75.4%, object_out_of_bound 21.0%, harmful_collision 2.9%, palm_flipped 0.7%, hand_too_far 0.0%
- Checkpoint: `logs/rl_games/dextrah_tekken_lstm/04-03_20-36-28/nn/last_dextrah_tekken_lstm_ep_6500_rew_15365.232.pth`
- Eval JSON: `logs/eval_tb_20260404_124555/eval_metrics_20260404_125132.json`
- Config: 2048 envs (vs documented 1024), single GPU

**Comparison vs Teacher 11:** lift 78.3% vs 85.8% (−7.5pp), unsafe 21.6% vs 23.1% (−1.5pp, slightly better). Competitive despite starting with much harder actuator constraints and only reaching ADR 13 vs 14.

**Next:** run2b — adjust reward curriculum to close the mid-ADR gap: `lift_weight` (40,20)→(40,30), `object_to_goal_sharpness` (-5,-10)→(-8,-13). Stronger goal gradient from step 0 + slower lift decay.

### run2b — reward curriculum + tighter finger gains (2026-04-04)

**Branch:** `fr3_agilehand_teacher_v2`
**Config:** 2048 envs, single GPU, headless, starting_adr=0

**Changes vs run2a:**
- `lift_weight`: (40, 20) → (40, 30) — slower decay
- `object_to_goal_sharpness`: (-5, -10) → (-8, -13) — stronger goal gradient from start
- Finger gain min: 0.5 → 0.7 (all finger groups: mcp_pitch, mcp_yaw, pip, thumb_rot)

**Result:** Same ADR 13 plateau. Starting goal sharpness at -8 was too aggressive — policy lifted objects consistently but couldn't earn goal reward at typical distances. Lift reward declined from 23.4 → 17.7, lift_success peaked at 0.53 (ADR 3) then decayed to ~0.35. Object-to-goal reward also declined from ~15 → ~10. Crucially, these metrics continued declining **within** ADR 13 (fixed randomization), confirming the reward curriculum itself was eroding the learning signal — not just the randomization.

**Diagnosis:** `lift_sharpness=2.0` too flat → policy earns lift reward by hovering near table without committing to full lift. Combined with increasing goal sharpness and finger curl penalty, the policy drifts toward a conservative, low-lifting strategy.

### run2c — lift sharpness + goal reward rebalance (2026-04-04)

**Branch:** `fr3_agilehand_teacher_v2`
**Config:** 2048 envs, single GPU, headless, starting_adr=0

**Changes vs run2b:**

| Parameter | run2b | run2c | Why |
|---|---|---|---|
| `lift_sharpness` | 2.0 | 4.0 | Steeper saturation — stop rewarding "hover near table", force policy to fully lift |
| `object_to_goal_sharpness` | (-8, -13) | (-5, -10) | Reverted start to -5 for broad early gradient; less aggressive terminal |
| `finger_curl_reg` | (-0.5, -1.2) | (-0.3, -0.8) | Previous penalty too aggressive, discouraged grasps at higher ADR |
| `success_bonus_weight` | 10 | 20 | Stronger discrete reward for reaching goal tolerance (0.1m) |

**Key principle:** Make lift reward saturate faster (sharpness 4.0 vs 2.0) so goal reward becomes the dominant signal once object is off table. Broader goal gradient at start (-5) for easier discovery, gentler curl penalty to allow aggressive grasps.

**Result:** Same ADR 13 plateau. in_success_region peaked at ~25%, never reaching 0.4 threshold. hand_to_object_distance increased sharply after ADR 12 (~5k epochs), with hand_to_object_reward dropping correspondingly — the hand couldn't reliably reach the object. Reward and lift_success continued declining within ADR 13 (fixed randomization level).

**Root cause identified:** `robot_spawn.joint_pos_noise` (0, 0.8) at ADR 13 gives ±0.21 rad, plus ±0.2 rad EventTerm = ±0.41 rad total (23°). kuka_allegro uses (0, 0.35) with no EventTerm AND has FABRICS smoothing. fr3_agilehand at ADR 13 has 4.5× more arm randomization than kuka_allegro at the same level, with no motion planner to smooth recovery.

### run2d — reduced spawn noise + reward tuning (2026-04-05)

**Branch:** `fr3_agilehand_teacher_v2`
**Config:** 2048 envs, single GPU, headless, starting_adr=0, visdex_selected (13 objects)
**Log dir:** `logs/rl_games/dextrah_tekken_lstm/04-05_14-05-55/`

**Changes vs run2c:**

| Parameter | run2c | run2d | Why |
|---|---|---|---|
| `robot_spawn.joint_pos_noise` | (0, 0.8) | (0, 0.35) | Matched to kuka_allegro; EventTerm ±0.2 rad still provides diversity |

**Cumulative changes from run2a (all still active):**
- `lift_sharpness`: 2.0 → 4.0
- `lift_weight`: (40, 20) → (40, 30)
- `object_to_goal_sharpness`: (-5, -10) — reverted to original
- `finger_curl_reg`: (-0.5, -1.2) → (-0.3, -0.8)
- `success_bonus_weight`: 10 → 20
- Finger gain min: 0.5 → 0.7
- `robot_spawn.joint_pos_noise`: (0, 0.8) → (0, 0.35)

**Hypothesis:** Reducing arm spawn noise will keep hand_to_object_distance manageable through ADR 13+, allowing the reward tuning from run2c to take effect.

**Result:** Same ADR 13 plateau despite reduced spawn noise. hand_to_object_distance still increased from ADR 9 (~4k epochs), hand_to_object_reward decreased correspondingly. vel_explosions increased from 10-20 to 25-50. `lifted_now` decayed from 55% (early ADR) → 40%, `in_goal_now` hovered 35-37%, `in_success_region` peaked at ~40% to trigger ADR but settled at 30% at ADR 13. Arm gain randomization (0.5, 2.0) likely the remaining culprit — 0.5× stiffness degrades arm tracking, 2× causes overshoot/vel_explosion, and no FABRICS to smooth it.

**Key insight:** With 13 curated objects (visdex_selected), hard objects (~4-5) rarely lift, permanently dragging `in_success_region` below 0.4 threshold. kuka_allegro trains on 150 objects where easy shapes dominate the average.

### run2e — full object set visdex_objects (2026-04-05)

**Branch:** `fr3_agilehand_teacher_v2`
**Config:** 2048 envs, single GPU, headless, starting_adr=0, visdex_objects (152 objects, ~13 envs/object)
**Log dir:** `logs/rl_games/dextrah_tekken_lstm/04-05_19-59-06/`

**Changes vs run2d:** Object set only — `visdex_selected` (13 curated) → `visdex_objects` (152 ShapeNet). All other params unchanged from run2d.

**Hypothesis:** More objects with many easy/compact shapes (mugs, boxes, bottles) will keep `in_success_region` average above 0.4 through higher ADR levels, matching kuka_allegro's training regime. Tradeoff: thin coverage per object (~13 envs each) may cause noisy gradients.

```bash
cd dextrah_lab/rl_games && \
CUDA_VISIBLE_DEVICES=1 python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=visdex_objects \
  env.use_cuda_graph=False
```

**Result:** Failed to learn — no lifting, no ADR progression, no in_success_region. 2048 envs / 152 objects = ~13 envs/object, too thin for stable gradients.

### run2f — visdex_objects with 4096 envs (2026-04-05)

**Branch:** `fr3_agilehand_teacher_v2`
**Config:** 4096 envs, single GPU, headless, starting_adr=0, visdex_objects (152 objects, ~27 envs/object)
**Log dir:** `logs/rl_games/dextrah_tekken_lstm/04-05_22-29-29/`

**Changes vs run2e:** Doubled envs 2048→4096, minibatch 4096→8192. All other params unchanged.

```bash
cd dextrah_lab/rl_games && \
CUDA_VISIBLE_DEVICES=1 python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 4096 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=visdex_objects \
  env.use_cuda_graph=False
```

**Hypothesis:** 27 envs/object should provide enough gradient signal per object. If OOMs on single 4090, will need multi-GPU.

**Result:** Same ADR 13 plateau. in_goal ~30% increasing slowly but never triggered ADR past 13. ~160 vel_explosions, 26 palm_flips at 4096 envs. lift_success peaked at 60% (ADR 6, ~2.5k epochs), decayed to 40% by ADR 13 (~5k epochs) and stayed flat. 152 objects didn't help — the ADR 13 wall is independent of object set.

**Conclusion after runs 2a–2f:** ADR 13 wall is consistent across all reward configs, spawn noise settings, and object sets. The one unchanged parameter: `arm_joint_stiffness_and_damping` (0.5, 2.0). vel_explosions climb at ADR 13 as arm gain randomization destabilizes physics.

### run2g — tighter arm gain randomization (2026-04-06)

**Branch:** `fr3_agilehand_teacher_v2`
**Config:** 2048 envs, single GPU, headless, starting_adr=0, visdex_selected (13 objects)
**Log dir:** `logs/rl_games/dextrah_tekken_lstm/04-06_12-49-15/`

**Changes vs run2d (back to visdex_selected baseline):**

| Parameter | run2d | run2g | Why |
|---|---|---|---|
| `arm_joint_stiffness_and_damping` | (0.5, 2.0) | (0.7, 1.5) | Reduce arm physics instability — 2× stiffness causes overshoot/vel_explosion, 0.5× makes arm too sluggish to reach objects |

All other params unchanged from run2d (lift_sharpness=4.0, lift_weight=(40,30), object_to_goal_sharpness=(-5,-10), finger_curl_reg=(-0.3,-0.8), success_bonus=20, finger gains min 0.7, robot_spawn noise (0,0.35)).

```bash
cd dextrah_lab/rl_games && \
CUDA_VISIBLE_DEVICES=1 python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
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

**Hypothesis:** Tighter arm gains (0.7, 1.5) will reduce vel_explosions and keep arm tracking stable through ADR 13+, finally breaking through the wall. This is the one parameter unchanged across all previous runs.

**Result:** Same ADR 13 plateau. Arm gain tightening did NOT break the wall — the issue is not arm physics instability.

**Eval (640 episodes, 32 envs, ADR 0):**

| Checkpoint | Lift | Unsafe | Failure breakdown |
|---|---|---|---|
| `dextrah_tekken_lstm.pth` (best reward auto-save) | **79.1%** | 25.3% | OOB 42%, phys_inst 37%, harmful_collision 14%, palm_flip 7% |
| ep 6500 (settled at ADR 13) | 75.9% | 23.1% | phys_inst 63%, OOB 29%, harmful_collision 8% |

- Eval JSONs: `logs/eval_tb_20260407_101145/eval_metrics_20260407_103836.json`, `logs/eval_tb_20260407_102855/eval_metrics_20260407_104305.json`
- `dextrah_tekken_lstm.pth` is the rl-games auto-saved best-reward checkpoint and is the best Teacher v2 result across all runs.

**Stored policy:** `stored_policies/fr3_agilehand/12_teacher_v2_hw_realistic_adr13_04-06_12-49-15/nn/dextrah_tekken_lstm.pth`

**Critical insight:** Eval at ADR 0 (no randomization) shows ep 6500 (later, more training, settled at ADR 13) has *worse* lifting than ep 2500 (earlier). A policy trained at ADR 13 should do **better** at ADR 0 (easier conditions), not worse. This proves the training process at ADR 13 is **actively degrading** lifting ability — the policy is forgetting how to lift while trying to handle harder conditions. The within-ADR reward decay (lift_weight 40→30) is washing out learned behavior.

**Final Teacher v2 results vs Teacher 11 baseline:**

| Run | Checkpoint | Lift | Unsafe | Notes |
|---|---|---|---|---|
| Teacher 11 (baseline) | — | **85.8%** | 23.1% | reached ADR 14, unrealistic starting limits |
| run2a | ep 6500 | 78.3% | **21.6%** | hardware limits, ADR 13 |
| **run2g** | **`dextrah_tekken_lstm.pth`** | **79.1%** | 25.3% | best Teacher v2, stored as policy 12 |
| run2g | ep 6500 | 75.9% | 23.1% | post-decay |

**~7pp lift gap to Teacher 11 — Teacher v2 hardware-realistic constraints pay a real cost.**

**Conclusion:** ADR 13 plateau is invariant across all tested parameter changes. The wall is structural — likely a combination of:
1. Lift reward decay actively degrading learned behavior within a fixed ADR level
2. `success_for_adr=0.4` threshold too strict for fr3_agilehand (vs kuka_allegro's 152-object easy-shape average)
3. No FABRICS smoothing means direct joint control is more sensitive to physics randomization

**Next options (untested):**
- **Flatline `lift_weight` (40, 40)** — remove the decay that's degrading lifting
- **Lower `success_for_adr` to 0.3** — let policy advance through ADR 13 with current performance
- **Resume from ep 2500 with `starting_adr_increments=14`** — skip the wall (caveat: LSTM hidden state lost on resume)

### Teacher v2 evaluated on visdex_top8 (2026-05-07)

First eval of Teacher v2 (run2g, policy 12, trained on visdex_selected = 13 objects) on `multi_objects/visdex_top8` — the same 8-object subset used as student environment in exp_04 distillation. Goal: get a teacher number directly comparable to student visdex_top8 lift rates without relying on post-hoc per-object subsetting.

**Surface bug found and fixed during this eval:** the `teacher_objects_dir` index-remap mechanism (introduced in main commit `0dc3362` to fix one-hot identity scrambling for distillation runs) was gated to `self.cfg.distillation` only. Teacher eval runs with `distillation=False`, so the gate skipped the remap and 7/8 objects got wrong one-hot indices, producing **26.9% lift / 35.0% unsafe** before the fix.

Generalized the gate (env.py line 672, dropped `and self.cfg.distillation`) and added `--teacher_onehot_size` and `--teacher_objects_dir` flags to `eval_teacher.py` to match the distillation runner's interface.

**Sanity check after fix** — env logged the expected visdex_selected-aligned mapping:
```
basketball_shoe→0, closed_fist→2, elephant_toy→3, mario→5,
milk_pot→6, teddy_bear→8, toy_bagger→9, tutle_candle_holder→12
```
(8-object subset, alphabetical visdex_selected positions, *not* sequential 0-7.)

**Eval command:**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 python eval_teacher.py \
  --task dextrah_fr3_agilehand --headless \
  --checkpoint stored_policies/fr3_agilehand/12_teacher_v2_hw_realistic_adr13_04-06_12-49-15/nn/dextrah_tekken_lstm.pth \
  --objects_dir multi_objects/visdex_top8 \
  --teacher_onehot_size 13 \
  --teacher_objects_dir multi_objects/visdex_selected \
  --num_envs 32 --eval_episodes 10 \
  --file_name_head teacher_v2_visdex_top8_FIXED
```

**Result (320 episodes, ADR 0):**

| Metric | Teacher v2 visdex_top8 | Teacher v2 visdex_selected (run2g baseline) | Teacher 11 visdex_top8 (main repo, post-hoc) |
|---|---|---|---|
| Lift success | **59.4%** | 79.1% | 99.0% |
| Unsafe rate | 46.6% | 25.3% | 22.4% |
| object_out_of_bound (% of unsafe) | 64.4% | — | — |
| physics_instability (% of unsafe) | 22.1% | — | — |
| harmful_collision (% of unsafe) | 10.7% | — | — |
| palm_flipped (% of unsafe) | 2.7% | — | — |

**JSON:** `dextrah_lab/rl_games/logs/eval_tb_20260507_211757/teacher_v2_visdex_top8_FIXED_20260507_212113.json`

**Apples-to-apples with Teacher 11.** The `multi_objects/visdex_top8` directory contents are bit-identical between this test repo and the main repo (`diff -rq` empty), so the 59.4% Teacher v2 number is directly comparable to the 99.0% Teacher 11 visdex_top8 baseline computed on main. **Teacher v2 is ~40pp behind Teacher 11** on the same 8 objects.

**Interpretation:** Teacher v2 drops ~20pp on visdex_top8 vs its native 13-object set (79.1% → 59.4%) — the hardware-realistic ADR 13 training generalizes worse on this subset than the easier-curriculum Teacher 11 (which actually peaks on top8 because the 13-set average was dragged down by two unliftable objects, train/chicken_head_in_car). Failure mode is dominated by `object_out_of_bound` (64.4% of unsafe episodes), suggesting the policy commits to lift trajectories that fling the object off the table on harder objects.

### Teacher v2 retrain attempt — replay run2a after state drift (2026-05-08)

**Branch:** `fr3_agilehand_teacher_v2`
**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_00-18-43/` (failed run, kept for forensics)

**Motivation:** During the visdex_top8 eval work on 2026-05-07, the original Teacher v2 (run2g, policy 12) measured **52.2% lift** on visdex_selected — a 27pp drop from the Apr 7 baseline of 79.1%. After eliminating env_cfg, env.py, eval_teacher.py, and dextrah_lab code as differences (all bit-identical), the only remaining variable was system state: IsaacLab was upgraded `v2.2.1 → v2.3.0` on 2026-05-06 (different env's needs), then reverted to v2.2.1 on 2026-05-07 with a `./isaaclab.sh -i` reinstall. The Apr 7 79.1% was on pre-toggle IsaacLab; today's measurements are on post-reinstall IsaacLab. Decided to retrain Teacher v2 from scratch under the (currently-installed) IsaacLab v2.2.1 to get a deployable teacher for hardware testing.

**First retrain attempt (05-08_00-18-43, failed):**
Used the committed env_cfg.py state at HEAD (which reflects run2g's *committed* values: lift_sharpness=4.0, success_bonus_weight=20, finger gain ADR (0.7,2.0), arm gain ADR (0.7,1.5), joint_pos_noise (0,0.35), lift_weight (40,30), finger_curl_reg (-0.3,-0.8)). Ran 1024 envs, single GPU, headless, seed 42, 18k epochs.

**Result:** Training never learned to lift. `lift_success` peaked at 0.008 around epoch 470, then collapsed to ~0 for the entire remaining 17k epochs. `num_adr_increases` stayed at 0 for the full run. Compared to run2g at the same step:

| Epoch | This run lift | run2g lift | This run ADR | run2g ADR |
|---|---|---|---|---|
| 100 | 0.000 | 0.014 | 0 | 0 |
| 500 | 0.002 | 0.160 | 0 | 0 |
| 1000 | 0.001 | **0.571** | 0 | 0 |
| 6000 | 0.000 | 0.433 | 0 | 13 |
| 18000 | 0.000 | 0.269 | 0 | 13 |

Shaped rewards were non-zero (`lift_reward≈5`, `contact≈6`) — the policy interacts with objects but never completes a sustained lift. Stuck local optimum.

**Diagnosis:** Discovered that the *committed* env_cfg state never matched what was actually used during run2a → run2g training. The env_cfg.py at commit `4fe7cb7` (Apr 7 13:53, "Teacher v2 runs 2a-2g") was committed *after* runs 2a-2g already finished training; the user accumulated uncommitted edits across the run2a→run2g iterations and committed them all at the end. Specifically:
- The committed state has run2g's *cumulative* settings (the iterative sequence's endpoint)
- Each individual run (2a, 2b, ..., 2g) used a different intermediate working-tree state
- run2a's actual training state = commit `32a8924` (Apr 3) — the only state that was both committed AND matched a documented successful run

This is why retraining against committed HEAD doesn't reproduce any documented run.

**Decision:** Replay run2a using its documented committed state (`32a8924`) — it reached ADR 13 with 78.3% lift / 21.6% unsafe (per exp_03 run2a entry above), which is a deployable teacher for hardware. run2a's settings were:
- `lift_sharpness=2.0`, `success_bonus_weight=10.0`, `finger_curl_reg_weight=-0.2` (base term weights)
- ADR ranges: `joint_pos_noise=(0., 0.8)`, `arm_joint_stiffness_and_damping=(0.5, 2.)`, `finger gains=(0.5, 2.)`, `lift_weight=(40., 20.)`, `finger_curl_reg=(-0.5, -1.2)`
- All hardware-realistic actuator limits and `arm_joint_init` EventTerm intact (these were always in 32a8924)

**Revert command:**
```bash
git checkout 32a8924 -- dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py
```

**Verified:** asset files (`FR3_tekkenadof_left.usd`, `fr3_tekken_left.py`) unchanged between 32a8924 and HEAD; env.py only adds the no-op `teacher_objects_dir` mechanism.

**Retrain command:**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Success criteria:**
- `lift_success` crosses 0.1 by ep ~370 (run2g's pattern; run2a expected similar)
- `num_adr_increases` starts climbing from ep ~2000
- Reaches ADR 13 around ep ~6500, with lift ~78%
- Best-reward checkpoint (`dextrah_tekken_lstm.pth`) becomes the deployable Teacher v2 replacement for policy 12

**Important caveat going forward:** The 79.1% historical baseline for Teacher v2 (policy 12 .pth) is no longer reproducible from current state — system-state drift between Apr 6 training and today's evals shifts measurement by ~25pp. Any new comparisons should use today's reproducible numbers (52.2% on visdex_selected with current code/IsaacLab). After this retrain succeeds, the new policy will replace policy 12 as the canonical Teacher v2.

#### Update — seed=42 retrain on run2a env_cfg also failed (2026-05-08)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_12-05-22/` (failed)

After reverting env_cfg.py to commit `32a8924` (run2a's documented committed state), retrained from scratch with same train command as run2a (1024 envs, single GPU, seed=42, IsaacLab v2.2.1). Training failed identically to last night's attempt — `lift_success` never crossed 0.003 through 2700+ epochs.

**Side-by-side vs run2a (Apr 3) at matched epochs:**

| Epoch | Today (run2a env_cfg, seed=42) | run2a (Apr 3) |
|---|---|---|
| 100 | 0.000 lift | 0.014 lift |
| 500 | 0.000 lift | 0.160 lift |
| 1000 | 0.000 lift | **0.571 lift** |
| 2000 | 0.000 lift | 0.547 lift, ADR 2 |

Identical "touch-but-don't-lift" basin: `lift_reward≈14`, `contact≈3.5`, `object_to_goal≈3.4` — policy interacts with the object and elevates it slightly (enough for the shaped lift reward to saturate at sharpness=2.0) but never reaches the 15cm `object_height_thresh` for binary lift_success.

**Conclusion:** env_cfg is definitively NOT the cause. The exact same env_cfg that worked in April fails today. Remaining suspects: IsaacLab v2.2.1 reinstall today (subtly different file state vs April's v2.2.1), GPU floating-point non-determinism, or other system-level state drift since April.

#### seed=1 attempt (2026-05-08, in progress)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_13-39-55/`
**GPU:** CUDA_VISIBLE_DEVICES=2

Same train command as the failed seed=42 retrain, only `--seed 1` and capped `max_epochs=2000` (~30 min). If a different seed escapes the touch-don't-lift basin, the failure was seed-specific GPU-non-determinism. If seed=1 also fails, environment drift is structural and we accept policy 12 (the existing Teacher v2 .pth) at 52-55% lift as the canonical Teacher v2 for hardware testing.

**Decision rule:**
- `lift_success > 0.1` by ep ~500 → seed=1 found a working basin, let it run longer to reproduce ADR 13
- Still flat at ep 500 → try seed=7 or accept policy 12 as-is

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 1 \
  --num_envs 1024 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=2000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

#### seed=1 result and final decision (2026-05-08)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_13-39-55/` (seed=1, run2a env_cfg, ~600 iters)

**Result:** Identical failure to seed=42. Same touch-don't-lift basin:
- `lift_reward` ramps to ~12-13 by iter 400 (object elevated slightly, lift-shape reward saturates flat)
- `lift_success` stays at ~0 (max spike ~1e-3, no episode crosses 15 cm threshold)
- `num_adr_increases` stuck at 0
- TensorBoard plots show monotonic plateau, not learning

**Three failed retraining attempts in a row** (all with the same touch-don't-lift basin):

| Run dir | Env_cfg state | Seed | Outcome |
|---|---|---|---|
| `05-08_00-18-43` | run2g (committed HEAD) | 42 | ADR 0, lift_success=0 through 18k epochs |
| `05-08_12-05-22` | **run2a (32a8924)** | 42 | ADR 0, lift_success=0 through 2.7k epochs |
| `05-08_13-39-55` | run2a (32a8924) | 1 | ADR 0, lift_success≈1e-3 through ~600 iters |

This is **structural, not random variance** (six consecutive successes in April, three consecutive failures today). Some system-level state has shifted since April that we cannot pinpoint with the diagnostics available. Verified-identical things between April and now:
- `env_cfg.py` content (reverted bit-identical to 32a8924)
- `env.py` content (only no-op `teacher_objects_dir` mechanism added; finger spawn-noise zeroing intact since 32a8924)
- Asset USDs and `fr3_tekken_left.py` (unchanged blob hashes)
- IsaacLab git ref (`v2.2.1`, `git status` clean, `__pycache__` cleared)
- `dextrah_test` conda env packages (Mar 25 mtime, identical pip freeze except numpy patch)

**Plausible mechanism (not testable cheaply):** Teacher v2 was trained at hardware-realistic actuator limits (15 deg/s thumb velocity, 12-87 Nm arm efforts) — sitting near the edge of what's physically possible to lift in sim. A subtle physics-engine numerical change (driver, CUDA toolkit, omni.physx state) invisible to Teacher 11's evaluation could push Teacher v2 across the "can't initiate lift" threshold both in eval (52.2% vs 79.1%) and in training (no learning at all from random init). Teacher 11 has actuator margin and is unaffected.

### Final decision: policy 12 is the canonical Teacher v2 (2026-05-08)

`stored_policies/fr3_agilehand/12_teacher_v2_hw_realistic_adr13_04-06_12-49-15/nn/dextrah_tekken_lstm.pth` is the deployable Teacher v2 for hardware testing.

**Reproducible measurements with current system state:**
- visdex_selected, hold-gated 0.5s: **55.0% lift / 40.8% unsafe** (640 ep)
- visdex_top8, hold-gated 0.5s: **59.4% lift / 46.6% unsafe** (320 ep)
- visdex_selected, instantaneous: **65.6% lift / 37.7% unsafe**

**Historical Apr 7 measurement (no longer reproducible):** 79.1% / 25.3%. This was on pre-system-drift state and shouldn't be referenced in comparisons going forward.

**For hardware testing:** policy 12 is ADR-13-trained with the hardware-realistic curriculum (15 deg/s thumb velocity, 12-87 Nm arm efforts) — exactly the constraint regime intended for sim2real transfer. The lower sim numbers reflect that this teacher operates at the edge of feasibility under realistic actuator constraints, which is what you want when transferring to a robot with the same constraints.

**Cleanup actions:**
- Failed run dirs preserved for forensics: `05-08_00-18-43/`, `05-08_12-05-22/`, `05-08_13-39-55/`
- env_cfg.py left at 32a8924 state (run2a settings) for future Teacher v2 retraining attempts after system state is investigated
- `pre_reset_recovery` git tag from 2026-05-07 reset experiment can be deleted once cleanup confirmed

#### Clean-state replay attempt — full revert to commit 32a8924 (2026-05-08)

**Why:** Three failed retraining attempts today on the post-merge HEAD (with various env_cfg states + seeds) all fell into the same touch-don't-lift basin. To rule out any working-tree contamination as the cause, did a full `git checkout 32a8924` putting EVERY dextrah_lab file (env.py, env_cfg.py, eval scripts, distillation scripts, ALL of it) into the exact state from Apr 3 20:16 — the run2a starting commit. This is the most aggressive possible repo-state revert.

**Pre-revert preservation:**
- Safety branch: `may08-pre-revert-snapshot-20260508_172739` (HEAD `8968c61`, all today's commits)
- File backup: `/tmp/dextrah_test_backup_20260508_172739/` (modified files + BC checkpoint .pth)

**State after `git checkout 32a8924`:**
- HEAD: `32a8924` Teacher v2: hardware-realistic actuator limits, reduced curriculum, arm init randomization (Apr 3 20:16)
- Detached HEAD (no branch)
- `lift_sharpness=2.0`, `success_bonus_weight=10.0` (run2a values)
- All dextrah_lab files at Apr-3 commit state (no working-tree modifications)
- All `__pycache__` cleared in test repo and IsaacLab
- IsaacLab still on `v2.2.1` tag, git status clean

**Train command (1024 envs, dextrah_test env, single GPU):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Expected (per run2a in April):**
- ADR climbs to 13 in ~4-7k epochs
- ep 6500 lift ~78%, unsafe ~22%

**Decision rule:**
- `lift_success > 0.1` by ep ~370 → previous failures were caused by working-tree contamination we couldn't pinpoint; this run will reach ADR 13 like run2a in April. New `dextrah_tekken_lstm.pth` becomes the canonical Teacher v2.
- `lift_success` still flat at ep ~500 → regression is below the dextrah_lab repo level (IsaacLab internals, system libs, drivers, hardware). Definitively accept policy 12 as the canonical Teacher v2 for hardware testing.

**Result:** *(to be filled in after training)*

**Restore today's investigation work after training finishes:**
```bash
git checkout may08-pre-revert-snapshot-20260508_172739
git stash pop
```

**Update — clean-state replay (32a8924, 1024 envs, seed=42) ALSO failed (2026-05-08 evening):**
- `lift_success` flat at ~0 throughout, with minor 1e-3 scale spikes (consistent with sparse per-object gradients pattern)
- Same touch-don't-lift basin as previous attempts
- Confirms the regression is below the dextrah_lab repo level — even at the bit-identical Apr 3 commit state, training fails

**The "minor spikes" pattern** matches CLAUDE.md's documented "16 envs in livestream mode" failure mode: with 13 objects across 1024 envs that's ~78 envs/object — should be enough on paper, but the spike-flatline pattern suggests gradient noise is dominating signal in some way today.

#### Replay attempt with 2048 envs (2026-05-08 evening)

**Hypothesis:** run2a's exp_03 entry notes "Config: 2048 envs (vs documented 1024), single GPU" — the original successful run actually used 2048 envs, NOT 1024. With 2048 envs / 13 objects = ~157 envs/object, double the gradient density of 1024. Earlier today's failed attempts all used 1024. Maybe the spike pattern is the symptom of insufficient per-object density and 2048 is the threshold that makes lifting discoverable.

**Train command (2048 envs, single GPU, scaled minibatch):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

`minibatch_size` doubled from 4096 → 8192 to match the doubled num_envs (per CLAUDE.md multi-GPU template scaling).

**VRAM:** 2048 envs on a 24GB 4090 should land ~18-20GB. Tight but should fit. If OOM, reduce minibatch back to 4096.

**Decision rule:**
- `lift_success > 0.1` by ep ~370 → was per-object density issue all along; 2048 envs reproduces run2a
- Still flat at ep ~500 → density was not the root cause; system-level regression confirmed; accept policy 12

**Result:** *(to be filled in)*

**Update — 2048-env replay also failed:** `lift_success` stayed flat with the same minor-spikes-around-zero pattern. Per-object density (2048/13 = 157 envs/obj) was not the differentiator either.

### Final decision (post-2048-env attempt)

**Total failed retraining attempts: 4.**
| Run | num_envs | Seed | env_cfg | Result |
|---|---|---|---|---|
| `05-08_00-18-43` | 1024 | 42 | run2g committed | flat lift, ADR 0 |
| `05-08_12-05-22` | 1024 | 42 | run2a reverted | flat lift, ADR 0 |
| `05-08_13-39-55` | 1024 | 1 | run2a reverted | flat lift, ADR 0 |
| `05-08_clean_state` | 1024 | 42 | 32a8924 full checkout | flat lift, ADR 0 |
| `05-08_2048env` | **2048** | 42 | 32a8924 full checkout | flat lift, ADR 0 |

Five distinct attempts, varying env_cfg state, seed, env count, and working-tree cleanliness — all fail in the identical touch-don't-lift basin. The regression is structural and below the dextrah_lab repo level. Without root-causing the system-level shift (driver/CUDA/IsaacLab internals between April and now), retraining Teacher v2 from scratch is not viable on this machine in its current state.

**Canonical Teacher v2 for hardware testing: policy 12** at
`stored_policies/fr3_agilehand/12_teacher_v2_hw_realistic_adr13_04-06_12-49-15/nn/dextrah_tekken_lstm.pth`

Reproducible eval today (IsaacLab v2.2.1, dextrah_test env):
- visdex_selected, hold-gated 0.5s: **55.0% lift / 40.8% unsafe**
- visdex_top8, hold-gated 0.5s: **59.4% lift / 46.6% unsafe**
- visdex_selected, instantaneous: 65.6% lift

The Apr 7 79.1% historical baseline is from pre-system-drift state and is no longer reproducible. Use 55% / 59.4% as the canonical Teacher v2 numbers going forward.

**Hardware deployment is the path forward.** Policy 12 was trained with hardware-realistic actuator limits (15 deg/s thumb, 12-87 Nm efforts), which is exactly the constraint regime intended for sim2real transfer. Lower sim numbers reflect the edge-of-feasibility operating point — testing on actual hardware is now the experiment that answers whether the approach works.

---

## Run 3 — Teacher v2 retraining attempts (2026-05-08 → 2026-05-11)

**Context:** During visdex_top8 eval work on 2026-05-07, discovered the stored Teacher v2 policy 12 measured 52.2% lift on visdex_selected — a 27pp drop from the Apr 7 79.1% baseline. After eliminating env_cfg, env.py, eval_teacher.py, IsaacLab tag, conda env packages, USDs, and asset files as differences (all bit-identical or LFS-content-identical to April), decided to retrain Teacher v2 from scratch to get a deployable teacher that's known-good in the current system state.

All run3x attempts share: IsaacLab v2.2.1 (reinstalled 2026-05-07), `dextrah_test` conda env, single GPU, max_epochs=100000, seed=42 (unless noted), wandb_activate=False, env.success_for_adr=0.4, env.objects_dir=multi_objects/visdex_selected.

### run3a — committed HEAD env_cfg (run2g state) (2026-05-08)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_00-18-43/`
**Config:** 1024 envs, env_cfg at HEAD commit `8968c61` (post-merge): lift_sharpness=4, success_bonus_weight=20, finger gain ADR (0.7, 2.0), arm gain ADR (0.7, 1.5), joint_pos_noise (0, 0.35).

**Result:** Trained 18,000 epochs. `lift_success` peaked at 0.008 around ep 470, then collapsed to ~0 for the entire remaining 17.5k epochs. `num_adr_increases` stuck at 0. Shaped `lift_reward` saturated at ~12-14 (object elevated slightly, never above 15cm threshold). Same "touch-don't-lift" basin throughout.

### run3b — revert env_cfg to run2a state (32a8924) (2026-05-08)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_12-05-22/`
**Config:** 1024 envs, env_cfg reverted via `git checkout 32a8924 -- env_cfg.py`: lift_sharpness=2, success_bonus_weight=10, finger gain ADR (0.5, 2.0), arm gain ADR (0.5, 2.0), joint_pos_noise (0, 0.8), lift_weight (40, 20), finger_curl_reg (-0.5, -1.2).

**Result:** Identical failure to run3a. ~2,700 epochs, `lift_success` never crossed 0.003.

**Conclusion:** Reward weight differences between run2a (committed) and run2g (subsequent uncommitted edits) are not the cause.

### run3c — different seed (seed=1) (2026-05-08)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_13-39-55/`
**Config:** Same as run3b but `--seed 1`, GPU 2, max_epochs=2000.

**Result:** Same flat lift pattern, max ~0.001 lift through ~600 iters.

**Conclusion:** Failure not seed-specific GPU non-determinism. Rules out random variance.

### run3d — full clean-state checkout to 32a8924 (2026-05-08)

**Run dir:** included in run3e (consolidated)
**Config:** `git checkout 32a8924` (detached HEAD) — every dextrah_lab file at Apr 3 state, working tree clean, IsaacLab and dextrah_lab `__pycache__` cleared, 1024 envs.

**Result:** Same flat pattern through training.

**Conclusion:** Rules out any working-tree contamination from today's edits.

### run3e — 2048 envs matching run2a's actual config (2026-05-08 → 2026-05-11)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-08_19-35-53/`
**Config:** Identical to run3d but `--num_envs 2048`, `minibatch_size=8192`, `central_value.minibatch_size=8192`. This matches exp_03 run2a's actual config (the entry notes "2048 envs (vs documented 1024)"), addressing the hypothesis that per-object gradient density (157 envs/obj vs 78) was the missing factor.

**Result:** Left running 3 days, reached ~33,700 epochs. `lift_success` max was **0.004 at ep 81**, never crossed 0.01 for the entire run. `lift_reward` saturated at ~14, `num_adr_increases` stuck at 0 throughout. Side-by-side vs run2g at matched epochs:

| Epoch | This run lift | run2g lift |
|---|---|---|
| 100 | 0.000 | 0.014 |
| 500 | 0.000 | 0.160 |
| 1000 | 0.000 | **0.571** |
| 6000 | 0.000 | 0.433 |

**Conclusion:** Per-object density is not the cause. Five distinct attempts in a row (varying env_cfg state, seed, env count, working-tree cleanliness, run length) all fail in the identical touch-don't-lift basin — the regression is structural and below the dextrah_lab repo level.

### Visual inspection (2026-05-11)

Replayed run3e's ep 33,500 checkpoint via `play_test.py --livestream 2` for direct observation. **Observed failure mode**: during approach phase, the object physically knocks the thumb's MCP pitch joint backward and the thumb buckles inward, hindering grasp formation. Identical to the symptom documented in `exp_01_thumb_rot_curriculum.md` (2026-03-23):
> "the object physically knocks the thumb's MCP pitch joint backwards (joint collapses toward 0 / open position). Once the thumb is forced open by the object, the hand can no longer form a grasp and the episode fails."

That exp_01 issue was supposedly solved with: thumb_rot init at -0.3491 rad (-20°), `thumb_rot_init` EventTerm randomization, `thumb_mcp_pitch` init at 0.0, and raised `mcp_pitch` actuator stiffness to 10.0. **Verified all four fixes ARE in current env_cfg and asset config** (bit-identical to 32a8924). USDs LFS-content-identical to April (closed_fist.usd SHA256 matches LFS pointer). Despite all the documented fixes being in place, the symptom is back — pointing at a system-level physics change (IsaacLab internal, omni.physx, CUDA, etc.) that we cannot bisect.

### run3f — remove `thumb_rot_init` EventTerm (2026-05-11)

**Hypothesis:** The random thumb_rot offset (-20° to 0° every reset) may be preventing the policy from developing a consistent approach geometry under current physics. With deterministic -20° init, approach trajectory is fixed and learnable.

**Change:** Removed `thumb_rot_init` EventTerm entirely from env_cfg.py (line 113 area). `revolute_thumb_rot` now always starts at -0.3491 rad (-20°, joint min).

**Result:** Per user observation — "didn't work too well." Same touch-don't-lift pattern, no improvement in lift_success.

**Conclusion:** Random thumb_rot offset is not the differentiating factor.

### Contact-gating root-cause finding (2026-05-11)

After run3f's failure, investigated the lift_reward formula itself:
```python
contact_mask = (contact_count > 0.0).to(contact_count.dtype)
lift_reward = lift_weight * exp(-lift_sharpness * object_vertical_error) * contact_mask
object_to_goal_reward = object_to_goal_weight * exp(...) * contact_mask
```

**Critical finding:** `contact_count` is the *sum* of all per-sensor contact registrations (palm, all four distal phalanxes, thumb tip). `contact_mask` fires when ANY one of those reports contact. The thumb sensor (`Thumb_Distal_Phalanx`) is a single rigid body — its contact sensor reports contact regardless of whether the inner or outer surface touches the object.

**This means**: when the policy buckles the thumb backward during approach (the symptom seen in livestream), the *outer* surface of the buckled thumb scrapes the object → `contact_count++` → `contact_mask=1` → `lift_reward` gates open. The policy can farm shaped lift reward by repeatedly bumping the object with the buckled thumb, with no need to actually grasp. This is a **degenerate local optimum that the gate makes available** — and the policy converges on it instantly because it's strictly easier than learning to grasp.

The reward function already receives a stricter alternative — `good_grasp_mask = (finger_count ≥ 2) & thumb_contact` — but it's only used as a standalone `good_grasp_reward` term, not as the gate for `lift_reward`.

**Hypothesis for why this exploit didn't dominate in April**: subtle contact-physics changes (between Apr's IsaacLab v2.2.1 and today's reinstalled v2.2.1) shifted where contact forces register on the buckled thumb. In April, the buckled-thumb-scrape may have been less reliable; today it's reliable enough that the policy converges on the exploit. This is consistent with eval shifts too (52% vs 79% lift on the same .pth), since the deployed policy similarly suffers from contact-physics differences.

### run3g — close the contact-gating exploit + amplify correct signals (2026-05-11, starting)

**Changes (this commit/working-tree state):**

1. **`contact_mask = good_grasp_mask.to(contact_count.dtype)`** in `compute_rewards()` (env.py:2243). lift_reward and object_to_goal_reward now only fire when the policy has thumb contact AND ≥1 other finger in contact. The buckled-thumb-scrape exploit no longer opens the gate.

2. **`lift_weight` ADR `(40., 20.)` → `(60., 30.)`** (env_cfg.py:922). +50% stronger lift signal. Once the policy DOES achieve a proper grasp, lift reward is dominant enough to compete with shaped exploration rewards.

3. **`finger_curl_reg` ADR `(-0.5, -1.2)` → `(-0.8, -1.8)`** (env_cfg.py:923). +60% stronger curl penalty. Discourages the buckled thumb pose at the source, in addition to closing its reward exploit.

**Why these three together:** The structural fix (1) closes the degenerate local optimum the policy has been falling into. (2) and (3) shift the reward landscape so that the proper-grasp behavior is the steeper gradient, making it the path of least resistance for early exploration.

**Train command:**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Expected behavior if working:**
- `lift_reward/iter` should stay near 0 for first ~200-500 epochs (no exploit to farm; good_grasp_mask is rarely True at random init)
- `good_grasp_reward/iter` and `object_contact_count` climb as policy learns proper grasp
- `lift_success/iter` (binary at 15cm threshold) should first cross 0.1 by ep ~500-1000 if proper-grasp signal works
- Successful trajectory: by ep ~6,500 should reach ADR 13 like the original run2a, with `lift_success` ~50-70%

**Decision rule:**
- `lift_success > 0.1` by ep ~1,000 → exploit-closure worked; let it run to ADR 13 plateau
- `lift_reward` and `good_grasp_reward` both stay flat through ep 5,000 → gate too strict (no learning signal). Loosen: gate by `(contact_count >= 2)` instead of `good_grasp_mask`, OR add `(thumb_contact_alone)` gate (thumb contact required but not multi-finger)
- `lift_reward` climbs but `lift_success` stays flat → grasp signal works but lift still fails. Try: hand spawn closer to object via fr3_joint init changes

**Fallback options if run3g fails:**
- run3h: gate on `(contact_count >= 2)` (more permissive than good_grasp_mask but stricter than `>0`)
- run3i: hand spawn closer to object (adjust fr3_joint2 / fr3_joint4 init positions to bring palm ~10cm closer on average)
- run3j: thumb mcp_pitch actuator stiffness 10.0 → 20.0 in fr3_tekken_left.py (mechanically resist buckling at the actuator level)

### run3h — slower finger actuators + bumped curl penalty (2026-05-11)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-11_17-19-52/`

**Hypothesis:** Run3g's `good_grasp_mask` gate closed the buckled-thumb-scrape reward exploit, but `lift_success` still stayed at 0 through ep 700+. Visual replay showed the thumb still buckling inward despite the gate. Suspect the buckling itself is a contact-physics phenomenon — joints flung backward by object contact forces faster than the PD spring can resist. With `mcp_pitch` velocity_limit_sim=8 rad/s (~458 deg/s, ~30× hardware-realistic), a single bad timestep at 60 Hz allows up to 7.6° of joint motion — enough to drive the thumb_mcp_pitch from open (0°) to fully buckled.

**Changes from run3g:**
1. **mcp_pitch/yaw/pip velocity_limit_sim: 8.0 → 1.0 rad/s** (~57 deg/s, per-step motion cap 0.96°). Closes sim2real gap (real hardware ~30-60 deg/s) AND prevents single-step buckling — contact forces can only nudge a joint ~1° per step before the PD controller has time to react.
2. **`finger_curl_reg` ADR: (-0.8, -1.8) → (-1.5, -3.0)**. Further amplifies curl penalty since visual replay showed buckling persisted at the previous level.

**Result:** Through ep 327. `lift_success` flat at 0 throughout. `lift_reward` saturated at 16.4 (down from run3g's 17.9), `good_grasp_reward` 2.08 (down from 2.3), `object_contact_count` 2.66 (down from 3.07). `finger_curl_reg=−2.84` — **saturating at the existing −3.0 cap**. Slower fingers reduced over-aggressive multi-finger contact but did not stop the policy from finding the buckled-thumb pose.

**Visual replay (ep 322 area):** thumb still curling inward, but less aggressively than run3g — slower fingers reduce the speed of buckling but the policy still drives the joints there. Confirms two things: (a) the velocity cap helped (less violent buckling), but (b) the curl penalty isn't strong enough — the policy still finds the buckled pose profitable.

### run3i — raise curl penalty cap + slow thumb non-rot joints further (2026-05-11)

**Hypothesis:**
1. The current `finger_curl_reg` is clamped at -3.0 (`finger_curl_reg_min`), and run3h already saturated it. Raising the cap lets the bumped ADR weight (-1.5, -3.0) actually push the penalty further negative when the deviation is large (the L2 distance from `curled_q` is dominated by the thumb because curled_q has thumb_mcp_pitch=0 / open while other fingers target slight curl=0.1).
2. The thumb's non-rotation joints (`thumb_mcp_pitch`, `thumb_mcp_yaw`, `thumb_pip`) share the general finger actuator group — same 1.0 rad/s velocity limit as other fingers. To make buckling physically impossible at the source, slow the thumb-specific joints to half (0.5 rad/s ~ 28 deg/s, ~0.48°/step).

**Changes from run3h:**

| Setting | run3h | **run3i** | Why |
|---|---|---|---|
| `finger_curl_reg_min` (env_cfg.py:749) | -3.0 | **-8.0** | Raise the penalty cap so the bumped ADR weight can actually bite (run3h saturated at -3.0) |
| `thumb_mcp_pitch` velocity_limit_sim (fr3_tekken_left.py) | shared 1.0 rad/s | **0.5 rad/s** | Half-speed for thumb-specific buckling-prone joint |
| `thumb_mcp_yaw` velocity_limit_sim | shared 1.0 rad/s | **0.5 rad/s** | Match for consistency |
| `thumb_pip` velocity_limit_sim | shared 1.0 rad/s | **0.5 rad/s** | Match for consistency |
| Other fingers' velocity_limit_sim | 1.0 rad/s | 1.0 rad/s (unchanged) | Kept so non-thumb fingers can still close in reasonable time |

Implementation: split the regex `revolute_.*_mcp_pitch` (etc.) into two actuator groups — `finger_mcp_pitch` for `revolute_(index|middle|ring|pinky)_mcp_pitch` at 1.0 rad/s, and `thumb_mcp_pitch` for `revolute_thumb_mcp_pitch` at 0.5 rad/s. Same for mcp_yaw and pip. The EventTerms (`finger_mcp_pitch_gains` etc.) still use the original regex `revolute_.*_mcp_pitch` so their gain-randomization covers all 5 joints uniformly.

**Velocity profile after run3i:**

| Joint group | velocity_limit_sim | ~deg/s | Per-step motion @60Hz |
|---|---|---|---|
| FR3 arm (all 7) | 2.175 rad/s | 125 | 2.07° |
| thumb_rot | 0.2618 rad/s | 15 | 0.25° |
| **thumb_mcp_pitch / yaw / pip** (new) | **0.5 rad/s** | **28** | **0.48°** |
| Other fingers (mcp_pitch / yaw / pip) | 1.0 rad/s | 57 | 0.96° |

**Train command:**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Decision rule:**
- `finger_curl_reg/iter` goes more negative than -3.0 (e.g., -5 to -7 range) → cap raise is working as intended; check if `lift_success` follows
- `lift_success > 0.1` by ep ~1000 → thumb buckling was indeed the root cause; let it run to ADR 13 plateau
- `lift_success` still flat at ep 1000 with `finger_curl_reg` saturating at the new -8.0 cap → curl penalty can't outweigh the policy's discovered local optimum; need a structural fix (hand spawn closer to object, OR larger thumb_mcp_pitch stiffness so it physically can't be deflected)
- Thumb still visibly buckling in livestream → even slower thumb (0.25 rad/s) or stiffer thumb_mcp_pitch actuator (stiffness 10 → 25) becomes next lever

**Result:** *(to be filled in)*

#### Update — run3i actuator split reverted (2026-05-11)

**Run dir of failed attempt:** `logs/rl_games/dextrah_tekken_lstm/05-11_18-14-16/`

After restarting training with the run3i changes (thumb actuator split into separate `thumb_mcp_pitch` / `thumb_mcp_yaw` / `thumb_pip` groups at 0.5 rad/s, finger groups at 1.0 rad/s, raised curl cap, thumb init at 3°), the env immediately hit **vel_explosion terminations at every reset**:

- `episode_lengths/iter` = **2.241** (vs normal hundreds — episodes dying after ~2 steps)
- `joint_velocity_penalty/iter` = **-12.541** (vs max -0.017 historically — finger joint velocities exploding to ~30+ rad/s despite the 0.5/1.0 cap)
- `lift_reward = contact_reward = good_grasp_reward = object_contact_count = 0` (hand never reaches the object before termination)

**Root cause hypothesis:** splitting the joint regex `revolute_.*_mcp_pitch` into multiple actuator groups (one matching thumb, one matching other fingers) causes IsaacLab actuator-vs-USD limit resolution to break down — PhysX falls back to USD-baked velocity limits OR fails to enforce the Python-side velocity_limit_sim correctly, producing immediate constraint violations at reset.

**Action: reverted the split.** Restored the unified actuator groups:
- `mcp_pitch`: regex `revolute_.*_mcp_pitch` (all 5 mcp_pitch joints), velocity_limit_sim=1.0
- `mcp_yaw`: regex `revolute_.*_mcp_yaw`, velocity_limit_sim=1.0
- `pip`: regex `revolute_.*_pip`, velocity_limit_sim=1.0

The thumb now shares the same 1.0 rad/s limit as the other fingers' non-rotation joints (thumb_rot stays at 0.2618 rad/s via its dedicated `thumb_rot` actuator group, which works because it has only one joint and no split).

**All other run3i changes retained:**
- `contact_mask = good_grasp_mask` gate (env.py:2243)
- thumb_mcp_pitch + thumb_pip init at 3°
- `finger_curl_reg` ADR (-1.5, -3.0)
- `finger_curl_reg_min` cap -6.0
- `lift_weight` ADR (60, 30)
- `thumb_rot_init` EventTerm still removed

**Lesson** (also captured in CLAUDE.md): avoid splitting a joint regex across multiple actuator groups in IsaacLab. Tune velocity_limit_sim at the unified-group level instead. If thumb-specific velocity is needed in the future, the only safe approach is to add a separate single-joint actuator (like `thumb_rot` already is) rather than splitting an existing regex.

**Result:** *(to be filled in once livestream confirms episodes are running normally and lift behavior is observable)*

#### run3i v2 result — thumb stability achieved (2026-05-12)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-12_10-17-22/`

After the actuator-split revert (yesterday), restarted livestream training (24 envs, seed 42) on the unified-actuator + run3i config. Through ep ~540:

**Episodes are healthy** (no more vel_explosion):
- `episode_lengths/iter` = **49.7** (max 127.8) — normal range, episodes terminating on legitimate conditions or running long
- `joint_velocity_penalty/iter` = **-0.041** — back to typical magnitude (was -12.541 with the broken split)

**Lift gate working correctly** (exploit closed):
- `lift_reward/iter` latest 0.0, max **3.087** — way below the 14-17 saturation range observed in runs 3a-3h with the old `contact_mask = (contact_count > 0)` gate. The good_grasp_mask gate is preventing the buckled-thumb shaped-reward farming.
- `good_grasp_reward/iter` max **0.625** — fires sometimes (thumb + ≥1 finger contact), indicating the policy is achieving proper multi-finger grasps occasionally.
- `hand_object_contact_reward/iter` max **3.0** — modest contact happening (~1 finger avg).
- `object_contact_count/iter` max **1.0** — single-finger contact on average (no over-aggressive multi-finger clenching).

**Approach behavior** (`hand_to_object_distance` decreasing from 0.88m → 0.21m):
- Hand is reaching toward the object.
- Visual livestream observation: **thumb no longer folds inwards on approach**. The 3° init + 1.0 rad/s velocity cap is preventing the contact-driven buckling that plagued runs 3a-3h.

**`lift_success` still 0 through ep 540**:
- Policy hasn't yet figured out how to actually grasp + lift, but for the first time in 7 retraining attempts it's not stuck in the touch-don't-lift basin. The reward landscape now correctly biases toward proper grasps — needs more training time for the policy to find them.

**Conclusion (preliminary):** the combination of changes in run3i v2 (good_grasp_mask gate + 3° thumb init + 1.0 rad/s finger velocity + raised curl cap + bumped lift weight + removed thumb_rot_init) has broken the policy out of the degenerate exploit. Visual stability confirms the thumb buckling root cause is addressed.

**Next: let training continue.** Watch for `lift_success > 0.1` by ep ~1000-2000 if the run is going to converge. If still 0 by ep 3000+, may need additional levers (hand spawn closer to object, longer episodes, lift_weight bumped further).

**Verified-working configuration after run3i v2:**
- `contact_mask = good_grasp_mask` (env.py:2243)
- Unified actuator groups: `mcp_pitch`/`mcp_yaw`/`pip` regex `revolute_.*_<joint>`, all at velocity_limit_sim=1.0 rad/s
- thumb_rot dedicated actuator group at velocity_limit_sim=0.2618 rad/s
- thumb_mcp_pitch + thumb_pip init at 0.0524 rad (3°)
- finger_curl_reg ADR (-1.5, -3.0), cap -6.0
- lift_weight ADR (60, 30)
- thumb_rot_init EventTerm removed

### run3j — livestream-tuned posture + scaled-up training (2026-05-12)

**Approach.** After run3i v2's promising-but-unfinished result, used the livestream debug mode (24 envs, no headless) to *visually* iterate on the env starting pose. The 24-env count gives noisy gradients but lets us actually SEE the robot+hand behavior in each reset. Iterated several parameters by observation, then once the visuals looked stable+productive, restarted in 2048-env headless mode for productive training.

**Livestream-tuning iterations done in real time (2026-05-12):**

1. **Re-add `thumb_rot_init` EventTerm** → **fatal sim crash** (random thumb_rot offset interacts badly with slow finger PD + 3° thumb_mcp_pitch init). Reverted, keeping thumb_rot deterministic at -20°.
2. **Lower robot base z: 0.25 → -0.05** (30 cm lower) — goal: maximize FR3 workspace toward object spawn region.
3. **fr3_joint4 elbow extension iteration** — needed to compensate for the lowered base:
   - -150° (original) → hand collides with table on reset
   - -120° → tried first, EE still too low
   - -60° → too much extension, EE position weird
   - -10° → way too much, EE in air
   - -30° → still too high
   - -90° → EE far from object (`hand_to_object_distance` 0.7m vs original 0.2m — far worse), vel_explosion terminations
   - **-120°** → settled here. Middle ground after observing the EE positioning vs object spawn region.
4. **All non-thumb PIP inits 0.0 → 0.0524 rad (3°)** — moved off joint min. Same reasoning as the thumb_mcp_pitch/pip move earlier (run3i): init at joint min causes PD oscillation against the hard stop, manifests as finger jitter at reset. Brings the policy's curled_q target to 3° on every PIP.
5. **Finger velocity_limit_sim 1.0 → 2.0 rad/s** — the 1.0 rad/s cap (yesterday) was producing PD limit-cycle instability (commands generated faster than the cap could resolve them). 2.0 rad/s gives PD enough headroom while still being 4× slower than the original 8 rad/s and well within hardware-safe range (~114 deg/s).

**Verified-working configuration after livestream tuning (entering run3j productive training):**

| Setting | Value | Why |
|---|---|---|
| Robot base `pos.z` | -0.05 m (was 0.25) | Lowered 30cm to maximize workspace toward object |
| `fr3_joint4` init | -2.0944 rad (-120°) | Tuned visually; compensates lowered base without overshoot |
| All hand PIP inits | 0.0524 rad (3°) | Off joint min, no PD oscillation |
| `revolute_thumb_mcp_pitch` init | 0.0524 rad (3°) | Off joint min |
| `revolute_thumb_rot` init | -0.3491 rad (-20°) | Deterministic (EventTerm removed) |
| `mcp_pitch/yaw/pip` velocity_limit_sim | 2.0 rad/s (~114 deg/s) | PD-stable, hardware-safe |
| `thumb_rot` velocity_limit_sim | 0.2618 rad/s | Hardware-realistic, unchanged |
| `contact_mask` gate | `good_grasp_mask` (thumb + ≥1 finger) | Closes buckled-thumb exploit |
| `finger_curl_reg` ADR | (-1.5, -3.0); cap -6.0 | Discourages buckled pose |
| `lift_weight` ADR | (60, 30) | Stronger lift signal post-grasp |
| `thumb_rot_init` EventTerm | removed | Random thumb_rot was breaking sim |

**Train command (productive, 2048 envs headless):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Decision rule:**
- `lift_success > 0.1` by ep ~370-500 → on track to reach ADR 13 like historical run2a
- `num_adr_increases` starts climbing from ep ~1500-2000 → ADR curriculum engaging
- Healthy `episode_lengths` (hundreds) → no vel_explosion regression
- If still flat at ep 2000: need to revisit hand spawn distance OR effort limits

**Result:** *(to be filled in)*

### run3k — thumb_rot init randomization, recentered around 0° (2026-05-12)

**Motivation:** Visual replay of run3j's best policy (ep ~1300) showed the thumb still getting in the way during approach, and the policy never crossed `lift_success = 0.1`. Metrics analysis:
- `episode_lengths=512` (healthy), `hand_to_object_distance=0.10m` (hand reaching object), `good_grasp_reward=2.31` (~77% envs with multi-finger grip) — basic mechanics working
- `lift_reward=17.99` (saturated), `object_to_goal_reward=2.22` (object moving toward goal) — partial lift happening, never enough
- `finger_curl_reg=-5.06` (near -6 cap, saturated) — curl penalty fighting against full finger closure
- `joint_velocity_penalty=-0.047` (low) — actuators NOT saturating, plenty of headroom

User hypothesis: the deterministic thumb_rot at -20° creates a thumb plane that's ~90° offset from the finger plane — an opposed-thumb grip that's mechanically correct for power grasps but visually a difficult pre-grasp configuration. Moving thumb_rot to 0° (thumb aligned with finger plane) puts it in a tip-pinch starting orientation, with ±10° randomization for grasp-configuration diversity.

**Changes from run3j (env_cfg.py):**

| Setting | run3j | **run3k** | Why |
|---|---|---|---|
| `revolute_thumb_rot` init | -0.3491 rad (-20°) | **0.0 rad (0°)** | More natural tip-pinch starting orientation; less 90° offset from fingers |
| `curled_q` for thumb_rot (= init) | -20° | **0°** | Regularizer now rewards thumb near 0° instead of -20° |
| `thumb_rot_init` EventTerm | removed | **active**, position_range (-0.1745, 0.1745) | ±10° randomization at reset, gives LSTM diverse approach geometries |

**Safety vs the previous crash:** the prior `thumb_rot_init` re-add attempt (earlier today) used `position_range=(0.0, 0.3491)` from -20° init → final [-20°, 0°]. The -20° end is **at the joint min**, suspect cause of the fatal sim crash. New configuration keeps thumb_rot strictly within [-10°, +10°] — at least 10° away from either joint limit at all times. No limit contact, no PD oscillation against limits.

**Train command (1024 envs — chose this over 2048 since 79 envs/object is plenty above the ≥64 stability threshold, and gradients still settle cleanly):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- Sim doesn't crash → ±10° EventTerm range is safe (joint stays away from limits)
- `lift_success > 0.1` by ep ~500-1000 → thumb_rot orientation was the missing piece; thumb-aligned-with-fingers pose unlocks the grasp+lift sequence
- If still flat at ep 2000 → curl penalty saturation hypothesis is the real bottleneck; reduce `finger_curl_reg` ADR back to (-0.5, -1.5) and raise the cap back to -3.0

**Result:** *(to be filled in)*

### run3k result + run3l setup — reward-shape tuning for lift incentive (2026-05-12)

**run3k observations (1024 envs, thumb_rot init=0° with ±10° randomization, finger_curl_reg reduced to (-1.0, -2.0), hand_to_object_weight 4→3):**

- Sim stable, episodes healthy
- Visual: thumb_rot diversity working (different starting angles per reset), but the thumb is still "getting in the way" — i.e., the policy still drives the thumb into a buckled-grip pose during approach
- Policy keeps achieving multi-finger contact with thumb included but doesn't progress to lift
- `lift_success` still 0 through several hundred epochs

**Diagnosis (full reward landscape audit, 2026-05-12 afternoon):**

Discovered two drift points from run2a baseline that were strangling the lift signal:
- `lift_sharpness` had drifted to 5.0 (comment said "5→2" but value was 5) — at sharpness=5, the lift_reward exponential drops off much faster with vertical_error. Policy sees less reward at low elevations, so the gradient toward "lift higher" is weak.
- `object_to_goal_weight` had dropped from 40 → 30 — 25% weaker goal-direction pull.

**Reverted (env_cfg.py, user-applied):**
- `lift_sharpness`: 5.0 → **2.0** (back to run2a value — flatter gradient so reward grows from any height)
- `object_to_goal_weight`: 30 → **40** (back to run2a)

**run3l setup — additional lift_weight bump after reverts didn't unlock lifting:**

Even with sharpness=2 and object_to_goal_weight=40 restored, the policy still camps at the object. Observed `lift_reward` saturating without lift_success climbing. Joint torques have 100× headroom (0.21 Nm needed at wrist, 20 Nm available), so actuator strength is not the bottleneck — the incentive landscape is.

**Change (env_cfg.py adr_custom_cfg_dict):**

| Parameter | run3k | **run3l** | Why |
|---|---|---|---|
| `lift_weight` ADR | (60., 30.) | **(100., 50.)** | Doubling the lift signal — once good_grasp_mask gate opens, lift_reward at 5cm off table = 100 × exp(-0.2) = 82 per step, way above the 12-18 the policy can get from just camping at the object with contact. The lift state should now be obviously the dominant reward path. |

**Current full reward landscape (ADR 0):**

| Signal | Max value | Gated |
|---|---|---|
| `lift_reward` | **100** × exp(-2 × vert_err) | yes (good_grasp_mask) |
| `object_to_goal_reward` | 40 × exp(-5 × pos_err) | yes (good_grasp_mask) |
| `hand_object_contact_reward` | 3 × num_contacts (~9-15) | no |
| `hand_to_object_reward` | 3 × exp(-5 × dist) | no |
| `good_grasp_reward` | 3 (binary) | (is the gate) |
| `success_bonus_reward` | 10 per step | yes (in goal region) |
| `in_success_region_at_rest_weight` | 10 per step | yes (in goal region, at rest) |
| `finger_curl_reg` | up to -6 (clamped) | no |
| `palm_direction_alignment_reward` | up to -0.7 × θ² | no |

**Train command:**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success > 0.1` by ep ~500-1000 → the incentive landscape was the bottleneck, 100 lift_weight unlocks the lift behavior
- `lift_reward/iter` grows past 60-80 (previously saturated at 18) → lift gate is opening more often / at better positions
- If `lift_success` still flat at ep 2000 → the issue is mechanical (object slipping during lift, or arm trajectory can't translate good_grasp into vertical motion under current posture). Next levers: bump object friction (static/dynamic), or adjust the arm pose so vertical motion is more "natural" for the policy.

**Result:** Bumping `lift_weight` ADR (60, 30) → (100, 50) had no visible effect — policy still completely oblivious to lifting in livestream. Confirmed the issue is not reward magnitude. Reverted in run3m.

### run3m — revert lift_weight + bump arm torques +20% (2026-05-12)

**Hypothesis (rejected after livestream):** wrist torque might be the lift bottleneck despite ~100× headroom calculation. Probe by bumping arm `effort_limit_sim` +20% over hardware spec.

**Changes (env_cfg.py + fr3_tekken_left.py):**

| Parameter | run3l | **run3m** |
|---|---|---|
| `lift_weight` ADR | (100., 50.) | (60., 30.) (reverted) |
| `franka_arm` effort_limit_sim | 90.0 | **108.0** Nm (+20%) |
| `franka_joints_ee` effort_limit_sim | 20.0 | **24.0** Nm (+20%) |
| `arm_14_effort_limit` ADR | (90.0, 87.0) | **(108.0, 104.4)** |
| `arm_57_effort_limit` ADR | (20.0, 12.0) | **(24.0, 14.4)** |

**Livestream observation:** policy "completely oblivious towards lifting" — not even attempting upward motion, despite plentiful torque.

**Diagnosis (mid-session reward inspection):**

```python
# env.py:2243
contact_mask = good_grasp_mask.to(contact_count.dtype)
lift_reward = lift_weight * exp(-lift_sharpness * vert_err) * contact_mask
object_to_goal_reward = ... * contact_mask  # also gated
```

`good_grasp_mask` = `(≥2 unique fingers in contact) AND (thumb is one of them)`. If the policy never achieves a real grasp during exploration, `lift_reward` and `object_to_goal_reward` are both **identically zero** for every episode — no gradient toward lifting exists in the policy's experience.

The April run2g baseline used `contact_mask = (contact_count > 0)` — ANY single contact unlocked lift_reward, including thumb-scraping. That gate was the bootstrap path: policy could exploit messy contact to discover the lift action, then refine. Closing that exploit in run3 also closed the bootstrap.

Run3m's torque bump was the wrong lever — torque isn't blocking lifting, the **reward signal itself never reaches the policy**.

**Result:** Confirmed torque is not the bottleneck. Visually the policy started "trying to engage" with the object slightly more once contact-weight dynamics changed mid-session, which motivated run3n.

### run3n — reduce contact weight + meaningful arm-torque curriculum (2026-05-12)

**Motivation:** in run3m livestream, policy was beginning to interact with object meaningfully. To preserve that improvement, scale back `hand_object_contact_weight` proportionally with the earlier `hand_to_object_weight` reduction (4→3 in run3k). The previous arm-torque ADR was nearly flat for joints 1-4; make both groups span a real range that actually challenges sim2real robustness.

**Changes (env_cfg.py):**

| Parameter | run3m | **run3n** |
|---|---|---|
| `hand_object_contact_weight` | 3.0 | **2.0** |
| `arm_14_effort_limit` ADR endpoint | 104.4 | **85.5** (5% below 90 Nm spec) |
| `arm_57_effort_limit` ADR endpoint | 14.4 | **19.0** (5% below 20 Nm spec) |

Now both arm groups span (start +20% above spec) → (end −5% below spec), so the curriculum actually tests robustness toward hardware-realistic torque. Previous endpoint of 14.4 Nm on wrist was overly aggressive (−28% below spec) for what's effectively a no-bottleneck regime.

**Current reward landscape (ADR 0):**

| Signal | Max | Gated |
|---|---|---|
| `hand_to_object_reward` | 3 × exp(-5 × dist) | no |
| `hand_object_contact_reward` | 2 × num_contacts (~6-10) | no (reduced from 3) |
| `good_grasp_reward` | 3 (binary) | (= the gate itself) |
| `lift_reward` | 60 × exp(-2 × vert_err) | yes (good_grasp_mask) |
| `object_to_goal_reward` | 40 × exp(-5 × pos_err) | yes (good_grasp_mask) |
| `finger_curl_reg` | up to -6 | no |

Now `good_grasp_reward` (3) > `hand_object_contact_reward` per sensor (2). Once policy achieves a real grasp, that path is strictly more rewarding than camping with mere contact.

**Train command:**
```bash
cd dextrah_lab/rl_games
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

**Decision rule:**
- `lift_success` shows non-zero by ep ~1000-2000 → reduced contact weight unblocked the good_grasp incentive path
- `extras/good_grasp_reward` rising over training → policy is finding thumb-opposed grasps
- If `lift_success` still flat by ep 3000 → the `good_grasp_mask` gate itself is the bootstrap problem (policy never exploration-stumbles into the joint config). Next experiment: soft gate, `contact_mask = 0.2 + 0.8 * good_grasp_mask` so partial-contact still leaks a 20% lift gradient, OR curriculum gate (start with `contact_count > 0`, transition to `good_grasp_mask` once basic grasping emerges)

**Result (2026-05-15 livestream of best ckpt from run dir `05-12_18-31-14`):** ~16000 epochs trained over ~24h, reward flat at 15-17k throughout, ADR never advanced past 0 (`success_for_adr=0.4` never met because `lift_success` stayed at 0). Livestream confirmed the bootstrap-gate hypothesis: policy converges to "contact + camp" steady-state, never explores upward — `good_grasp_mask` is satisfied transiently in some envs but the lift gradient never makes it into the policy. Torque is not the bottleneck (verified at hardware-spec 87/12 Nm — robot moves fine, see run3o). Confirmed run3n's decision rule branch: bootstrap-gate is the blocker; next experiment reverts the strict gate.

### run3o — revert lift gate to any-contact, cut hand_to_object weight, remove thumb_rot randomization (2026-05-15)

**Motivation:** run3n confirmed three things: (1) `good_grasp_mask` gate prevents bootstrap — the policy never explores into the gated lift signal; (2) `hand_to_object_reward` at weight 3 + `contact_reward` at weight 2 give a ~9-13/step always-on positive signal that makes camping-at-object a stable equilibrium with no need to ever lift; (3) thumb_rot randomization at reset may be confusing the LSTM during early grasp learning (different thumb plane every episode before any grasp policy exists). Run3o targets all three.

The good_grasp gate was originally added in run3 (2026-05-11) to close the thumb-buckling exploit from the April baseline (`contact_count > 0` let policies farm lift reward with a buckled-thumb scrape). The bet for run3o: `good_grasp_reward` as a +3 shaping bonus is sufficient to make real grasps more rewarding than buckled scrapes, *without* gating lift_reward on it.

**Changes from run3n (3 files):**

| Setting | run3n | **run3o** | Why |
|---|---|---|---|
| `hand_to_object_weight` | 3.0 | **1.0** | Cut always-on approach reward by 2/3. Camping equilibrium drops from ~9/step to ~7/step; lift path peak (60 + 40 + 10) becomes overwhelmingly dominant once any contact is made. |
| `contact_mask` for lift/goal (env.py:2243) | `good_grasp_mask` | **`(contact_count > 0)`** | Bootstrap fix. Any contact opens the lift gradient; policy can discover "wrist up = object up" without first satisfying a complex multi-finger criterion it never explores into. |
| `thumb_rot_init` EventTerm | active (±10°) | **disabled (commented out)** | Removes inter-episode thumb-plane variability while LSTM learns basic grasp. Plan: re-introduce as ADR curriculum (start (0, 0), ramp to (-0.1745, 0.1745)) once `lift_success > 0.3`. |
| `franka_arm` effort_limit_sim | 108 Nm | **87 Nm** | Match actual USD hardware spec (user verified motion still clean at 87). Discussed in run3m as +20% probe — confirmed not the bottleneck. |
| `franka_joints_ee` effort_limit_sim | 24 Nm | **12 Nm** | Match actual USD hardware spec. |
| `arm_14_effort_limit` ADR | (108, 85.5) | **(87, 87)** | Flat at hardware spec. No curriculum span while debugging bootstrap. |
| `arm_57_effort_limit` ADR | (24, 19) | **(12, 12)** | Flat at hardware spec. |

**Current reward landscape (ADR 0):**

| Signal | Max value | Gate |
|---|---|---|
| `hand_to_object_reward` | **1** × exp(-5 × dist) | none |
| `hand_object_contact_reward` | 2 × num_contacts (~6-10) | none |
| `good_grasp_reward` | 3 (binary) | (= the shaping signal for real grasps) |
| `lift_reward` | 60 × exp(-2 × vert_err) | **any contact** |
| `object_to_goal_reward` | 40 × exp(-5 × pos_err) | **any contact** |
| `success_bonus_reward` | 10 per step | in_success_region |
| `finger_curl_reg` | up to -6 (clamped) | none |
| `palm_alignment_reward` | up to -0.7 × θ² | none |
| `early_term_penalty` | -1.0 | terminated AND no_contact |

**Predicted policy trajectory** (if hypothesis holds):
1. Approach phase: policy moves hand toward object (still incentivized at +1/step max)
2. First touch: any contact opens lift gate at +60 — even a finger graze, even thumb-only contact
3. Discovery of "wrist up + contact = +50 per step" via random exploration is now trivially reachable
4. As lift gradient takes hold, `good_grasp_reward` (+3 only for real grasps) gradually pulls thumb+finger pose toward proper grip — this is the slow-shaping mechanism rather than a hard gate
5. ADR advances once `lift_success > 0.4` on the 13-object set

**Train command (1024 envs, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success > 0.1` by ep ~500-1000 → bootstrap-gate was the bottleneck and any-contact gate unlocks lift; continue
- `lift_reward/iter` climbs past 30 by ep 1000 → contact gate is opening reliably and policy is finding upward motion
- Livestream shows thumb-buckling exploit (thumb scraping object without grasp, farming lift reward) → bump `good_grasp_weight` 3 → 6 to strengthen the real-grasp shaping signal
- `lift_success` still flat at ep 2000 → contact gate isn't the issue either; suspect arm trajectory / posture, next experiment is init-pose sweep
- Once `lift_success > 0.3` stably → re-introduce `thumb_rot_init` EventTerm as a curriculum (ADR range starting (0, 0), ramping to (-0.1745, 0.1745))

**Result (run dir `05-15_12-32-16`, ~5000 epochs, ~3h):** Reward catastrophically negative throughout (-0.74 → -0.24 → -0.74, mean episode reward). Diagnosis: `hand_to_object_weight=1` cut the approach signal too aggressively — at weight 1, peak `hand_to_object_reward` is ~1 (vs ~3 in run3n), but `finger_curl_reg` (-1 to -2) + `palm_align` (~-0.5) + `action_rate` (~-0.05) + `early_term_penalty` (-1 on terminations without contact) sum to ~-2 to -3.5. Net per-step reward is negative regardless of position, so there's no positive gradient pulling the hand toward the object. Policy never bootstraps to contact. Confirmed via livestream: no contact attempts.

This is a different failure mode than run3n. run3n was "contact + camp" (positive equilibrium, no exploration into lift). run3o is "freeze far from object" (negative equilibrium, no exploration into approach). The lift-gate change (`good_grasp_mask` → `contact_count > 0`) wasn't tested because contact never happened.

### run3p — restore approach signal, remove velocity throttles (2026-05-15)

**Motivation:** Run3o demonstrated that `hand_to_object_weight=1` is below the threshold needed to overcome the always-on negative regularizers. We need to restore enough approach signal to actually reach the object — but not so much that camping-at-object becomes the stable equilibrium again (run3n's failure). Compromise: 2.0 (halfway between run3n's 3.0 and run3o's 1.0). At weight 2, peak approach signal is ~2 which exceeds the ~-2 negative-regularizer floor, giving a positive gradient toward the object.

Additionally, run3o livestream raised a separate concern: the policy's grasp closure looked sluggish even when reaching the object. Investigation showed finger `velocity_limit_sim=2.0 rad/s` (~114 deg/s) was throttling the PD-commanded closure speed. Real AgileHand fingers can move at ~360 deg/s (verified by user). Remove the sim throttle so finger closure speed matches hardware capability.

For symmetry and to remove a potentially hidden constraint: arm `velocity_limit_sim=2.175 rad/s` removed too. Let `effort_limit_sim` (87/12 Nm) + PD (stiffness=200, damping=40) be the only motion constraints on the arm.

**Changes from run3o (asset + env_cfg):**

| Setting | run3o | **run3p** | Why |
|---|---|---|---|
| `hand_to_object_weight` | 1.0 | **2.0** | Restore positive net per-step reward; ~2/step peak approach signal exceeds the ~-2 always-on regularizer floor. |
| `franka_arm` velocity_limit_sim | 2.175 rad/s | **removed** (USD default) | No software throttle on arm motion; PD + effort_limit shape it. |
| `franka_joints_ee` velocity_limit_sim | 2.175 rad/s | **removed** (USD default) | Same. |
| `mcp_pitch` velocity_limit_sim | 2.0 rad/s | **6.2832 rad/s (360 deg/s)** | Match hardware capability — fingers were artificially slow during grasp closure. |
| `mcp_yaw` velocity_limit_sim | 2.0 rad/s | **6.2832 rad/s (360 deg/s)** | Same. |
| `pip` velocity_limit_sim | 2.0 rad/s | **6.2832 rad/s (360 deg/s)** | Same. |
| `thumb_rot` velocity_limit_sim | 0.2618 rad/s (15 deg/s) | unchanged | Hardware spec; sim2real lever already set. |

**Important caveat:** Isaac Lab's `ImplicitActuatorCfg` falls back to USD's baked-in velocity limit when `velocity_limit_sim` is omitted. If FR3's USD has 2.175 rad/s baked in, the arm constraint will silently persist. **TODO during livestream: verify arm joints can exceed 2.175 rad/s in practice** — if not, set to a large explicit value (e.g. 100.0 rad/s).

**Current reward landscape (ADR 0):**

| Signal | Max value | Gate |
|---|---|---|
| `hand_to_object_reward` | **2** × exp(-5 × dist) | none |
| `hand_object_contact_reward` | 2 × num_contacts (~6-10) | none |
| `good_grasp_reward` | 3 (binary) | shaping for real grasps |
| `lift_reward` | 60 × exp(-2 × vert_err) | any contact |
| `object_to_goal_reward` | 40 × exp(-5 × pos_err) | any contact |
| `success_bonus_reward` | 10 per step | in_success_region |
| `finger_curl_reg` | up to -6 (clamped) | none |
| `palm_alignment_reward` | up to -0.7 × θ² | none |
| `early_term_penalty` | -1.0 | terminated AND no_contact |

**Net per-step reward estimate by phase:**
- Far from object: +0.2 (hand_to_object) - 1.5 (finger_curl + palm_align) = **-1.3/step** — still slightly negative, but gradient toward the object is positive (closer → higher reward)
- At object, no contact: +2 - 1.5 = **+0.5/step** — positive, but small enough that contact is the better option
- With contact, no lift: +2 + 6 (contact) - 1.5 = **+6.5/step** — strong incentive to maintain contact
- With contact, lifting: +2 + 6 + 50 (lift, at 5cm) = **+56.5/step** — overwhelming incentive to lift

The gradient stack is now: far → at object → contact → lift, each step strictly more rewarding than the previous.

**Train command (1024 envs, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- Episode mean reward turns positive by ep ~500 → approach signal is now strong enough to bootstrap toward the object
- `extras/hand_object_contact_reward` > 0 by ep ~500-1000 → contact is being made
- `lift_success > 0.1` by ep ~1000-2000 → contact-gate (from run3o) actually opens lift gradient; this finally tests the run3o hypothesis
- Livestream early on: verify arm joints can exceed 2.175 rad/s (no USD-baked limit) and finger joints can exceed 2.0 rad/s. If they can't, set explicit large values.
- If `lift_success` flat at ep 2000 but contact is happening → buckling exploit returned; bump `good_grasp_weight` 3 → 6 or re-strict the gate

**Result (run dir `05-15_17-25-57`, stopped at ~1870 epochs by user, 2026-05-15):** Bootstrap worked. Episode reward climbed steadily 12k → 14k → 15.5k from ep_500 to ep_1500 (positive throughout, vs run3o's catastrophic -0.5). Livestream + replay confirmed: contact happening reliably, `good_grasp_reward` firing in many envs, `lift_reward` accumulating per step. **But `lift_success` remained flat at zero** — policy reaches object, makes contact, fingers curl, sometimes good_grasp fires, but never actually lifts the object off the table.

Diagnosis: residual lift_reward at table-touch. With `lift_sharpness=2` (run3p value), `lift_reward` at table-sit = 60 × exp(-2 × 0.47) ≈ **23/step** — about 40% of max lift_reward harvested without lifting. Combined with `hand_object_contact_reward` (~6-10/step) the policy has a healthy ~30/step camping equilibrium and no clear gradient toward upward motion since lifting only gains +6 to +13/step against the risk of losing the entire ~30 if contact slips.

### run3q (config only, never trained, 2026-05-15)

After the run3p diagnosis, three rebalancing changes were committed in the same commit (`220112c`):
- `lift_sharpness` 2.0 → **5.0** (steeper gradient, table-touch lift_reward drops 23 → ~5.7)
- `hand_object_contact_weight` 2.0 → **1.5** (smaller camp signal)
- `good_grasp_weight` 3.0 → **2.5** (rescaled with contact cut)

No fresh training run was started before pivoting to run3r — these changes carry forward into run3r as the new baseline. With sharpness=5 alone, table-touch lift_reward is still ~5.7/step which the policy may still find satisfying given the camp stack. Hence run3r below.

### run3r — quarter-cut lift_weight magnitude (2026-05-18)

**Motivation:** The run3p diagnosis identified residual `lift_reward` at table-touch as the camping driver. run3q addressed the *shape* of the lift gradient (sharpness 2 → 5). run3r addresses the *magnitude*: even at sharpness=5, table-sit lift_reward is ~5.7/step (with weight 60) — still enough to make camping a comfortable equilibrium when added to contact + good_grasp + hand_to_object.

The hypothesis: the policy is *satisfied* with the contact+residual-lift reward and never explores upward. By cutting lift_weight by 4×, residual lift_reward at camp drops to ~1.4/step — basically negligible. The remaining "lift incentive" shifts to `object_to_goal_reward` (max 40 at goal, sharpness=5) and `success_bonus_reward` (10/step at goal) which both require actual goal-reaching, not just contact.

**Change (env_cfg.py:926):**

| Setting | run3p/q | **run3r** | Why |
|---|---|---|---|
| `lift_weight` ADR | (60, 30) | **(15, 7.5)** | Quarter of previous magnitudes. Drops table-touch lift_reward from ~5.7 to ~1.4/step. Shifts lifting incentive to object_to_goal + success_bonus, both of which require actual height gain. |

**Reward landscape comparison (table-sit with contact, ADR 0):**

| Signal | run3p | run3q | **run3r** |
|---|---|---|---|
| `hand_to_object` | 2.0 | 2.0 | 2.0 |
| `hand_object_contact` (3-5 sensors) | 6-10 | 4.5-7.5 | 4.5-7.5 |
| `good_grasp` | 3.0 | 2.5 | 2.5 |
| `lift_reward` at camp | **23.4** | **5.7** | **1.4** |
| `object_to_goal` at camp | 3.3 | 3.3 | 3.3 |
| Total camp ~ | ~38 | ~21 | **~15** |
| Total goal ~ | ~120 | ~88 | **~73** |
| Camp:goal ratio | 1:3.2 | 1:4.2 | **1:4.9** |

Camping yields less of the available reward; goal becomes ~5× better than camp.

**Train command (1024 envs, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success > 0.1` by ep ~1000-2000 → reduced camp signal worked; policy now explores upward to harvest object_to_goal + success_bonus
- `extras/object_to_goal_reward` rising over training → goal-reaching is the main lift driver as designed
- Livestream shows the policy attempting upward motion (even briefly) → confirms the diagnosis was magnitude, not gradient shape or gate type
- If `lift_success` still flat at ep 2000 → the issue isn't lift_reward at all; object-on-table contact gating (run3t) is the next test

**Result (run dir `05-18_12-43-51`, stopped at ~1500 epochs, 2026-05-18):** Episode reward grew 2024 → 5207 → 5416 over ep_500/1000/1500 — positive and rising but much lower absolute than run3p (15.5k) because all weights had been cut. **Lift behavior did not emerge**: livestream confirmed no upward motion attempts. The policy still focused on engaging with the object via maximum contact-count + good_grasp, but the thumb was visibly getting stuck inside the object during closure (likely buckling against the object surface). Diagnosis: even with lift_weight quartered, `contact_reward` (~6/step at 4 sensors w/ weight 1.5, dropped to ~4/step with the live mid-run weight cut to 1.0) was the dominant signal — policy still over-rewarded for sensor-count farming rather than vertical motion. Additionally, `object_to_goal_reward` at camp (~3.3/step with sharpness=-5) was parking the policy near the object.

### run3s — rebalance reward stack toward goal-only signals + livestream sanity (2026-05-18)

**Motivation:** Run3r showed cutting `lift_weight` alone wasn't enough — the policy was still finding a stable camping equilibrium driven by `contact_reward` (~4-6/step) + `good_grasp_reward` (2.5) + `object_to_goal_reward` at camp (~3.3). The lift gradient was steep enough but the camping stack was high enough to satisfy the policy without lifting.

Two compounding fixes for run3s:
1. Bump `lift_weight` back up partially so the goal-side reward becomes more visible (camp residual stays low because of sharpness=5).
2. Reduce all the "engagement at object" rewards (`contact`, `good_grasp`, `object_to_goal` at camp) so camping has no payoff path.

**Changes from run3r (env_cfg.py):**

| Setting | run3r | **run3s** | Why |
|---|---|---|---|
| `lift_weight` ADR | (15, 7.5) | **(25, 12.5)** | Quarter cut was too aggressive — goal-side lift_reward at 15 was below contact+good_grasp+object_to_goal stack at camp. 25 makes lift_reward at goal more visible. |
| `hand_object_contact_weight` | 1.0 (mid-run cut from 1.5) | **0.8** | Further reduce per-sensor reward. At 4 sensors: 3.2/step (was 4.0). |
| `good_grasp_weight` | 2.5 | **1.5** | Scale down with contact cut. Still a real-grasp shaping bonus but no longer the second-biggest signal at camp. |
| `object_to_goal_sharpness` ADR | (-5, -10) | **(-8, -12)** | Sharpens the goal-distance gradient. At sharpness=-5, camp reward (err=0.5m) was 3.3/step. At -8, camp drops to 0.73/step. Goal reward (err=0) unchanged at 40. |

**Run3s reward landscape at camp (table-sit, with contact, ADR 0):**

| Signal | run3r value | **run3s value** |
|---|---|---|
| `hand_to_object_reward` | 2.0 | 2.0 |
| `hand_object_contact` (4 sensors) | 4.0 | **3.2** |
| `good_grasp` | 2.5 | **1.5** |
| `lift_reward` at camp (sharpness=5) | 1.4 | **2.4** |
| `object_to_goal` at camp | 3.3 | **0.73** |
| **Total camp ~** | ~13 | **~9.8** |

**Run3s reward landscape at goal (object in goal region):**

| Signal | run3r | **run3s** |
|---|---|---|
| `contact` (still gripping) | ~4 | ~3.2 |
| `good_grasp` | 2.5 | 1.5 |
| `lift_reward` at goal | 15 | **25** |
| `object_to_goal` at goal | 40 | 40 |
| `success_bonus` | 10 | 10 |
| **Total goal ~** | ~72 | **~80** |

Camp:goal ratio: **1:8.2** (was 1:5.5 in run3r). Stronger pull toward goal-reaching, weaker pull toward camp-on-object.

**Side observation from run3r livestream — thumb getting stuck:** During the run3r playback, the thumb visibly got jammed inside/against the object during closure. Hypothesis: with finger velocity_limit_sim=6.28 rad/s (360 deg/s) and aggressive closure incentivized by contact+good_grasp, the thumb is being driven into the object surface faster than the contact physics can resolve. Three contributing factors:
1. `revolute_thumb_rot` deterministic at 0° at every reset (thumb_rot_init EventTerm disabled in run3o)
2. `finger_curl_reg` at ADR (-1, -2) — weak penalty against over-curl
3. `penetration_penalty_weight` = 0 (penetration penalty is configured but disabled)

For now run3s addresses only the lift incentive issue; the thumb-stuck issue is queued for a separate experiment.

**Train mode: livestream (sanity check, not converged training)** — running with 16 envs and 3 objects to visually verify the new reward landscape produces lift-attempting behavior before committing GPU time to a full 1024-env run. Per CLAUDE.md, 16 envs across 13 objects is gradient-unstable; using `multi_objects/3` for cleaner per-step gradients.

**Livestream train command (16 envs, 3 objects, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=False \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/3 \
  env.use_cuda_graph=False
```

**Decision rule:**
- Livestream shows the policy attempting any upward motion within ~ep 200-500 → reward landscape is healthy, kick off a full 1024-env run
- Livestream shows the policy still camping with no vertical attempts → reward stack still pulls toward camp; consider activating run3t (table-contact gate) before committing to full training
- Livestream shows thumb still getting stuck → run3s alone didn't address the closure-physics issue, queue a separate thumb experiment (re-enable thumb_rot_init, raise finger_curl_reg cap, or activate penetration_penalty)

**Result:** *(to be filled in)*

### run3s.1 — palm_flip penalty refactor (mid-run patch, 2026-05-18)

**Motivation:** Run3s livestream showed `palm_flip` counts climbing in the TERMINATIONS row of the training status table — the policy was repeatedly flipping its palm during contact (which is never legitimate, but the existing `early_term_penalty` was gated by `no_contact` so palm_flips while gripping the object incurred *zero* penalty). The policy had no signal to discourage this behavior.

**First attempt (rejected):** Added a dedicated `palm_flip_penalty = -5.0` reward term that fires whenever `last_palm_flipped == True`, independent of contact state. Two penalty mechanisms running in parallel: `early_term_penalty` (-1.0, gated by no_contact) for object_out/hand_too_far/palm_flipped, and `palm_flip_penalty` (-5.0, unconditional) for palm_flipped only.

**Refactored to single mechanism (final):** Folded into `early_term_penalty`. palm_flip now triggers the existing -1.0 penalty unconditionally (without the `no_contact` gate); object_out and hand_too_far keep their no_contact gating. One penalty value (-1.0), one mechanism, cleaner logic.

**Changes (env.py + env_cfg.py):**

| File | Change |
|---|---|
| `env_cfg.py:785` | Removed `palm_flip_penalty` config field; clarified `early_termination_penalty` docstring |
| `env.py:1175-1186` | `penalty_mask = (self._penalty_terminated & no_contact) \| self.last_palm_flipped` — palm_flip joins via OR (unconditional), object_out/hand_too_far still need no_contact |
| `env.py:1234-1246` | Added `("early_term", early_term_penalty.mean().item())` to the training status table |

**Updated penalty table:**

| Termination cause | Penalized? | Trigger |
|---|---|---|
| `object_out` | -1.0 | no_contact (unchanged) |
| `hand_too_far` | -1.0 | no_contact (unchanged) |
| `palm_flipped` | -1.0 | **always (new — no_contact gate removed)** |
| `hand_too_close`, `arm_table_contact`, `robot_unstable`, `vel_explosion` | none | physics artifacts / exploration side-effects |

**Monitoring during run3s livestream:**
- `early_term` value in the rewards table should approximately match the rate of penalized terminations × -1.0. E.g. 1 in 16 envs hitting -1.0 = mean -0.0625
- `palm_flip=N` count in TERMINATIONS row should trend down as the penalty signal reaches the policy
- If `palm_flip` count stays high, the -1.0 magnitude isn't a strong enough signal; bump `early_termination_penalty` (e.g. to -3.0 or -5.0). All three penalty causes will scale together.

**Result:** *(to be filled in along with run3s)*

### run3s.2 — scale to 2048 envs after 1024-env camp plateau (2026-05-18)

**Motivation:** Run3s + run3s.1 ran at 1024 envs (`05-18_16-53-22`) for ~3000 iterations. Tensorboard analysis shows the policy parked in a stable camp equilibrium:

| Signal | start (<100 iter) | end (>2800 iter) | What it means |
|---|---|---|---|
| `hand_to_object_distance` | 0.377 m | 0.086 m | Policy approaches object to ~9cm |
| `hand_object_contact_reward` | 0.000 | 2.62 | ~3 sensors with weight 0.8 (= consistent contact) |
| `good_grasp_reward` | 0.000 | 1.00 | Real grasp fires in ~67% of envs (1.0 / 1.5 weight) |
| `lift_reward` | 0.000 | **2.31** | **Parked at the camping residual** = 25 × exp(-5 × 0.47) = 2.4 |
| `object_to_goal_reward` | 0.000 | 0.93 | Parked at residual = 40 × exp(-8 × 0.48) = 0.86 |
| `in_success_region` | 0 | 0 | Object never at goal |
| `lift_success` (per env) | noise (1/1024) | noise (1/1024) | **Never reliably lifted** — max value seen ever was 0.000977 (one env) |
| `early_term_penalty` | -0.08 | -0.00 | ✓ palm_flip refactor working — penalty signal almost zero |
| `palm_align` | -1.07 | -0.34 | Palm well-oriented |
| `episode_lengths` | 23 | 548 | Full-duration episodes (no early terminations) |
| `rewards/iter` (mean ep) | -54 | 3435 | Stable positive equilibrium |

The policy learned approach + grip + palm orientation + termination avoidance but never explored upward. The camp residual (`lift_reward` ≈ 2.3/step) is the highest single positive signal in the steady-state stack and the policy is satisfied with harvesting it.

**Agent config sanity check:** compared `rl_games_ppo_lstm_cfg.yaml` between fr3_agilehand and dextrah_kuka_allegro — diff is 4 cosmetic lines (experiment name, max_epochs, save_frequency, wandb_project). Every PPO-relevant parameter matches: `entropy_coef=0.002`, `learning_rate=3e-4` (both override to 1e-4 in command), `sigma_init.val=0`, `fixed_sigma=True`, identical LSTM 1024 actor + LSTM 2048 critic. Exploration capacity is the same — the bottleneck is in env/reward, not in the agent.

**Key reward differences vs the working dextrah_rgb (kuka_allegro) setup:**
- `lift_weight`: kuka uses **(5, 0)** (small, decays to zero) vs our **(25, 12.5)** (large, decays slowly)
- `finger_curl_reg`: kuka uses **(-0.01, -0.01)** (essentially off) vs our **(-1, -2)** with -6 floor (saturating to -2.94 in practice). 100-200× stronger curl penalty in our setup actively fights against finger closure.
- Action space: kuka uses FABRICS + FGP (smooth, low-dim PCA basis) — fundamentally more exploration-efficient. We use direct joint position control.
- Object set: kuka uses `visdex_objects` (152 objects) — but envs/object is similar (27 there vs 79 here).

**For now run3s.2 leaves the reward landscape unchanged and tests whether 2× the envs unlocks lift-exploration.** Curl penalty reduction is queued as a separate next step if 2048 envs also plateaus. The hypothesis: 2048 envs across 13 objects (=157 envs/object) doubles the chance any single env randomly stumbles into a wrist-up motion that propagates through the gradient before the policy converges into the camp local optimum.

**Train command (2048 envs, single GPU, headless):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

Same structure as the 1024-env run (4 minibatches per epoch: 2048 × 16 / 8192 = 4).

**Decision rule:**
- `lift_success` sustainably > 0.001 (i.e. multiple envs lifting, not single-env noise) within ep ~2000 → 2× env scaling helped; let it ride
- Plateaus the same way as 1024-env (camp equilibrium, lift_success at noise floor) → confirms env count isn't the bottleneck; activate run3t (table-contact gate) which makes lift_reward = 0 unless object is off the table

**Result:** *(to be filled in)*

### run3t (planned, deferred) — gate lift_reward on object-not-touching-table

**Motivation:** Even with rebalanced rewards (run3s), the current contact_mask is `(hand_object_contact > 0)`. This means `lift_reward` and `object_to_goal_reward` fire continuously while the policy holds the object *against the table* with contact — they pay out even when the object hasn't been physically lifted. With sharpness=5 the residual is small (~2.4/step at camp), but it's still "false hope" — the policy can interpret partial reward as progress toward lifting when actually the object is sitting still.

A cleaner gate: `lift_reward` only fires when the object is *demonstrably off the table*.

**Option A (z-threshold proxy):** `object_lifted = (object_pos[:, 2] > table_top_z + epsilon)` with epsilon ~0.005-0.01m. Simple, no sensor wiring needed. Edge case: thin objects whose centroid is below table_top_z when resting (rare with visdex objects).

**Option B (table contact sensor):** add a contact sensor on the table, gate on `table_contact_count == 0`. More robust, but requires adding a sensor and wiring it through the env. Higher implementation cost.

Recommend Option A as a first pass. Threshold value matters: too tight (0.001m) and bouncy contact creates flickering reward; too loose (0.05m) and the gate doesn't fire until the object is already nearly at lift_success threshold.

**Planned change (env.py:2243 region):**
```python
# Existing:
contact_mask = (contact_count > 0.0).to(contact_count.dtype)
# Add:
table_top_z = self.cfg.table_cfg.init_state.pos[2] + 0.5 * self.cfg.table_size_z
object_lifted = (self.object_pos[:, 2] > table_top_z + 0.01)
lift_gate = contact_mask * object_lifted.to(contact_mask.dtype)
# Use lift_gate instead of contact_mask for lift_reward only (not object_to_goal_reward — dragging toward goal on the table is still a valid sub-skill).
```

**Decision rule:**
- run3s `lift_success > 0.1` → run3t probably not needed (run3s solved the issue)
- run3s `lift_success` flat but livestream shows some upward attempts (even brief) → run3t might help by removing false hope
- run3s `lift_success` flat with no attempts → run3t alone won't help; need to reconsider the reward stack entirely

**Status:** Not yet started. Wait for run3r results before deciding.

### run3s.3 — revert lift landscape + bump approach anchor (working teacher v2 values, 2026-05-19)

**Motivation:** run3s.2 (2048 envs, 3297 iters) confirmed the camp basin is structural to the current reward landscape, not a sample-size problem. `lift_success` stayed at noise floor (max 0.00195 = 4/2048 envs) throughout. `lift_reward` parked at exactly **2.25** = the camp residual (`25 × exp(-5 × 0.48)`), matching contact (2.0) + good_grasp (0.9) combined. With sharpness=5 the lift gradient is essentially flat for any vertical_err > 0.2m — the policy cannot discover lifting because the gradient is too steep close to the goal and too dead at table height.

Diff against working teacher v2 (commit 32a8924, policy 12 stored at `12_teacher_v2_hw_realistic_adr13_04-06_12-49-15`) showed the **decisive landscape change was `lift_sharpness: 2 → 5`** plus the matching `lift_weight: (40,20) → (25,12.5)` cut. The working policy got lift_reward = 14.7/step at table height (40 · exp(-2·0.5)) — a clear, discoverable signal **6× the combined contact bonus**. The current policy gets lift_reward = 2.25/step at table height — **equal** to the combined contact bonus → no escape gradient.

First attempt was a minimal 3-param revert (lift_sharpness, lift_weight, finger_curl_reg) keeping `hand_to_object_weight=2.0`. Livestream showed the policy turning **away** from the object and getting palm_flips — the new lift gradient was so strong it pulled the policy toward grasping-positions-without-an-object, and the 2.0 approach weight couldn't anchor the hand. Bumped `hand_to_object_weight` to 3.0 (working v2 used 4.0) as a second change in the same run.

**Changes (cumulative from run3s.2):**

| param | run3s.2 | run3s.3 | rationale |
|---|---|---|---|
| `lift_sharpness` | 5.0 | **2.0** | flatten — discoverable from table height |
| `lift_weight` ADR | (25, 12.5) | **(40, 20)** | bigger pot, slower decay |
| `finger_curl_reg` ADR | (-1.0, -2.0) | **(-0.5, -1.2)** | halve curl floor so fingers can fully close |
| `hand_to_object_weight` | 2.0 | **3.0** | stronger anchor to counter the bigger lift attractor |

**Resulting reward landscape at ADR 0, vertical_err = 0.5 m:**

| signal | value/step |
|---|---|
| `lift_reward` | **14.7** (was 2.25) |
| `hand_object_contact_reward` (≈4 sensors × 0.8) | 3.2 |
| `good_grasp_reward` (full grasp × 1.5) | 1.5 |
| `object_to_goal_reward` (at err ≈ 0.5) | 0.73 |
| `hand_to_object_reward` (at object) | **1.33** (was 0.89) |
| `finger_curl_reg` (full curl, ADR 0) | -0.5 (was -1.0) |
| **net positive** | **21.2** (was 8.0) |
| **net negative** | **-3.0** (was -4.6) |

The lift signal is now ~3× the combined contact stack, ~20× the goal residual — clearly the dominant gradient when contact is present. At goal (vertical_err = 0), lift_reward = 40 (vs 25 before). At weight 3, hand_to_object pays +1.33/step at the object × ~500-step episode = +665 — comparable scale to lift_reward at table height, providing a real anchor against drift.

**Unchanged from run3s.2 (intentionally NOT reverted):**
- `hand_object_contact_weight = 0.8`, `good_grasp_weight = 1.5` (working had 3.0/3.0) — keep contact subordinate to lift
- Arm effort fixed at 87/12 (working had 90→87 / 20→12 curriculum) — hardware-realistic from step 0
- Contact gate using `(contact_count > 0)` not `good_grasp_mask` (working used good_grasp_mask)
- `palm_flip` folded into `early_term_penalty` (run3s.1)

If run3s.3 lifts, we know the lift landscape + approach anchor combo was the blocker. If it doesn't lift, we revert the next layer (contact weights to 3.0/3.0, then arm curriculum).

**Train command (livestream first to sanity-check anchor + lift attempt, then scale to 2048 envs headless):**
```bash
# Livestream sanity check (16 envs, visible):
cd dextrah_lab/rl_games
/home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False

# Headless 1024 envs (after livestream sanity — chosen over 2048 to reduce vel_explosion frequency):
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- Livestream: hand anchors on object, contact forms, upward motion attempts → kill livestream, start headless
- Livestream: hand still drifts / palm_flips → bump hand_to_object_weight to 4.0 (full working-teacher-v2 value)
- Headless: `lift_success > 0.05` and `num_adr_increases ≥ 1` within ep 2000 → lift landscape + anchor combo was the whole problem
- Headless: `lift_reward` climbs above ~15 (table-height residual) but `lift_success` low → policy is *trying* to lift but not yet succeeding; let it train longer, possibly LR-bump
- Headless: `lift_reward` parks at ~14-15 and `lift_success` stays at noise floor → revert contact weights next (3.0/3.0 instead of 0.8/1.5)
- Negative reward / vel_explosion → curl penalty cut too aggressive, bump to (-0.7, -1.5)

**Livestream observations (2026-05-19):**
- **First positive sign in the whole run3 saga**: with the reverted lift_sharpness=2 + lift_weight=(40,20) + hand_to_object=3, the policy is now visibly **attempting upward motion** after contact. The lift gradient is being followed — not just sat on.
- Lift attempts are **unsuccessful**: object is barely cleared from the table, never reaches `lift_success` threshold, no completions.
- **vel_explosion territory**: as the policy tries harder to lift, finger and arm joints start hitting velocity limits → vel_explosion terminations creeping up. Free unpenalized resets are likely going to become attractive again if the rate stays high.
- Decision: kill the 16-env livestream, scale to **1024 envs headless** (not 2048 — fewer envs may reduce vel_explosion cascade frequency from contact instabilities and give the policy more stable trajectories to learn lifting). Same reward config.

**Next step: headless 1024-env run**, same reward config as logged above. Watch for `lift_success` actually firing now that the gradient is discoverable, AND for the `term_real` vs `term_physics_instability` ratio in TensorBoard — if vel_explosion dominates terminations, we'll need to address joint velocity stability before lift can converge.

**Result:** *(headless 1024-env run pending — see run3s.4)*

### run3s.4 — scale run3s.3 to 1024 envs headless (2026-05-19)

**Motivation:** First livestream of run3s.3 (16 envs) showed the policy attempting upward motion after contact — first positive signal in the whole run3 saga. Lifts unsuccessful and vel_explosion creeping up, but the gradient is now being followed. Scaling to 1024 envs headless to get a proper training signal: more parallel trajectories means higher chance some envs survive the vel_explosion regime and reach lift_success, and the gradient-averaging across 1024 envs should be smoother than 16. Chose **1024 over 2048** because vel_explosion is likely contact-instability triggered — fewer envs may reduce per-step physics solver load and give cleaner contact resolution.

**Reward config:** unchanged from run3s.3 (lift_sharpness=2, lift_weight=(40,20), finger_curl=(-0.5,-1.2), hand_to_object=3, hand_object_contact=0.8, good_grasp=1.5, arm effort fixed at 87/12). Only delta vs the livestream is `--num_envs 16 → 1024` and `--livestream 2` removed.

**Train command:**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

Single GPU (1), 4 minibatches per epoch (1024 × 16 / 4096), matching the run3s.2 horizon/minibatch structure.

**What to watch:**
- `lift_success/iter` — must rise above noise floor (>0.01 sustained). At 1024 envs, noise floor is 1/1024 ≈ 0.001.
- `lift_reward/iter` climbing past **14.7** (table-height residual = 40·exp(-2·0.5)) → policy following the gradient *upward*, not just sitting on contact.
- `termination/physics_instability` vs `termination/real_unsafe` ratio → if physics dominates (>50% of terms), vel_explosion is the bottleneck and we need joint velocity stability fixes (e.g. tighter `velocity_limit_sim` on fingers, lower `max_depenetration_velocity`).
- `episode_lengths` recovering past **60** → unlocks the `min_num_episode_steps=60` lift_reward warmup gate. Below 60, lift_reward is forced to zero by env.py:1085-1091.
- `num_adr_increases` ticking up → cleared `success_for_adr=0.4` threshold on some objects.

**Decision rule:**
- `lift_success > 0.1` and `num_adr_increases ≥ 1` within ep 2000 → success, ride it out, watch for ADR climb
- `lift_success > 0` but slow (e.g. 0.01-0.05 sustained) → gradient working but undersamped; let train longer (5000+ epochs)
- `lift_success ≈ 0` and `lift_reward` parks at 14.7 → policy stuck at table-touch camp again; lower `min_num_episode_steps` 60 → 0 so short-episode envs can also receive lift gradient
- physics_instability > 50% of terms → tighten finger `velocity_limit_sim` 6.2832 → 3.0 rad/s (still ~170 deg/s, way above hardware 60 deg/s spec but more PD-resolvable)
- Episode rewards go negative for >500 epochs → in_grip_alignment is again driving early termination; gate it on hand-near-object (`hand_to_object_pos_error < 0.15`)

**Result (15,645 iters, 2026-05-19 night):** Policy converged to a stable camp at table height. `lift_success = 0` throughout (max ever = 0.00098 at iter 26 = noise floor, > 0 in only 79/15645 iters). `episode_lengths` recovered to 588 (full duration), `hand_to_object_distance` = 9 cm (palm anchored), `object_contact_count` = 2.6 (multiple grasping contacts), `good_grasp_reward` = 1.07 (real grasp formed), `lift_reward` parks at **14.15/step** — exactly the table-touch residual (40·exp(-2·0.5) = 14.7). Net reward ~11,332/episode — policy converged to a new, much higher local optimum: approach + grasp + hold-at-table. Replays of ep 2500 and ep 15000 both show the same "settle for touching" behavior; no upward attempts in either.

Diagnosis: value-function trap. With camp residual = 14.7/step × ~500 steps = ~7,350 secured reward, any upward exploration risks dropping the object (and losing the camp). PPO correctly values "don't move" higher than "risk it". Subsequent audit found 3 silent drifts from working teacher v2 not addressed by run3s.3 — feeding into run3s.5.

### run3s.5 — restore working v2 upward gradient (object_to_goal_sharpness + arm effort curriculum) (2026-05-19)

**Motivation:** Number-by-number audit of current env_cfg vs working teacher v2 (commit 32a8924) found that three parameters had silently drifted away from v2 over the run3s saga, and these collectively weaken the upward reward gradient by ~24%:

1. **`object_to_goal_sharpness` ADR: (-5, -10) → (-8, -12)** — sharpened during run3s to reduce a "camp at object" residual (3.28/step at table → 0.73/step), but this *also* removed a second upward gradient stack. At table height, v2's object_to_goal gradient toward goal was 16.4/m; current is 5.86/m. Lift_reward alone (29.4/m gradient) is doing all the "go up" work. In v2, lift+goal stacked for ~46/m total — that's the gradient the working policy used to discover lifting.

2. **`arm_14_effort_limit`: (90, 87) curriculum → (87, 87) fixed**. Working v2 used hardware spec as the curriculum *end*, with 90 Nm as a small warm-up. Current is flat at 87 from step 0.

3. **`arm_57_effort_limit`: (20, 12) curriculum → (12, 12) fixed**. Same story — wrist torque had 20 Nm warm-up in v2; current is fixed at 12.

Other silent drifts NOT addressed in run3s.5: `hand_to_object_sharpness 5` (v2 was 4), `finger_curl_reg_min -6` (v2 was -3). Left alone — these are second-order effects compared to the upward-gradient issue.

**Changes (cumulative from run3s.3 = run3s.4):**

| param | run3s.4 | run3s.5 |
|---|---|---|
| `object_to_goal_sharpness` ADR | (-8, -12) | **(-5, -10)** |
| `arm_14_effort_limit` ADR | (87, 87) fixed | **(90, 87)** curriculum |
| `arm_57_effort_limit` ADR | (12, 12) fixed | **(15, 12)** curriculum (15 < v2's 20 — keep close to hardware spec) |

**Resulting reward landscape at ADR 0, vertical_err = 0.5 m, hand on object:**

| signal | run3s.4 value | run3s.5 value | delta |
|---|---|---|---|
| `object_to_goal_reward` (camp at table) | 0.73 | **3.28** | +2.55/step |
| upward gradient (lift + goal) | 35.3 / m | **45.8 / m** | **+30% upward pull** |
| arm joints 1-4 effort | 87 Nm | 90 Nm at ADR 0, ramping to 87 | small warm-up |
| arm joints 5-7 effort | 12 Nm | 15 Nm at ADR 0, ramping to 12 | small warm-up |

**Unchanged from run3s.3/3s.4:** lift_sharpness=2, lift_weight=(40,20), finger_curl_reg=(-0.5,-1.2), hand_to_object_weight=3, hand_object_contact_weight=0.8, good_grasp_weight=1.5.

**Train command (1024 envs, single GPU, headless):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success > 0.05` sustained by ep 2000 → restored upward gradient broke the camp, ride it out and watch for ADR climb
- `lift_success > 0` intermittent but trending up → gradient working but undersamped; let train longer
- `lift_success ≈ 0` and `lift_reward + object_to_goal` parks at ~18/step (= 14.7 + 3.28 camp) → camp trap is structural even with stacked gradients; **time to do run3t (z-threshold gate on lift_reward)**
- Episode lengths drop below 60 → the run3s.3 stability has regressed; check whether arm effort warm-up (90/15) destabilized contact

**Result (livestream training, 2026-05-19):** Predicted decision-rule outcome triggered. Status table showed `lift_reward = 14.075` while `lifted_now = 0.0%` and `in_goal_now = 0.0%`. Decoding:
- `lift_reward = 40 × exp(-2 × 0.47) × P(contact) = 15.6 × 0.90 ≈ 14.0` — object sitting on table, ~90% of envs with at least 1 sensor contact, harvesting **39% of the maximum lift_reward (40) WITHOUT lifting**.
- `obj_to_goal = 3.43` — consistent with object at table near spawn (3D dist ~0.31m, weight 40, sharpness -5: 40 × exp(-5 × 0.31) ≈ 8.5 × P(contact)).
- `total = 18.9` — stable camp equilibrium parking value.

Confirms structurally: at `lift_sharpness=2 + lift_weight=40`, the camp residual is so large (15.6/step at table) that it dominates the camp-stack and removes any incentive to lift. The historical teacher v2 win was not a clean reward-gradient result — it relied on specific stochastic exploration. The flat sharpness=2 gradient is fundamentally a camping trap with these weights.

### run3u — bump lift_sharpness 2 → 5 while keeping working v2 reverts (2026-05-19)

**Motivation:** run3s.5 livestream confirmed the camp residual was the actual signal the policy was farming. The reverts to working-v2 values (lift_weight=40, finger_curl=-0.5/-1.2, hand_to_obj=3, object_to_goal_sharpness=-5/-10, arm warm-up curricula) restored the *upward-pull magnitudes* but at sharpness=2 the camp residual at the table (15.6/step) is itself large enough to be a stable equilibrium. **Sharpness alone is the camp-vs-lift discriminator; weights set the magnitude.** Bumping sharpness back to 5 drops the camp residual to 3.8/step (9.5% of max) while keeping the goal-side reward (40) and all other run3s.x reverts intact.

**Change (env_cfg.py:760):**

| param | run3s.5 | **run3u** |
|---|---|---|
| `lift_sharpness` | 2.0 | **5.0** |

All other config carries forward from run3s.5 (unchanged).

**Full state of run3u env_cfg vs the previously-committed run3s state (a577c6f):**

| Variable | committed (run3s) | run3u | run history |
|---|---|---|---|
| `hand_to_object_weight` | 2.0 | **3.0** | bumped in run3s.3 (palm_flip stability) |
| `palm_direction_alignment_weight` | 0.6 | **0.7** | reverted to working v2 |
| `hand_action_rate_penalty_scale` | 1.2 | **1.5** | reverted to working v2 |
| `lift_sharpness` | 5.0 | **5.0** | dropped to 2 in run3s.3, bumped back in run3u |
| `object_to_goal_sharpness` ADR | (-8, -12) | **(-5, -10)** | reverted in run3s.5 (restore stacked upward gradient) |
| `lift_weight` ADR | (25, 12.5) | **(40, 20)** | reverted in run3s.3 (working v2 ADR) |
| `finger_curl_reg` ADR | (-1.0, -2.0) | **(-0.5, -1.2)** | reverted in run3s.3 (reopen closure space) |
| `arm_14_effort_limit` ADR | (87, 87) flat | **(90, 87)** curriculum | re-introduced in run3s.5 (warm-up) |
| `arm_57_effort_limit` ADR | (12, 12) flat | **(15, 12)** curriculum | re-introduced in run3s.5 (warm-up; 15 not v2's 20 — closer to hardware) |

**Resulting reward landscape at ADR 0 (sharpness=5, weight=40):**

| Object pose | lift_reward | % of max |
|---|---|---|
| Sitting on table (z=0.28, vert_err=0.47) | **3.8** | 9.5% |
| Lift_success threshold (z=0.40, vert_err=0.35) | 6.9 | 17% |
| Halfway lifted (z=0.50, vert_err=0.25) | 11.4 | 29% |
| Halfway higher (z=0.60, vert_err=0.15) | 18.9 | 47% |
| At goal (z=0.75, vert_err=0) | 40 | 100% |

Camp:goal ratio is now **1:10.5** (was 1:2.6 at sharpness=2). The flat residual problem from run3s.5 livestream is gone; lift signal grows steeply with vertical motion.

`object_to_goal_reward` at camp (sharpness=-5, weight=40, 3D dist ≈ 0.49 m): 40 × exp(-2.45) = 3.4. Still provides ~16/m goal-side gradient stacked on top of lift's ~38/m → ~54/m total upward pull from table.

**Livestream train command (16 envs, visdex_selected, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
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

**Decision rule:**
- Livestream `lift_reward` settles meaningfully below 10/step at the table → camp residual cleared, policy must work for the lift signal
- Policy visibly attempts upward motion → sharpness=5 was the missing piece; queue full 1024+ env training
- Policy still parks at table (lift_reward ~3.8 = max camp residual) with no upward motion → camp is structurally attractive even at 9.5% residual; **activate run3t (table-contact gate) which zeros lift_reward when object is on table**
- Episode lengths short / palm_flips spike → the reward landscape change destabilized the existing learned palm orientation; check `palm_align` and termination counts

**Result (livestream, stopped by user, 2026-05-19):** Camp residual cleared (`lift_reward` settled in the 3-5/step range at table, ~10% of max — matching the math) but **the policy still did not attempt to lift the object**. The bigger observation from livestream: the **palm approach speed was visibly too high** — the hand rushed in toward the object and the policy had to spend the rest of the episode bleeding velocity / oscillating. This suggests the missing element is not lift signal magnitude but exploration *quality*: the policy lands on the object too fast and unstably to even attempt the vertical-lift action sequence before contact is lost.

Two leverage points identified for run3v: (a) re-enable the dormant `approach_speed_penalty` to discourage fast palm approach, (b) ease sharpness slightly from 5 → 4 so the policy gets a bit more carrot at low elevations once it does engage stably.

### run3v — re-enable approach_speed_penalty + ease lift_sharpness 5 → 4 (2026-05-19)

**Motivation:** Run3u livestream showed the policy approaching the object too fast and never settling into a lift attempt. The `approach_speed_penalty` formula (`-weight × max(0, palm_velocity · approach_dir)²`) has been computed and logged but commented out of the `reward_terms` dict since the run3p-era "don't move" diagnosis. The asymmetric clamp means only INWARD motion is penalized — moving away from the object is free — so it's a clean "slow down on approach" signal that doesn't recreate the freeze-everything failure mode.

Secondary tweak: sharpness=5 was working as designed (camp residual gone) but the gradient between "object on table" and "lift_success" is so steep that brief upward nudges don't produce visible reward feedback. Easing to sharpness=4 gives a slightly bigger reward jump for partial lifts (e.g. at vert_err=0.42 / 5cm up: 4.6 → 7.4) without re-introducing the camping problem (camp residual goes 9.5% → 15% of max, still far from sharpness=2's 39% trap).

**Changes from run3u (2 files):**

| Where | Before | **After** | Why |
|---|---|---|---|
| `env.py:1201` (reward_terms dict) | `# "approach_speed_penalty": approach_speed_penalty` (commented) | **`"approach_speed_penalty": approach_speed_penalty`** (active) | Re-enable. Weight stays at 0.001 — squared, so only fast approaches bite. |
| `env_cfg.py:760` | `lift_sharpness = 5.0` | **`lift_sharpness = 4.0`** | Slightly more carrot for partial lifts; camp residual still small. |

**Resulting lift_reward landscape (weight=40, sharpness=4):**

| Object pose | sharpness=5 (run3u) | **sharpness=4 (run3v)** |
|---|---|---|
| Table sit (vert_err=0.47) | 3.8 (9.5%) | **6.1 (15%)** |
| 5 cm above table (vert_err=0.42) | 4.6 | **7.4** (60% more carrot at low elevation) |
| Lift_success threshold (vert_err=0.35) | 6.9 (17%) | **9.9 (25%)** |
| Halfway lifted (vert_err=0.25) | 11.4 (29%) | **14.7 (37%)** |
| At goal (vert_err=0) | 40 | 40 |

Camp:goal ratio: 1:6.6 (was 1:10.5 at sharpness=5, 1:2.6 at sharpness=2).

**approach_speed_penalty magnitudes at weight=0.001:**

| Palm closing speed | penalty/step |
|---|---|
| 0.1 m/s (slow) | -0.00001 (negligible) |
| 0.3 m/s (typical approach) | -0.00009 (negligible) |
| 0.6 m/s | -0.00036 (small) |
| 1.0 m/s (fast) | -0.001 (mild) |
| 2.0 m/s (ramming) | -0.004 (starts to bite) |

If 0.001 is too gentle and livestream still shows fast approach, bump to 0.005 or 0.01.

**Livestream train command (16 envs, visdex_selected, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
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

**Decision rule:**
- Livestream approach visibly slower / smoother and policy attempts lift → both levers worked; queue full training
- Approach still fast (`extras/approach_speed_penalty` mean small in absolute terms, indicating policy not really feeling the penalty) → bump weight from 0.001 → 0.005 or 0.01
- Approach OK but still no lift attempts → easing sharpness wasn't the missing piece; **activate run3t (table-contact gate)** — at this point we've exhausted shaping options and need to force the policy off the table to discover lift
- Approach better but `palm_align` worsens / palm_flips spike → policy is trading speed against orientation; might need to bump palm_align weight in parallel

**Result (livestream, 2026-05-19):** Approach speed visibly improved — palm slowed down on the approach as designed. Critically, **the policy DID start lifting after contact** — wrist went up toward the goal. But: **the object stayed on the table while the hand lifted empty**. The fingers never closed enough to maintain grip during the upward motion. New insight from the user: this is consistent with a *finger_curl_reg trap* — the regularizer's target `curled_q = init_joint_pos` ≈ open-hand (3° at PIPs/mcp_pitch), so the per-step penalty actively pulls fingers AWAY from a grasp pose. With ADR (-0.5, -1.2), closing 13 finger joints to a ~0.5 rad grasp costs roughly 1.6/step ramping to 3.9/step — directly competitive with the lift gradient. The policy correctly identified the local optimum: "lift hand WITHOUT curling fingers", since curling cost more reward than the marginal extra contact provided.

Also added during run3v iteration: **run3v.1 — palm_flip removed from termination + penalty path** ([env.py:1376, 1180, 1412](dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py)). Hypothesis: termination on palm_flip created an "escape hatch" where the policy could short-circuit a bad rollout. Continuous `palm_direction_alignment_reward` (-0.7 × θ²) remains as the only palm-orientation signal. `term_counts["palm_flip"]` still tracked for diagnostics. Effect on the empty-lift behavior: not the bottleneck (the empty lift was still observed after this change), but episodes did get longer with fewer palm_flip terminations.

### run3w — relax finger_curl_reg to 10× weaker (allow grip to form) (2026-05-19)

**Motivation:** Run3v livestream pinpointed the *finger closure* problem. The reward landscape successfully drives the wrist upward (lift gradient at sharpness=4 is fine), and approach speed is in check, but `finger_curl_reg` with `curled_q = init_joint_pos` actively punishes the policy for closing fingers. The naming is misleading — the "curl reg" is really an "open-hand reg" because `curled_q` was set to the slightly-open init pose (3° offsets to avoid joint-min PD oscillation, per CLAUDE.md note). At ADR (-0.5, -1.2), this is competitive with the lift gradient: closing 13 finger joints to ~0.5 rad bent costs ~1.6/step ramping to ~3.9/step, while the marginal lift_reward gain from actually carrying the object up is similar. Policy optimum: lift empty.

**Change (env_cfg.py:927):**

| Parameter | run3v | **run3w** |
|---|---|---|
| `finger_curl_reg` ADR | (-0.5, -1.2) | **(-0.05, -0.1)** |

10× softer. Still leaves a small "default pose" pull (for sim2real stability) but no longer fights grasp formation. Reference: kuka_allegro uses (-0.01, -0.01) — effectively off — as a working proven config.

**Magnitude impact at a typical grasp (13 finger joints each ~0.5 rad bent → finger_curl_dist² ≈ 3.25):**

| ADR step | run3v (-0.5 → -1.2) | **run3w (-0.05 → -0.1)** |
|---|---|---|
| ADR 0 (start) | -1.6/step | **-0.16/step** |
| ADR end | -3.9/step | **-0.33/step** |

The lift gradient (~3-6/step at table → ~15/step at lift_success threshold → 40 at goal) now dominates by a wide margin even with a full grip — closing fingers is no longer punished competitively with lifting.

**Livestream train command (16 envs, visdex_selected, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
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

**Decision rule:**
- `finger_curl` in rewards table drops from ~-1.6 → ~-0.16 magnitude (10× smaller — confirms config applied)
- Livestream visibly shows finger closure during contact — the key behavioral test
- Object lifts WITH the hand (vs the empty-hand lift in run3v) → curl_reg was the bottleneck; queue full training
- Fingers close but object slips out → friction issue; either bump object/finger friction or restore some `good_grasp` weight
- Fingers still don't close → curl_reg wasn't the dominant blocker; try (-0.01, -0.01) matching kuka_allegro, then investigate finger PD strength

**Result (livestream, stopped at ~2500 epochs, 2026-05-19):** Mixed outcome. Fingers now visibly curl (the curl_reg fix worked as designed). **Contact reward started to emerge by ep ~2500** — the policy was learning to engage the object with the curled hand. **But no lift attempts were made** — the wrist-up motion that emerged in run3v didn't reappear here. Hypothesis: the run3s-era cuts to `hand_object_contact_weight` (0.8) and `good_grasp_weight` (1.5) left the contact shaping signal too weak to align the curl timing with the object position, so the policy spent epochs learning to make ANY contact rather than progressing toward the contact-then-lift sequence. The previous "lift empty hand" behavior from run3v was tied to the heavier curl_reg producing a flatter hand that incidentally lifted along the lift gradient — once fingers were freed to curl, the policy had to relearn the approach geometry.

### run3x — restore contact + good_grasp weights to shape grip timing (2026-05-19)

**Motivation:** Run3w showed the policy now CAN curl fingers (curl_reg fix), but the cut contact (0.8) and good_grasp (1.5) signals weren't strong enough to coordinate WHEN to close. After 2500 epochs the policy was just starting to make contact, with no lift attempts at all. The fix: restore these signals to roughly their pre-camp-fight values so they actively shape "curl AT the object, not in mid-air, then lift."

**Changes (env_cfg.py:750-751):**

| param | run3w | **run3x** |
|---|---|---|
| `hand_object_contact_weight` | 0.8 | **1.5** |
| `good_grasp_weight` | 1.5 | **3.0** |

`good_grasp_reward` fires only when thumb + ≥1 other finger touch — the exact "real grip formed" signal. Doubling it makes that pattern decisively more rewarding than "fingers closed near the object but missing it".

**New reward stack (typical contact with 3 sensors + good_grasp firing, object on table):**

| Signal | run3w | **run3x** |
|---|---|---|
| `hand_to_object` | ~2 | ~2 |
| `contact_reward` (3 sensors) | 2.4 | **4.5** |
| `good_grasp_reward` | 1.5 | **3.0** |
| `lift_reward` at table | 6.1 | 6.1 |
| `object_to_goal` at camp | ~0.7 | ~0.7 |
| `finger_curl_reg` (grip pose) | -0.16 | -0.16 |
| **Camp-with-grasp total** | ~13/step | **~18/step** |
| **Goal-with-grasp total** | ~52/step | **~60/step** |

Camp:goal ratio: 1:3.3 → 1:3.3. The bumps don't disproportionately favor camp — they raise both endpoints by similar amounts (lift gradient dominates the goal-side payoff). What changes is the *coordination*: the policy gets more reward feedback specifically for "fingers in contact with the object" patterns, which should shape when/where the curl happens.

**Livestream train command (16 envs, visdex_selected, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
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

**Decision rule:**
- Contact and good_grasp values in the rewards table climb past run3w's "suck" baseline within ~500-1000 epochs → grip shaping signals are taking hold
- Visible finger closure ON the object (not in air) followed by lift attempts → the cuts were the problem; queue full training
- Contact rises but still no lift after ~2500 epochs → contact+good_grasp shaping isn't enough; might need to tighten approach (palm_finger_alignment?) OR finally activate run3t (table-contact gate on lift_reward)
- Camp regresses (contact rises but policy parks at object) → bumped good_grasp/contact too much relative to lift_reward; partial backoff (good_grasp 3 → 2.0)

**Result (livestream, run dir `05-19_14-45-23`, stopped at ~1934 epochs, 2026-05-19):** Real progress for the first time. Most metrics climbing healthily AND **some envs achieved actual lifts** (lift_success max = 0.0625 = 1 env out of 16, with `lift_reward` max spike of 0.5/step — both well above the pure 1/1024 noise floor of prior runs).

| Metric | start (<50 ep) | end (last 50 ep) | Trend |
|---|---|---|---|
| `rewards/iter` (ep mean) | -97 | **+123** | Strong climb |
| `hand_to_object_distance` | 0.54 m | **0.21 m** | Hand reaching object |
| `hand_object_contact_reward` | 0.006 | **0.69** | Contact emerging |
| `good_grasp_reward` | 0.000 | **0.19** | ~6% of envs forming grasps |
| `object_to_goal_reward` | 0.015 | **1.30** | Object motion toward goal |
| `lift_reward` (mean) | 0.025 | 0.035 (peaks at 0.5) | Mostly camping, occasional lifts |
| `lift_success` (mean) | 0.001 | 0.0 (peaks at 0.0625) | 1/16 envs lifting in good moments |
| `finger_curl_reg` | -0.14 | -0.055 | Curl_reg fix confirmed working |
| `early_term_penalty` | -0.003 | 0.000 | Penalized terminations not firing |
| `episode_lengths` | **263** | **39** | ❌ Crashed |

Two concerning patterns:
- **Episode lengths crashed from 263 → 39**. The policy regressed into a strategy that terminates fast. `early_term_penalty ≈ 0` rules out the *penalized* terminations (object_out, hand_too_far), so episodes are dying via unpenalized causes — most likely `hand_too_close`, `arm_table_contact`, or `vel_explosion`. Consistent with a "rush at object, jam fingers" behavior that's productive per-step but fragile.
- **Lift success intermittent, not sustained**. Some envs find the lift but the behavior doesn't generalize across the policy at 16-env scale.

Net: **the reward shaping is now demonstrably producing lift behavior** — just not consistently at livestream scale. Worth scaling to 1024 envs to see if the lift discovery consolidates across the policy with 64× more parallel exploration. Episode-length and ramming concerns can be addressed in a follow-up run if they persist at scale.

### run3x.1 — scale to 1024 envs (consolidate the emerging lift signal) (2026-05-19)

**Motivation:** Run3x livestream proved the reward landscape can produce real lifts in some envs. At 16 envs across 13 objects (~1.2 envs/object), the gradient is too noisy to consolidate the discovered behavior — the lift behavior fired intermittently and never crossed into a stable strategy across the policy. At 1024 envs (~79 envs/object), parallel exploration should let the gradient lock in the lift trajectories that were already emerging.

**Config unchanged from run3x** — same lift landscape (sharpness=4, weight=40), same grip shaping (contact=1.5, good_grasp=3.0), same curl_reg (-0.05, -0.1), same palm_flip-removed termination set, same approach_speed_penalty active.

**Train command (1024 envs, headless, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` mean rises and sustains above 0.05 (= 50/1024 envs) by ep ~2000 → lift behavior consolidating; let it ride toward `success_for_adr=0.4` threshold and ADR climb
- `lift_success` stays in noise (< 0.005) but `lift_reward` mean rising → lift attempts are happening but not crossing the height threshold; bump `object_height_thresh` down (0.15 → 0.10) OR ramp lift_weight via ADR
- `episode_lengths` crashes the same way (< 100 by ep 1000) → ramming is still killing rollouts at scale; bump `approach_speed_penalty_weight` 0.001 → 0.005 and/or activate `penetration_penalty_weight`
- Camp regression (no lifts, contact_reward growing past good_grasp_reward) → contact weight too high, partial backoff 1.5 → 1.2

**Result (run dir `05-19_15-36-39`, stopped at ~1146 epochs, 2026-05-19) — catastrophic policy collapse:** The run3x reward shaping worked at first then was forgotten. Per-window means:

| Window (iter) | rewards | contact | lift_reward | object_to_goal | hand_to_obj_dist | episode_lengths |
|---|---|---|---|---|---|---|
| 0-100 | 659 | 0.87 | 1.42 | 1.00 | 0.28 | 230 |
| **100-200 (peak)** | **4391** | **2.71** | **4.26** | **2.64** | **0.12** | 360 |
| 200-400 | 4421 | 1.79 | 3.37 | 2.02 | 0.19 | 456 |
| 400-600 | 1301 | 0.24 | 0.62 | 0.38 | 0.24 | 548 |
| 600-800 | 531 | 0.025 | 0.10 | 0.06 | 0.23 | 523 |
| 800-1000 | 261 | **0.001** | 0.006 | 0.003 | 0.25 | 539 |
| 1000-1147 | 455 | 0.003 | 0.017 | 0.010 | 0.24 | 574 |

Policy trajectory:
1. **ep 100-400 — working policy.** Hand approaching to 12cm, contact firing at 2.7 (~1.8 sensors), good_grasp at 0.83 (~28% of envs gripping), lift_reward 4.3.
2. **ep 400-800 — abandoning engagement.** Contact drops 100×. hand_to_obj_distance climbs from 0.12m to 0.24m. *Episode lengths simultaneously grow.* Policy is finding longer-survival strategies that involve LESS contact.
3. **ep 800+ — locked into avoidance.** Contact = 0.001 (basically never touches object), hovering at safe distance. rewards/iter slowly recovering (261 → 635) on the wrong axis — the policy is milking small positive signals (hand_to_object ~1.0, palm_align ~-0.23) over long-duration episodes.

`early_term_penalty ≈ 0` across all windows means the penalized terminations (object_out, hand_too_far) weren't firing. So engagement was being implicitly punished by UNPENALIZED early terminations (`hand_too_close`, `arm_table_contact`, `vel_explosion`) — the policy lost future reward from short rollouts and PPO gradients pulled it toward the safe-hover equilibrium.

Structural diagnosis: with the current termination set, **commitment to engagement is a high-variance bet** (sometimes lift = big reward, sometimes terminate early = small reward), while **hovering near the object** is a low-variance positive return (~1/step from `hand_to_object_reward` × ~540 steps ≈ 540 reward/episode). The policy correctly minimized variance and abandoned the high-variance lift trajectory.

### run3y — remove approach_speed_penalty, retry 1024 envs (2026-05-19)

**Motivation:** After the run3x.1 collapse, the priority is reducing "don't move" pressure on the policy that might compound with the implicit-termination cost it already faces. The approach_speed_penalty was re-enabled in run3v with weight 0.001 — magnitude was tiny in practice (-0.0007/step at peak, totaling maybe -0.4 over an episode) but conceptually it added another small headwind against approach motion. Removing it simplifies the gradient. Doesn't solve the structural collapse problem (those are run3t / terminate-penalty changes), but removes one confounder before deciding whether to invoke those bigger changes.

**Change (env.py:1203):**

| | run3x.1 | **run3y** |
|---|---|---|
| `approach_speed_penalty` in reward_terms | active (weight 0.001) | **commented out** |

Penalty term still computed and logged to `extras/approach_speed_penalty` for diagnostics — just removed from `total_reward`.

**Train command (1024 envs, headless, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- Reward curve tracks similar to run3x.1 (peak ep 200-400, then collapse) → approach_speed_penalty wasn't the bottleneck; structural problem confirmed; **activate run3t (table-contact gate) and/or penalize unpenalized terminations**
- Reward keeps climbing past ep 400 without collapse → simpler reward stack helped; let it run toward ADR ramp
- Collapse happens *earlier* than last time (e.g. by ep 200) → some other parameter regression; check for asset/code changes since last run3x.1
- `lift_success` sustained > 0.05 by ep 2000 → lift behavior consolidated; first real success since this saga began

**Watch checkpoint:** if reward peaks similarly to run3x.1 (ep 200-400 around 4000+), check `dextrah_tekken_lstm.pth` immediately after that window — it's likely the auto-best snapshot of the working policy before collapse, useful for replay analysis.

**Result (run dir `05-19_16-27-37`, stopped at ~1500 epochs, 2026-05-19) — engaged camp equilibrium, no collapse, no lift:** Run3y survived past the run3x.1 collapse zone (ep ~400) without abandoning engagement. Removing approach_speed_penalty was sufficient to prevent the avoidance basin from emerging.

| Window (iter) | rewards | contact | good_grasp | lift_reward | hand_to_obj_dist | episode_lengths |
|---|---|---|---|---|---|---|
| 0-100 | 679 | 0.68 | 0.18 | 1.35 | 0.25 | 234 |
| 100-200 | 3836 | 2.31 | 0.72 | 4.07 | 0.12 | 342 |
| 200-300 | 5456 | 3.05 | 1.05 | 4.54 | 0.11 | 403 |
| 300-400 | 5983 | 2.58 | 1.08 | 4.52 | 0.13 | 446 |
| 400-500 | 5060 | 2.19 | 1.01 | 4.34 | 0.15 | 428 |
| **500-1500 (plateau)** | **~5500-6000** | **~2.5-2.9** | **~1.0-1.3** | **~5.0-5.5** | **~0.12** | **~430** |

Hand approaches reliably (12cm distance), 84% of envs form real grasps (good_grasp ~1.0 at weight 3.0), but **lift_success stayed at noise floor** (max 0.0029 = 3/1024 envs at peaks, never sustained). `lift_reward` plateaued at 5.5/step which decodes to ~90% of envs in contact × `40·exp(-4·0.47) = 6.1` camp residual = 5.5. **The lift_reward is parked at the camp residual ceiling, not from lifting.** Policy is risk-averse: marginal gain from "actually lift" (~34/step at goal) is small compared to the guaranteed ~5/step from "camp with contact", risk-adjusted equilibrium = camp.

`object_to_goal` plateaued at 3.3/step = same camp residual story for that signal at sharpness=-5, dist=0.5m.

`episode_lengths` stable at ~430 (vs the run3x.1 climbing-while-engagement-collapsing pattern). No avoidance, just engagement+camp.

Net diagnosis: same structural camping problem we've been chasing since run3p. The two reward exponentials (lift_reward and object_to_goal) both have camp residuals because their shape pays out at table-touch. The policy correctly identified that "camp and harvest residuals" beats "risk a lift attempt".

### run3z — soften lift/goal sharpness toward "linear-style" gradient (2026-05-19)

**Motivation:** The user's intuition: maybe the previous sharpness values created reward cliffs the policy couldn't traverse. Partial lifts paid out small marginal reward (at sharpness=4, lifting 5cm gains only +1.3/step over camp), so the policy stayed at camp. Try a much flatter exponential — closer to "linear" — where every cm of lift pays a more constant marginal reward (~0.3/cm at sharpness=1).

This is the *opposite* direction from killing the camp residual (which would mean *higher* sharpness or a table-contact gate). The hypothesis: the policy isn't camp-trapped because the residual is too big — it's camp-trapped because partial lifts feel like "no progress" due to the small marginal gradient. A flatter curve provides more uniform gradient feedback during any upward motion.

CAVEAT: this makes the camp residual much LARGER (25/step vs 6.1/step at sharpness=4). If the hypothesis is wrong, run3z will camp even harder than run3y.

**Changes (env_cfg.py):**

| Param | run3y | **run3z** | Why |
|---|---|---|---|
| `palm_direction_alignment_weight` | 0.7 | **0.5** | Reduce palm-orientation pressure now that palm_flip terminations are off (run3v.1). |
| `lift_sharpness` | 4.0 | **1.0** | Smoother lift gradient; constant ~0.3 reward per cm of lift. Caveat: camp residual 6.1 → 25.0. |
| `object_to_goal_sharpness` ADR | (-5, -10) | **(-3, -8)** | Smoother goal-distance gradient; ~16/step at lift_success threshold (was 5). Caveat: object_to_goal camp residual 3.3 → 8.9. |

**Resulting lift_reward landscape (weight=40, sharpness=1):**

| Object pose | vert_err | lift_reward | % of max |
|---|---|---|---|
| Table sit | 0.47 | 25.0 | 62.5% |
| 5cm above table | 0.42 | 26.3 | 65.8% |
| Lift_success threshold (15cm) | 0.35 | 28.2 | 70.5% |
| Halfway lifted (25cm) | 0.25 | 31.1 | 77.8% |
| At goal (50cm) | 0 | 40 | 100% |

Marginal gain per cm of lift: **constant ~0.3 reward**. Total gain from camp to goal: 15 reward over 50cm.

**Resulting object_to_goal_reward landscape (weight=40, sharpness=-3):**

| 3D distance to goal | reward |
|---|---|
| 0.5 m (camp) | 8.9 |
| 0.4 m | 12.0 |
| 0.3 m | 16.3 |
| 0.2 m | 21.9 |
| 0.1 m | 29.6 |
| 0 (at goal) | 40 |

Combined camp residual at table-touch: ~25 + ~9 = **~34/step** of lift+goal signals alone (was 5.5 + 3.3 = 8.8/step in run3y). The total camp equilibrium reward is now MUCH higher.

**Train command (1024 envs, headless, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` consolidates above 0.05 sustained by ep 2000 → smooth gradient hypothesis confirmed; flatter exponential was the missing piece
- `lift_success` stays at noise floor + camp metrics inflate (`lift_reward` jumps to ~22-25/step from camp residual alone) → confirms camp trap is structural, regardless of curve shape; **NEXT MOVE: activate run3t (table-contact gate)** to zero lift_reward when object is on table
- Total ep reward climbs past 10k (vs run3y's 5500-6000) but lift_success still zero → policy farming the inflated camp residual; same diagnosis as above
- Policy collapses to avoidance like run3x.1 → unlikely without approach_speed_penalty back, but worth watching

**Result (run dir `05-19_18-18-53`, 7595 epochs, single-object chicken_head_in_car, 2026-05-19) — decisive confirmation of camp residual trap, smooth-gradient hypothesis falsified:** This run was the cleanest possible test: single-object training removes object-diversity confounds, 1024 envs gives dense gradients. Result is the worst outcome of any run in the saga.

| Window (iter) | rewards | **lift_reward** | **good_grasp** | contact | obj_to_goal | hand_to_obj_dist | episode_lengths |
|---|---|---|---|---|---|---|---|
| 0-500 | 14,536 | 18.1 | **0.69** | 2.17 | 7.3 | 0.16 | 460 |
| 500-1500 | 18,389 | 21.5 | **0.08** | 1.47 | 8.5 | 0.15 | 540 |
| 1500-3000 | 17,779 | 21.8 | **0.002** | 1.35 | 8.7 | 0.15 | 521 |
| 3000-4500 | 19,339 | 22.4 | **0.000** | 1.38 | 8.8 | 0.15 | 555 |
| 4500-6000 | 20,156 | 22.5 | **0.000** | 1.46 | 9.2 | 0.15 | 567 |
| **6000-7596** | **20,482** | **22.5** | **0.000** | 1.43 | 9.2 | 0.15 | **578** |

Decoding:
- `lift_reward = 22.5` = `40 × exp(-1 × 0.47) × P(contact=0.9) = 25.0 × 0.9 = 22.5` — **exactly the predicted camp residual at sharpness=1**
- `object_to_goal = 9.2` = `40 × exp(-3 × 0.5) × P(contact=1.0) = 8.9` — **exactly the predicted camp residual at sharpness=-3**
- Total camp harvest from lift+goal alone = 31.7/step × 578 steps = ~18,000 reward/episode (matches `rewards/iter ≈ 20,000`)
- `lift_success = 0` for entire run after iter ~200 — no lifts at all

The bombshell: **good_grasp REGRESSED to zero.** Started at 0.69 (23% of envs grasping early on), collapsed to literally 0.0 by ep 1500. The policy actively *unlearned* grasping. Why? At sharpness=1, lift_reward fires at 22.5/step (near max) for ANY single-finger contact (`contact_mask` is binary). Forming a real grasp (thumb + finger) provides no additional lift_reward — only the marginally smaller good_grasp_reward (3/step max). So the policy converged to the *minimum-effort contact* strategy: hover at 15cm, tap one finger, harvest 31.7/step residual indefinitely. `object_contact_count = 0.95` (less than 1 sensor avg) confirms the lazy contact.

Replay confirms visually: hand hovers stable at ~15cm from object, occasionally brushes with one finger, never attempts grasp or lift.

**Conclusion from the run3o → run3z arc:** Every reward shape we've tried — sharpness 2, 4, 5, 8, 10, 1, 0.5 — produces the same outcome with different operating points: the policy finds the laziest way to harvest the camp residual, and the residual is structurally guaranteed to exist as long as `lift_reward = weight × exp(...) × contact_mask` fires for any table-touch. **No exponential shape solves this.** The reward formula needs a hard gate (run3t) to zero out lift_reward when the object is on the table.

The good_grasp degradation in run3z is the new evidence that this saga can't continue with shaping alone — at smooth sharpness the policy is *incentivized to do less*, not more. Run3t is now structurally required, not optional.

### run4a — fresh retrain from run2g baseline post-revert (2026-05-19)

**Motivation:** The run3o→run3z saga of reward shaping decisively failed (full diagnosis in exp_03 commit acafa2b). Rather than continue exploring shape variants or adding the table-contact gate to an env config that has drifted far from the working baseline, do a clean reset: revert the three diverged files (`env.py`, `env_cfg.py`, `fr3_tekken_left.py`) to commit `4fe7cb7` — the end of the original run2a-g exploration that produced Teacher v2's best result (79.1% lift / 25.3% unsafe at run2g ep 2500). All experiment logs and CLAUDE.md are preserved (see commit `990c395`).

This is a direct re-test of the question: **can run2g's training setup still reproduce its historical result on current IsaacLab v2.2.1?** Per CLAUDE.md, this hasn't been reproducible since the IsaacLab upgrade — recent measurements with the same .pth + same env_cfg give 55.0% lift / 40.8% unsafe on visdex_selected. So the realistic expectation isn't 79.1% — it's whether the working config can at least produce LIFTING behavior again, even if the absolute number is lower.

**Configuration (as reverted from 4fe7cb7):**

Key reward weights:
- `lift_sharpness = 4.0` (not 1.0 from run3z, not 5.0 from run3u — the run2g-tuned middle value)
- `success_bonus_weight = 20.0` (was 10 pre-run2g)
- `object_to_goal_weight = 40` static
- `hand_to_object_weight = 4.0` (run2g value, was bumped to 3.0 then back)
- `hand_object_contact_weight = 3.0` (run2g value)
- `good_grasp_weight = 6.0` (run2g value — much higher than run3 era's 1.5-3.0)

Key ADR ranges:
- `lift_weight` ADR: (40, 30) — slower decay than run3u's (40, 20)
- `object_to_goal_sharpness` ADR: (-5, -10)
- `finger_curl_reg` ADR: (-0.3, -0.8) — between run3w's (-0.05, -0.1) and run3 pre-revert (-0.5, -1.2)
- Arm + finger gains: (0.7, 2.0) — tightened from run3's (0.5, 2.0)
- `joint_pos_noise`: (0, 0.35) — was (0, 0.8) in some run3 attempts

Termination set: palm_flip is BACK in `out_of_reach` (reverts run3v.1). approach_speed_penalty is BACK in reward_terms (reverts run3y).

Asset state: `fr3_tekken_left.py` at 4fe7cb7 — slightly different from recent edits.

**Run directory:** `05-19_22-32-41`

**Train command (1024 envs, headless, GPU 1):**
```bash
cd dextrah_lab/rl_games
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Expected vs run3z (single-object, sharpness=1, smooth-gradient regression):**

| Signal | run3z plateau | run4a expectation |
|---|---|---|
| `rewards/iter` | 20,500 (inflated camp harvest) | 3,000-6,000 (run2g-era reward magnitudes) |
| `lift_reward` | 22.5/step (camp residual) | ~3-4/step at table + scaling toward 40 at goal |
| `lift_success` | 0.000 (no lifts) | Expect intermittent lifts emerging by ep 500-1000; consolidation by ep 2000-3000 if reproducible |
| `good_grasp` | 0.000 (regressed) | Expect ~1.0-3.0 (good_grasp_weight=6 here) |
| `hand_to_obj_dist` | 0.15m (lazy hover) | Expect to drop below 0.12m as approach learns |

**Decision rule:**
- `lift_success` sustained > 0.05 by ep 2000 → run2g baseline is still trainable on current IsaacLab → can proceed to distillation
- `lift_success` 0.01-0.05 plateau → partial success, same structural ceiling as the cited "55% lift" measurement from CLAUDE.md
- `lift_success` < 0.005 indefinitely → the run2g baseline IS NOT trainable from scratch on current IsaacLab; the historical 79.1% was either a fluke run or required IsaacLab v < 2.2.1; **next step: table-contact gate (run3t) on this baseline**, since we've now isolated the issue from the run3 shape drift
- Collapse pattern emerges (rewards peak then decline like run3x.1) → IsaacLab drift introduced the same termination-cost asymmetry that broke run3x.1; investigate which terminations are firing

**Result:** *(to be filled in — training in progress)*

### run4b — Teacher v1 (exp 1m, non-hardware-realistic) sanity check (2026-05-19)

**Motivation:** Before assuming the run4a v2 retrain plateau (if it happens) is system-level drift below the repo, isolate whether the dextrah_clean env / current IsaacLab v2.2.1 / current physx + driver stack can still train *any* fr3_agilehand teacher to lift behavior. Teacher v1 (exp 1m) is the pre-hardware-realistic baseline that historically reached ~85.8% lift (Teacher 11 benchmark). If v1 trains today, the regression is v2-specific; if v1 also fails to lift, the cause is below the repo.

This pairs with the diagnostic snapshot tooling added in `dextrah_lab/docs/scripts/diagnostic_snapshot.py` (2026-05-19) — the snapshot was designed to surface system-level drift (IsaacLab HEAD, rl_games commit, driver version, USD LFS state, omni.physx version). The v1 sanity check is the empirical complement to that static comparison.

**Setup:** Main repo (`/home/carsten.oertel/code/tg2_dexman_isaac_co`), `dextrah_clean` env, Teacher v1 (exp 1m) config — i.e. the original non-hardware-realistic FR3+AgileHand teacher setup, NOT the run4a v2 setup in the test repo. Different repo, different config, same underlying stack (Isaac Sim 5.1.0.0, IsaacLab v2.2.1, driver 555.42.06, torch 2.7.0+cu128, RTX 4090).

**Result (epoch 180):** **6% lift_success**, climbing.

**Result (epoch 13500, 2026-05-20):** **ADR level 13 reached.** v1 climbed through the full ADR curriculum and settled at the same structural plateau as the historical Teacher 11 benchmark (which converged at ADR ~13, 85.8% lift). This is the healthy, expected outcome — v1 has historically been the strongest fr3_agilehand teacher.

**Interpretation (confirmed at ep 13500):**
- The stack is **not** broken at the system level — IsaacLab v2.2.1 + current physx + driver 555.42.06 + dextrah_clean env are still capable of training a full fr3_agilehand teacher through the entire ADR curriculum to ADR 13.
- **Decisively rules out** the "below-the-repo drift" hypothesis from the diagnostic checklist (categories A1–A6 in `diagnostic_snapshot.py`): IsaacLab commit, rl_games version, PyTorch/CUDA, driver, apt-installed libs, omni.physx version. None of those can be the cause if the same stack trains v1 to ADR 13 today.
- Narrows the regression to **v2-specific config or v2-specific actuator+physics interaction**: hardware-realistic effort/velocity limits, arm init randomization, tighter starting joint limits, soft_joint_pos_limit=0.8, etc. — these are present in v2 and absent in v1.
- Early signal (6% at ep 180) was consistent with healthy Teacher 11 trajectory; ADR 13 at ep 13500 confirms the full curriculum is traversable on current stack.

**Implications for run4a (in progress, test repo):**

run4a is testing v2 at the **4fe7cb7-state revert** — but a closer look (via the diagnostic_snapshot script) revealed that the env_cfg at 4fe7cb7 had already drifted from the actual policy-12 training state (`32a8924`, Apr 3) within the 2a-2g runs themselves. **10 fields** drifted between 32a8924 and 4fe7cb7:

| Field | 32a8924 (policy 12 training) | 4fe7cb7 / current run4a |
|---|---|---|
| `success_bonus_weight` | 10.0 | 20.0 |
| `lift_sharpness` | 2.0 | 4.0 |
| `arm_joint_stiffness_and_damping` | (0.5, 2.0) | (0.7, 1.5) |
| `finger_mcp_pitch_gains` / `_yaw` / `pip` / `thumb_rot_gains` | (0.5, 2.0) | (0.7, 2.0) |
| `robot_spawn.joint_pos_noise` | (0., 0.8) | (0., 0.35) |
| `lift_weight` ADR | (40., 20.) | (40., 30.) |
| `finger_curl_reg` ADR | (-0.5, -1.2) | (-0.3, -0.8) |

Applied a hand-edit revert to the **test repo only** (`code/test/tg2_dexman_isaac_co/dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py`) to restore all 10 fields to 32a8924 state. `git diff 32a8924 -- ...env_cfg.py` is now clean except for one trailing-whitespace difference. All other training-path files (`env.py`, `train.py`, USDs, agent yaml configs) are bit-identical to 32a8924 by `git diff --stat`. Verified via `diagnostic_snapshot.py` re-run.

**Decision rule for next experiment:**
- v1 (run4b) climbs to >50% lift by ep 2000+ → confirms system works.
- THEN run v2 at the 32a8924-reverted env_cfg in the test repo (parallel GPU). If v2 lifts → drift across 2a-2g and run3 investigations was the entire cause (lesson: accumulated config tweaks during exploration silently broke trainability). If v2 still won't lift even at the 32a8924-exact config → narrows to v2-specific actuator/physics interaction that became incompatible with current physx (the only remaining suspect).
- Either way, run4a (the 4fe7cb7 revert) is *not* the same experiment as a 32a8924-exact revert — the 10-field drift makes that distinction important.

**Status (2026-05-20):** v1 reached ADR 13 at ep 13500 → system healthy. Next step is unblocked: kick off v2 training at the 32a8924-reverted env_cfg in the test repo on a parallel GPU and observe whether it lifts. That experiment will isolate "drift across run3 was the entire cause" from "v2 actuator/physics interaction is itself the problem".

### run4c — fresh v2 retrain at exact policy-12 baseline (32a8924) (2026-05-20)

**Motivation:** run4a is anchored at `4fe7cb7`, which the diagnostic_snapshot.py-driven audit revealed had **10 fields drifted** from the actual policy-12 training state at `32a8924` (Apr 3, 2026). Those drifts accumulated *within* the 2a-2g run cycle itself — so run4a is *not* the same experiment as a true policy-12 baseline retrain. run4c corrects that: hand-edit env_cfg.py to exactly match `32a8924` (verified via `git diff 32a8924 -- ...env_cfg.py` clean except trailing whitespace). All other training-path files (`env.py`, `train.py`, USDs, agent yaml configs, `fr3_tekken_left.py`) are already bit-identical to 32a8924 — verified via `git diff --stat 32a8924 -- dextrah_lab/tasks/fr3_agilehand/` and the asset file. So run4c is the **closest empirical retrain of policy 12 possible from the current repo**.

This run is paired with the v1 sanity check (run4b): v1 already confirmed the stack (IsaacLab v2.2.1, Isaac Sim 5.1.0.0, driver 555.42.06, torch 2.7.0+cu128) trains a teacher to ADR 13 today. So run4c's failure modes are now disambiguated — if v2 also fails at the 32a8924-exact config, the cause must be v2-specific.

**Configuration (32a8924-exact, the 10 reverted fields):**

Scalar reward weights:
- `success_bonus_weight = 10.0` (was 20.0 in 4fe7cb7)
- `lift_sharpness = 2.0` (was 4.0 in 4fe7cb7) — the prime suspect for "touch-don't-lift" basin in run3
- `object_to_goal_weight = 40` (unchanged)
- `hand_to_object_weight = 4.0` (unchanged; CLAUDE.md's "expect 3.0" was a false alarm based on later tuning notes)

ADR ranges:
- `lift_weight` ADR: (40., 20.) (was (40., 30.) in 4fe7cb7)
- `finger_curl_reg` ADR: (-0.5, -1.2) (was (-0.3, -0.8))
- `robot_spawn.joint_pos_noise`: (0., 0.8) (was (0., 0.35) — note CLAUDE.md considers 0.35 "correct" for sim2real but 0.8 is what policy 12 actually trained with)
- `arm_joint_stiffness_and_damping`: (0.5, 2.0) (was (0.7, 1.5))
- `finger_mcp_pitch_gains` / `_yaw_gains` / `pip_gains` / `thumb_rot_gains`: (0.5, 2.0) (were (0.7, 2.0))

Everything else (env.py reward computation, contact gating, termination set, USDs, agent yaml) is unchanged from current HEAD = bit-identical to 32a8924.

**Setup:**
- Repo: test repo (`code/test/tg2_dexman_isaac_co`)
- Env: `dextrah_test` (test repo's env, distinct from main repo's `dextrah_clean`, but same underlying Isaac Sim / IsaacLab / driver — validated healthy by run4b)
- GPU: 0 (free; v1 was on GPUs 1-3 in main repo)
- Seed: 42 (matches v1 + prior v2 attempts for direct comparability)

**Train command:**
```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 30% by ep 2000-3000 → accumulated config drift across run3 was the entire cause; v2 is trainable at the 32a8924 baseline; can resume distillation.
- `lift_success` plateau at near-zero through ep 5000+ → narrowed to **v2-specific actuator/physics interaction** (hardware-realistic effort/velocity limits, soft_joint_pos_limit=0.8, arm init randomization EventTerm) incompatible with current physx state. Next move would be to bisect those one at a time, starting with the v2-only differences from v1's config.
- Partial lift (5-20% plateau) with same ADR 13 wall as historical → matches the "55% lift current measurement" pattern; suggests policy 12's 79.1% historical number was a fluke run or required IsaacLab < v2.2.1, but the config is otherwise sound.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_07-35-01/` (test repo, GPU 0)

**Result (ep 4800, 2026-05-20): FAILURE. Same touch-don't-lift basin as run3.**

TensorBoard at iter sample points:

| Signal | ep ~200 | ep ~1000 (peak rew) | ep ~2500 | ep ~4500 |
|---|---|---|---|---|
| **lift_success** | 0.000 | 0.000 | 0.000 | 0.000 |
| **in_success_region** | 0.000 | 0.000 | 0.000 | 0.000 |
| **num_adr_increases** | 0 | 0 | 0 | 0 |
| rewards (raw) | 6754 | **13870** | 12690 | 12771 |
| lift_reward (shaped) | 11.2 | 13.8 | 13.9 | 14.1 |
| hand_object_contact_reward | 6.1 | 6.9 | 3.5 | 4.6 |
| good_grasp_reward | 1.10 | 1.53 | 0.28 | 1.02 |
| object_contact_count | 2.0 | 2.3 | 1.2 | 1.5 |
| hand_to_object_distance (m) | 0.116 | 0.111 | 0.153 | 0.129 |
| episode_lengths | 306 | 519 | 554 | 529 |

Checkpoints by reward: ep 500 (5413), **ep 1000 (14411 — peak)**, ep 1500 (13256), ep 2000 (12070), ep 2500 (11932), ep 3000 (11938), ep 3500 (10112 — dip), ep 4000 (13016), ep 4500 (12981).

**Observations:**
1. **Zero lifts across 4800 epochs.** `lift_success = 0.000` and `in_success_region = 0.000` for the entire run. Identical to the run3 touch-don't-lift basin.
2. **ADR stuck at level 0** for the entire run — curriculum never advanced because `in_success_region < success_for_adr=0.4` was never met. So this isn't even a meaningful "hardware-realistic v2 with curriculum" test — it's failing at ADR 0 (the easiest possible setting).
3. **Peak-then-decline pattern** at ep 1000 — same regression structure as run3x.1. Between ep 1000 and 2500, `hand_object_contact_reward` dropped 6.9 → 3.5 (-49%) and `good_grasp_reward` dropped 1.5 → 0.28 (-81%). The policy briefly explored grasping at ep 1000 then *abandoned* it because farming the shaped `lift_reward` (~14/step) without committing to grasping pays better than risking a failed grasp.
4. **`lift_reward ≈ 14` despite zero lifts** — the shaped distance-from-table term fires from micro-displacement / vibration without actual lifting. Same exploit closed by `good_grasp_mask` gating during run3, but that fix isn't in the 32a8924 baseline code.
5. **Replay of `ep_1000_rew_14411.823.pth`** (highest-reward checkpoint) confirmed visually: same behavior as run3 era — hand approaches, fingers contact, thumb buckles or fails to close, no lift.

**Implication — decision rule trigger from earlier in this log fired:**
> "v2 won't lift even at 32a8924-exact config → narrowed to v2-specific actuator/physics interaction."

Accumulated config drift across run3 was **not** the cause. The original policy-12 baseline state itself can't be retrained to lift on current IsaacLab v2.2.1 + current physx state. Combined with run4b (v1 reached ADR 13 on same stack), this rules out below-the-repo system drift AND in-repo config drift as primary causes — narrows decisively to **v2-specific actuator/physics interaction** with the current Isaac Sim state.

The historical 79.1% policy 12 result is either a fluke run that benefitted from random init luck, or required an Isaac Sim / IsaacLab / physx state that no longer exists.

**v1 ↔ v2 diff analysis (post-failure, 2026-05-20):**

Diffing the test repo (v2/32a8924) against the main repo (v1, currently at ADR 13) surfaces exactly **5 training-relevant differences**:

| # | Surface | v1 (main, trains) | v2/32a8924 (test, fails) | Priority |
|---|---|---|---|---|
| 1 | `arm_joint_init` EventTerm | absent | ±0.2 rad on `fr3_joint.*` every reset | HIGH |
| 2 | thumb_rot starting vel limit | 10.0 rad/s (573 deg/s) | **0.2618 rad/s (15 deg/s)** | **VERY HIGH** |
| 3 | arm 5-7 starting effort | 50.0 Nm | **20.0 Nm** (hardware spec) | MEDIUM |
| 4 | arm 1-4 starting effort | 100.0 Nm | 90.0 Nm | LOW |
| 5 | Finger spawn noise zeroing | absent — noise on all 19 joints | finger noise zeroed (arm-only) | LOW (counter-direction) |

All four curriculum *endpoints* are identical between v1 and v2 — the differences are entirely in the **starting state** of the actuator curriculum. So v2 trains in a much harder regime from epoch 0 (38× tighter thumb_rot velocity, 2.5× tighter wrist effort) and the curriculum has narrower headroom to ramp down from.

Bisection plan as run5a-5d: re-introduce each v2 change on top of v1's config one at a time. Prime suspect is **thumb_rot starting velocity limit** (run5a) — 15 deg/s from epoch 0 may be too tight for the policy to learn grasp timing, since v1 trains with a 38× faster thumb throughout.

### run5a — double thumb_rot starting velocity limit (2026-05-20)

**Motivation:** Isolate whether the 15 deg/s thumb_rot starting velocity floor is what blocks grasping in v2 (run4c's failure mode). Double the starting value from 0.2618 → 0.5236 rad/s (~15 → ~30 deg/s). Keep the endpoint at 0.1396 rad/s (8 deg/s, hardware-realistic) so the curriculum still terminates at the same hardware target — only the *headroom during early learning* changes. Still **38× tighter than v1's 10 rad/s starting value**, so this is the most conservative possible bisection step (a single-knob increase, not a full revert to v1).

**Configuration delta from run4c:**
- `thumb_rot_vel_limit`: (0.2618, 0.1396) → **(0.5236, 0.1396)** ([env_cfg.py:937](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py))
- Everything else identical to run4c / 32a8924 baseline (arm_joint_init EventTerm still active, arm 5-7 effort still 20 Nm start, all reward weights unchanged, all other ADR ranges unchanged).

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024, visdex_selected — identical to run4c.

**Train command:**
```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` > 0.05 by ep 2000 → 15 deg/s starting floor was the blocker. Can then tighten back toward 15 deg/s start incrementally once learning is established (run5a.1 with start=0.3927 = ~22 deg/s, run5a.2 with start=0.2618 = 15 deg/s).
- Lift peaks then declines (run4c pattern) → thumb headroom alone is insufficient; reward shape also pulls toward camp-don't-lift. Compare contact/good_grasp peak heights to run4c's 6.9/1.5.
- Still flat-zero lift through ep 4000+ → thumb_rot is **not** the structural cause. Revert to (0.2618, 0.1396), then run5b: remove `arm_joint_init` EventTerm.

**Run directory:** *(launched on test repo GPU 0, terminated early at ep ~1000)*

**Result (ep 1000, terminated early, 2026-05-20):** `lift_success = 0.000`. Identical trajectory to run4c at the same epoch — flat-zero lifting, no early sign of grasp-timing emerging despite 30 deg/s thumb headroom. Did not reach the 2000-epoch decision point per the rule above. Inconclusive on whether thumb_rot is the structural cause (decision rule required ep 4000+ of flat-zero to definitively rule it out), but the absence of any positive signal at ep 1000 (run4c's contact_reward peak epoch) is a discouraging early indicator. User chose to switch tracks rather than wait — moving to an environment-isolation test (run4d) to rule out test-repo / dextrah_test env as the cause before continuing the actuator bisection.

### run4d — replicate run4a (4fe7cb7) in main repo + dextrah_clean env (2026-05-20)

**Motivation:** run4a (test repo, dextrah_test env, 4fe7cb7 state) and run4c (test repo, dextrah_test env, 32a8924 state) both failed to lift. run5a (test repo, doubled thumb velocity) shows no early grasping at ep 1000 either. Before continuing the v2 actuator bisection, isolate one more environmental variable: **does the same v2/4fe7cb7 config train differently when run from the main repo with the dextrah_clean env (the env that just trained v1 to ADR 13)?** If main+dextrah_clean also fails at 4fe7cb7 → repo/env can be ruled out as the cause. If it succeeds → something in the test repo install or dextrah_test env is contaminating training.

This is a pure environment-swap experiment with config controlled (same commit as run4a, same train hyperparams as run4c).

**Setup:**
- Repo: **main** (`/home/carsten.oertel/code/tg2_dexman_isaac_co`)
- Branch state: detached HEAD at `4fe7cb7` (Teacher v2 runs 2a-2g wrap-up commit)
- Env: **`dextrah_clean`** (`/home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python`)
- GPU: 0 (free)
- Seed: 42, num_envs 1024, visdex_selected — matches run4a/4c/5a for direct comparability

**Commands:**

```bash
# 1. Switch to main repo and checkout the last working 2a-g commit
cd /home/carsten.oertel/code/tg2_dexman_isaac_co
git checkout 4fe7cb7
# (detached HEAD warning is expected and fine)

# 2. Launch training from main repo with dextrah_clean env
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- Same failure pattern (0 lift through ep 2000+, peak-then-decline reward) → repo + env confirmed irrelevant; v2 actuator/physics interaction is the root cause regardless of which env/repo runs it. Resume bisection (run5b: remove arm_joint_init EventTerm).
- Materially different result (any lift signal, or different reward trajectory) → test repo or dextrah_test env had a contaminating factor; investigate dextrah_test pip freeze vs dextrah_clean, USD copy state, asset symlinks, and other test-repo-specific state.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_11-54-27/` (main repo, GPU 0, dextrah_clean env)

**Result (ep 2500, 2026-05-20): FAILURE, but different failure mode than run4c.**

TensorBoard at iter sample points:

| Signal | ep 500 | ep 1000 | ep 1500 | ep 2000 | ep 2500 | PEAK |
|---|---|---|---|---|---|---|
| **lift_success** | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | **0.008 @ ep 474** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 @ ep 479 |
| rewards (raw) | 7418 | 7870 | 6924 | 7158 | 6898 | 9334 @ ep 996 |
| good_grasp_reward | 0.54 | 0.28 | 0.25 | 0.12 | 0.13 | **1.11 @ ep 351** |
| hand_object_contact_reward | 4.77 | 6.47 | 5.52 | 5.36 | 4.87 | 7.43 @ ep 942 |
| in_grip_alignment_reward | -1.00 | -1.12 | -1.12 | -1.24 | **-1.34** | -0.15 @ ep 62 |
| finger_curl_reg | -1.08 | -1.55 | -1.51 | -1.53 | -1.55 | -0.01 @ ep 75 |
| hand_to_object_distance (m) | 0.148 | 0.133 | 0.151 | 0.152 | 0.146 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | — |
| episode_lengths | 500 | 456 | 444 | 469 | 472 | 599 @ ep 38 |

**Observations:**
1. **Brief lifting signal emerged then collapsed.** Peak `lift_success = 0.008 (0.8%) at ep 474`; `good_grasp_reward = 1.11 at ep 351`. By ep 1000 both had collapsed to ~0 and stayed there through ep 2500.
2. **Reward landscape peaked at ep 996** (9334 raw) then declined to 6898 at ep 2500. Not climbing anymore.
3. **`in_grip_alignment_reward` worsened over time** (-1.00 → -1.34): the object is *increasingly not in the grip zone*. The policy learned to keep fingers near the object without putting it between them — same "camp at contact" failure pattern as run4c, just shifted later.
4. **`hand_object_contact_reward` stays 4.8-6.5**: policy makes contact but doesn't grasp. Touch-don't-lift basin fully locked in.
5. **ADR stuck at level 0** through 2500 epochs — same as run4c, run5a.

**Comparison to run4c:**

| Signal | run4c (test, 32a8924) | run4d (main, 4fe7cb7) |
|---|---|---|
| Lift signal ever > 0? | NO (flat zero entire 4800 ep) | **YES, briefly (peak 0.8% @ ep 474)** |
| Peak reward | 14411 @ ep 1000 | 9334 @ ep 996 |
| Failure mode | never grasps | grasps briefly then abandons |
| Env / repo | test, dextrah_test | main, dextrah_clean |
| Config | 32a8924 | 4fe7cb7 |

**Decision rule triggered:**
> "Same failure pattern (0 lift through ep 2000+, peak-then-decline reward) → repo + env confirmed irrelevant; v2 actuator/physics interaction is the root cause regardless of which env/repo runs it."

Conclusions reinforced:
- Below-the-repo stack ruled out (v1 trains)
- Config drift across run3 ruled out (both 32a8924 and 4fe7cb7 fail)
- **Now also ruled out: repo state (test vs main) and conda env (dextrah_test vs dextrah_clean)** — same v2 config fails identically in either.
- Brief lifting in run4d but not run4c suggests the 4fe7cb7 reward shape (steeper `lift_sharpness=4.0`, higher `success_bonus=20`) does *initially* steer the policy toward lifting better than 32a8924's shape — but neither holds the basin past ep ~1000.

Remaining surface: the 5 v2-only differences from v1's working config (arm_joint_init EventTerm, thumb_rot starting velocity, arm 5-7/1-4 starting effort, finger spawn noise zeroing). Bisection should now isolate which of these breaks v2 on the current stack.

### run4e — 32a8924 config + main repo + dextrah_clean env (2026-05-20)

**Motivation:** Complete the 2×2 grid (config × env/repo) before continuing the actuator bisection:

| Config \ env+repo | test + dextrah_test | main + dextrah_clean |
|---|---|---|
| **32a8924** | run4c (failed: flat zero) | **run4e (this run)** |
| **4fe7cb7** | run4a (failed: flat zero) | run4d (failed: brief 0.8% lift, collapse) |

run4e checks the only remaining unfilled cell: **does 32a8924 also produce a brief lifting signal when run in the env that v1 succeeded in?** If yes, it would mean repo/env *does* contribute to the early exploration phase (just not enough to break the basin). If no, the 32a8924 config is structurally worse than 4fe7cb7 for lifting regardless of env.

This is also the closest practical reproduction of "policy 12's actual training conditions" given the lack of an Apr 6 commit: 32a8924 is the only env_cfg state committed before policy 12 trained, and dextrah_clean is the env that just trained v1 to ADR 13.

**Setup:**
- Repo: **main** (`/home/carsten.oertel/code/tg2_dexman_isaac_co`)
- Branch state: detached HEAD at `32a8924` (Apr 3, 2026 — Teacher v2 setup commit)
- Env: **`dextrah_clean`** (conda-activated, not direct-binary path)
- GPU: 0
- Seed: 42, num_envs 1024, visdex_selected — matches run4a/4c/4d/5a

**Commands:**

```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co
git checkout 32a8924
conda activate dextrah_clean
cd dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- Same flat-zero lift through ep 2000+ as run4c → 32a8924 config is structurally non-trainable on current stack regardless of repo/env. Both Apr 6 bracketing states fail; policy-12 reproduction is impossible from current dextrah_lab state. Confirms cause is v2-specific actuator/randomization deltas. Move to **run5b: remove arm_joint_init EventTerm**.
- Brief lift signal like run4d → 32a8924 is salvageable in dextrah_clean; run4c's flat-zero was contaminated by test repo / dextrah_test specifics. Worth investigating what's different about that env (pip freeze diff, asset copy state, etc).
- Sustained climbing past run4d's 0.8% peak → policy 12 was actually trained at this 32a8924 config (Apr 3 commit) and the wrap-up 4fe7cb7 drift introduced the regression. Then the fix is to revert env_cfg to 32a8924 and continue distillation.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_13-39-06/` (main repo, GPU 0, dextrah_clean env)

**Result (ep 579, terminated early, 2026-05-20): brief lift signal — DIFFERENT failure mode than run4c.**

TensorBoard at iter sample points:

| Signal | ep 250 | ep 500 | PEAK |
|---|---|---|---|
| **lift_success** | 0.000 | 0.001 | **0.005 @ ep 147** |
| in_success_region | 0.000 | 0.000 | 0.000 |
| rewards (raw) | 8317 | 12434 | 13991 @ ep 552 |
| lift_reward | 12.77 | 13.49 | 14.12 @ ep 430 |
| hand_object_contact_reward | 6.99 | 7.68 | 8.10 @ ep 531 |
| good_grasp_reward | 1.21 | 1.53 | **1.75 @ ep 531** |
| object_contact_count | 2.33 | 2.56 | 2.70 @ ep 531 |
| hand_to_object_distance (m) | 0.110 | 0.109 | — |
| finger_curl_reg | -2.15 | -2.27 | -0.19 @ ep 78 |
| num_adr_increases | 0 | 0 | 0 |
| episode_lengths | 332 | 458 | 520 |

**Observations:**
1. **Brief lift signal emerged early**, peaking at `lift_success = 0.005` at ep 147, then collapsing to ~0 by ep 250. Same pattern as run4d's 0.8% peak but smaller and earlier.
2. **Rewards still climbing when terminated** (8317 → 12434 → 13991 across the run) — unlike run4d which peaked at ep 996 then declined. Hard to say if run4e would have followed the same collapse pattern past ep 1000 because it didn't reach that point.
3. **Good_grasp and contact rewards climbing** through ep 531 — policy was actively learning to engage the object. The early lift collapse pattern is similar to run4d but the larger reward landscape hadn't peaked yet.
4. **ADR stuck at 0** through 579 ep — same as every v2 retrain to date.

**Decision rule outcome — second branch fires (with caveat):**

> "Brief lift signal like run4d → 32a8924 is salvageable in dextrah_clean; run4c's flat-zero was contaminated by test repo / dextrah_test specifics."

Comparison to run4c definitively confirms env/repo affects the early exploration phase:

| | run4c (test, 32a8924) | run4e (main, 32a8924) |
|---|---|---|
| Lift signal ever > 0? | NO (flat zero, 4800 ep) | **YES, briefly (0.5% @ ep 147)** |
| Peak reward through ep 500 | ~9300 @ ep 1000 | 12434 @ ep 500 (still climbing) |
| Failure mode | never grasps | grasps briefly then abandons |

**The env/repo difference is real but does not break the basin.** Same config (32a8924), same v2-specific actuator deltas, different early exploration outcome — but neither reaches sustainable lifting. This is consistent with the run4d pattern: the policy can briefly find lifting, but the touch-don't-lift basin pulls it back. Env/repo determines *how easily* the policy stumbles into the brief lift exploration window — main+dextrah_clean does it earlier but smaller (0.5% @ ep 147), test+dextrah_test does it later but bigger or not at all.

**Net implication:** Env/repo is a *minor* variable affecting exploration phase, NOT the structural cause of the v2 regression. The bisection should continue focusing on v2-specific actuator/randomization deltas (arm_joint_init, thumb_rot velocity floor, arm 5-7 effort).

**Note:** User terminated this run at ep 579 to free GPU 0 for run5b. Did not reach the 2000-epoch decision point — strictly speaking, can't rule out the third branch (sustained climbing) without continuing past ep 1000. But given run4d's full 2500-epoch trajectory at the related 4fe7cb7 config showed the same touch-don't-lift collapse, it's safe to assume run4e would have followed the same path.

### run5b — remove arm_joint_init EventTerm (2026-05-20, terminated early)

**Motivation:** Isolate whether the ±0.2 rad arm joint randomization at every reset (the qualitatively biggest v2↔v1 difference) is what blocks grasp-timing consolidation in v2.

**Result (ep 580, terminated early):** Did not show meaningful improvement. User stopped early — "didn't seem good." Per the user's qualitative assessment, the policy was not showing emerging lift signal at 580 epochs (vs run4d which had peak `lift_success=0.008` at ep 474). Suggests `arm_joint_init` alone is not the structural cause — removing it didn't unlock lifting. Bisection moves on to reward-magnitude experiments (run5c).

### run5c — 10× lift_weight ADR bump on clean 4fe7cb7 / run2g state (2026-05-20)

**Motivation:** Three actuator-side bisection attempts (run5a doubled thumb_rot, run5b removed arm_joint_init, both implicitly v2 config) failed to unlock lifting. Run4d showed the 4fe7cb7 reward shape *does* briefly steer the policy toward grasping (0.8% lift peak @ ep 474) — but the basin pulls it back into camp-don't-lift by ep 1000. So the issue may be **insufficient reward magnitude**, not actuator constraints: even when grasping briefly emerges, the marginal value of committing to a full lift isn't high enough vs the camping baseline.

Test: **10× multiply the `lift_weight` ADR range from (40., 30.) to (400., 300.)** while keeping everything else at the clean run2g end-state (4fe7cb7 / 990c395). This makes successful lifting ~10× more rewarding than any other shaped term. If the touch-don't-lift basin is fundamentally a reward-shape problem (not an actuator-feasibility problem), this should break out of it by sheer reward gradient.

Historical reference: CLAUDE.md notes that run3l bumped `lift_weight` ADR from (60, 30) → (100, 50) and "made the lift action 60% more rewarding than camping." That ~67% bump didn't break the basin. A 10× bump is qualitatively different — it's testing whether the reward landscape itself is the bottleneck.

**Risks of 10× bump:**
- Reward saturation — lift_reward alone could dominate total reward enough that the gradient signal from other terms becomes negligible.
- Unsafe lifts — policy may learn to flick the object up at any cost (even causing OOB / palm flip terminations) because the lift bonus outweighs termination penalties.
- These are tolerable for this experiment: we want to see if the policy *can* lift at all. If it does, we can then dial the bump back to find a sustainable level.

**Configuration delta from clean 4fe7cb7 / 990c395 baseline:**
- `lift_weight` ADR: (40., 30.) → **(400., 300.)** ([env_cfg.py:922](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py))
- Everything else identical to 4fe7cb7 / 990c395 (arm_joint_init EventTerm active, thumb_rot start 15 deg/s, arm 5-7 start 20 Nm, lift_sharpness=4.0, success_bonus=20).

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024, visdex_selected.

**Train command:**
```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past run4d's 0.8% peak and consolidates (>5% by ep 2000) → reward magnitude was the structural bottleneck. Then iterate: dial the bump back in run5c.1 (e.g., 5× = (200, 150)) and run5c.2 (2× = (80, 60)) to find the smallest bump that still works.
- Brief lift peak then collapse, similar to run4d → 10× helps initial exploration but the basin still pulls back. Means there's an additional structural issue beyond reward magnitude (likely actuator-side).
- Flat zero lift through ep 2000+ → reward magnitude is *not* the issue. Confirms the cause is in v2's actuator/randomization deltas. Strong indication that bisection should pivot to a *combined* test (multiple v2 changes reverted at once).
- Unsafe rate explodes (palm flips, OOB) → reward gradient is too strong, policy learning to flick rather than lift. Need to add safety terms or reduce bump.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_14-18-57/` (test repo, GPU 0, dextrah_test env)

**Result (ep 1213, 2026-05-20): FAILURE. Reward magnitude confirmed NOT the bottleneck — basin more unstable, not broken.**

TensorBoard at iter sample points:

| Signal | ep 50 | ep 200 | ep 300 (peak engagement) | ep 500 | ep 750 (avoidance) | ep 1000 | PEAK |
|---|---|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | **0.002 @ ep 164** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewards (raw) | -745 | 6423 | 16235 | 19561 | **2505** | 23374 | 32221 @ ep 1173 |
| lift_reward | 0.0 | 30.4 | 36.4 | 27.1 | **5.2** | 50.3 | 55.1 @ ep 1082 |
| hand_object_contact_reward | 0.0 | 2.94 | 3.80 | 2.20 | **0.53** | 5.35 | 6.01 @ ep 1068 |
| good_grasp_reward | 0.0 | 0.39 | 0.54 | 0.32 | **0.10** | 1.18 | 1.41 @ ep 1062 |
| object_contact_count | 0.0 | 0.98 | 1.27 | 0.74 | **0.18** | 1.78 | 2.00 @ ep 1068 |
| hand_to_object_distance (m) | 0.77 | 0.16 | 0.14 | 0.25 | **0.41** | 0.13 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 481 | 186 | 368 | 484 | 470 | 382 | 599 |

User qualitative observation at ep 1200: "0 lift. high contact and bad good_grasp."

**Observations:**
1. **`lift_success` peaked at 0.002 (0.2%) at ep 164 — *lower* than run4d's 0.8% peak with the same env_cfg minus the 10× bump.** The 10× lift_weight bump did not unlock lifting; if anything, the bigger reward gradient hurt convergence.
2. **Policy oscillates between engagement and avoidance phases:** approach learning by ep 200 (contact_count 0.98) → engagement peak ep 300 (contact 1.27, lift_reward 36) → **avoidance regression ep 500-750** (contact dropped to 0.18, hand walked 0.27m away from object) → re-engagement ep 1000+ (contact back to 1.78). The 10× reward gradient is making policy updates more violent, causing larger swings in/out of engagement basin.
3. **`lift_reward` saturates at the 10× camp residual (~50/step)** at engagement peaks — exactly the math: 400 · exp(-4 · 0.5) ≈ 54 for object-at-table-touch. Policy is fully exploiting the shaped lift_reward without ever actually lifting.
4. **`good_grasp_reward` at peak = 1.41** (vs weight=6.0 cfg ceiling). Moderate but not high — user's "bad good_grasp" observation is consistent with this.
5. **ADR stuck at 0** through 1213 ep — same as every other v2 retrain on this stack.

**Decision rule outcome — second branch fires:**

> "Brief lift peak then collapse, similar to run4d → 10× helps initial exploration but the basin still pulls back. Means there's an additional structural issue beyond reward magnitude (likely actuator-side)."

Worse than that branch: the 10× bump didn't help initial exploration (peak 0.2% vs run4d's 0.8%) AND introduced new oscillation pathology. Reward magnitude is now decisively ruled out as the bottleneck. Strong evidence that the structural cause lies in v2's actuator/randomization deltas.

**Comparison against all v2 retrains to date:**

| Run | Config | Env | Peak lift_success | Pattern |
|---|---|---|---|---|
| run4c | 32a8924 | test+dextrah_test | 0.000 | flat zero entire 4800 ep |
| run4e | 32a8924 | main+dextrah_clean | 0.005 @ ep 147 | brief peak, terminated at 579 |
| run4d | 4fe7cb7 | main+dextrah_clean | 0.008 @ ep 474 | peak then decline |
| run5a | 32a8924 + thumb 2× | test+dextrah_test | 0 @ ep 1000 | flat zero, terminated |
| run5b | 4fe7cb7 − arm_joint_init | test+dextrah_test | 0 @ ep 580 | no improvement, terminated |
| **run5c** | **4fe7cb7 + lift 10×** | **test+dextrah_test** | **0.002 @ ep 164** | **oscillation, basin worse** |

Every single-knob revert has now been tested. None unlock lifting. **The next experiment should be a multi-knob test** — revert several v2 deltas simultaneously to v1's values. Candidates: remove `arm_joint_init` + restore thumb_rot start to v1's 10.0 rad/s + restore arm 5-7 effort to v1's 50 Nm. If that lifts, narrow down which combination matters. If that *also* fails, the cause is below the config level (LSTM hidden state interaction with these constraints? early termination cascade?).

### run6a — bump good_grasp_weight 3 → 15 to dominate contact (2026-05-20)

**Motivation:** Pivoting from actuator-side bisection to reward-shaping based on user qualitative observation during run5c. Hypothesis: the touch-don't-lift basin is caused by the policy **lifting before forming a proper grasp** — it commits to upward motion when finger contact is insufficient, the object slips, hand ends up empty. This explains the contact-then-avoidance oscillation observed in run5c: contact emerges → policy tries to lift → grasp slips → contact drops → policy retreats → cycles back.

Evidence from run5c TB at peaks: `hand_object_contact_reward = 6.0`, `good_grasp_reward = 1.4`. With weights `hand_object_contact_weight=3.0` and `good_grasp_weight=3.0`, good_grasp condition fires only ~23% of contact time. The policy is rewarded ~4× more for *touching* than for *properly grasping*, so it never has structural pressure to commit to thumb+finger closure before attempting lift.

Fix: **bump `good_grasp_weight` 3 → 15 (5×)** so good_grasp can deliver up to 15/step (vs contact's ~6/step at 5 sensors). With good_grasp dominating, the policy's reward gradient should pull it toward thumb+finger commit before any vertical motion is profitable. lift_weight stays at baseline (40, 30) — undoing run5c's 10× bump that destabilized the policy.

This is identical to the run3-era hypothesis that produced the `contact_mask = good_grasp_mask` gate (per CLAUDE.md), but tested via weight-shaping rather than hard gating — softer and reversible.

**Configuration delta from previous (cd6ecfe / run5c):**
- `good_grasp_weight`: 3.0 → **15.0** ([env_cfg.py:747](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py))
- `lift_weight` ADR: (400., 300.) → (40., 30.) — reverting run5c's 10× bump back to baseline
- Net diff vs 990c395 (run2g baseline): one line, `good_grasp_weight` only.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 0.05 by ep 2000+ → good_grasp shaping was the missing piece; the basin was a grasp-formation problem, not actuator-side. Then iterate: try a smaller bump (run6a.1 with 9.0 to find minimum sufficient weight), check for unsafe rate, then resume distillation pipeline.
- Brief lift peak then collapse, similar to run4d/5c → good_grasp helps initial exploration but contact oscillation still drives the policy out of engagement. Means the issue is *also* in approach stability, not just grasp commit. Next: combine good_grasp bump with reduced approach speed / arm_joint_init removal.
- Contact stabilizes higher (>2.0 average), good_grasp jumps proportionally, BUT lift still doesn't emerge → grasp is being formed but policy still doesn't lift. Then it's an actuator/feasibility issue. Pivot back to multi-knob actuator revert (run6b).
- Flat zero lift through ep 2000+, no improvement in good_grasp/contact ratio → good_grasp_weight isn't the lever even at 5×. Strong push to multi-knob actuator revert next.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_15-39-08/` (test repo, GPU 0, dextrah_test env)

**Result (ep 1393, stopped early, 2026-05-20): HIGHEST v2 LIFT PEAK EVER (1.6%), then catastrophic collapse — hypothesis validated but magnitude too aggressive.**

TensorBoard at iter sample points:

| Signal | ep 200 | ep 400 (engaging) | ep 500 (peak) | ep 700 (CRASH) | ep 900 | ep 1100 | ep 1300 (recovering) | PEAK |
|---|---|---|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.001 | 0.004 | **0.000** | 0.000 | 0.000 | 0.000 | **0.016 @ ep 517** |
| rewards (raw) | 2819 | 9023 | 7132 | **-722** | -344 | -717 | 459 | 10371 @ ep 373 |
| lift_reward | 3.51 | 4.43 | 2.99 | **0.00** | 0.004 | 0.004 | 0.085 | 4.86 |
| hand_object_contact_reward | 3.55 | 4.75 | 3.06 | **0.00** | 0.002 | 0.002 | 0.045 | 5.60 @ ep 313 |
| good_grasp_reward | 3.01 | 5.13 | 3.20 | **0.00** | 0.00 | 0.00 | 0.006 | **6.06 @ ep 401** |
| object_contact_count | 1.19 | 1.58 | 1.02 | **0.00** | 0.001 | 0.001 | 0.015 | 1.87 @ ep 313 |
| hand_to_object_distance (m) | 0.14 | 0.14 | 0.27 | **0.38** | 0.24 | 0.37 | 0.21 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 226 | 472 | 524 | **580** | 530 | 563 | 293 | 599 |

**Observations:**
1. **`lift_success` peaked at 0.016 (1.6%) at ep 517 — 2× the previous v2 record (run4d's 0.8%).** User observed 1.8% in debug window around the peak. The grasp-before-lift hypothesis is validated: incentivizing proper thumb+finger grasps over single-finger taps produced the strongest lift signal of any v2 retrain.
2. **`good_grasp_reward` jumped 4.3× from run5c (1.41 → 6.06).** The 5× weight bump translated into a ~4× reward increase — the policy was actually getting credit for forming proper grasps, and exploration responded.
3. **Catastrophic policy collapse at ep ~700.** Total reward went from +7132 → -722 in one window (-110% swing). All engagement metrics dropped to zero: contact 3.06 → 0.00, good_grasp 3.20 → 0.00, lift_reward 2.99 → 0.00. Hand walked from 0.27m → 0.38m. Episode_lengths spiked to 580 (timeout-only, no contact-triggered ends). This is the worst collapse pattern of any v2 run to date.
4. **Mechanism of collapse:** with `good_grasp_weight = 15`, the policy's reward expectations grew large. A few failed grasps (with grasp event delivering 0 reward vs the expected ~15) likely produced strongly negative advantage estimates, pushing the policy away from approach behavior entirely. The combination of "high promised reward for success" + "zero reward for partial success" + "no positive baseline for staying near" created a deep avoidance gradient.
5. **Slow recovery starting ep 1300** (rewards back to +459, distance dropping). But based on every prior v2 collapse pattern, recovery from -700 rewards back to engagement is extremely slow and unlikely to surpass the original peak.

**Decision rule outcome — first branch fires with caveat:**

> "`lift_success` climbs past 0.05 by ep 2000+ → good_grasp shaping was the missing piece"

Did not quite cross 0.05 (peak 0.016 = 1.6%), but **2× the previous v2 record on a single-knob change** is strong validation that good_grasp shaping is the right lever. The hypothesis is correct; the magnitude was too aggressive. Next: dial back to find the sustainable magnitude.

**Implications:**
- Grasp-before-lift hypothesis is structurally correct — verified by the strongest lift signal in any v2 retrain.
- 5× bump (3→15) is too aggressive: reward gradient becomes too violent and collapses the policy after the brief engagement window.
- A more moderate bump (3→9 = 3×) should retain the grasp incentive without the catastrophic failure cost.
- Once a sustainable magnitude is found, run6a.x can stack with other improvements (e.g., reduced approach speed, longer training horizon).

### run6a.1 — dial good_grasp_weight back from 15 → 9 (2026-05-20)

**Motivation:** Run6a confirmed the grasp-before-lift hypothesis (1.6% lift peak, the strongest v2 signal ever) but the 5× bump (3→15) was too aggressive — caused catastrophic policy collapse at ep ~700 (rewards +7132 → -722, all engagement to zero). Find the sustainable magnitude. **3× bump (3→9)** should still let good_grasp dominate contact (max 9/step vs contact's ~6/step) while halving the reward-expectation gradient violence that destabilized run6a.

If 3× works → look for `lift_success` peak comparable to run6a (>1%) but *sustained* — no collapse to negative rewards. If 3× also collapses (just smaller magnitude) → grasp shaping isn't a stable lever on its own; need to combine with other adjustments (gate lift on good_grasp_mask, reduce approach speed, etc).

**Configuration delta from previous (85799c0 / run6a):**
- `good_grasp_weight`: 15.0 → **9.0** ([env_cfg.py:747](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py))
- Net vs 990c395 baseline: still single-knob (good_grasp 3 → 9, 3× bump vs baseline).
- Everything else identical to run2g baseline.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` reaches >0.005 by ep 500 AND sustains without collapsing past ep 1000 → 3× is the sustainable magnitude. Then let it run longer to see if ADR finally advances past 0 (success_for_adr=0.4).
- Same peak-then-collapse pattern as run6a, just smaller peak → 3× still too aggressive; try 3 → 6 (2× bump = just matching contact weight).
- No lift signal at all, peak < run6a's 1.6% → 3× isn't enough incentive; the 5× peak from run6a was the natural ceiling for this approach without combining with other shaping. Pivot to combination experiments (run6b: good_grasp 3→9 + gate lift on good_grasp_mask).
- Catastrophic collapse to negative rewards (like run6a) → good_grasp bump in *any* magnitude triggers the avoidance basin. Need to add a positive baseline reward (e.g., increased hand_to_object_weight) to prevent the policy from retreating.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_16-49-54/` (test repo, GPU 0, dextrah_test env)

**Result (ep 4178, stopped by user, 2026-05-20): FAILURE. Moderate bump (3×) didn't break the basin — peak 4× lower than run6a's 5× peak.**

TensorBoard at iter sample points:

| Signal | ep 200 | ep 500 | ep 1000 (eng. peak) | ep 1500 (CRASH) | ep 2500 (recovery) | ep 3500 | ep 4000 | PEAK |
|---|---|---|---|---|---|---|---|---|
| **lift_success** | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.004 @ ep 926** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 @ ep 983 |
| rewards (raw) | 2696 | 6186 | 8795 | **95** | 2917 | 7176 | 6149 | 12437 @ ep 831 |
| lift_reward | 3.32 | 3.26 | 4.95 | **0.011** | 4.10 | 5.36 | 4.95 | 5.60 @ ep 3519 |
| hand_object_contact_reward | 3.18 | 3.47 | 5.32 | **0.006** | 1.90 | 3.18 | 3.23 | 8.33 @ ep 753 |
| good_grasp_reward | 1.30 | 2.08 | 3.34 | **0.001** | 0.001 | 1.72 | 1.86 | 5.97 @ ep 764 |
| object_contact_count | 1.06 | 1.16 | 1.77 | **0.002** | 0.63 | 1.06 | 1.08 | 2.78 @ ep 753 |
| hand_to_object_distance (m) | 0.149 | 0.195 | 0.145 | **0.377** | 0.194 | 0.155 | 0.170 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 256 | 460 | 469 | 528 | 280 | 454 | 407 | 599 |
| finger_curl_reg | -1.15 | -1.46 | -1.48 | -0.58 | -0.53 | -0.48 | -0.95 | -0.01 |

**Observations:**
1. **`lift_success` peaked at 0.004 (0.4%) at ep 926 — 4× LOWER than run6a's 1.6% peak.** The 3× bump (vs run6a's 5×) wasn't enough incentive to push the policy into the lift basin even briefly.
2. **Same collapse pattern as run6a, less severe magnitude.** Rewards 8795 → 95 at ep 1500 (>99% drop but not negative this time). Hand walked 0.145m → 0.377m, all engagement vanished. Confirms the basin pulls the policy back regardless of weight magnitude.
3. **Recovery into stable touch-don't-lift state.** By ep 3500-4000 the policy re-engaged (contact 1.06, good_grasp 1.7-1.9, rewards 6000-7000) — but **lift_success stays at zero**. The policy found a stable equilibrium that's not lifting.
4. **ADR stuck at 0** through 4178 epochs — same as every v2 run.

**Decision rule outcome — third branch fires:**

> "No lift signal at all, peak < run6a's 1.6% → 3× isn't enough incentive; the 5× peak from run6a was the natural ceiling for this approach without combining with other shaping. Pivot to combination experiments (run6b: good_grasp 3→9 + gate lift on good_grasp_mask)."

**Three-data-point picture of the good_grasp_weight landscape:**

| Bump | Weight | Peak lift_success | Collapse severity | Stable end-state |
|---|---|---|---|---|
| 1× (baseline 4fe7cb7) | 3.0 | ~0.8% (run4d) | mild collapse to declining | touch-don't-lift |
| **3× (run6a.1)** | **9.0** | **0.4%** | rewards to +95 | touch-don't-lift, recovered |
| 5× (run6a) | 15.0 | **1.6%** | rewards to -722 | partial recovery |

**Non-monotonic in peak, monotonic in collapse severity.** Higher good_grasp incentive → higher peak lift signal AND deeper collapse. There's no monotonic sweet spot on this single knob.

**Implication:** good_grasp_weight alone isn't a sustainable lever. The lift basin is pulled back regardless of shaping magnitude. **Next move: structural pressure via gating** — make `lift_reward` conditional on `good_grasp_mask`, so the policy literally cannot earn the lift_reward camp residual without forming a proper thumb+finger grasp first. This is the run3-era fix described in CLAUDE.md.

### run6b — lift_sharpness 4 → 2 + good_grasp 3 → 15 (run6a's bump value) (2026-05-20)

**Motivation:** Combine the two reward-shape changes that produced the most signal so far:
- **lift_sharpness 4 → 2** — flatter, more sustained lift gradient. CLAUDE.md flags 2.0 as the run2a baseline value that drifted to 4.0 during run-2a-g. At sharpness=4 the lift_reward saturates fast (gradient collapses by h≈25cm); at sharpness=2 ~80% of d(reward)/dh remains at h=15cm.
- **good_grasp_weight 3 → 15** — match run6a's 5× bump, the value that produced the **strongest v2 lift peak (1.6% @ ep 517)** before catastrophically collapsing at ep 700.

**Hypothesis:** run6a's collapse was driven by the steep lift_sharpness=4 landscape — once the policy briefly engaged and lifted, the lift gradient flattened sharply past h=25cm, leaving the policy without continuous pull. The high good_grasp expectation (15) combined with the diminishing lift gradient created an unstable "high promised reward, vanishing gradient" zone that the policy retreated from. By flattening the lift gradient (sharpness=2), the upward pull is more sustained, so the good_grasp=15 incentive can be paired with a lift landscape that doesn't punish the policy for getting partway up.

In short: take the run6a peak that worked, but fix the gradient collapse that destabilized it.

**Configuration delta from previous (5ece6da / run6a.1):**
- `lift_sharpness`: 4.0 → **2.0** ([env_cfg.py:756](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py))
- `good_grasp_weight`: 9.0 → **15.0** (bumped back to run6a's value)
- Net vs 990c395 baseline: two lines, both reward-shaping. **2-knob test by user direction.**

Historically novel combination: 32a8924's lift_sharpness (2.0) on top of 4fe7cb7's other values (success_bonus=20, tightened ADR ranges, etc.) + run6a's good_grasp=15. Neither was tested in the 2a-g range.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` peak exceeds run6a's 1.6% AND sustains past ep 1000 without collapsing to negative rewards → the combination worked. Both shaping changes are validated; let it run long enough for ADR to advance past 0 for the first time in any v2 retrain.
- Lift peak ~run6a's 1.6% but collapse delayed or avoided → flatter gradient helped sustain but didn't push higher. Still a major win; iterate down on good_grasp (run6b.1 with 12) to find the smallest sufficient bump.
- Lift peak < run6a's 1.6% and same collapse pattern → flatter gradient didn't help the collapse mechanism; the policy is still being yanked out of engagement by something other than the lift_reward saturation. Pivot to structural gating (run6c: gate lift on good_grasp_mask, lift_sharpness=4 restored).
- No lift signal at all → 2-knob combination broke something (e.g., flatter lift gradient + high good_grasp pulled the policy into a different bad equilibrium). Revert and isolate: run6b.1 with sharpness=2 + good_grasp=3 to test sharpness alone.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_19-14-18/` (test repo, GPU 0, dextrah_test env)

**Result (ep 3225, stopped by user, 2026-05-20): Major partial breakthrough — no collapse, sustained engagement, partial-lift basin replaces touch-don't-lift. But lift_success still capped at 1.0%.**

TensorBoard at iter sample points:

| Signal | ep 200 | ep 500 | ep 1000 | ep 1500 | ep 2000 | ep 2500 | ep 3000 | PEAK |
|---|---|---|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.010 @ ep 486** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 @ ep 467 |
| rewards (raw) | 4450 | 10625 | 14728 | 13241 | 13799 | 13886 | **15676** | **17184 @ ep 2720** |
| lift_reward | 9.08 | 10.27 | 12.45 | 11.82 | 12.34 | 11.96 | 12.91 | 13.75 @ ep 3225 |
| **good_grasp_reward** | 2.83 | 3.85 | 8.47 | 5.39 | 7.23 | 7.19 | 8.39 | **10.25 @ ep 3216** |
| hand_object_contact_reward | 3.61 | 4.23 | 4.73 | 3.97 | 4.56 | 4.20 | 4.58 | 5.35 |
| object_contact_count | 1.20 | 1.41 | 1.58 | 1.32 | 1.52 | 1.40 | 1.53 | 1.78 |
| hand_to_object_distance (m) | 0.135 | 0.154 | 0.142 | 0.136 | 0.142 | 0.160 | 0.145 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| finger_curl_reg | -1.16 | -1.35 | -2.51 | -2.60 | -2.60 | -2.24 | -2.37 | — |

**Observations:**

1. **No collapse — monotone reward climb.** Rewards grew 4450 → 17184 over 3225 epochs with no negative-reward window. This is the first v2 retrain that didn't crash at some point.
2. **`good_grasp_reward` peaked at 10.25 — highest of any v2 run** (vs run6a's 6.06, run6a.1's 5.97). With weight=15, that's a ~68% grasp rate (vs run6a's 40%). The policy is forming proper grasps consistently.
3. **`lift_reward` climbed to 13.75 — implies object lifted ~21cm off the table on average.** The policy IS lifting, just to a moderate height. With sharpness=2: `13.75 / 40 = 0.34 = 1 - exp(-2*h)` → h ≈ 0.21m.
4. **`lift_success` peak 0.010 (1.0%) at ep 486 — second-highest v2 peak ever** (below run6a's 1.6%), but the trajectory is different: after the peak it dropped to 0 while everything else continued climbing. The peak was *transient exploration*, not the policy's learned behavior.
5. **Hand stayed near object throughout** (0.13-0.16m). No avoidance phases. Sustained engagement.
6. **ADR stuck at 0** through 3225 epochs.
7. **User livestream qualitative observation (best ckpt `ep_3000_rew_15616`):** policy uses **thumb + ring finger** as the primary grasp (ring is geometrically opposite the thumb in this configuration). **Middle and index fingers are fully curled at the MCP and not contributing to the grasp** — they may be incidentally brushing the object for `contact_mask=True`, but they're not forming the grip. The policy found the simplest valid grasp under the current good_grasp_mask definition (`thumb + ≥1 other finger`) and ignores the unnecessary fingers.

**Mechanism — why partial lift but no full success:**

Two factors compound:
1. **`lift_reward` is gated on `contact_mask` (any contact)**, not `good_grasp_mask`. So at altitude ≈ 21cm, the policy earns lift_reward 13.75/step just for keeping any sensor touching. Higher altitude doesn't pay much more (gradient at h=21cm with sharpness=2 is `40*2*exp(-0.42) ≈ 52` per meter — still meaningful but the policy has already exited the steep part of the curve).
2. **lift_reward 13.75 > good_grasp_reward 8.39 at the "moderate lift" equilibrium.** Lifting to camp altitude is more rewarding than maintaining a tighter grasp. The policy settled here because it's the local optimum.

The flatter sharpness=2 gradient + grasp shaping cleared the camp-at-table basin (huge progress vs runs 4c-6a.1), but introduced a NEW basin: **camp-at-moderate-height**. Object never reaches the goal region, in_success_region stays at 0, ADR can't advance.

**Decision rule outcome — first branch fires partially:** the combination prevented the collapse and produced sustained partial-lift behavior. But lift_success didn't exceed run6a's 1.6%, so it's not a full win yet. The remaining issue is the lift_reward camp residual at moderate altitude — addressed in run6c.

**Path forward (run6c):** gate `lift_reward` on `good_grasp_mask` so the policy can only earn lift_reward when a proper thumb+finger grasp is formed. This breaks the "any contact + slight lift = camp residual" exploit at moderate height. The run3-era structural fix per CLAUDE.md, but applied here on top of the run6b reward shape that prevents the collapse pattern.

### run6c — gate lift_reward on good_grasp_mask (env.py edit) (2026-05-20)

**Motivation:** Run6b's failure mode was the partial-lift camp at ~21cm altitude. The policy gets `lift_reward = 13.75/step` from the sharpness=2 gradient just for keeping any sensor touching at moderate altitude, with no incentive to grasp tighter or lift higher. Confirmed by livestream: thumb+ring grip with middle/index curled and not contributing (just brushing the object for `contact_mask=True`).

Apply the run3-era structural fix per CLAUDE.md: **change `lift_reward`'s gate from `contact_mask` (any sensor) to `good_grasp_mask` (thumb + ≥1 finger)**. With this gate, lift_reward fires only when a proper grip is formed. The middle/index brushing the object no longer earns the lift residual — the policy must form a real grasp to get any lift gradient signal at all.

`good_grasp_mask` definition is intentionally left unchanged (`thumb + ≥1 unique finger`). Per user direction: "the policy should find its own optimum" — we're not forcing tripod (thumb+index+middle) specifically, just requiring SOME proper grip rather than incidental contact.

**Hypothesis:** with the gate, the moderate-altitude camp residual disappears (because the policy must actively maintain a grip to keep lift_reward flowing). Combined with run6b's existing partial-lift behavior, this should push the policy from "lift to 21cm with weak contact" → "lift to higher altitudes while maintaining the grip" → eventually reach the goal region.

**Risk:** if the policy currently spends most of its time at "moderate lift with weak contact," gating lift_reward might zero out a large fraction of the reward gradient, potentially destabilizing the policy back into avoidance.

**Configuration delta from previous (12400f2 / run6b):**
- `env.py` line 2225: `lift_reward = ... * contact_mask` → `... * good_grasp_mask.to(contact_count.dtype)` ([env.py:2226](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py))
- `env_cfg.py`: **unchanged from run6b** (good_grasp_weight=15, lift_sharpness=2 preserved).
- Net vs 990c395 baseline: env_cfg.py 2-line diff (run6b values) + env.py 1-line gate change.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, **num_envs 16 livestream mode** (qualitative observation — see if the gate changes grasping/lifting behavior visually). Note per CLAUDE.md: 16 envs × 13 objects ≈ 1.2 envs/object is well below the ≥64-envs-per-object stable-training threshold, so TB data will be noisy and the policy won't converge. Use this run for qualitative livestream debugging only; if behavior looks promising, switch to the 1024-env headless command for actual training.

**Train command (16-env livestream — current launch):**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=100000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Headless 1024-env command (for follow-up if livestream looks promising):**

```bash
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 0.01 sustained AND `in_success_region` starts firing (>0) AND ADR finally moves off 0 → structural fix worked. Camp-at-moderate-height basin broken. Let it cook to confirm goal-region behavior consolidates.
- `lift_reward` stays around 10-14/step but only when good_grasp is also high (>5/step) → policy correctly couples grasp + lift. Sign of healthy progression even if lift_success peak is still moderate (~1-2%).
- Lift_reward drops dramatically (e.g. <5/step average) AND policy retreats from object → gate removed too much gradient signal; policy lost the partial-lift behavior. Then revert env.py and try a softer alternative: keep contact_mask gate but ADD a good_grasp multiplier (e.g., `lift_reward *= (0.5 + 0.5 * good_grasp_mask)` for half-reward without grasp, full with).
- Same partial-lift pattern as run6b (lift_reward ~13 at ~21cm, lift_success 0) → the camp basin isn't gate-related; the lift gradient itself flattens too much past 25cm. Pivot to lift_sharpness=3 (between run6b's 2 and 4fe7cb7's 4).

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_21-30-39/` (test repo, GPU 0, dextrah_test env, **16-env livestream**)

**Result (ep 2404, stopped by user at ep 2300, 2026-05-20): INCONCLUSIVE — 16-env training under-resourced; no contact-and-hold behavior emerged.**

TensorBoard at iter sample points (note: 16 envs / 13 objects = ~1.2 envs/object, well below CLAUDE.md's ≥64-envs-per-object stability threshold — data is dominated by gradient noise):

| Signal | ep 200 | ep 500 | ep 1000 | ep 1500 | ep 2000 | ep 2300 | PEAK |
|---|---|---|---|---|---|---|---|
| lift_success | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000 (never lifted)** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewards (raw) | 125 | 242 | 358 | 272 | 51 | 164 | 431 @ ep 693 |
| lift_reward | 0.17 | 0.34 | 0.25 | 0.05 | 0.02 | 0.15 | 3.08 @ ep 1182 |
| good_grasp_reward | 0.38 | 0.49 | 0.95 | 0.11 | 0.09 | 0.52 | 4.69 @ ep 1007 |
| hand_object_contact_reward | 1.00 | 1.03 | 1.24 | 0.25 | 0.42 | 0.98 | 3.38 @ ep 222 |
| object_contact_count | 0.33 | 0.34 | 0.41 | 0.08 | 0.14 | 0.33 | 1.13 |
| hand_to_object_distance (m) | 0.214 | 0.214 | 0.176 | 0.210 | 0.254 | 0.237 | — |
| episode_lengths | 103 | 75 | 84 | 125 | 35 | 44 | 284 |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Observations:**

1. **Magnitudes 5-10× lower than run6b (1024 envs).** `lift_reward` peak 3.08 (run6b had 13.75), `good_grasp_reward` peak 4.69 (run6b had 10.25), `rewards` 125-430 (run6b 4450-17184). The policy never developed the partial-lift behavior run6b had.
2. **Approach incomplete.** `hand_to_object_distance` hovers at 0.18-0.25m vs run6b's 0.14m — the hand doesn't reliably reach the object. With 1.2 envs/object, each object only gets 1-2 envs per batch and the policy gradient is too noisy to consolidate even basic approach.
3. **Episode lengths short (35-280 steps vs 470-530 in run6b)** — episodes terminating early via early_termination penalty paths (hand_too_far, palm_flip, etc.). The policy can't even sustain engagement long enough to test the gate hypothesis.
4. **Gate hypothesis NOT tested.** Run6c's data is dominated by the env-count problem, not by the lift_reward gate change. With 16 envs the policy never reaches the state where the gate matters (i.e., proper good_grasp formation that would or wouldn't be paired with lift). Need 1024 envs to actually test the gate.

**Decision rule outcome — none of the branches cleanly fire** because the experiment didn't reach the failure modes the rules describe. The run was under-resourced for multi-object training. Useful primarily as a livestream sanity check: confirmed the gate change doesn't break anything immediately (no NaN, no immediate divergence), but didn't produce learnable signal.

**Next:** run6c.1 — re-launch the EXACT same code (env.py + env_cfg.py unchanged) with **1024 envs headless**, the configuration where run6a/6b actually produced meaningful learning. This will be the real test of whether the lift_reward gate breaks the partial-lift basin from run6b.

### run6c.1 — same gate config, scaled to 1024 envs headless (2026-05-20)

**Motivation:** Run6c at 16 envs was inconclusive (under-resourced for 13-object training per CLAUDE.md's ≥64-envs-per-object threshold). The gate-on-good_grasp_mask hypothesis from run6c hasn't been tested yet at a viable env count. Re-launch the same env.py + env_cfg.py state with 1024 envs headless to actually test whether gating lift_reward on good_grasp_mask breaks the partial-lift basin from run6b.

**Configuration delta from previous (976e982 / run6c):**
- **No code changes.** env.py and env_cfg.py are bit-identical to run6c's committed state.
- Launch only: `--num_envs 16 --livestream 2` → `--num_envs 1024 --headless`, and minibatch_size 256 → 4096 to match the larger env count.
- Net vs 990c395 baseline (run2g): same as run6c — 2 lines env_cfg (run6b values: good_grasp=15, lift_sharpness=2) + 1 line env.py gate change.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024 headless, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule (the real run6c hypothesis test):**
- `lift_success` climbs past 0.01 sustained AND `in_success_region` starts firing AND ADR moves off 0 → **gate fix broke the partial-lift basin**. The structural fix per CLAUDE.md works on top of the run6b reward shape. Let it cook to confirm goal-region consolidation.
- `lift_reward` and `good_grasp_reward` climb together (both >5/step) but `lift_success` still stuck at 0 → policy correctly couples grasp + lift but can't physically reach goal altitude. Means actuator constraints (thumb_rot velocity at 15 deg/s start, arm 5-7 effort 20 Nm start) limit lift height. Pivot to actuator bisection on top of this gate.
- `lift_reward` stays near zero while `good_grasp_reward` climbs → policy grasps but doesn't try to lift (because lift_reward is now rare/conditional, gradient signal weak). Need to add an unconditional "any lift effort" reward term, or relax the gate (e.g., `lift_reward *= 0.3 + 0.7 * good_grasp_mask` for partial credit without grasp).
- Both `lift_reward` and `good_grasp_reward` drop AND policy retreats from object → gate removed too much gradient. Either of the run3-era fallbacks: revert env.py, or add a positive baseline reward to keep the policy engaged.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_22-12-07/` (test repo, GPU 0, dextrah_test env, 1024 envs headless)

**Result (ep 799, stopped by user at ~750, 2026-05-20): Gate works mechanically but policy farms a NEW exploit — buckled-thumb pose satisfies good_grasp_mask without enabling lift.**

TensorBoard at iter sample points:

| Signal | ep 100 | ep 250 | ep 500 | ep 700 | PEAK |
|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.000 | 0.000 | 0.000 | **0.003 @ ep 320** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | ~0 |
| rewards (raw) | -72 | 4220 | 14195 | 14488 | ~16k+ |
| **lift_reward** (gated) | 0.21 | 5.63 | 8.92 | 9.13 | ~9-10 |
| **good_grasp_reward** | 0.21 | 6.56 | 8.39 | 8.55 | ~10 |
| hand_object_contact_reward | 0.47 | 6.07 | 6.98 | 7.03 | ~8 |
| object_contact_count | 0.16 | 2.02 | 2.33 | 2.34 | ~2.5 |
| hand_to_object_distance (m) | 0.487 | 0.110 | 0.110 | 0.109 | — |
| episode_lengths | 287 | 200 | 464 | 471 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 |

**Observations:**

1. **The gate works mechanically.** `lift_reward` and `good_grasp_reward` are now firing TOGETHER (9.13 and 8.55 at ep 700) — confirming the gate couples them. Contact_reward is climbing in parallel (7.03) but no longer driving lift_reward independently.
2. **Higher good_grasp than run6b** (8.55 vs run6b's similar value 8.39 at same ep but with no gate) — the policy is forming "good_grasp_mask=True" states slightly more often when forced to by the gate.
3. **BUT zero lift_success.** Peak 0.003 (0.3%) at ep 320, which is *lower* than run6b's 1.0% peak. The gate prevented the partial-lift altitude (~21cm) that run6b achieved without unlocking actual lifts.
4. **User livestream qualitative observation:** policy is **curling the thumb inward to satisfy `good_grasp_mask`** — thumb tip + ≥1 finger contact achieved by a buckled-thumb pose, NOT a real outside-wrap grip. The policy found a NEW exploit: the gate requires thumb contact, so the policy buckles the thumb into the palm-side of the hand to maintain contact without forming a graspable pose. Lift_reward fires (gate satisfied) but the geometry can't actually elevate the object.

**Mathematical analysis (per user):** in this run, `lift_reward ≈ 9` and `good_grasp_reward ≈ 8.5` — they're roughly equal. With the gate, **they're functionally indistinguishable signals from the policy's perspective**: both fire when `good_grasp_mask=True`. The marginal reward for actually LIFTING (vs just maintaining grasp at table) is small (Δlift ≈ 4-5/step across 13cm). The marginal reward for maintaining the grasp (good_grasp constant 15 when firing) is large. **The grip is 3.5× more rewarding per step than the lift gradient** — the math says buckling thumb to maintain "grip" is optimal.

**Decision rule outcome — third branch fires:**

> "`lift_reward` stays near zero while `good_grasp_reward` climbs → policy grasps but doesn't try to lift..."

Except modulated: lift_reward isn't near zero (9 is significant), but lift_success IS near zero. The policy is exploiting the lift_reward at low altitude through buckled-thumb satisfying the gate. The gate is necessary but not sufficient — needs to be paired with a reward shape where actually lifting pays clearly more than holding-at-table.

**Open hypotheses for run6d:** (1) bump lift_weight to make lift gradient dominate, (2) bump arm 5-7 effort to ensure torque isn't the cap, (3) reduce contact and good_grasp weights to remove competing signals, (4) restore sharpness=4 to concentrate lift gradient near the goal. User selected options (1)+(3)+(4) combined.

**Asset-level finding (separate from main investigation):** user provided AgileHand hardware spec: finger velocities = 360 deg/s = 6.283 rad/s. Current asset value of 8.0 rad/s is 27% over spec. CLAUDE.md's previous "Use 2.0 rad/s" recommendation (based on assumed 30-60 deg/s hardware) was wrong — corrected in commit `17b1677`. Asset velocity_limit_sim updated to 6.283 rad/s for the next experiment (run6d).

### run6d — multi-knob: lift dominance + asset velocity fix (2026-05-20)

**Motivation:** Run6c.1 confirmed the gate works mechanically but the policy farms a buckled-thumb exploit because `lift_reward ≈ good_grasp_reward ≈ 9/step` — the two signals are functionally identical when the gate fires, so there's no marginal incentive to actually lift vs maintain a static "grip". Per user analysis, the fix needs to (a) make lift_reward MUCH more dominant than the competing signals, (b) concentrate lift gradient near the goal (steeper sharpness), and (c) match asset finger velocities to hardware spec.

**5-knob change** (by user direction, multi-knob test):

1. **Asset finger velocity_limit_sim 8.0 → 6.283 rad/s** (360 deg/s, AgileHand hardware spec) — applies to `mcp_pitch`, `mcp_yaw`, `pip` joint groups. Sim2real accuracy fix + may reduce the buckled-thumb fling-back artifact slightly.
2. **`hand_object_contact_weight` 3.0 → 1.5** — halve contact's competing signal. Contact_reward at run6c.1 peaks was ~7/step; this cuts it to ~3.5/step.
3. **`good_grasp_weight` 15.0 → 6.0** — back to ~run2g original value. Cuts good_grasp peak from ~10/step to ~4/step.
4. **`lift_sharpness` 2.0 → 4.0** — restore 4fe7cb7 value. Steeper gradient near goal, less reward at table. At weight=60 + sharpness=4: camp residual at table ≈ 8/step, max at goal ≈ 60/step. Marginal lift gain across 50cm = ~52/step (vs run6c.1's ~4/step over 13cm).
5. **`lift_weight` ADR (40., 30.) → (60., 30.)** — 1.5× starting bump per user direction. End unchanged.

**Reward landscape comparison at ADR 0, "engaging" state (gate firing):**

| Component | run6c.1 (gate, sharp=2, w=40, grasp=15, contact=3) | run6d (gate, sharp=4, w=60, grasp=6, contact=1.5) |
|---|---|---|
| good_grasp_reward (when gate fires) | 15 | **6** (-60%) |
| contact_reward (5 sensors max) | 15 (5×3) | **7.5** (-50%) |
| lift_reward at table (h≈0) | 40·exp(-1) ≈ 14.7 | **60·exp(-2) ≈ 8.1** (-45%) |
| lift_reward at mid (h=0.25m) | 40·exp(-0.5) ≈ 24.3 | **60·exp(-1) ≈ 22.1** (similar) |
| lift_reward at goal (h=0.5m) | 40 | **60** (+50%) |
| Δlift (table → goal) | 25.3 | **51.9** (+105%) |

**Net effect:** lift gradient across the full trajectory more than doubles, while competing signals (good_grasp + contact) drop by ~50%. The policy can no longer farm buckled-thumb-grip for big reward — it has to actually lift to get the dominant signal.

**Net diff vs 990c395 baseline (run2g):**
- env.py: 1-line good_grasp_mask gate (from run6c)
- env_cfg.py: 4 lines (contact_weight, good_grasp_weight, lift_sharpness, lift_weight ADR)
- fr3_tekken_left.py: 3 lines (mcp_pitch, mcp_yaw, pip velocity_limit_sim)

**5-knob test by user direction.** Skill anti-pattern warning acknowledged — if this works, we won't know which knob is responsible. If it doesn't work, we'll need to bisect down.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024 headless, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 0.05 sustained by ep 1500 AND `in_success_region > 0` AND ADR finally advances → 5-knob combination broke the basin. Let it cook; subsequent runs can bisect which knobs matter.
- `lift_reward` dominates good_grasp by >2× consistently (per the math expected — lift should reach 20+ at engaged state while good_grasp caps at 6) AND lift_success climbs slowly → reward shape is correct, just needs more training time.
- Same buckled-thumb pattern (good_grasp=6 firing constantly, lift_reward stuck low) → policy STILL exploits the gate with crushed grip. Means even with reduced grasp incentive, the local optimum is buckled-thumb. Pivot to a structural fix on the mask itself (require specific fingers).
- Catastrophic collapse / negative rewards → 5-knob change destabilized something. Bisect: revert good_grasp+contact reductions first (run6d.1 with only lift bump + sharpness fix).
- Brief lift peak >1% then collapse → reward shape works for initial exploration but basin re-emerges. Means the gate has a fundamental issue beyond reward magnitude.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-20_23-20-46/` (test repo, GPU 0, dextrah_test env, 1024 envs headless)

**Result (ep 1671, stopped by user at ~1.6k, 2026-05-21): Dominance fix partially worked but basin still pulls back. Peak lift 1.3%, same collapse-recover pattern.**

TensorBoard at iter sample points:

| Signal | ep 100 | ep 250 | ep 500 | ep 750 (CRASH) | ep 1000 | ep 1250 | ep 1500 | PEAK |
|---|---|---|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | **0.013 @ ep 369** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 @ ep 426 |
| rewards (raw) | 4243 | 7168 | 8986 | **879** | 9616 | 10306 | 9813 | 11415 @ ep 1237 |
| **lift_reward** | 4.40 | 5.75 | 5.67 | 0.59 | 6.71 | 6.49 | 6.66 | 7.67 |
| **good_grasp_reward** | 2.95 | 3.65 | 3.50 | 0.37 | 4.17 | 4.02 | 4.10 | 4.78 |
| hand_object_contact_reward | 3.66 | 3.71 | 3.14 | 0.30 | 3.67 | 3.56 | 3.50 | 4.17 |
| object_to_goal_reward | 3.10 | 3.12 | 2.90 | 0.34 | 3.39 | 3.39 | 3.39 | 3.53 |
| object_contact_count | 2.44 | 2.47 | 2.09 | 0.20 | 2.45 | 2.37 | 2.34 | 2.78 |
| hand_to_object_distance (m) | 0.100 | 0.124 | 0.135 | **0.274** | 0.106 | 0.110 | 0.118 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 281 | 407 | 518 | 547 | 485 | 524 | 507 | 587 |

**Observations:**

1. **Weight cuts took effect as designed.** good_grasp_reward dropped to ~4/step (from run6c.1's ~9) ✓. contact_reward dropped to ~3.5/step (from ~7) ✓. The reward landscape was reshaped as intended.
2. **lift_reward / good_grasp ratio improved to 1.63×** (lift 6.7 / good_grasp 4.1). Better than run6c.1's ~1:1, but not the >2× dominance the math predicted (which assumed lifting to mid-trajectory). Suggests the policy is still spending most of its time at table altitude where lift_reward = ~8 ≈ good_grasp_reward × 1.3.
3. **Peak lift_success 0.013 (1.3%) at ep 369** — higher than run6c.1's 0.3% peak, comparable to run6b's 1.0%. The dominance fix produced *some* additional exploration but didn't break the basin.
4. **Catastrophic collapse at ep 750.** Rewards 8986 → 879 (-90%). All engagement metrics dropped to zero (contact 3.14 → 0.30, good_grasp 3.50 → 0.37). Hand walked away (0.135m → 0.274m). Same pattern as run6a's run5c-era collapse. Recovered by ep 1000 but lift_success never returned.
5. **ADR stuck at 0** through 1.6k epochs.

**Decision rule outcome — first branch did NOT fire (no >0.05 sustained), second branch did NOT fully fire (ratio not >2×), fifth branch fired:** "Brief lift peak >1% then collapse → reward shape works for initial exploration but basin re-emerges. Means the gate has a fundamental issue beyond reward magnitude."

**Summary of all v2 retrains' lift peaks:**

| Run | Config summary | Peak lift_success | Pattern |
|---|---|---|---|
| run4c | 32a8924 baseline | 0.000 | flat zero entire 4800 ep |
| run4d | 4fe7cb7 main+clean | 0.008 | peak then decline |
| run4e | 32a8924 main+clean | 0.005 | brief peak terminated |
| run5a | thumb 2× | 0.000 | flat |
| run5b | rm arm_joint_init | ~0 | terminated 580 ep |
| run5c | lift 10× | 0.002 | oscillation, no help |
| run6a | grasp 5× | **0.016** | catastrophic collapse |
| run6a.1 | grasp 3× | 0.004 | mild collapse |
| run6b | sharp 2 + grasp 5× | 0.010 | partial-lift basin, no collapse |
| run6c.1 | + lift gate | 0.003 | buckled-thumb exploit |
| **run6d** | **5-knob: lift-dominant + asset fix** | **0.013** | **brief peak + collapse + flat** |

**No v2 retrain has exceeded 1.6% lift_success peak. The basin appears robust to ANY combination of reward magnitudes + gating + sharpness + asset velocity tweaks tested so far.** Strong evidence that the issue lies at a level deeper than reward shape — possibly the gate definition (good_grasp_mask too lenient with the buckled-thumb pose), or v2 actuator constraints making physical lifting infeasible regardless of reward signal.

**Path forward suggestions:**
- **run6e (option 4 from earlier):** tighten `good_grasp_mask` itself to require specific fingers (thumb + index + middle) or `finger_count >= 3 & thumb_contact`. Prevents the buckled-thumb exploit by making the mask geometrically demanding.
- **run6f (option 1 from earlier):** verify torque isn't the cap — bump arm 5-7 effort start from 20 to 50 Nm (v1 value) while keeping run6d reward shape. If lift_success jumps, torque was the constraint.
- **Alternative:** explicitly visualize the failure — replay ep_500 (closest to lift peak window) AND ep_1500 (best reward) to see what's qualitatively different between the brief-lift state and the recovered-engagement state.

### run6e — symmetric thumb_rot init + tighter hardware-realistic thumb velocity (2026-05-21)

**Motivation:** User livestream observation during run6d revealed the policy gravitates to thumb_rot ≈ -30° (joint min, thumb pointing AWAY from other fingers). Investigation showed the reset state was asymmetric and biased toward the joint min:

- `init_state.joint_pos["revolute_thumb_rot"] = -0.3491` rad = **-20°** (only 10° clear of -30° hard stop)
- `thumb_rot_init` EventTerm `position_range = (0.0, 0.3491)` = offset 0° to +20° (asymmetric, no negative offset)
- Net reset range: **-20° to 0°** (thumb at reset is ALWAYS in the lower half of its joint range)

Combined with the fact that joint range is `[-30°, +20°]` (50° wide), the policy was being exposed only to thumb positions from "near joint min" to "mid-range" — zero exposure to the upper half where the thumb points toward the other fingers (needed for a real outside-wrap grasp).

Also: user provided updated hardware spec — real `revolute_thumb_rot` rotational speed is **2 deg/s** (much tighter than the 8 deg/s used in the previous curriculum endpoint).

**4-knob change:**

1. **`init_state.joint_pos["revolute_thumb_rot"]`**: -0.3491 → **0.0** (mid-range). Symmetric default, equal exposure to "thumb toward fingers" and "thumb away" at reset.
2. **`thumb_rot_init` EventTerm `position_range`**: (0.0, 0.3491) → **(0.0, 0.0)** (no randomization at ADR 0). Lets the policy learn a clean default first.
3. **New `adr_cfg_dict["thumb_rot_init"]["position_range"]` = (-0.1745, 0.1745)** — the curriculum target. ADR linearly widens reset randomization from (0, 0) to (-10°, +10°) over `num_increments`. Keeps the joint well clear of both joint limits (-30°, +20°) even at max ADR.
4. **`adr_custom_cfg_dict["actuator_curriculum"]["thumb_rot_vel_limit"]`**: (0.2618, 0.1396) → **(0.0873, 0.0349)** = 5 deg/s → 2 deg/s. Matches updated hardware spec; tighter than the previous 15→8 deg/s curriculum.

**Hypothesis:** the buckled-thumb / -30° default isn't an inherent policy preference — it's a reset-distribution artifact. With symmetric reset and a clear default at mid-range, the policy should explore the upper half of the thumb_rot range (toward fingers) where a real outside-wrap grasp is geometrically possible. Combined with the tighter hardware velocity (2 deg/s instead of 8), the policy must learn careful timing on thumb closure rather than fast flailing.

**Configuration delta from previous (7fbd1fd / run6d):**
- env_cfg.py 4 lines (init pos, EventTerm range, new adr_cfg_dict entry, vel curriculum)
- env.py: unchanged (gate from run6c persists)
- Asset: unchanged from run6d (finger velocity 6.283 rad/s persists)
- Reward weights: unchanged from run6d (contact=1.5, good_grasp=6, lift_sharpness=4, lift_weight=(60,30))

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024 headless, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 0.05 sustained AND livestream shows thumb wrapping AROUND object (not buckled inward) → centered init was the missing piece. The combined run6d shape + run6e geometry fix unsticks the basin. Let it cook to confirm ADR advances.
- `lift_success` matches run6d (~1% peak) but livestream shows new thumb geometry → reset bias contributed but isn't the dominant cause. Need to tighten `good_grasp_mask` (require specific fingers) to fully close the exploit.
- Policy explores upper thumb_rot range early but reverts to buckled at -30° later → the asymmetry was symptomatic, not causal. Pivot to good_grasp_mask tightening.
- thumb velocity at 2 deg/s causes finger oscillation / PD instability → drop velocity bump (revert to 5 → 4 deg/s) but keep init position fix.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-21_00-32-27/` (test repo, GPU 0, dextrah_test env, 1024 envs headless)

**Result (ep 753, stopped by user at ~750, 2026-05-21): FAILURE. Centered thumb init didn't break the basin. Lower lift signal than run6d, no qualitative breakthrough.**

TensorBoard at iter sample points:

| Signal | ep 100 | ep 250 | ep 500 | ep 750 | PEAK |
|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.000 | 0.000 | 0.000 | **0.003 @ ep 249** |
| in_success_region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewards (raw) | -382 | 3666 | 6968 | 6999 | 7801 @ ep 548 |
| lift_reward | 0.14 | 2.50 | 4.45 | 4.31 | 5.30 @ ep 531 |
| good_grasp_reward | 0.13 | 1.63 | 2.78 | 2.69 | 3.33 @ ep 531 |
| hand_object_contact_reward | 0.25 | 1.61 | 2.30 | 2.34 | 2.71 |
| object_contact_count | 0.17 | 1.07 | 1.54 | 1.56 | 1.81 |
| object_to_goal_reward | 0.40 | 1.88 | 2.82 | 2.64 | 3.06 |
| hand_to_object_distance (m) | 0.365 | 0.142 | 0.120 | 0.132 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 418 | 401 | 506 | 501 | 599 |
| finger_curl_reg | -0.83 | -1.33 | -1.41 | -1.33 | — |

**Observations:**

1. **Centered thumb init didn't unlock lifting.** Peak `lift_success = 0.003 @ ep 249` — lower than run6d's 1.3% peak. The geometric fix (thumb starting at 0° instead of -20°) helped slightly with approach (hand_to_object reaches 0.12m by ep 500) but didn't translate into actual lifts.
2. **Lower reward magnitudes overall** (good_grasp 2.78 vs run6d's 4.10, lift_reward 4.45 vs 6.66, contact 2.30 vs 3.50). Likely because the thumb at 5→2 deg/s velocity is *slow* — the policy has less margin to form grasps within episode time. Same shape of reward landscape but lower amplitudes.
3. **No collapse** (unlike run6d's ep 750 crash) — reward monotonically climbed 3666 → 6968 → 6999. The slower thumb may be preventing the instability/oscillation that caused run6d's avoidance regression.
4. **ADR stuck at 0** — same as every v2 run.

**Decision rule outcome — second/third branch fires:** "Policy explores upper thumb_rot range early but reverts to buckled at -30° later → the asymmetry was symptomatic, not causal" — except in this run, lift didn't even reach run6d's peak. Suggests centered init alone isn't enough.

**Most likely remaining bottleneck — user's wrist torque hypothesis:**

User raised the FR3 joints 5-7 effort limit (20 Nm at ADR 0) as a candidate constraint. Mechanism: when fingers (especially thumb_rot with 10 Nm capacity) generate gripping torques, the reaction torques on the hand base must be reacted by the wrist. v1 has 50 Nm wrist effort (2.5× headroom); v2 has 20 Nm. If finger reaction loads consume 5-10 Nm of the 20 Nm wrist budget, very little is left for actual lifting + orientation control. Next experiment (run6f) tests this directly.

### run6f — bump arm_57_effort_limit 20 → 50 Nm (match v1) (2026-05-21)

**Motivation:** v1 (Teacher 11) trained successfully to ADR 13 with `arm_57_effort_limit = 50 Nm` at ADR 0. v2 baseline has `arm_57_effort_limit = 20 Nm` — a 2.5× tighter wrist budget. Mechanistic hypothesis:

```
τ_wrist_total = τ_lift (object + hand weight at moment arm)
              + τ_orient (palm orientation control)
              + Σ τ_finger_reactions (Newton-3rd-law reactions to finger torques)
```

Finger reaction loads on the hand base can be substantial: thumb_rot motor capacity is 10 Nm; combined with 4 finger MCP+PIP joints at 2 Nm each, the cumulative reaction torque on the hand base could be 10-20 Nm. v2's 20 Nm wrist budget gets eaten by finger reactions, leaving little for lift+orient. v1's 50 Nm has 30+ Nm headroom.

**Single-knob change** on top of run6e:
- `arm_57_effort_limit` ADR: (20.0, 12.0) → **(50.0, 12.0)**
- Start matches v1's Teacher 11 value (Apr 2026 successful run).
- End unchanged at 12 Nm (hardware spec).

**Configuration delta from previous (378d69c / run6e):**
- env_cfg.py 1 line (arm_57_effort_limit start 20 → 50)
- Everything else identical to run6e (thumb centered init, ±10° curriculum, 5→2 deg/s velocity, lift_weight=(60,30), sharpness=4, good_grasp=6, contact=1.5, asset finger velocity 6.283 rad/s, env.py lift_reward gate on good_grasp_mask)

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024 headless, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 0.05 sustained AND ADR moves off 0 → wrist torque budget was the structural bottleneck. v2 trainable at v1's wrist headroom. Then iterate down: try (35, 12) to find smallest start value that still works.
- `lift_reward` magnitudes climb (15+/step during engaged states) AND livestream shows real wrist lift attempts (object actually leaving table > 5cm) → headroom unlocked physical lifting even if curriculum hasn't advanced yet. Strong positive signal.
- Same flat-zero lift pattern as run6e → wrist torque NOT the bottleneck; lift constraint is elsewhere. Strong evidence for tighter `good_grasp_mask` (require thumb+index+middle specifically) as next move.
- Lift signal emerges but degrades as ADR drops effort 50 → 12 across curriculum → wrist torque matters but the v1-realistic endpoint (12 Nm) is too tight. Then question whether 12 Nm endpoint is actually correct for sim2real, or if it needs to stay higher.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-21_00-59-49/` (test repo, GPU 0, dextrah_test env, 1024 envs headless, overnight)

**Result (ep 18857, 2026-05-21): FAILURE on lift_success, but reveals a striking 2-phase trajectory. Wrist torque helped engagement emerge eventually but did NOT unlock lifting.**

TensorBoard at iter sample points:

| Signal | ep 500 | ep 5000 | ep 8000 | **ep 10000** | ep 12000 | ep 14000 | ep 18000 | PEAK |
|---|---|---|---|---|---|---|---|---|
| **lift_success** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | **0.0029 @ ep 10436** |
| in_success_region | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0010 |
| rewards (raw) | 1022 | 1346 | **705** | **3772** | 9493 | 10171 | 10066 | 11750 @ ep 12470 |
| lift_reward | 0.0001 | 0.0000 | 0.0000 | **4.22** | 6.90 | 6.87 | 6.80 | 7.51 @ ep 16965 |
| good_grasp_reward | 0.0001 | 0.0000 | 0.0000 | **2.61** | 4.24 | 4.22 | 4.17 | 4.61 @ ep 16965 |
| hand_object_contact_reward | 0.001 | 0.000 | 0.000 | **2.53** | 3.71 | 3.84 | 3.37 | 4.50 |
| object_contact_count | 0.001 | 0.000 | 0.000 | **1.69** | 2.47 | 2.56 | 2.25 | 3.00 |
| object_to_goal_reward | 0.003 | 0.001 | 0.000 | 2.63 | 3.24 | 3.29 | 3.29 | 3.43 |
| hand_to_object_distance (m) | 0.223 | 0.181 | 0.188 | **0.133** | 0.102 | 0.117 | 0.125 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 537 | 573 | 584 | **278** | 469 | 516 | 521 | 599 |
| finger_curl_reg | -0.54 | -0.68 | -1.48 | -1.27 | -1.20 | -1.63 | -1.65 | — |

**Observations:**

1. **Two-phase trajectory.** Phase 1 (ep 0-8000): policy did almost nothing — all contact/grasp/lift metrics ≈ 0, hand wandered at 0.18-0.22m from object. Phase 2 (ep 10000+): sudden breakthrough into engagement — contact_count 0 → 1.7, good_grasp 0 → 2.6, lift_reward 0 → 4.2. The transition was rapid (~1500 epochs).
2. **Episode lengths dropped at the engagement transition** (584 → 278 around ep 10000) then recovered (520 by ep 14000). Suggests early engagement attempts were causing terminations (out_of_reach or palm_flip from initial unstable grips), then the policy stabilized.
3. **Steady-state at lift_reward ≈ 7, good_grasp ≈ 4** is similar to run6d's mid-training values. Same partial-grasp basin, just reached on a 10000-epoch delay.
4. **Peak `lift_success` = 0.0029 (0.29%) at ep 10436** — right at the engagement breakthrough. *Lower* than run6d (1.3%), run6b (1.0%), and even run6c.1 (0.3%). This is among the worst lift peaks of any v2 retrain. After ep 10500, lift_success drops to ~0.
5. **ADR stuck at 0** through 18,857 epochs (longest training of any v2 retrain to date).

**Decision rule outcome — wrist torque NOT the lift bottleneck:**

The 20 → 50 Nm wrist effort bump (matching v1) demonstrably affected exploration dynamics — the policy eventually found engagement, which it didn't in earlier runs at similar epoch counts. But it did NOT translate into more lifting. The basin trap is robust to wrist torque headroom.

Cumulative evidence after 12 v2 retrains (run4c through run6f):

| Hypothesis tested | Run | Outcome |
|---|---|---|
| Config drift (32a8924 vs 4fe7cb7) | run4c, run4d, run4e | ruled out |
| Repo/env swap | run4d, run4e | ruled out |
| Thumb_rot velocity bump (2×) | run5a | ruled out |
| Remove arm_joint_init EventTerm | run5b | ruled out |
| 10× lift_weight | run5c | ruled out (destabilized) |
| 5× good_grasp_weight | run6a | ruled out (collapse) |
| 3× good_grasp_weight | run6a.1 | ruled out (insufficient) |
| Flatter lift gradient (sharp 2) + grasp 5× | run6b | partial (no collapse, basin held) |
| lift_reward gated on good_grasp_mask | run6c.1 | gate works mechanically, buckled-thumb exploit |
| Lift dominance (multi-knob shaping) | run6d | partial (1.3% peak) |
| Centered thumb init + 5→2 deg/s velocity | run6e | no improvement |
| **Wrist torque to v1 (20→50 Nm)** | **run6f** | **ruled out — partial engagement, no lift** |

**The basin is robust to every reward-shape, actuator, and randomization change tested.**

Strong push to investigate the structural good_grasp_mask itself — the buckled-thumb exploit is the consistent qualitative failure mode. Tightening the mask definition (require specific fingers thumb + index + middle, or `≥3 fingers + thumb`) would prevent the policy from satisfying the gate via crushed thumb geometry. This is the next planned experiment (run6g).

### run6g — revert lift_reward gate (good_grasp_mask → contact_mask) (2026-05-21)

**Motivation:** User livestream observation on run6f's best policy revealed the failure mode is NOT buckled-thumb (as we hypothesized in run6c-6e). The policy actually forms **proper grasps with thumb opposite the fingers**. The issue is **lack of lift incentive** — the policy parks at "good grasp held, no lift" because the reward landscape makes lifting feel risky.

**v1 vs v2 reward comparison** revealed 7 differences (6 weights + 1 structural gate). Termination landscape is **identical**. The most consequential difference is the **`lift_reward` gate**:

| Field | v1 (works) | v2 (run6f, 0.3% peak) |
|---|---|---|
| `lift_reward` gate | `contact_mask` (any sensor) | `good_grasp_mask` (thumb + ≥1 finger) |

**Math of why the gate punishes lifting:**

At v2's "good grasp held at table" steady state:
- `good_grasp_reward` (weight 6) = 6/step
- `lift_reward` (weight 60, sharp 4, gated, at h=0): 60·exp(-2)·1 ≈ 8/step
- `contact_reward` (weight 1.5) = 1.5/step
- **Total: ~15.5/step**

If the policy tries to lift and the grip slips for one timestep (good_grasp_mask = False):
- `good_grasp_reward` = 0 (−6 vs status quo)
- `lift_reward` = 0 (gated, −8 vs status quo)
- `contact_reward` = 1.5 (still firing on residual single-finger touch)
- **Total: ~1.5/step → 14/step LOSS per slip**

The marginal gain from a successful incremental lift is small (~ a few /step) compared to the **risk of losing 14/step on grip slips during the motion**. The policy correctly identifies "stay in good_grasp state" as the local maximum.

**v1's no-gate landscape avoids this**: lift_reward fires on ANY contact, so partial-contact lifts still earn lift_reward — even when good_grasp momentarily slips during lift transitions.

**Single-knob test:** revert ONLY the lift_reward gate (env.py line 2230). Keep all other v2 changes (reward weights, asset velocity, wrist torque, thumb_rot init, etc.). If this alone unlocks lifting, the gate was the structural blocker. If not, the additional 6 weight differences also matter.

**Configuration delta from previous (ac285b2 / run6f):**
- env.py line 2230: `lift_reward = ... * good_grasp_mask.to(contact_count.dtype)` → `... * contact_mask` ([env.py:2230](../tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py))
- env_cfg.py: unchanged (good_grasp_weight=6, lift_sharpness=4, lift_weight=(60,30), arm_57_effort=(50,12), thumb centered init, etc. all persist)

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, num_envs 1024 headless, visdex_selected.

**Train command:**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
  --headless --task=dextrah_fr3_agilehand --seed 42 \
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

**Decision rule:**
- `lift_success` climbs past 0.05 sustained AND `in_success_region > 0` AND ADR finally advances → **gate was the structural blocker.** Combined with v2's hardware-realistic actuators + run6d/6e reward magnitudes, removing the gate produces a working v2 teacher. Direct path to distillation.
- `lift_success` climbs past 1-2% but doesn't reach 5% sustained → gate is partial cause; remaining lift bottleneck is in the weight imbalances. Next: stage run6h with v1's exact reward weights (`lift_sharpness=2`, `contact_weight=3`, `good_grasp_weight=3`, `lift_weight=(40,20)`, `success_bonus=10`, `finger_curl_reg=(-0.5,-1.2)`).
- Lift peak ≤ run6d's 1.3% → gate removal alone doesn't fix it, weights are the dominant factor. Go to full v1 reward landscape adoption.
- Policy regresses to buckled-thumb behavior (lift_reward still high without proper grasp) → confirms our earlier run6c era concern about no-gate, but with the current actuator setup (slower thumb), this should be less of a risk.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-21_10-21-17/` (test repo, GPU 0, dextrah_test env, 1024 envs headless)

**Result (ep 3744, stopped by user, 2026-05-21): HIGHEST SUSTAINED v2 LIFT SIGNAL OF THE SAGA — 2.7% peak @ ep 2726, first in_success_region > 0.**

TensorBoard at iter sample points:

| Signal | ep 200 | ep 1000 | ep 2000 | ep 2500 | ep 3000 | ep 3500 | PEAK |
|---|---|---|---|---|---|---|---|
| **lift_success** | 0.000 | 0.000 | 0.000 | 0.002 | 0.013 | 0.012 | **0.027 @ ep 2726** |
| **in_success_region** | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.005 | **0.017 @ ep 2964** |
| rewards (raw) | 25 | 1242 | 1495 | 4873 | 8176 | 8871 | 10233 @ ep 3634 |
| lift_reward | 0.03 | 0.00 | 0.46 | 5.16 | 7.70 | 7.95 | 8.59 |
| hand_object_contact_reward | 0.01 | 0.00 | 0.08 | 1.34 | 2.20 | 2.31 | 2.61 |
| good_grasp_reward | 0.00 | 0.00 | 0.01 | 0.15 | 0.72 | **2.39** | 3.38 @ ep 3716 |
| object_contact_count | 0.01 | 0.00 | 0.06 | 0.89 | 1.47 | 1.54 | 1.74 |
| hand_to_object_distance (m) | 0.31 | 0.18 | 0.17 | 0.13 | 0.12 | 0.12 | — |
| finger_curl_reg | -0.12 | -0.18 | -0.38 | -0.68 | -0.85 | -1.37 | — |
| num_adr_increases | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| episode_lengths | 30 | 490 | 463 | 449 | 498 | 498 | — |

**Observations:**
1. **Peak lift_success = 2.7% (0.027) @ ep 2726 — the highest sustained v2 lift signal of the entire saga.** Beats run6a's 1.6% (collapsed), run6d's 1.3%, run6b's 1.0%. Most importantly, no catastrophic collapse afterward — settled at 1.2% by ep 3500, not zero.
2. **First `in_success_region > 0` of the entire v2 saga** — peak 0.017 @ ep 2964. The policy briefly reached the goal region, which is the precondition for ADR finally advancing past 0. Still below the 0.4 threshold (`success_for_adr`) so no ADR increment yet, but qualitatively new behavior.
3. **Late-emergent learning pattern:** essentially nothing through ep 2000 (matches user's "no change" impression — visually flat in livestream), then explosive engagement ep 2000-2700: contact climbed from 0.055 → 1.47, good_grasp from 0.01 → 0.72, lift_reward 0.46 → 7.70. The gate removal needed ~2000 epochs of exploration before producing usable signal.
4. **`finger_curl_reg = -1.37` at ep 3500** — fingers are curling harder, indicating active grasping attempts. The visual "nothing change" likely means lifts are small-amplitude in a small fraction of envs.
5. **Slight decline from peak ep 2726 to ep 3500** (lift_success 0.027 → 0.012) — not collapse but plateau / slow regression. Could indicate the ungated lift_reward starting to attract a "small camp + tiny lift" exploit, similar to run6c.1 concern but at low amplitude.

**Decision rule outcome — second branch fires:**

> "`lift_success` climbs past 1-2% but doesn't reach 5% sustained → gate is partial cause; remaining lift bottleneck is in the weight imbalances. Next: stage run6h with v1's exact reward weights."

Gate removal alone delivered the strongest v2 signal yet but plateaued at 2.7%. Next: adopt v1's exact reward landscape on top of the gate revert to push past the 5% threshold and trigger ADR.




