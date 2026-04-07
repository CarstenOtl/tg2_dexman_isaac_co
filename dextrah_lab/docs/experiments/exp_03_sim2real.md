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
| ep 2500 (best reward, pre-ADR-13) | **79.1%** | 25.3% | OOB 42%, phys_inst 37%, harmful_collision 14%, palm_flip 7% |
| ep 6500 (settled at ADR 13) | 75.9% | 23.1% | phys_inst 63%, OOB 29%, harmful_collision 8% |

- Eval JSONs: `logs/eval_tb_20260407_101145/eval_metrics_20260407_103836.json`, `logs/eval_tb_20260407_102855/eval_metrics_20260407_104305.json`
- ep 2500 is the best Teacher v2 checkpoint across all runs (highest lift)

**Critical insight:** Eval at ADR 0 (no randomization) shows ep 6500 (later, more training, settled at ADR 13) has *worse* lifting than ep 2500 (earlier). A policy trained at ADR 13 should do **better** at ADR 0 (easier conditions), not worse. This proves the training process at ADR 13 is **actively degrading** lifting ability — the policy is forgetting how to lift while trying to handle harder conditions. The within-ADR reward decay (lift_weight 40→30) is washing out learned behavior.

**Final Teacher v2 results vs Teacher 11 baseline:**

| Run | Checkpoint | Lift | Unsafe | Notes |
|---|---|---|---|---|
| Teacher 11 (baseline) | — | **85.8%** | 23.1% | reached ADR 14, unrealistic starting limits |
| run2a | ep 6500 | 78.3% | **21.6%** | hardware limits, ADR 13 |
| **run2g** | **ep 2500** | **79.1%** | 25.3% | best Teacher v2 lift |
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
