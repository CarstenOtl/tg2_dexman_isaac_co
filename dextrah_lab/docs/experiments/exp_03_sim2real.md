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

**Result:** *(to be filled in — about to start)*
