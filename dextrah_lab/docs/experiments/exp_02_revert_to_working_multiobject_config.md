# Experiment 02 — Revert to Working Multi-Object Config (f15593c)

**Date started:** 2026-03-23
**Branch:** `fr3_agilehand`
**Author:** Carsten Oertel
**Status:** In Progress

---

## Motivation

All recent attempts at training a multi-object teacher policy have failed. The last known working multi-object config was at commit `f15593c` (2026-03-09, "multiobject training success! ADR 21 steps"), stored as `stored_policies/fr3_agilehand/06_multiple_objects_higher_dep_vel_ADR_8_03_09_19_48_54`.

Since then, many changes were made to reward weights, physics materials, lift reward formula, joint init positions, and ADR ranges. This experiment reverts all tuning parameters back to the f15593c values while **keeping structural bug fixes** (NaN guards, teacher_onehot, vel_explosion termination, multi-object env distribution).

**Key question:** Can the f15593c config still produce a working multi-object teacher with the current USD assets and code structure?

---

## What Was Reverted

### Reward weights (env_cfg.py)

| Parameter | After experiments (broken) | Reverted to (f15593c) |
|---|---|---|
| hand_to_object_weight | 4.0 | **5.0** |
| good_grasp_weight | 1.5 | **5.0** |
| finger_curl_reg_weight | -0.2 | **-0.5** |
| object_to_goal_weight | 10 | **20** |
| lift_sharpness | 3.0 | **7.5** |
| action_rate_penalty_weight | 0.008 | **0.01** |

### Physics materials (env_cfg.py)

| Parameter | After experiments | Reverted to (f15593c) |
|---|---|---|
| robot restitution_range (EventTerm) | (0.05, 0.05) | **(1.0, 1.0)** |
| object restitution_range (EventTerm) | (0.05, 0.05) | **(1.0, 1.0)** |
| sim physics_material | friction_combine_mode="max", restitution_combine_mode="max" | **removed (PhysX defaults)** |
| fingertip_physics_material EventTerm | present (static=2.0, dynamic=1.5) | **removed entirely** |
| ADR robot restitution_range | (0.0, 0.2) | **(0.8, 1.0)** |
| ADR object restitution_range | (0.0, 0.2) | **(0.8, 1.0)** |
| ADR fingertip_physics_material | present | **removed entirely** |

### Joint init positions (env_cfg.py)

Reverted from the "FABRICS-derived" arm pose back to the original f15593c homing pose:

| Joint | After experiments | Reverted to (f15593c) |
|---|---|---|
| fr3_joint1 | 1.4748 (84.5°) | **0.3491 (20°)** |
| fr3_joint2 | 0.7941 (45.5°) | **0.6109 (35°)** |
| fr3_joint3 | -1.0996 (-63°) | **-0.8727 (-50°)** |
| fr3_joint4 | -1.7436 (-99.9°) | **-0.8727 (-50°)** |
| fr3_joint5 | 0.9373 (53.7°) | **-0.3491 (-20°)** |
| fr3_joint6 | 3.3967 (194.6°) | **2.6180 (150°)** |
| fr3_joint7 | -0.8920 (-51.1°) | **0.0 (0°)** |
| thumb_rot | -0.3491 (-20°) | **-0.5** |

### Lift reward formula (env.py)

| | After experiments | Reverted to (f15593c) |
|---|---|---|
| Formula | `lift_weight * (exp(sharpness * height_above_table) - 1.0) * contact_mask` | `lift_weight * exp(-sharpness * vertical_error) * contact_mask` |
| Input | `object_height_above_table` (height above table surface) | `object_vertical_error` (abs distance to goal Z) |

The new formula rewarded absolute height gain; the old formula rewards proximity to goal height. Fundamentally different reward signal.

### ADR custom reward ranges

| Parameter | After experiments | Reverted to (f15593c) |
|---|---|---|
| object_to_goal_sharpness | (-3., -10.) | **(-5., -10.)** |
| lift_weight | (30., 10.) | **(10., 5.)** |
| finger_curl_reg | (-0.05, -1) | **(-0.1, -1)** |

### Removed features

- `thumb_velocity_limit` ADR curriculum (env_cfg + env.py) — removed entirely
- `finger_unstable_vel_thresh` termination — **kept** (bug fix, not a tuning change)

### Structural changes KEPT (bug fixes)

- NaN/Inf sanitization on observations and joint velocities
- `teacher_onehot` (size-1 placeholder) instead of `multi_object_idx_onehot` in teacher/critic obs
- `vel_explosion` termination condition
- Multi-object env distribution (modular indexing across all envs)
- Student obs size separated from teacher obs size
- `joint_vel_penalty.clamp(min=-30.0)`

---

## What Was NOT Reverted

- **USD assets** — keeping current USD files (joint gains in fr3_tekken_left.py are NOT reverted). The f15593c config had finger joint stiffness=10.0 vs current ~1.77/0.28/0.24. If training fails, this is the next thing to revert.
- **Code structure** — all multi-object routing, observation space computation, and NaN safety code stays.

---

## Complete Change List

Everything below was reverted from HEAD back to the f15593c values. Structural bug fixes (NaN guards, teacher_onehot, vel_explosion termination, multi-object distribution) were **kept**.

### 1. Reward weights (`env_cfg.py`)
- [x] `hand_to_object_weight`: 4.0 → **5.0**
- [x] `good_grasp_weight`: 1.5 → **5.0**
- [x] `finger_curl_reg_weight`: -0.2 → **-0.5**
- [x] `object_to_goal_weight`: 10 → **20**
- [x] `lift_sharpness`: 3.0 → **7.5**
- [x] `action_rate_penalty_weight`: 0.008 → **0.01**

### 2. Physics materials (`env_cfg.py`)
- [x] Robot `restitution_range` EventTerm: (0.05, 0.05) → **(1.0, 1.0)**
- [x] Object `restitution_range` EventTerm: (0.05, 0.05) → **(1.0, 1.0)**
- [x] Removed `friction_combine_mode="max"` from sim physics_material
- [x] Removed `restitution_combine_mode="max"` from sim physics_material
- [x] Removed `fingertip_physics_material` EventTerm entirely (didn't exist at f15593c)

### 3. ADR physics ranges (`env_cfg.py` → `adr_cfg_dict`)
- [x] Robot `restitution_range`: (0.0, 0.2) → **(0.8, 1.0)**
- [x] Object `restitution_range`: (0.0, 0.2) → **(0.8, 1.0)**
- [x] Removed `fingertip_physics_material` ADR entry entirely

### 4. ADR custom reward ranges (`env_cfg.py` → `adr_custom_cfg_dict`)
- [x] `object_to_goal_sharpness`: (-3., -10.) → **(-5., -10.)**
- [x] `lift_weight`: (30., 10.) → **(10., 5.)**
- [x] `finger_curl_reg`: (-0.05, -1) → **(-0.1, -1)**
- [x] Removed `thumb_velocity_limit` ADR entry entirely

### 5. Joint init positions (`env_cfg.py`)
- [x] `fr3_joint1`: 1.4748 → **0.3491** (84.5° → 20°)
- [x] `fr3_joint2`: 0.7941 → **0.6109** (45.5° → 35°)
- [x] `fr3_joint3`: -1.0996 → **-0.8727** (-63° → -50°)
- [x] `fr3_joint4`: -1.7436 → **-0.8727** (-99.9° → -50°)
- [x] `fr3_joint5`: 0.9373 → **-0.3491** (53.7° → -20°)
- [x] `fr3_joint6`: 3.3967 → **2.6180** (194.6° → 150°)
- [x] `fr3_joint7`: -0.8920 → **0.0** (-51.1° → 0°)
- [x] `thumb_rot`: -0.3491 → **-0.3491** (-20°, kept — within joint limits [-0.34, 0.34])

### 5b. Hotfix — thumb_rot instability (vel_explosion on every episode)

Initial revert set `thumb_rot` init to -0.5 (from f15593c), but this is **outside the joint range** `[-0.34, 0.34]`. Combined with `velocity_limit_sim=1.7453` (from the thumb curriculum experiment), the PD controller oscillated against the joint stop, causing `vel_explosion=64` on every reset after the policy started outputting non-trivial actions.

**Fixes applied:**
- [x] `thumb_rot` init: -0.5 → **-0.3491** (-20°, at joint min, within limits)
- [x] `thumb_rot` `velocity_limit_sim` (`fr3_tekken_left.py`): 1.7453 → **20.0** rad/s (restored to f15593c value)

### 6. Lift reward formula (`env.py` → `compute_rewards`)
- [x] Reverted from `lift_weight * (exp(sharpness * height_above_table) - 1.0)` → **`lift_weight * exp(-sharpness * vertical_error)`**
- [x] Input changed from `object_height_above_table` back to **`object_vertical_error`** (abs distance to goal Z)
- [x] Removed `lift_reward.clamp(max=50.0)` hard cap

### 7. Thumb velocity limit code (`env.py`)
- [x] Removed `self.thumb_rot_dof_idx` init
- [x] Removed thumb velocity limit application in `_reset_idx()`
- [x] Removed thumb velocity limit update in ADR increment block

### 8. Asset config (`fr3_tekken_left.py`) — fully reverted
- [x] Full `git checkout f15593c -- fr3_tekken_left.py`
- [x] Arm: restored single group `fr3_joint[1-7]` stiffness=400, damping=40, effort=200, vel=2.175
- [x] `thumb_rot` `velocity_limit_sim`: 1.7453 → **20.0** rad/s
- [x] `max_depenetration_velocity`: 100.0 → **5.0**
- [x] `solver_position_iteration_count`: 16 → **10**
- [x] `solver_velocity_iteration_count`: 6 → **4**
- [x] Finger stiffness/damping: already 10.0/1.0 (matches f15593c)
- [ ] Object set — using `multi_objects/14` (14 objects) vs f15593c's 11 visdex objects in `test_object`

### 9. USD file restored (`FR3_tekkenadof_left.usd`)
- [x] `git checkout f15593c -- dextrah_lab/assets/fr3_tekken_adof/FR3_tekkenadof_left.usd`
- Hotfix 5b didn't resolve vel_explosion — USD has baked-in joint physics properties (damping, joint limits, collision meshes) that Python config doesn't fully override. Restoring the binary USD to the exact f15593c version.

---

## Test Plan

### Run 1 — Multi-object (14 objects)
```bash
cd dextrah_lab/rl_games
python train.py --task=dextrah_fr3_agilehand --seed 42 --livestream 2 \
  --num_envs 16 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.multi_gpu=False \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/14 \
  env.use_cuda_graph=False
```
Objects: female_knight, flat_toy_car, homer, large_5_cyl, large_8_cyl, locomotive, plane, random_warrior, shoe, soccer_cleat, teacup, toy_logic_board, truck, waving_diddy

**Success criterion:** Single teacher lifts multiple objects, ADR progresses.

### Fallback — If Run 1 fails, revert USD (joint gains)
Restore finger stiffness to 10.0 in `fr3_tekken_left.py` to fully match f15593c.

---

## Results Log

### Run 1a — vel_explosion death loop (2026-03-23)
- Config: reverted env_cfg reward/physics to f15593c, but `fr3_tekken_left.py` and USD still had post-f15593c values
- Result: **vel_explosion=64 on every episode** (avg_ep_step=2)
- Root cause: `thumb_rot` init=-0.5 outside joint range [-0.34, 0.34] + `velocity_limit_sim=1.7453` → thumb oscillates against joint stop on reset

### Run 1b — after thumb fix, still exploding (2026-03-23)
- Fixed thumb init to -0.3491 (-20°) and velocity_limit to 20.0
- Result: **still vel_explosion** — USD file and `fr3_tekken_left.py` had different solver iterations, depenetration velocity, and arm gains vs f15593c
- Fix: restored both `FR3_tekkenadof_left.usd` and `fr3_tekken_left.py` from f15593c, then re-split arm into joints [1-4] and [5-7] (same gains, for future curriculum)

### Run 1c — policy targets fr3_link1 for distance minimization (2026-03-23)
- Config: full f15593c revert (USD + Python config), arm split [1-4]/[5-7] with same gains (effort=200, stiffness=400, damping=40)
- Observation: policy minimises `hand_to_object` reward by moving `fr3_link1` (arm base) toward object instead of reaching with the hand. The `hand_to_object_pos_error` uses `hand_object_distance_body_names` which includes `base_link` (palm) — but the policy found it cheaper to move the whole arm base.

### Run 1d — env_cfg fully restored from f15593c, vel_explosion removed (2026-03-24)
- Config: env_cfg.py checked out from f15593c (only diffs: valid_objects_dir, thumb_rot init=-0.3491, gt_marker prim path). Removed `finger_unstable_vel_thresh` / vel_explosion termination. USD + fr3_tekken_left.py also from f15593c (arm split [1-4]/[5-7] with identical gains).
- **env.py still differs from f15593c** — NaN guards, teacher_onehot, student obs separation, multi-object distribution, lift reward formula (reverted but not verified).
- Result: **policy runs but does not learn to approach**. Across seeds (42, 1), the policy optimises for `hand_far` termination with minimum penalties rather than approaching the object. No lift, no contact.
- **Key finding**: there is still an unknown discrepancy between current state and f15593c causing training failure. The piecemeal revert of env_cfg + assets was not sufficient — env.py changes remain and may be the cause.
- Note: f15593c `_setup_objects` in non-distillation mode only loaded `sub_dirs[0]` for all envs, but the commit message says "training with 11 objects" — needs investigation whether distillation=True was used or if the code path was different.

### Run 1e — 5969c04 USD + full config restore, multi_objects/14, seed 1 (in progress)
- Date: 2026-03-24
- Objects: multi_objects/14
- Seed: 1
- num_envs: 16
- Config:
  - USD: restored from `5969c04` (`git checkout 5969c04 -- FR3_tekkenadof_left.usd`)
  - Arm gains: stiffness=400, damping=40, effort=200, vel=2.175 (split [1-4]/[5-7] with identical values)
  - Finger joints: stiffness=10.0, damping=1.0 (matches 5969c04)
  - Reward weights: fully restored to 5969c04 values
  - thumb_rot init: -0.3491 rad (-20°) — kept from exp_01 improvement (within joint limits)
- Training command:
  ```bash
  python train.py --task=dextrah_fr3_agilehand --seed 1 --livestream 2 \
    --num_envs 16 \
    agent.params.config.minibatch_size=256 \
    agent.params.config.central_value_config.minibatch_size=256 \
    agent.params.config.learning_rate=0.0001 \
    agent.params.config.horizon_length=16 \
    agent.params.config.mini_epochs=4 \
    agent.params.config.multi_gpu=False \
    agent.wandb_activate=False \
    env.success_for_adr=0.4 \
    env.objects_dir=multi_objects/14
  ```
- Max ADR reached: 0
- Lift success: 0
- Epochs run: ~850
- Outcome: **Inconclusive / no exploration** — policy minimises penalties (action rate, joint vel) without approaching the object. No contact, no lift reward. Same behaviour as all previous failed runs.
- Caveats: only 850 epochs, single seed (1) — too short to conclude definitively.
- Notes: `fr3.usd` was still the post-5969c04 version during this run. Now also restored `fr3.usd` from `5969c04`.

---

### Run 1f — fr3.usd restored + finger mcp_pitch init 0.1 (abandoned)
- Date: 2026-03-24
- Objects: multi_objects/14
- Seed: 1
- num_envs: 16
- Changes vs Run 1e:
  - `fr3.usd` restored from `5969c04`
  - All finger `mcp_pitch` joints init: 0.0 → 0.1 rad
- Max ADR reached: 0
- Lift success: 0
- Outcome: **Abandoned** — policy optimises `palm_direction_alignment` reward by moving the arm upward until the palm is normal to the table. No approach, no contact, no lift.
- Root cause: `palm_direction_alignment` provides a dense, easy-to-maximise signal that does not require approaching the object. Policy finds this local optimum before ever discovering the approach/contact rewards.
- Next: try different seed to check if this is a seed-specific collapse or systematic.

### Run 1g — seed 42, palm_direction_alignment_weight 0.5, closer arm init pose (abandoned)
- Date: 2026-03-24
- Objects: multi_objects/14
- Seed: 42
- num_envs: 16
- Changes vs Run 1f:
  - `palm_direction_alignment_weight`: 1.0 → **0.5**
  - Arm init pose: real-robot pose (closer to object): [84.5°, 45.5°, -63°, -99.9°, 53.7°, 194.6°, -51.1°]
- Max ADR reached: —
- Lift success: —
- Outcome: **Abandoned** — policy still fails to approach/lift. Same reward-hacking pattern: policy minimises penalties without ever discovering approach/contact rewards.
- Notes:
  - Reducing `palm_direction_alignment_weight` to 0.5 did not prevent exploitation of that reward term
  - Identified additional failure mode: policy exploits **short episodes** — triggering `out_of_reach` early terminations gives the policy many short episodes with low penalty, rather than committing to an approach trajectory. No early termination penalty was in place.

---

### Run 1h — early termination penalty + rl_games isaac-sim fork
- Date: 2026-03-24
- Objects: —
- Seed: —
- num_envs: 16
- Changes vs Run 1g:
  - Added `early_termination_penalty = -5.0` in `env_cfg.py` — flat penalty applied only on `out_of_reach` terminations (not natural timeouts)
  - Implemented penalty in `_get_dones` (sets `self._early_terminated`) and `_get_rewards` (applies penalty tensor)
  - Installed rl_games isaac-sim fork (`6b3534f`) to match pinned requirements — replaces pip 1.6.1
- Max ADR reached: 0
- Lift success: Yes (actively grasping and learning to lift)
- Epochs run: ~6k before crash
- Outcome: **Promising start, then crashed**
- Notes:
  - **Policy was actively pursuing grasping** — visible approach, contact, and lifting behaviour. First run across all exp_02 attempts where the policy explores rather than minimising penalties.
  - Best recorded checkpoint was actively grasping objects — usable as a resume point for future runs.
  - Root cause of prior failures was likely a combination of: (1) rl_games pip 1.6.1 diverging from the isaac-sim fork used during original successful training, and/or (2) absence of early termination penalty allowing short-episode exploitation.
  - Both changes applied together — cannot yet isolate which was the decisive factor.
  - **Crash at ~6k epochs**: hand started spawning in invalid configurations. Root cause unknown. The best checkpoint from before the crash can be used to resume training.

### Run 1h-resume — current setup with checkpoint resume (completed)
- Date: 2026-03-24 → 2026-03-25
- Resumed from: best checkpoint from Run 1h (before crash)
- Config: same as Run 1h (early_termination_penalty=-5.0, dep_vel=5.0, solver 10/4)
- Changes vs Run 1h:
  - Reverted `teacher_onehot` → `multi_object_idx_onehot` in teacher and critic obs (matching kuka_allegro, tg2_inspirehand, and test repo)
  - Removed `vel_explosion` print spam (termination logic kept)
- Max ADR reached: **36**
- Outcome: **Best result so far — policy grasps and lifts, ADR progresses to 36**
- Physics issue: **significant depenetration artifacts** observed during training. Likely caused by `max_depenetration_velocity=5.0` (vs f15593c's 100.0) — low dep_vel prevents PhysX from resolving interpenetration fast enough, leading to unrealistic contact behaviour.
- **Obs space issue**: policy was trained with **single-object observation** (obs space declared for N=1) while running multi-object training (`multi_objects/14`). The obs space was not adapted to include multi-object identity (like `multi_object_idx_onehot` in kuka_allegro / tg2_inspirehand). This means the teacher had no way to condition on which object it was manipulating — a fundamental mismatch.
- Next: increase `max_depenetration_velocity` to 100.0 and fix obs space to multi-object before retraining.

### Run 1i — test repo (f15593c full checkout), from scratch (completed)
- Date: 2026-03-24 → 2026-03-25
- Repo: `/home/carsten.oertel/code/test/tg2_dexman_isaac_co/`
- Config: full f15593c checkout, no modifications, dep_vel=100.0, solver 16/6
- From checkpoint: No (training from scratch)
- Max ADR reached: 0
- Outcome: **Underperformed** — policy never reached grasping. Optimised for getting close to the object (hand_to_object reward) but did not learn to grasp or lift. No ADR progression.
- Notes:
  - Same conda environment as current repo
  - Uses `multi_object_idx_onehot` (original f15593c behaviour)
  - Despite matching f15593c exactly, this run did not replicate the original success. May need more epochs, or the key improvements in the current repo (early_termination_penalty, vel_explosion termination) are actually load-bearing for learning to grasp.

---

### Remaining differences between current repo and test repo

| Parameter | Current repo | Test repo (f15593c) |
|---|---|---|
| `max_depenetration_velocity` | 5.0 | 100.0 |
| `solver_position_iteration_count` | 10 | 16 |
| `solver_velocity_iteration_count` | 4 | 6 |
| `early_termination_penalty` | -5.0 | not present |
| NaN/Inf sanitization | yes | no |
| `vel_explosion` termination | yes (no print) | no |
| `joint_vel_penalty.clamp(min=-30)` | yes | no |
| Arm actuator groups | split [1-4]/[5-7] (same gains) | single [1-7] |
| `hand_to_object_sharpness` | 4.0 | 5.0 |
| `palm_direction_alignment_weight` | 0.5 | 1.0 |
| `hand_object_contact_weight` | 3.0 | 2.0 |
| `mcp_pitch` init | 0.1 | 0.0 |

### Run 1j — increase dep_vel to 100.0 (planned)
- Date: 2026-03-25
- Based on: Run 1h-resume config (early_termination_penalty=-5.0, solver 10/4, multi_object_idx_onehot)
- Changes vs Run 1h-resume:
  - `max_depenetration_velocity`: 5.0 → **50.0** (conservative step toward f15593c's 100.0 — reduces artifacts without jumping straight to maximum)
  - **Obs space: adapted to multi-object** — `multi_object_idx_onehot` is now a proper N-dim one-hot (one entry per object) in teacher and critic obs, matching the pattern in `kuka_allegro` and `tg2_inspirehand`. Previously a fixed size-1 constant ones tensor (`teacher_onehot`) that gave no object-identity signal.
  - **Arm init pose: real-robot pose** (closer to object) — restored from commit `4c9beb5` (2026-03-23): `[84.5°, 45.5°, -63°, -99.9°, 53.7°, 194.6°, -51.1°]`. Previous f15593c pose `[20°, 35°, -50°, -50°, -20°, 150°, 0°]` started the arm further away, making approach harder. Closer init reduces the distance the policy needs to learn to cover before first contact.
  - **Contact-gated early termination penalty (-3.0)** — penalty only applied when episode ends early AND no object contact was made. If the hand was touching the object at termination, no penalty is applied. Grasping attempts will frequently trigger early terminations (fingers slipping, palm flip, etc.) and should not be discouraged. Pure avoidance / short-episode exploitation (no contact at all) is still penalised at -3.0.
- Rationale: four issues addressed vs 1h-resume: (1) unrealistic contact physics from low dep_vel, (2) missing object-identity signal in obs, (3) arm starting too far from the object, (4) penalty too heavy suppressing contact exploration.
- Max ADR reached: 0
- Outcome: **Failed** — policy found a risk-mitigation strategy of hovering above the object. Contact-gated penalty was not sufficient to pull the policy toward touching: hovering near the object collects `hand_to_object` reward without risking any penalised termination. No contact, no lift.
- `thumb_mcp_pitch` init also reduced from 0.1 → 0.05 rad during this run (thumb folding inward) — did not change outcome.

### Run 1k — increased contact reward, reduced finger curl reg, flatter lift gradient (failed)
- Date: 2026-03-25
- Changes vs Run 1j:
  - `hand_object_contact_weight`: 3.0 → **5.0** — make contact actively more valuable than hovering
  - `finger_curl_reg_weight`: -0.5 → **-0.2** — loosen curl penalty so ADR can widen it; was suppressing finger motion needed for contact
  - `lift_sharpness`: 7.5 → **5.0** — flatter lift gradient makes the reward easier to discover initially
- Rationale: policy is hovering optimally — contact reward must outweigh the value of hovering risk-free. Looser finger curl allows fingers to move more freely toward the object.
- Outcome: **Failed** — policy still hovers near object, does not attempt contact. Confirmed the hovering optimum is not a reward weight issue but a physics stability issue: every time fingers touch the object, the simulation becomes unstable → early termination → policy learns that contact = punishment.
- Root cause identified: **finger joint physics instability at contact**. Comparison with `kuka_allegro` config revealed:
  - `effort_limit_sim` on finger joints was 10.0 Nm (kuka_allegro uses 0.5 Nm) — PD controller generates explosive contact forces
  - `damping` was 1.0 (critically underdamped at stiffness=10.0) — oscillations on contact
  - `solver_velocity_iteration_count=4` — velocity iterations can amplify contact instability
  - `max_depenetration_velocity=50.0` — too low, PhysX can't resolve interpenetration fast enough
  - AgileHand has mimic joints (DIP follows PIP) — high PIP effort amplified through mimic chain at contact
  - Isaac Sim docs confirm: mimic joints must have drive stiffness/damping=0 (already correctly set in USD); instability comes from actuated PIP effort being too high

### Run 1l — physics stability overhaul + single object test (in progress)
- Date: 2026-03-25
- Objects: `test_object` (plane — simplest possible, isolates reward from object complexity)
- Changes vs Run 1k:
  - `effort_limit_sim` on mcp_pitch/mcp_yaw/pip: 10.0 → **2.0 Nm** (matches kuka_allegro scale)
  - `damping` on mcp_pitch/mcp_yaw/pip: 1.0 → **3.0** (prevent oscillation at contact)
  - `velocity_limit_sim` on mcp_pitch/mcp_yaw/pip: 15.0 → **8.0 rad/s** (clamp runaway velocities)
  - `max_depenetration_velocity`: 50.0 → **30.0** (moderate value between tg2's 2.0 and kuka_allegro's 1000.0)
  - `solver_velocity_iteration_count`: 4 → **0** (velocity iterations worsen contact instability)
  - `palm_direction_alignment_weight`: 0.5 → **0.7**
  - `thumb_mcp_pitch` init: 0.1 → **0.05 rad** (thumb was folding inward)
- Rationale: address root cause — make contact physically stable so the policy can learn from it rather than being punished by it. Single object training to verify reward structure in isolation.

**Physics comparison — working configs vs fr3_agilehand run1l:**

| Parameter | kuka_allegro | tg2_inspirehand | fr3_agilehand run1l |
|---|---|---|---|
| `max_depenetration_velocity` | 1000.0 | 2.0 | 30.0 |
| `solver_position_iteration_count` | 8 | 8 | 10 |
| `solver_velocity_iteration_count` | **0** | 4 | 0 |
| `enabled_self_collisions` | — | **False** | True |
| finger stiffness | 3.0 | 10.0 | 10.0 |
| finger damping | 0.1 | 1.0 | 3.0 |
| finger effort_limit_sim | **0.5 Nm** | 3.0 Nm | 2.0 Nm |
| finger velocity_limit_sim | not set | 15.7 rad/s | 8.0 rad/s |

Key observations:
- dep_vel varies wildly (2.0 to 1000.0) between working configs — likely not the critical stability parameter
- tg2 disables self collisions — reduces constraint complexity when fingers cluster around an object
- kuka_allegro uses 0 velocity solver iterations and very low effort (0.5 Nm) — aligns with our run1l changes
- Neither working config has mimic joints (AgileHand does) — mimic chain amplifies contact forces through DIP joints

**Note on dep_vel via ADR curriculum:** Technically feasible — `max_depenetration_velocity` is a `RigidBodyPropertiesCfg` field updateable dynamically via `write_body_physx_props_to_sim()`. Could be added as a custom ADR event term (~50 lines). However, dep_vel is a physics accuracy setting, not a difficulty parameter — there is no benefit to ramping it. Recommended: fix it to 100.0 immediately.

---

## Things to Add After Validated Baseline

- [ ] Arm joint split [1-4] / [5-7] (same gains initially, for future curriculum)
- [ ] Multi-object distribution across envs (currently only first object used in teacher training)
- [ ] NaN/Inf sanitization on observations and joint velocities
- [ ] Student obs size separated from teacher obs
- [x] Reverted `teacher_onehot` → `multi_object_idx_onehot` (matches kuka_allegro, tg2, test repo)
- [ ] `gt_pos_marker` separate prim path (for distillation)
- [ ] `finger_unstable_vel_thresh` termination (if needed, was causing false positives)
- [ ] Thumb init at -20° (pre-rotated away from approach path)
- [ ] Realistic arm effort limits via ADR curriculum

---

## Related Files

- Env config: `dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py`
- Env: `dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py`
- Asset config: `dextrah_lab/assets/fr3_tekken_adof/fr3_tekken_left.py`
- Reference checkpoint: `stored_policies/fr3_agilehand/06_multiple_objects_higher_dep_vel_ADR_8_03_09_19_48_54/`
- Reference commit: `f15593c` (2026-03-09)
