---
name: exp_02 run log
description: Log of experiment 2 runs for FR3+AgileHand multi-object training
type: project
---

# Experiment 02 — FR3 + AgileHand

## Run history (from git log)
- **run1a–run1j**: early iterations (see git log for details)
- **run1j**: contact-gated early termination penalty, closer arm init
- **run1k**: increase contact reward, loosen finger curl, flatter lift
- **run1l**: physics stability overhaul + single object test

## run1m — 2026-03-25
**objects_dir**: `multi_objects/visdex_selected` (13 objects: mario, milk_pot, tutle_candle_holder, toy_bagger, teddy_bear, closed_fist, basketball_shoe, chicken_head_in_car, elephant_toy, train, homer, toy_cow, plane)

**Reward changes vs run1l:**
- `hand_object_contact_weight`: 5.0 → 10.0 (doubled)
- `good_grasp_weight`: 5.0 → 8.0
- ADR `lift_weight` range: (20→5) → (40→20) — lift stays dominant throughout ADR

**Command:**
```bash
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
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False
```

**Why:** Training wasn't converging fast enough; policy was hovering without making contact. Contact and lift signals strengthened to push faster discovery.

**Result:** Good contact, but not lifting. Environments breaking on contact (physics explosions). Arm stiffness/effort too high.

## run1n — 2026-03-25
**objects_dir**: `multi_objects/visdex_selected` (same 13 objects)

**Init pose change:** Arm now starts **above the object** (top-down approach) instead of to the side. Joint angles: j1=-5°, j2=-35°, j3=0°, j4=-150°, j5=-30°, j6=170°, j7=0°.

**Changes vs run1m:**
- Arm joints 1-4: effort_limit 200→100 Nm, stiffness 400→200
- Arm joints 5-7: effort_limit 200→50 Nm, stiffness 400→200
- Finger mcp/pip damping: 3→6 (smoother contact, less oscillation)
- `hand_object_contact_weight`: 10→8 (-2)
- `good_grasp_weight`: 8→6 (-2)
- Lift reward unchanged (ADR 40→20)

**Why:** Arm was pushing through objects with 200 Nm + stiffness 400, causing env explosions on contact. Reduced to real-robot torque limits. Contact/grasp rewards halved since contact is already working — lift needs to be the dominant signal now.

**How to apply:** Watch if arm can still lift with reduced torques. If arm is too weak, raise effort_limit slightly.

## run1o — 2026-03-25
**objects_dir**: `multi_objects/visdex_selected` (13 objects)

**Changes vs run1n:**
- `early_termination_penalty`: -3.0 → -1.0 (subtle signal, policy was hovering to avoid penalty)
- Early term penalty now only applies to: object out of bounds, hand out of bounds, palm flip — NOT physics crashes (vel_explosion, robot_unstable, arm_table_contact, hand_too_close)
- `hand_to_object_weight`: 5 → 4 (-1, reduce hovering incentive)

**Why:** Policy was hovering — early term penalty was too strong and the hand_to_object reward was keeping it near the object without committing to contact. Penalty scoped to intentional bad behaviour only.

**Result:** Policy CAN lift — periodic spikes in lift_success (~0.024) but collapses cyclically. object_to_goal_reward oscillates 0.1–0.6. Root cause: only 16 envs across 13 objects → ~1 env per object per batch, causing gradient oscillation (learns one object, overwrites on next batch).

## run1p — 2026-03-25
**objects_dir**: `multi_objects/visdex_selected` (13 objects)

**Changes vs run1o:**
- `num_envs`: 16 → 512 (headless), ~39 envs per object for stable gradients
- Resume from best checkpoint of run1o

**Command:**
```bash
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 512 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=2048 \
  agent.params.config.central_value_config.minibatch_size=2048 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=20000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=multi_objects/visdex_selected \
  env.use_cuda_graph=False \
  --checkpoint logs/rl_games/dextrah_tekken_lstm/03-25_19-01-51/
```

**Why:** 16 envs was too few for 13 objects — policy oscillated between object-specific strategies. 512 envs gives stable multi-object coverage per batch.

**Result:** Steady lifting improvement, reached ADR 5 by epoch 11124 (manually paused). First ADR step at epoch 8850. Policy reliably contacts and lifts across objects. Checkpoint: `logs/rl_games/dextrah_tekken_lstm/03-25_21-15-48/`
