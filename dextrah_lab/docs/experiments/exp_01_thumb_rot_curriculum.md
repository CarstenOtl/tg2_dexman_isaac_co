# Experiment 01 — Thumb Rotation Speed Curriculum for Multi-Object Single Teacher

**Date started:** 2026-03-23
**Branch:** `fr3_agilehand`
**Author:** Carsten Oertel
**Status:** Planning

---

## Motivation

The FR3+AgileHand task previously supported a single teacher policy that could grasp multiple objects (commit `5969c04`, ADR 8, 12k epochs). After retuning joint gains to match real hardware specs (lower stiffness on finger joints — see commit `693330e`), training a single multi-object teacher became unreliable. Per-object single teachers (N=1 environment) still train well.

**Hypothesis:** The thumb rotation joint (`revolute_thumb_rot`) is a bottleneck. The thumb must reposition between objects — its speed and range directly affect how quickly the policy can adapt its grasp posture to different object shapes. When thumb rotation is slow/constrained, the policy may not be able to generalise across objects within a single training run. By starting with a fast (unrestricted) thumb and progressively constraining it via ADR, the policy may be able to learn a more robust multi-object strategy before the difficulty ramp-up kicks in.

---

## Goal

1. Train a single teacher policy that successfully grasps **multiple objects** (starting with `multi_objects/3`: plane, shoe, female_knight).
2. Use an ADR curriculum on `velocity_limit_sim` of `thumb_rot`: start high (fast/easy), gradually reduce (slow/realistic).
3. Understand whether thumb rotation speed is the limiting factor in multi-object generalisation.

---

## Current State (Baseline)

### Actuator config (`fr3_tekken_left.py` — `FR3_TEK_LEFT_CONFIG`)

| Joint | stiffness | damping | velocity_limit_sim |
|---|---|---|---|
| thumb_rot | 20.0 | 2.0 | 20.0 rad/s |
| mcp_pitch | 1.77531 | 0.5 | 15.0 rad/s |
| mcp_yaw | 0.28467 | 0.2 | 15.0 rad/s |
| pip | 0.24299 | 0.2 | 15.0 rad/s |

### ADR max ranges (no thumb velocity term currently)

```python
"robot_joint_stiffness_and_damping": {
    "stiffness_distribution_params": (0.5, 2.),
    "damping_distribution_params": (0.5, 2.),
},
```

### Last known multi-object result

**Commit `f15593c`** — "multiobject training success! ADR 21 steps" (2026-03-09 19:43)
Stored policy: `stored_policies/fr3_agilehand/06_multiple_objects_higher_dep_vel_ADR_8_03_09_19_48_54`

Actuator config at that commit (the config that **worked**):

| Joint | stiffness | damping | velocity_limit_sim |
|---|---|---|---|
| thumb_rot | 20.0 | 2.0 | 20.0 rad/s |
| mcp_pitch | **10.0** | **1.0** | 15.0 rad/s |
| mcp_yaw | **10.0** | **1.0** | 15.0 rad/s |
| pip | **10.0** | **1.0** | 15.0 rad/s |

After joint gain retuning (`693330e`, 2026-03-10): finger joint stiffness was dropped to match real hardware (`mcp_pitch`: 10.0→1.77, `mcp_yaw`: 10.0→0.28, `pip`: 10.0→0.24). This made fingers much more compliant. Single-object teachers still converge; multi-object does not.

**Note:** `thumb_rot` stiffness (20.0) was NOT changed between the working and broken configs — it is relatively stiff compared to the other fingers. The finger joints are now ~10× less stiff than they were at the multi-object success point.

---

## Planned Changes

### Phase 1 — Fast thumb, easy physics (plane only)

**Goal:** Confirm that a high velocity limit makes training fast and stable. Use `test_object` (plane).

Changes to `fr3_tekken_left.py`:
- Increase `velocity_limit_sim` for `thumb_rot`: **20.0 → 100.0 rad/s**
- Keep stiffness/damping unchanged

ADR curriculum: add a thumb-specific `velocity_limit` ADR term that starts at `(1.0, 1.0)` × 100 rad/s and gradually reduces toward `(0.1, 0.3)` × 100 rad/s = **[10, 30] rad/s** at full ADR.

Training command:
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
  env.objects_dir=test_object \
  env.use_cuda_graph=False
```

**Success criterion:** Lift success > 60%, ADR reaches level ≥ 5.

---

### Phase 2 — Fast thumb, 3 objects

Once Phase 1 policy converges (or with the Phase 1 checkpoint as warm-start), switch to `multi_objects/3` and test whether the fast thumb + ADR curriculum enables a single multi-object teacher.

Training command:
```bash
# Resume from Phase 1 checkpoint or start fresh:
env.objects_dir=multi_objects/3
```

**Success criterion:** Single teacher lifts all 3 objects with >40% success, ADR progresses.

---

### Phase 3 — Progressive physics difficulty

Once Phase 2 converges, increase object/robot physics randomization ranges (friction, mass, restitution) to test robustness. This mirrors the original `5969c04` success but with the thumb curriculum in place.

---

## Physics Curriculum Design

| ADR level | thumb velocity_limit (effective range) | Physics randomization |
|---|---|---|
| 0 | 100 rad/s (fixed) | None |
| 2 | [80, 100] rad/s | friction ±10% |
| 5 | [50, 100] rad/s | friction ±30%, mass ±20% |
| 8 | [20, 60] rad/s | full ADR ranges |
| 10 | [10, 30] rad/s | full ADR ranges (at target real values) |

---

## Results Log

### Run 02 — Restored finger stiffness + fast thumb + thumb init at -20° (partial improvement)
- Date: 2026-03-23
- Objects: test_object (plane) and multi_objects/14/
- Config changes vs Run 01:
  - `mcp_pitch`: stiffness 1.77531→10.0, damping 0.5→1.0
  - `mcp_yaw`: stiffness 0.28467→10.0, damping 0.2→1.0
  - `pip`: stiffness 0.24299→10.0, damping 0.2→1.0
  - `thumb_rot`: velocity_limit_sim=1.7453 rad/s (100 deg/s, ADR curriculum active)
  - `revolute_thumb_rot` init: 0.0 → -0.3491 rad (-20°)
- Max ADR reached: —
- Lift success: low
- Outcome: **Partial improvement, abandoned**
- Notes:
  - **Positive**: Thumb init at -20° eliminated the approach collision — thumb no longer gets knocked back by the object. Pre-rotation gives a noticeably better grasping position.
  - **Negative**: Policy still fails to lift. Classic reward hacking — policy discovers it can maximise approach/contact/alignment rewards without committing to a full lift. Lift reward not dominant enough to drive the behaviour through.
  - **Pattern**: This reward-hacking failure (maximise dense rewards, skip lift) has appeared across multiple experiments. The dense shaping rewards may be providing sufficient return without ever needing to lift.

---

### Run 01 — Thumb rotation curriculum baseline (abandoned)
- Date: 2026-03-23
- Objects: test_object (plane)
- Checkpoint: `logs/rl_games/dextrah_tekken_lstm/03-23_18-18-23/nn/dextrah_tekken_lstm.pth`
- Max ADR reached: —
- Lift success: —
- Outcome: **Abandoned** — primary failure mode identified as thumb MCP pitch collapsing on object contact during approach (see Observations below), not thumb rotation speed. Training further with this config would not address the root cause.

---

## Observations & Failure Modes

### 2026-03-23 — Thumb MCP pitch collapse during approach

**Observation:** The primary failure mode is not thumb rotation speed. During the approach phase, the object physically knocks the thumb's MCP pitch joint backwards (joint collapses toward 0 / open position). Once the thumb is forced open by the object, the hand can no longer form a grasp and the episode fails.

**Implication:** Thumb rotation velocity is likely not the bottleneck. The more fundamental issue is that the thumb MCP pitch joint is too compliant (stiffness=1.77531 Nm/rad, damping=0.5) to resist object contact forces during approach. The finger buckles rather than pushing through.

**Hypothesis update:** The experiment may need to also address thumb MCP pitch stiffness, not just thumb rotation speed. The current low stiffness (matching real hardware) makes the thumb mechanically too weak to maintain grip posture when the object pushes back.

**Possible fixes to test:**
1. Increase thumb MCP pitch stiffness selectively (above hardware spec) during early training, then reduce via ADR — similar to the thumb rotation curriculum.
2. Add a reward term for maintaining thumb MCP pitch above a minimum threshold during approach.
3. Accept collapse during approach as unavoidable with current gains and rely on the policy learning to pre-position the thumb out of the collision path.

---

## Open Questions for Thesis

1. Is velocity limit the right proxy for "thumb difficulty"? Stiffness affects tracking bandwidth; velocity limit is a hard cap. Worth comparing.
2. Does starting from a fast thumb + curriculum actually generalise to the constrained regime, or does the policy overfit to fast motion?
3. Could the multi-object failure be primarily due to **object visual similarity in teacher obs** (object scale/shape variation) rather than thumb kinematics?
4. Is a per-object teacher + multi-teacher distillation (current plan) better than solving the multi-object single teacher — trade-off in training cost vs. student quality?

---

## Related Files

- Asset config: `dextrah_lab/assets/fr3_tekken_adof/fr3_tekken_left.py`
- Env config (ADR): `dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py`
- ADR manager: `dextrah_lab/tasks/fr3_agilehand/dextrah_adr.py`
- Stored policies: `dextrah_lab/stored_policies/fr3_agilehand/`
- Best plane teacher: `stored_policies/fr3_agilehand/08_ADR_7_single_object_plane_success_03_22_10_14_17/nn/last_dextrah_tekken_lstm_ep_16000_rew_15253.331.pth`
