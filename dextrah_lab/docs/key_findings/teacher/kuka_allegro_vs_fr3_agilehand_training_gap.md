# Key Finding: Why kuka_allegro Teachers Train ~5x Faster Through ADR Than fr3_agilehand

**Date:** 2026-04-02
**Branch:** fr3_agilehand
**Context:** kuka_allegro reaches 30 ADR steps within 6k epochs on a single 4090 (4096 envs). fr3_agilehand struggles to advance through ADR at comparable speed.

---

## Root Cause Analysis (Ranked by Impact)

### 1. FABRICS Controller = Free Safety Net + Low-Dimensional Actions

The kuka_allegro task uses a **geometric FABRICS controller** (`KukaAllegroPoseFabric`) that sits between the RL policy and the robot. This is the dominant structural difference.

| Aspect | kuka_allegro | fr3_agilehand |
|--------|-------------|---------------|
| Action dimensionality | **11** (6D palm pose + 5D hand PCA) | **23** (raw joint targets) |
| Arm control | 6D task-space (xyz + rpy) | 7D joint-space |
| Hand control | 5D PCA grasp primitives | 16D individual joints |
| Joint limit enforcement | FABRICS (hard constraints) | Policy must learn |
| Collision avoidance | FABRICS (built-in via WorldMeshesModel) | Policy must learn |
| IK feasibility | FABRICS (guaranteed) | Not guaranteed |

**Impact:** The policy explores a structured 11D manifold where every action is kinematically valid, vs. a raw 23D joint space where most random configurations cause immediate termination.

### 2. Termination Conditions: 3 vs 11

| kuka_allegro (3 conditions) | fr3_agilehand (11 conditions) |
|---|---|
| Object out of XY bounds | Object out of XY bounds |
| Object too low (Z < 0.2m) | Object too low (Z < 0.2m) |
| Timeout (600 steps) | Timeout (600 steps) |
| | **Hand too far (XYZ workspace box)** |
| | **Hand too close to table (<1cm)** |
| | **Palm flipped (cos < -0.3, >108deg)** |
| | **Arm-table contact (ContactSensor)** |
| | **Physics unstable (NaN/Inf)** |
| | **Velocity explosion (hand vel > 50 rad/s)** |

FABRICS prevents 6 of the 8 additional failure modes from ever occurring. In fr3_agilehand, early training is dominated by robot self-destruction rather than object manipulation learning.

### 3. Actuator Curriculum (fr3_agilehand only)

fr3_agilehand **actively tightens hardware constraints** as ADR advances — making each ADR step harder beyond randomization:

| Parameter | ADR 0 (start) | ADR 50 (terminal) | Reduction |
|---|---|---|---|
| `thumb_rot_vel_limit` | 10.0 rad/s | 0.14 rad/s (8 deg/s) | **71x slower** |
| `arm_57_effort_limit` | 50 Nm | 12 Nm | **4x weaker** |
| `arm_14_effort_limit` | 100 Nm | 87 Nm | 1.15x weaker |

kuka_allegro has **no actuator curriculum**. FABRICS handles motion constraints implicitly via damping (10 -> 20, a gentle 2x change).

The thumb velocity reduction alone can invalidate previously learned grasping strategies mid-curriculum — the agent must re-learn grasping at each ADR level instead of building on prior solutions.

### 4. Reward Structure: 4 Clean Terms vs 12 Competing Terms

**kuka_allegro (4 terms, clean gradient):**
| Term | Weight | ADR Modulation |
|------|--------|---------------|
| hand_to_object | 1.0, sharpness=10 | No |
| object_to_goal | 5.0, sharpness -15 -> -20 | Yes (sharpness only) |
| lift | 5.0 -> **0.0** (vanishes) | Yes |
| finger_curl_reg | -0.01 -> -0.005 (tiny) | Yes |

**fr3_agilehand (12+ terms, many competing):**
| Term | Weight | Notes |
|------|--------|-------|
| hand_to_object | 4.0, sharpness=4 | |
| object_to_goal | **40.0**, sharpness -5 -> -10 | 8x larger than kuka |
| lift | **40.0 -> 20.0** (stays dominant) | Never vanishes |
| finger_curl_reg | **-0.5 -> -1.2** (100x kuka) | Actively fights grasping |
| good_grasp | 3.0 (binary) | *Not in kuka* |
| palm_align | -0.7 | *Not in kuka* |
| in_grip_align | -1.0 | *Not in kuka* |
| action_rate_penalty | -0.01 (hand 1.5x) | *Not in kuka* |
| joint_velocity_penalty | -5e-4 (hand 3.0x) | *Not in kuka* |
| success_bonus | 10.0 | *Not in kuka* |
| episode_length | 0.005 | *Not in kuka* |
| early_termination_penalty | -1.0 | *Not in kuka* |

Key observations:
- kuka_allegro fades `lift -> 0`, creating a clean two-phase curriculum (lift first, then hold). fr3_agilehand keeps lift at 20, so lift and goal rewards compete throughout.
- fr3_agilehand `finger_curl = -1.2` is **100x** kuka's `-0.01`, actively penalizing the finger configurations needed for grasping.
- 12 terms with large magnitudes (40.0 goal, 40.0 lift, 10.0 success, -1.2 curl, -1.0 grip align) create gradient conflicts that slow convergence.

### 5. Robot Spawn Noise

| | kuka_allegro | fr3_agilehand |
|---|---|---|
| Joint position noise (terminal) | +/-0.35 rad (+/-20 deg) | +/-0.8 rad (+/-46 deg) |

2.3x more initial state randomization, combined with no safety controller, means many reset states start in configurations that immediately trigger termination.

### 6. Per-Joint-Group Gain Randomization

kuka_allegro uses **1 event term** for all joint gains (single correlated scale factor). fr3_agilehand uses **5 independent event terms** (arm, MCP pitch, MCP yaw, PIP, thumb rot) — the policy must handle 5 independent randomization axes where each finger joint group can have different gain scaling in the same episode.

---

## Proposed Improvements for FR3 AgileHand Teacher Training

### Tier 1: Immediate Changes (No New Code Required)

#### 1A. Start with Realistic Physics, Randomize Don't Curricularize

**Current problem:** The actuator curriculum starts with generous physics (50 Nm wrist, 10 rad/s thumb) and progressively tightens to realistic values. The agent learns strategies that work with generous physics, then must unlearn and relearn as constraints tighten.

**Proposed approach:**
- Start at **realistic hardware values from episode 0**: thumb_rot_vel = 0.14 rad/s, arm_57_effort = 12 Nm
- **Randomize around realistic values** (e.g., uniform [0.1, 0.2] rad/s for thumb vel, [10, 15] Nm for wrist effort) instead of curriculum from easy -> hard
- Keep physics material randomization via ADR (friction, mass, damping) but decouple actuator limits from ADR progression

**Why this helps:** The agent learns one grasping strategy that works under realistic constraints from the start. ADR then adds noise/spawn/wrench randomization on top, but the fundamental motor skills transfer directly to hardware.

**Implementation:** In `env_cfg.py`, replace the `actuator_curriculum` ADR block with fixed values:
```python
# Instead of ADR curriculum:
"actuator_curriculum": {
    "thumb_rot_vel_limit": [0.14, 0.14],   # fixed at hardware value
    "arm_14_effort_limit": [87.0, 87.0],
    "arm_57_effort_limit": [12.0, 12.0],
}
```
Then add uniform randomization as EventTerms (e.g., +/-20% around realistic values) so the policy sees variation without curriculum.

#### 1B. Simplify Reward Structure

Reduce to 5-6 core terms matching kuka_allegro's clean gradient structure:

| Term | Proposed Weight | Rationale |
|------|----------------|-----------|
| hand_to_object | 2.0 | Keep, but reduce magnitude |
| object_to_goal | 10.0 | Keep, reduce from 40 |
| lift | 10.0 -> 0.0 (ADR fade) | **Fade to zero** like kuka, clean two-phase learning |
| good_grasp | 3.0 | Keep, clear binary signal |
| action_rate | -0.005 | Keep, reduce slightly |
| finger_curl | **-0.05** | Reduce 10-25x from current -0.5 |

**Remove:** palm_align, in_grip_align, joint_velocity_penalty, episode_length, success_bonus, penetration_penalty. These are either redundant with other terms or add gradient noise without improving final policy quality.

#### 1C. Reduce Robot Spawn Noise

Change terminal `joint_pos_noise` from 0.8 rad to 0.35 rad (match kuka_allegro). The agent shouldn't need to recover from 46-degree joint perturbations — that's a sim-only challenge with no sim2real benefit.

### Tier 2: Controller Abstraction (Medium Effort)

#### 2A. Task-Space Arm Controller (No FABRICS Required)

Replace the 7D joint-space arm control with a 6D task-space controller using **Isaac Lab's built-in differential IK**:

```
Policy outputs: [palm_xyz(3), palm_rpy(3), finger_joints(16)] = 22D
                              |
                     Differential IK solver
                              |
                 arm_joint_targets(7) + finger_targets(16) = 23D
                              |
                        PD controller
```

**Isaac Lab provides `DifferentialInverseKinematicsAction`** — a drop-in action term that converts 6D EE deltas to joint commands with built-in joint limit clamping and singularity handling. This:
- Reduces effective arm action space from 7D to 6D
- Guarantees kinematically valid arm configurations
- Prevents arm-table collisions via workspace limits in task space
- Eliminates `hand_too_far`, `arm_table_contact`, and most `vel_explosion` terminations for the arm

**Implementation sketch:**
```python
# In env_cfg.py, replace ImplicitActuatorCfg with:
from isaaclab.controllers import DifferentialIKControllerCfg

arm_ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",           # 6D pose targets
    ik_method="dls",               # damped least squares
    position_limit=[[x_min, x_max], [y_min, y_max], [z_min, z_max]],
)
```

This is a lighter-weight alternative to FABRICS that doesn't require the external `fabrics_sim` package.

#### 2B. Hand PCA Basis for AgileHand

Compute a proper PCA basis for the AgileHand (the current values are Allegro placeholders marked `# TODO`):

1. Record 10k+ grasp demonstrations in sim (random objects, successful grasps only)
2. Collect 16D finger joint configurations at grasp contact
3. Run PCA, keep top 5-7 components (target >95% variance explained)
4. Set PCA mins/maxs from the demonstration distribution

This reduces hand action space from 16D to 5-7D. Combined with task-space arm control:
- **Total action space: 11-13D** (matching kuka_allegro's 11D)
- Policy explores a structured grasp manifold instead of raw 16D joint space

#### 2C. FABRICS Integration (High Effort, Best Performance)

**Status: Not directly feasible.** `fabrics_sim` only has `KukaAllegroPoseFabric` — no FR3/Franka variant exists. The AgileHand PCA basis is also missing (TODO placeholder).

**Required upstream work:**
1. Create `FR3AgileHandPoseFabric` class in fabrics_sim (or fork)
2. Add FR3 URDF to fabrics_sim robot registry
3. Map AgileHand joint/link names to fabric task maps
4. Compute AgileHand PCA basis (see 2B)

**Timeline estimate:** 2-4 weeks if fabrics_sim internals are well-understood. Depends on upstream maintainer support.

**Recommendation:** Pursue 2A (differential IK) + 2B (hand PCA) first — these achieve ~80% of FABRICS' benefit with standard Isaac Lab tools and no external dependencies.

### Tier 3: Structural Changes (Longer Term)

#### 3A. Decouple ADR Axes

Current fr3_agilehand ADR increments **all parameters simultaneously** — when the agent hits 40% success, physics, spawn, noise, actuator limits, AND reward weights all change at once. This creates compounding difficulty spikes.

Proposed: **Staged ADR with independent axes:**
1. **Phase 1 (ADR 0-15):** Physics randomization only (friction, mass, gains)
2. **Phase 2 (ADR 15-30):** Add spawn position and rotation randomization
3. **Phase 3 (ADR 30-45):** Add observation noise and wrench disturbances
4. **Phase 4 (ADR 45-50):** Final reward weight adjustments

This lets the agent master each type of randomization before the next layer is added, similar to how kuka_allegro effectively has fewer "axes of difficulty" due to FABRICS absorbing the kinematic challenges.

#### 3B. Reduce Gain Randomization Granularity

Merge the 5 separate finger gain event terms into 2:
- `finger_gains` (all MCP + PIP joints, single scale)
- `thumb_rot_gains` (thumb rotation, separate — this joint is mechanically different)

This reduces per-episode variance while maintaining the sim2real-relevant randomization.

---

## Summary: Recommended Implementation Order

| Priority | Change | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| 1 | Fix actuator values to hardware-realistic from start | 1 hour | High — eliminates strategy invalidation |
| 2 | Simplify reward to 5-6 terms, fade lift to 0 | 2 hours | High — cleaner gradient signal |
| 3 | Reduce spawn noise to +/-0.35 rad | 5 min | Medium — fewer wasted episodes |
| 4 | Task-space arm control via DifferentialIK | 1-2 days | Very High — 7D -> 6D arm + safety |
| 5 | Compute AgileHand PCA basis | 2-3 days | Very High — 16D -> 5D hand |
| 6 | Staged ADR (independent axes) | 0.5 days | Medium — smoother curriculum |
| 7 | Merge finger gain event terms | 30 min | Low-Medium |
| 8 | Full FABRICS integration | 2-4 weeks | Highest (but blocked on upstream) |

**Expected combined impact of items 1-5:** Bring fr3_agilehand ADR progression speed to within 2x of kuka_allegro, down from the current ~5x gap.

---

## Key Takeaway for Thesis

> The kuka_allegro task's training advantage is **not** primarily due to better hyperparameters or reward tuning — it's a structural advantage from the FABRICS controller that reduces the RL problem dimensionality (23D -> 11D), eliminates 6/11 termination conditions, and guarantees kinematic feasibility of every action. For fr3_agilehand to compete, the solution is not harder training but **smarter action abstractions** — task-space arm control and hand PCA bases that give the policy the same structural advantages without requiring the external FABRICS package. Starting with realistic physics (randomize, don't curricularize) prevents the additional problem of strategy invalidation during actuator curriculum progression.
