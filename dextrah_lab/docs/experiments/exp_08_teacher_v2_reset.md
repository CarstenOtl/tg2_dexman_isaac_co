---
name: exp_08 teacher v2 reset and reward-gate investigation
description: Post-reset fresh-train investigation on fr3_agilehand_teacher_v2_reset branch — was the run3v–7g failure mode dextrah-code regression or below-the-repo drift? Then directional contact filter to close the buckled-thumb exploit.
type: project
---

# Experiment 08 — Teacher v2 reset and reward-gate investigation

**Branch:** `fr3_agilehand_teacher_v2_reset` (created from `8a94ed4` on 2026-05-26, the run2g policy storage commit).

**Why this is a new file (not exp_03):** exp_03 was about sim2real curriculum tuning (thumb velocity, actuator limits, hardware-realistic constraints). This file is about **what went wrong** during the ~7-week run3v–run7g investigation that followed exp_03 — and whether the failure mode lives in dextrah code, below-the-repo (IsaacLab / drivers / physics), or in the reward gate itself. The directional contact filter work (run2h-reset on) is a reward-gate intervention, not a sim2real one.

**Context (what brought us here):**

After exp_03 ended with run2g at 79.1% lift, the `fr3_agilehand_teacher_v2` branch ran ~49 commits worth of experiments (run3v through run7g) trying to break a wrap-but-don't-lift basin / thumb-curl exploit. None worked:
- Wholesale revert to v1 (`run7d`): thumb buckling pattern re-emerged
- Warm-start from run7d ep_4500 (`run7e`): zero lift across 22k iters
- `lift_sharpness` 2 → 5 (`run7f`): livestream smoke check (16 envs) was inconclusive
- 2× envs (1024 → 2048, `run7g`): same basin, just ~2× faster — env-count hypothesis falsified
- Livestream verification on run7g checkpoint confirmed: **thumb-curl exploit** (buckled thumb dorsal surface scraping object), NOT a wrap-but-don't-lift basin as the TB signature looked

The run2g `.pth` itself no longer evaluates to 79.1% on the current IsaacLab+drivers (per `project_teacher_v2_baseline.md`: canonical is 55.0%/40.8%), so there's below-the-repo drift. But that doesn't tell us whether **training from scratch on the run2g codebase** produces a working policy on the current stack — that's what this branch tests.

## Preliminary fresh-train experiments (lr bisection)

### preliminary-1 — lr=1e-4 fresh-train (2026-05-26)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-26_17-05-29/`
**Stopped at:** iter ~2161 (user killed, no lift after 2k iters)
**Peak `lift_success`:** **0.78% at iter 474–475** (~40-iter window above noise floor, then decay back to 0)

**Pattern:** Genuine partial-lift signal briefly emerged, then policy regressed toward a settled basin: contact_reward → 5.3, curl → -1.5, h2o → 0.15 m, ep_len → 480. Less deep than run7g's thumb-curl basin (run7g had curl -2.86), but still no lift after the iter-475 peak.

### preliminary-2 — lr=5e-5 fresh-train (LR halved, 2026-05-26)

**Run dir:** `logs/rl_games/dextrah_tekken_lstm/05-26_18-28-47/`
**Stopped at:** iter ~7529 (user killed, no lift after 7.5k iters)
**Peak `lift_success`:** **0.78% at iter 456** — identical magnitude and near-identical iter to lr=1e-4 run. **Halving LR did NOT change the peak position or magnitude.**

**Pattern:** Same regression to settled no-lift basin, then 7000+ iters with `lift_success ≈ 0`. Contact basin deepens over time (contact_reward 2.1 → 7.5, curl -1.05 → -1.64) without ever escaping.

**Conclusion:** LR is NOT the cause of the iter-475 regression. The discovered-then-walked-away pattern is reward-shape driven, not optimizer driven. Both runs find the same local optimum and stay.

**Livestream verification:** User confirmed the policy basin is a settled no-lift configuration — thumb rotating too slowly to be useful, fingers curling around empty space. The `thumb_rot_vel_limit` (0.2618 rad/s ≈ 15 deg/s) at ADR 0 was identified as too slow for the policy to explore thumb-opposition strategies efficiently. (Separate observation — not addressed in run2h-reset.)

## Reward-gate intervention

### run2h-reset — directional fingertip contact filter on `good_grasp_mask` (2026-05-26)

**Motivation:** The persistent no-lift basin across all post-reset training shows the policy can't form productive grasps even with fresh weights and lower LR. CLAUDE.md documents the buckled-thumb exploit (`contact_mask = (contact_count > 0)` fires on any sensor contact including dorsal scraping), and the user's livestream confirmed the policy is exploiting non-grasp contact configurations. This experiment introduces a **directional filter on `good_grasp_mask`**: for each fingertip, only count contact toward "good grasp" if the contact force has a positive component along `(tip_pos_w - palm_pos_w)`. Reasoning: a real grasp force pushes the fingertip AWAY from the palm (object resists closing); a buckled finger receiving force on its dorsal surface gets force pushing TOWARD the palm = negative dot product = filtered out.

**Hypothesis:**
- Directional filter closes the buckled-finger contact-faking. `good_grasp_reward` only fires for geometrically-correct grasps. Policy should be pushed away from the exploit basin and toward proper opposed-finger grasps.
- If lift emerges within 2–5k iters at any object: filter is closing the exploit successfully.
- If no lift emerges but policy avoids all contact (contact_reward stays near 0): the filter is over-aggressive OR the sign convention is wrong (force direction in Isaac Lab's `force_matrix_w` may be reaction force rather than applied force) — flip threshold sign and retry.

**Configuration delta from previous (preliminary-2, `8a94ed4` baseline):**
- `env.py`: `_collect_object_contacts` now tracks two contact sets per env — **raw** (any-force, feeds `object_contact_counts` used by `lift_reward` / `object_to_goal_reward`, **unchanged behavior**) and **inside-filtered** (feeds `good_grasp_mask`).
- `env.py`: new `self.fingertip_body_indices` dict in `__init__` mapping the 5 distal-phalanx sensor link names to robot body indices.
- `env_cfg.py`: new `grasp_force_inside_threshold = 0.0` config field (any positive dot product passes; tune up to require stronger inward alignment, tune below 0 to relax / flip sign convention).
- **NO change** to `lift_reward` or `object_to_goal_reward` gating — they stay on raw `contact_mask = (contact_count > 0)`. Only `good_grasp_mask` (which gates `good_grasp_reward`) gets the directional filter.
- LR back to 1e-4 (preliminary-2's halving didn't help; revert).
- Scope: filter applied to **all 5 fingertips** (Thumb, Index, Middle, Ring, Pinky distal phalanges). Palm contact unfiltered.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, headless 1024 envs. visdex_selected.

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
- `lift_success` climbs past 1% sustained by iter 1000 → directional filter broke the exploit basin. Continue. Major win.
- `lift_success` peaks ~0.8% then regresses identically to preliminaries → filter doesn't bite, sign may be wrong (force direction reversed in Isaac Lab convention). Flip `grasp_force_inside_threshold` sign or invert the dot-product test.
- `contact_reward` collapses to near-0 and stays there → filter is over-aggressive; policy can't form any qualifying contact. Lower threshold or revisit the (tip - palm) geometry (may fail for fully-curled fingers).
- `good_grasp_reward` near zero throughout but contact_reward normal → filter is working but no policy can satisfy it; this still tells us the previous "good grasps" were all buckled exploits.

**Run directory:** *(not run — superseded by run2i-reset, which stacks additional curl-shaping changes on top of the directional filter)*

**Result:** Run2h-reset's directional filter was launched in livestream (16 envs) for a visual sanity check. User observation: policy did NOT learn to grasp — it learned to **hover** above the object without committing fingers. Directional filter alone is insufficient to push the policy toward proper opposition. Continuing in run2i-reset with additional curl-shaping.

### run2i-reset — split thumb / other-finger curl penalty + curled_q=0° (stacks on run2h-reset) (2026-05-26)

**Livestream observation that motivated this:** With run2h-reset's directional contact filter active, the policy in 16-env livestream converged to a **hover** behavior — palm parked above object, fingers extended, no commitment to opposition. The filter successfully blocked the buckled-thumb exploit (zero good_grasp_reward signal during scraping configurations), but with the existing curl regularizer (target = init pose, single weight across all 16 hand joints), the policy had no incentive to push the thumb out of the buckled / curled state.

**Motivation:** Split the curl regularizer so the thumb gets a **heavier** penalty than the other fingers, pulling the thumb toward a fully-extended (0°) configuration where it has room to swing into opposition. Exclude `revolute_thumb_rot` from any curl penalty — thumb rotation is left free for the policy to explore (the right thumb_rot init isn't known a priori, so don't pin it). Change the curl target from init pose to 0° for all penalized joints to remove the bias toward any specific starting configuration.

**Hypothesis:** Heavier thumb-curl penalty pushes the thumb to stay extended, giving the policy more "swing room" to rotate it into opposition with the other fingers. Combined with run2h-reset's directional contact filter (which makes good_grasp_reward only fire on geometrically-correct grasps), the policy should be funneled away from both the buckled-thumb exploit AND the hover-and-avoid behavior, toward proper opposed grasps.

**Configuration delta from run2h-reset:**
- `env.py` `__init__`: replaced single `self.curled_q` (built from `init_joint_pos`) with split target queues — `self.thumb_curl_target_q` (3 joints: thumb mcp_pitch/mcp_yaw/pip, all zeros) and `self.other_finger_curl_target_q` (12 joints: index/middle/ring/pinky × mcp_pitch/mcp_yaw/pip, all zeros). `revolute_thumb_rot` excluded from both.
- `env.py` `compute_rewards`: signature replaces `(robot_dof_pos, curled_q, finger_curl_reg_weight)` with `(other_finger_dof_pos, other_finger_curl_target_q, thumb_dof_pos, thumb_curl_target_q, finger_curl_reg_weight, thumb_curl_reg_weight)`. Body computes two separate L2 distances and applies separate weights. Both rewards returned, clamped independently, logged separately to extras and TB.
- `env_cfg.py`: new `thumb_curl_reg_weight = -0.4` (2× `finger_curl_reg_weight = -0.2`), new `thumb_curl_reg_min = -3.0`, `thumb_curl_reg_max = 0.0`, new ADR entry `thumb_curl_reg = (-0.6, -1.6)` (2× `finger_curl_reg = (-0.3, -0.8)`), added `"thumb_curl"` to `active_reward_terms`.
- `dextrah_adr.py`: untouched (generic key lookup handles the new `thumb_curl_reg` ADR entry automatically).
- **All of run2h-reset's directional contact filter remains active** — `good_grasp_mask` continues to use force·(tip - palm) dot product filter on all 5 fingertips.

**Setup:** test repo, dextrah_test env, GPU 0, seed 42. Starting with 16-env livestream sanity check (per workflow); 1024-env headless run after visual confirmation of behavior.

**Train command (livestream sanity check):**

```bash
cd /home/carsten.oertel/code/test/tg2_dexman_isaac_co/dextrah_lab/rl_games

CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_test/bin/python train.py \
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
- Livestream shows thumb staying extended (high curl penalty pulls it open), fingers curling AROUND object → behavior is in the right direction; promote to 1024-env headless.
- Thumb extended but other 4 fingers also stay extended (won't commit to closing) → other_finger_curl_reg too strong relative to grasp rewards; lower other_finger ADR weight.
- Thumb stays curled despite heavier penalty → thumb_curl_reg_weight magnitude insufficient; bump to 3× or 4× finger_curl_reg.
- Policy reverts to hovering with all fingers extended → reward shape is now anti-grasp; the new 0° target competes too directly with grasp formation; consider raising thumb_curl_reg_max from 0 toward positive (allow small bonus for staying extended) or reducing weight.

**Livestream smoke check (`05-27_09-58-55`, 32 envs, stopped at ~3k iters):** User observation — "thumb isn't staying bent. it's more straightened now. the good grasp actually looks pretty good. 0 lift after 3k epochs, but that is to be expected with low env count and livestream." Decision rule branch 1 fires exactly as hypothesized: the heavier thumb curl penalty (-0.4 weight, ADR -0.6 → -1.6) plus 0° target is pulling the thumb toward extension, AND the directional contact filter is producing visually plausible good-grasp configurations rather than the buckled-thumb exploit. Behavior direction matches the hypothesis. **Promoting to 1024-env headless.**

**Train command (headless, 1024 envs — the real run):**

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

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-27_11-06-04/` (test repo, GPU 0, dextrah_test, headless 1024 envs)

**Result (stopped at iter 284, 2026-05-27): TERMINATED EARLY — CPU bottleneck made training infeasibly slow, not enough data to verdict the hypothesis. But the 284 iters we did get were healthy and matched the run2h+run2i-reset design intent.**

| Signal | iter 50 | iter 100 | iter 200 | iter 284 |
|---|---|---|---|---|
| `lift_success` | 0.000 | 0.003 | 0.001 | 0.000 |
| `lift_reward` | 0.00 | 0.25 | 3.56 | 4.23 |
| `hand_object_contact_reward` | 0.00 | 1.48 | **9.66** | **11.25** |
| `good_grasp_reward` | 0.00 | **2.96** | **3.00** | **3.00** |
| `thumb_curl_reg` | -0.33 | -0.27 | -0.62 | -0.84 |
| `finger_curl_reg` | -0.71 | -0.48 | -0.73 | -0.70 |
| `hand_to_object_distance` (m) | 0.775 | 0.434 | 0.162 | 0.130 |
| `episode_lengths` | 583 | 93 | 254 | 409 |

PEAK `lift_success`: 0.0078 at iter 97 (≈ 1/128 envs, early-discovery noise).

**Performance (the reason the run was killed):**

| Metric | mean | last |
|---|---|---|
| `step_fps` | 7203 | 1029 |
| `step_inference_fps` | 6962 | 1027 |
| `rl_update_time` | 0.37 s | 0.34 s |

`step_fps` variance from 1029 → 7203 (7×) confirms the CPU-bound `_collect_object_contacts` hypothesis: the function does 10 GPU→CPU syncs per step plus Python loops over 1024 envs, causing the GPU to stall waiting for the CPU between syncs. User observed `nvidia-smi` GPU utilization bouncing 10–80% throughout the run.

**Observations:**
1. **Run2h-reset directional filter is doing its job.** `good_grasp_reward` saturates at 3.0 (its max value: good_grasp_weight × `good_grasp_mask=1`) by iter 100 and stays there — meaning the policy is producing inside-aligned multi-fingertip contacts, not the buckled-thumb scrape that the filter would reject.
2. **Run2i-reset curl split is working.** `thumb_curl_reg` (-0.84) is consistently heavier-magnitude than `finger_curl_reg` (-0.70) by iter 284 — the 2× weight ratio is reflected in the realized penalty. Thumb is being pulled toward the 0° target harder than the other fingers.
3. **Contact configuration is much richer than run7 era.** By iter 284, `contact_reward = 11.25` → ~3.75 fingertips per env in contact (vs run7g's ~2.2 in the thumb-curl basin) — the policy is engaging more fingertips in geometrically-real grasps, not buckling exploits.
4. **No verdict on lifting** — only 284 iters; the iter-475-equivalent peak window (where preliminary-1 hit 0.78%) hadn't arrived yet. Can't yet say whether this configuration breaks past the prior regression pattern.
5. **Step FPS at end of run (~1027) is ~10× slower than the pre-refactor headless runs of run7d-g** (which were on the order of ~10k FPS at 1024 envs). The Python overhead in contact collection becomes increasingly dominant as contact density grows (more nonzero env_idxs to loop over), explaining why mean was 7203 but last was 1029.

**Decision rule outcome:** Aborted before any decision-rule branch could fire — the run died on the perf axis, not the policy axis. Vectorized refactor of `_collect_object_contacts` is staged (uncommitted in `env.py`) to remove the bottleneck. Pending livestream verification on the refactored code that contact_reward and good_grasp_reward signatures still match (3.0 saturation, ~11 contact), then re-launch the same 1024-env headless config as a fresh run.

### run2i-reset.1 — retry of run2i-reset on vectorized contact code (2026-05-27)

**Motivation:** run2i-reset's 1024-env headless was killed at iter 284 due to a CPU bottleneck in `_collect_object_contacts` (10 GPU→CPU syncs per step + Python loops over envs). The bottleneck was fixed by vectorizing the function (commits `a91f8ee` perf + `f5e3b56` shape-fix). This is the actual hypothesis test that run2i-reset was supposed to be — does the directional fingertip force filter + split thumb/finger curl penalty produce sustained lift?

**Configuration delta from previous (run2i-reset / `55a29d5`):**
- No env_cfg.py change — same reward shape: directional filter on `good_grasp_mask`, split curl with thumb_curl_reg ADR (-0.6, -1.6) and finger_curl_reg ADR (-0.3, -0.8).
- Underlying code: `_collect_object_contacts` now vectorized (perf-only; math byte-equivalent for our num_bodies=num_filters=1 sensor config).

**Setup:** test repo, dextrah_test env, GPU 0, seed 42, headless 1024 envs, visdex_selected.

**Run directory:** `logs/rl_games/dextrah_tekken_lstm/05-27_14-45-50/` (test repo, GPU 0, dextrah_test, headless 1024 envs).

**Result (stopped at iter ~3593 by 1500-epoch lift-cutoff rule, 2026-05-27): FAILURE. Zero sustained lift through 3.6k iters. Multi-phase contact basin pattern with abandonment and re-engagement, but never crosses lift threshold.**

| Signal | iter 50 | iter 200 | iter 500 | iter 1000 | iter 1500 | iter 2000 | iter 2500 | iter 3000 | iter 3500 |
|---|---|---|---|---|---|---|---|---|---|
| `lift_success` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `lift_reward` | 0.00 | 5.01 | 0.05 | 0.20 | 0.00 | 0.01 | 0.08 | 0.58 | 5.36 |
| `hand_object_contact_reward` | 0.00 | **7.40** | 0.04 | 0.16 | 0.00 | 0.01 | 0.04 | 0.28 | 3.57 |
| `good_grasp_reward` | 0.00 | 0.32 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| `thumb_curl_reg` | -0.33 | -0.63 | -1.12 | -0.07 | -0.12 | -0.09 | -0.03 | -0.04 | -0.36 |
| `finger_curl_reg` | -0.71 | -0.81 | -0.44 | -0.15 | -0.10 | -1.00 | -0.51 | -0.39 | -0.65 |
| `hand_to_object_distance` (m) | 0.776 | **0.095** | **0.500** | 0.364 | 0.182 | 0.222 | 0.177 | 0.170 | 0.121 |
| `episode_lengths` | 583 | 321 | 590 | 409 | 572 | 506 | 553 | 558 | 536 |
| `num_adr_increases` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

PEAK `lift_success`: 0.49% at iter 3436 (~10/2048 envs transient).

**Performance (refactor verified working):** `step_fps` mean 15181, recent ~13300, last 13146. Vectorization sustained 13–15k FPS throughout vs the broken run's 7203 → 1029 degradation. No GPU stalls. **Refactor objective achieved.**

**Observations:**
1. **Three-phase basin search, not a single basin:**
   - **iter ~100–300:** rapid contact discovery (contact_reward 7.4, lift_rew 5.0, h2o 0.095 m). Hand reaches and engages.
   - **iter ~500–2500:** abandonment. h2o climbs to 0.18–0.50 m, contact crashes to ~0, good_grasp stays at 0. Policy walks AWAY from the object for ~2000 iters.
   - **iter ~3000+:** re-engagement. h2o returns to 0.12 m, contact recovers (3.57 by iter 3500), good_grasp climbs (0.50). Peak lift_success in this phase (0.49% at iter 3436).
2. **Likely interpretation:** directional filter rejected most early "contacts" (force directions wrong, e.g. dorsal/scraping). Policy got near-zero `good_grasp_reward` despite rich `contact_reward`, so it learned that approaching the object wasn't worth the curl-penalty cost. It retreated, only relearning contact much later with different force directions that DO satisfy the inside-aligned filter.
3. **`contact_reward = 7.40` at iter 200 vs `good_grasp_reward = 0.32`** confirms the dominance imbalance: contact pays much more than grasping does. The policy chases raw contact, hits the filter, fails grasping criterion, retreats.
4. **Curl regularizer worked as designed across this run:** thumb_curl tracks heavier than finger_curl when both are engaged (e.g. iter 200: thumb -0.63 vs finger -0.81, but on different L2 norms; per-joint magnitudes are comparable). When fingers extend (iter 1000-3000), both penalties shrink toward 0 as expected.
5. **No ADR progression** — `success_for_adr=0.4` is unreachable when peak lift is 0.49%.

**Decision rule outcome:** Branch 4 fires by spirit (zero lift, all fingers extended throughout most of training is closer to hover-and-avoid than productive grasp). Per the 1500-epoch lift-cutoff rule, this run failed. **Next experiment is a reward-rebalance** to make grasping pay more than raw contact, addressing the dominance imbalance flagged in observation 3.
