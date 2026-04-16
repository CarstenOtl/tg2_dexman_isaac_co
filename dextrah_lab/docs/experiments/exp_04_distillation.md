---
name: exp_04 distillation
description: Experiment 04 — SafeDagger distillation from multi-object teacher to vision-based student for FR3+AgileHand
type: project
---

# Experiment 04 — SafeDagger Distillation

**Goal:** Distill the multi-object teacher from exp_02 run1p into a vision-based student policy using SafeDagger.

## ⚠️ Teacher 11 baseline correction (2026-04-16)

Earlier evals used `eval_metrics_20260331_193003.json` (all 13 objects aggregated, no per-object breakdown) reporting Teacher 11 at **85.8% lift / 23.1% unsafe**. This was used as the reference in all comparison tables below, but it's **not a fair baseline** for students evaled on the 8-object `visdex_top8` subset (used from run7 onwards).

Re-eval on 2026-04-14 (`eval_metrics_20260414_031655.json`, per-object, 200 eps/object) shows:

| Subset | Lift | Total Unsafe | Real Unsafe (excl. physics) |
|--------|------|--------------|----------------------------|
| **All 13 objects** | **82.7%** | 22.3% | 11.2% |
| **visdex_top8 (fair vs students)** | **99.0%** | 22.4% | **12.2%** |
| 11 lifted (teacher-feasible) | 97.7% | — | 12.2% |
| Failed objects (`train`, `chicken_head_in_car`) | 0.0% | — | — |

**The fair Teacher 11 baseline for students evaled on visdex_top8 is 99.0% lift / 22.4% total unsafe / 12.2% real unsafe.** Compare student lift against 99.0%, not 85.8%.

Using this fair baseline, **no student surpasses the teacher on either lift or unsafe metric.** Best student (run14d at t=3.0) achieves 97.0% lift, still 2pp below teacher. Best safety (run14h at t=2.6) achieves 35.0% real unsafe, still 22.8pp worse than teacher.

Run14+ tables have been updated with corrected numbers. Older tables (run1-run11) reference the legacy 85.8% benchmark; interpret accordingly.

## Summary — All standalone student evaluations

**Teacher 11 benchmark (visdex_top8, fair baseline):** 99.0% lift, 22.4% total unsafe, 12.2% real unsafe
**Teacher 11 (all 13 legacy):** 85.8% lift, 23.1% unsafe (`eval_metrics_20260331_193003.json`)
**Teacher 10 benchmark** (80 episodes): 96.3% lift, 53.8% unsafe
**Best student: run14d (t=3.0) — 97.0% lift, 75.3% total unsafe, 38.8% real unsafe** — still 2pp below teacher on lift

| Run | Method | Loss | Teacher | Iters | Lift | Unsafe | Key Failure (% all eps) | Notes |
|---|---|---|---|---|---|---|---|---|
| run1b | SafeDagger | L2 | T10 | 100k | 0% | — | beta=0.875, student barely acted | — |
| run1c | SafeDagger | L2 | T10 | 350k | 36% | 67.5% | harmful_collision (27%) | — |
| run2a | DAgger | KL | T10 | 350k | 50% | 75% | harmful_collision (34%) | — |
| run2b | DAgger | KL | T10 | 700k | 0% | 100% | Collapsed | LR too high, no decay |
| run3a | SafeDagger | L2 | T11 | ~120k | 59% | 72.7% | physics (45%), collision (14%) | — |
| run3b | DAgger | KL | T11 | ~105k | 57.7% | 70.6% | physics (46%), collision (13%) | — |
| run4a | SafeDagger | L2 | T11 | 100k | 36.9% | 79.6% | physics (49%), collision (16%) | per-object logging |
| run4b | DAgger | KL | T11 | 100k | 52.1% | 44.8% | physics (24%), collision (7%) | **lowest unsafe** |
| **run5a** | **SafeDagger** | **L2** | **T11** | **100k** | **61.3%** | **75.4%** | physics (41%), collision (14%) | **best lift** |
| run5b | DAgger | KL | T11 | 100k | 49.2% | 58.5% | object_oob (18%), palm (17%) | real term logging |
| run6a | SafeDagger | L2 | T11 | 100k | — | — | — | scaled L2 threshold (0.665) |
| run6b | DAgger | L2 | T11 | 100k | — | — | — | AverageMeter logging |
| run7a | SafeDagger | L2 | T11 | 100k | 27.9% | 79.0% | physics (58%), collision (14%) | threshold=3.0, visdex_top8 |
| **run7b** | **DAgger** | **L2** | **T11** | **100k** | **56.9%** | **59.4%** | physics (39%), object_oob (11%) | visdex_top8, 3 envs/obj |
| run8a | SafeDagger | L2 | T11 | 100k | 41.7% | 70.0% | physics (44%), collision (11%) | threshold=2.0, top8, 10s eps |
| **run8b** | **DAgger** | **L2** | **T11** | **100k** | **55.2%** | **42.3%** | physics (21%), collision (12%) | top8, 10s eps |
| **run9a** | **SafeDagger** | **L2** | **T11** | **100k** | **72.7%** | **62.7%** | physics (32%), collision (16%) | **FIXED one-hot, top8, 10s, threshold=2.0** |
| **run9b** | **DAgger** | **L2** | **T11** | **100k** | **71.7%** | **58.3%** | physics (26%), object_oob (19%) | **FIXED one-hot, top8, 10s** |
| **run9c** | **BC** | **L2** | **T11** | **100k** | **87.8%** | **90.8%** | physics (49%), object_oob (21%), collision (19%) | **pure BC baseline ablation (32 envs)** |
| run10a | SafeDagger | L2 | T11 | 100k | 62.3% | 72.9% | physics (37%), collision (19%) | arm randomization ON |
| **run10b** | **SafeDagger** | **L2** | **T11** | **100k** | **79.2%** | **64.6%** | physics (27%), object_oob (15%) | **32 envs (4/obj), NEW BEST** |
| run11a | SafeDagger | L2 | T11 | 100k | 63.9% | 73.3% | physics (39%), collision (19%) | ADR 5, 32 envs |
| run11b | DAgger | L2 | T11 | 100k | 48.8% | 73.0% | collision (26%), object_oob (23%) | ADR 5, 32 envs |
| run12 | SafeDagger | L2 | **Tv2** | 100k | — | — | — | *running*, 32 envs, same config as run10b |
| **run14a** | **DAgger** | **L2** | **T11** | **100k** | **83.4%** | **68.3%** | object_oob (35%), physics (16%), collision (15%) | **32 envs DAgger baseline (β=0) — beats run10b SafeDagger lift** |
| **run14b** | **SafeDagger** | **L2** | **T11** | **100k** | **90.8%** | **80.2%** | physics (33%), collision (26%), object_oob (17%) | **threshold=2.2 — sharp recovery from the dip** |
| **run14c** | **SafeDagger** | **L2** | **T11** | **100k** | **87.3%** | **86.7%** | **palm flip (37%)**, object_oob (20%), physics (17%) | **threshold=2.4 — anomalous palm-flip dominant failure mode** |
| **run14h** | **SafeDagger** | **L2** | **T11** | **100k** | **88.8%** | **86.7%** | physics (52%), object_oob (18%), collision (13%) | **threshold=2.6 — palm-flip back to normal (4.5%)** |
| **run14i** | **SafeDagger** | **L2** | **T11** | **100k** | **82.0%** | **87.0%** | physics (45%), collision (19%), object_oob (17%) | **threshold=2.8 — lift drops back to DAgger-territory** |
| **run14d** | **SafeDagger** | **L2** | **T11** | **100k** | **🎯 97.0%** | **75.3%** | physics (37%), object_oob (20%), collision (15%) | **threshold=3.0 — NEW PEAK, best lift of entire sweep** |
| **run14j** | **SafeDagger** | **L2** | **T11** | **100k** | **89.5%** | **84.1%** | physics (45%), collision (22%), object_oob (15%) | **threshold=3.2 — drops from t=3.0 peak (spike is narrow)** |
| **run14k** | **SafeDagger** | **L2** | **T11** | **100k** | **🎯 95.3%** | **76.7%** | physics (35%), object_oob (21%), collision (13%) | **threshold=3.4 — high-lift region confirmed (not just t=3.0 spike)** |
| run14e,l,m | SafeDagger | L2 | T11 | 100k | — | — | — | *running/planned*, thresholds 3.6 / 3.8 / 4.0 |
| **run14f** | **SafeDagger** | **L2** | **T11** | **100k** | **90.9%** | **86.4%** | physics (48%), object_oob (23%), collision (14%) | **threshold=0.5 (β≈0.95) — slightly beats BC on both axes** |
| **run14g** | **SafeDagger** | **L2** | **T11** | **100k** | **79.4%** | **74.5%** | physics (37%), object_oob (20%), collision (11%), palm (7%) | **threshold=1.0 (β≈0.8) — landed in the dip region** |

## run14 — Safety threshold calibration ablation (2026-04-15)

**Goal:** Properly derive the SafeDagger unsafe_l2_threshold from the uncorrected DAgger L2 distribution, then validate via a threshold sweep at uniform 32-env config.

### Motivation

The original threshold calibration in run8a (`unsafe_l2_threshold=2.0`) was derived from **run7's L2 distribution** (run7 was SafeDagger with threshold=3.0). This is a partial circular reasoning issue: run7's L2 was already *suppressed* by teacher overrides — the high-L2 tail was cut off by intervention. Calibrating against this distribution underestimates the true L2 a fully-uncorrected student would experience.

**L2 distribution comparison (l2_loss_mean from existing TB exports):**

| Run | Method | Early L2 (mean / p80) | Late L2 (mean / p80) |
|------|--------|----------------------|---------------------|
| run9b | **DAgger (uncorrected, 24 envs)** | **3.46 / 3.95** | **2.05 / 2.37** |
| run10b | SafeDagger threshold=2.0 (32 envs) | 1.49 / 1.67 | 1.03 / 1.16 |
| run9c | BC, teacher always (32 envs) | 1.37 / 1.50 | 0.83 / 0.95 |

The DAgger L2 (true uncorrected baseline) is much higher than what was used for calibration. With threshold=2.0, run10b's late L2 is only 1.03 (mean) — meaning the threshold was almost never triggered late in training. Effectively, run10b ran like DAgger from ~30k onwards, which is consistent with the small 8pp gap between run10b (79.2%) and run9b (71.7%).

### Calibration logic

**Philosophy: DAgger gives us a principled range to sweep, not a prescriptive threshold.**

The threshold should be expressed in units of typical student L2 loss when acting alone. The reference distribution must come from a method where the student's L2 is *not* artificially suppressed by interventions — that is, **DAgger**. Running DAgger once (run14a) gives us an **objective scale** on which to pick meaningful threshold values.

**What DAgger's L2 distribution lets us conclude:**

| Claim | Supported by DAgger L2? |
|-------|------------------------|
| A **ballpark range** of meaningful thresholds (roughly 2 to 4) | ✅ Yes |
| **Relative ordering**: higher threshold → more DAgger-like, lower → more BC-like | ✅ Yes |
| Eliminating values that are clearly **outside** the useful range (<1 or >5) | ✅ Yes |
| The **single optimal threshold** for this task | ❌ No — requires empirical validation |
| **Exact β** per threshold (we only have per-step mean L2, not per-env distribution) | ❌ No — approximate at best |
| Whether the SafeDagger paper's 20–30% early target is actually best for this task | ❌ No — task differs from the paper's autonomous driving domain |

**The paper's ~20–30% early intervention recipe is a hypothesis, not a conclusion.** Translating it to an exact threshold has two layers of ambiguity:
1. **Temporal**: "early" is loosely defined — somewhere between iter 0 (student is random, β→1 regardless) and ~20% through training (student is actively learning, β decays steeply)
2. **Distributional**: `l2_loss_mean` is averaged across 32 envs per step; the threshold check is on per-env L2. Without per-env variance data, the mapping from "threshold X" to "β = Y%" is approximate.

**Conclusion: use DAgger L2 as a prior to scope the sweep, then validate empirically.**

The DAgger L2 percentiles suggest testing thresholds roughly spanning the late-p50 to early-p90 range (≈2.0 to 4.0). Within that range, we pick 4 equally-spaced values and run the sweep. The winning threshold tells the story — we report it *with its observed early-β* as a post-hoc data point, without pre-committing to a specific paper-recipe interpretation.

### Step 1 — run14a: DAgger baseline at 32 envs

Why this is needed: we don't currently have a 32-env DAgger + L2 + ADR-off run. Closest is run9b (24 envs) and run11b (32 envs but +ADR 5). For uniform comparison with run10b and the threshold sweep, all calibration data must come from the same 32-env / no-ADR config.

```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --vanilla_dagger \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directory:** `runs/dextrah-fr3-agilehand-dagger-stereo-transformer_15-12-11-49/`

**Status (2026-04-15):** ✅ complete.

**Standalone eval (640 episodes = 20 rollouts × 32 envs):**

| Metric | run14a DAgger (β=0) | run10b SafeD (t=2.0) | run9c BC (β=1) | Teacher 11 (fair TOP8) |
|---|---|---|---|---|
| **Lift success** | **83.4%** | 79.2% | 87.8% | **99.0%** |
| **Unsafe rate** | **68.3%** | 64.6% | 90.8% | 23.1% |
| Physics (% all eps) | 16.1% | 17.5% | 49.4% | 11.0% |
| Object OOB (% all eps) | **35.0%** | 9.4% | 19.1% | 9.4% |
| Collision (% all eps) | 15.3% | 8.5% | 19.3% | 1.7% |
| Palm flipped (% all eps) | 1.8% | 5.9% | 1.2% | 0.6% |

**Eval JSON:** `eval_results/eval_metrics_20260415_170358.json`

**Key findings:**
- **DAgger outperforms run10b SafeDagger by +4.2pp lift** (83.4% vs 79.2%) with only +3.7pp more unsafe. This surprised us — more evidence that run10b's threshold=2.0 was *more aggressive than we assumed*.
- **Failure mode shift is diagnostic:**
  - DAgger: **object OOB dominates (35.0% of all eps)** — student learned aggressive lift motion but can't stabilize / knows when to stop
  - SafeDagger run10b: physics dominates (17.5%) — student's state was kept close to teacher via interventions, but when it did fail it failed hard
  - BC: physics catastrophically dominant (49.4%) — student has no stability training at all
- **This confirms the BC → SafeDagger → DAgger spectrum**: as β decreases (less teacher intervention), physics instability drops but object OOB rises. DAgger trades one failure mode for another.

### Calibration — L2 percentiles from run14a

Extracted via tbparse from the run14a TB summaries (99979 logged `l2_loss_mean` values, step range 672 to 3.2M):

| Phase | mean | p50 | p80 | p90 | p95 | p99 |
|-------|------|-----|-----|-----|-----|-----|
| very_early (first 5%) | 3.66 | 3.30 | 4.03 | 4.75 | 6.69 | 11.10 |
| **early (first 10%)** | **3.20** | **2.97** | **3.57** | 4.03 | 4.75 | 7.97 |
| mid (40-60%) | 2.09 | 2.07 | 2.37 | 2.55 | 2.74 | 3.07 |
| **late (last 10%)** | **1.93** | **1.90** | **2.21** | 2.38 | 2.53 | 2.86 |
| very_late (last 5%) | 1.93 | 1.92 | 2.20 | 2.36 | 2.49 | 2.80 |
| overall | 2.24 | 2.16 | 2.59 | 2.90 | 3.19 | 4.04 |

**Critical insight: previous calibration was OFF.**

The original threshold=2.0 was derived from run7's L2 distribution (itself SafeDagger-suppressed). We assumed threshold=2.0 gave ~5% late intervention. Real DAgger data shows late p50=1.90, p80=2.21 — meaning threshold=2.0 triggers on ~50% of late-training steps, not 5%. The previous calibration was ~10x too aggressive.

This explains:
- Why run10b (threshold=2.0) has lower lift than DAgger — it was intervening far more than intended, pushing toward BC-like state distributions
- Why the BC→SafeDagger→DAgger story has a gradient instead of a clean separation — threshold=2.0 was partially collapsing into BC territory

### After run14a finishes, extract `l2_loss_per_env` percentiles from TensorBoard (via tbparse) at:
- Early phase: first 10% of training (steps 0 – ~320k frames)
- Late phase: last 10% of training (steps ~2.88M – 3.2M frames)

### Step 2 — run14b/c/d/e: threshold sweep (planned)

**Dense sweep (step 0.2) within the DAgger-derived meaningful range (2026-04-15):**

11 thresholds spanning 2.0 → 4.0, giving enough resolution for a smooth lift-vs-threshold plot.

| Run | Threshold | Where it falls in run14a L2 distribution | Rough β prediction (to be validated) |
|------|-----------|-------------------------------------------|--------------------------------------|
| run10b | 2.0 | below late p80 (2.21) | Moderate-high intervention (β≈0.5 late, confirmed) |
| run14b | **2.2** | ≈ late p80 (2.21) | Constant-ish intervention across training |
| run14c | **2.4** | between late p80 (2.21) and late p95 (2.53) | Moderate early, low late |
| run14h | **2.6** | ≈ late p95 (2.53) | Moderate-low intervention |
| run14i | **2.8** | ≈ mid p99 (3.07) / early p50 (2.97) | Low-moderate intervention |
| run14d | **3.0** | ≈ early p50 (2.97) | Aggressive early, near-zero late |
| run14j | **3.2** | between early p50 and p80 | Low intervention |
| run14k | **3.4** | between early p50 and p80 | Low intervention |
| run14e | **3.6** | ≈ early p80 (3.57) | Very low — closest approximation of paper's 20% early target |
| run14l | **3.8** | just above early p80 | Very low intervention |
| run14m | **4.0** | ≈ early p90 (4.03) | Near-DAgger — catch only catastrophic envs |

Plus existing endpoints (BC, t=0.5, t=1.0, DAgger=∞).

**These are hypotheses about β, not guarantees.** The actual observed β per run is logged to TensorBoard (`beta` scalar). Post-sweep we cross-reference winning threshold against its observed β and compare to paper's recipe in the thesis discussion.

Together with existing endpoints this produces an 8-point curve:

| β regime | Run | Threshold | Lift / Unsafe | Status |
|----------|-----|-----------|---------------|--------|
| β=1 (BC) | run9c | — | 87.8% / 90.8% | ✅ done |
| β≈0.95 | run14f | 0.5 | 90.9% / 86.4% | ✅ done (beats BC slightly) |
| β≈0.8 | run14g | 1.0 | **79.4% / 74.5%** | ✅ done (in the dip) |
| β≈0.5 (mid) | run10b | 2.0 | 79.2% / 64.6% | ✅ done (previously miscalibrated) |
| β≈0.3 | **run14b** | **2.2** | **90.8% / 80.2%** | ✅ done (sharp recovery from dip) |
| β≈0.2 | **run14c** | **2.4** | **87.3% / 86.7%** | ✅ done (palm-flip anomaly) |
| β≈0.15 | **run14h** | **2.6** | **88.8% / 86.7%** | ✅ done (palm-flip resolved) |
| β≈0.1 | **run14i** | **2.8** | **82.0% / 87.0%** | ✅ done (lift drops) |
| β≈0.07 | **run14d** | **3.0** | **🎯 97.0% / 75.3%** | ✅ done — **BEST LIFT** |
| β≈0.05 | **run14j** | **3.2** | **89.5% / 84.1%** | ✅ done (post-peak decline) |
| β≈0.03 | **run14k** | **3.4** | **🎯 95.3% / 76.7%** | ✅ done — **second high-lift confirmation** |
| β≈0.02 | **run14e** | **3.6** | **90.6% / 82.0%** | ✅ done (transitioning down from 3.4 peak) |
| β≈0.01 | **run14l** | **3.8** | **85.9% / 71.6%** | ✅ done (lift drops toward DAgger level) |
| β≈0.005 (?) | **run14m** | **4.0** | — | ⏳ running (launched 17:40 after 11:27 run was accidentally t=3.4) |
| β=0 (DAgger) | run14a | ∞ | 83.4% / 68.3% | ✅ done |

**Non-monotonic curve confirmed (after 5 data points).** Sweeping β from 1 → 0:

```
β:    1.0 →  0.95 →  0.8  →  0.5  →  0
lift: 87.8 → 90.9  → 79.4 → 79.2 → 83.4
                      ↑                ↑
                   dip starts       recovery
```

**Key finding: U-shaped tradeoff.** Very high intervention (β>0.9, BC-like) and zero intervention (β=0, DAgger) both outperform the middle range. The worst performance is at β≈0.5–0.8 (thresholds 1.0–2.0), where the student alternates between teacher-driven and self-driven state distributions without fully learning either — a form of **destructive interference**.

**Thesis implication:** The SafeDagger paper's canonical ~20% early intervention recipe (corresponding to thresholds ~3.0-3.6 in our data, still pending) is *near* the DAgger regime, which is fine. But the naïve "just tune the threshold" approach lands many people in the dip. The dense sweep (14b–m in 2.0–4.0 range) will tell us where the recovery boundary actually is and how wide the dip region is.

### run14f — SafeDagger threshold=0.5 (low-end anchor, parallel)

Bonus run launched in parallel on GPU 1 to anchor the low end of the threshold sweep with a "borderline-BC" datapoint. Threshold=0.5 is below even the BC late L2 mean (0.83), so β should saturate near 1.0 throughout — meaning the teacher overrides essentially every step. If results are indistinguishable from run9c BC, that confirms "very low threshold ≈ BC". If they differ, even rare student steps still meaningfully shape learning.

```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 0.5 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directory:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_15-12-22-01/`

**Status (2026-04-15):** ✅ complete.

**Standalone eval (640 episodes = 20 rollouts × 32 envs):**

| Metric | run14f (t=0.5, β≈0.95) | run9c BC (β=1) | Teacher 11 (fair TOP8) |
|---|---|---|---|
| **Lift success** | **90.9%** | 87.8% | **99.0%** |
| **Unsafe rate** | **86.4%** | 90.8% | 23.1% |
| Physics (% all eps) | 47.8% | 49.4% | 11.0% |
| Object OOB (% all eps) | 22.6% | 19.1% | 9.4% |
| Collision (% all eps) | 13.7% | 19.3% | 1.7% |
| Palm flipped (% all eps) | 2.2% | 1.2% | 0.6% |

**Eval JSON:** `eval_results/eval_metrics_20260415_171638.json`

**Key finding: rare student practice beats pure BC.**

Even with teacher overriding ~95% of steps, the ~5% of remaining student-driven steps yield **+3.1pp lift and −4.4pp unsafe vs pure BC**. This resolves the ambiguity posed in the motivation: pure BC is NOT equivalent to very-low-threshold SafeDagger — the tiny amount of student-distribution gradient signal matters. This is consistent with the DAgger distribution-shift-correction argument: even sparse samples from the student's own action distribution meaningfully improve the policy.

Also notable: run14f's **lift (90.9%) is below Teacher 11's fair TOP8 baseline (99.0%)** — the earlier claim "student exceeds teacher on lift" was based on a wrong baseline (using the all-13 eval including 2 unliftable objects). With correct fair-baseline comparison, no student exceeds the teacher on lift.

### run14g — SafeDagger threshold=1.0 (high-intervention regime, parallel)

Third parallel run on GPU 2 to fill the largest gap in the sweep — between threshold=0.5 (β≈100%) and threshold=2.0 (β≈80% early, ~5% late). At threshold=1.0, expected β ≈ 95% early and ~50% late (since DAgger late L2 mean is ~2.0 and BC late L2 mean is ~0.83 — most envs early in training would exceed 1.0 but stabilize around it late). This probes the "moderate-to-high intervention" regime where SafeDagger should genuinely shape learning differently from both BC and DAgger.

**Standalone eval (640 episodes = 20 rollouts × 32 envs, 2026-04-15):**

| Metric | run14g (t=1.0, β≈0.8) | run14f (t=0.5, β≈0.95) | run10b (t=2.0, β≈0.5) |
|---|---|---|---|
| **Lift success** | **79.4%** | 90.9% | 79.2% |
| **Unsafe rate** | **74.5%** | 86.4% | 64.6% |
| Physics (% all eps) | 36.6% | 47.8% | 17.5% |
| Object OOB (% all eps) | 20.1% | 22.6% | 9.4% |
| Collision (% all eps) | 11.3% | 13.7% | 8.5% |
| Palm flipped (% all eps) | 6.6% | 2.2% | 5.9% |

**Eval JSON:** `eval_results/eval_metrics_20260415_201147.json`

**Key finding: threshold=1.0 is the start of the dip region.**

Lift drops sharply from 90.9% (t=0.5) to 79.4% (t=1.0) — an 11.5pp drop for a small threshold change. This is strong evidence that **moderate-intervention SafeDagger suffers from destructive interference**: the teacher overrides enough of the student's actions that the student can't build coherent self-distribution experience, but not enough to give pure BC's clean imitation signal.

Between t=1.0 (79.4%) and t=2.0 (79.2%) the lift is essentially flat — the entire β ≈ 0.5–0.8 range is a "dead zone" for lift performance. Recovery happens only at β→0 (DAgger, 83.4%) or β→1 (BC, 87.8%).

This motivates the dense sweep (2.0 to 4.0, step 0.2) — we need to identify *where* lift recovers to tell if it's a gradual transition or a sharp boundary.

```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=2 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 1.0 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directory:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_15-14-48-42/`

**Status (2026-04-15):** *running* on GPU 2, started 14:48.

### run14b — SafeDagger threshold=2.2 (late p80)

Constant-ish safety net — threshold sits at the 80th percentile of late-training L2 mean, so intervention rate should be roughly stable across training rather than decaying steeply.

**Run directory:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_15-17-51-09/`

**Status (2026-04-16):** ✅ complete.

**Standalone eval (640 episodes = 20 rollouts × 32 envs):**

| Metric | run14b (t=2.2) | run10b (t=2.0, dip) | run14g (t=1.0, dip) | run14a DAgger (∞) |
|---|---|---|---|---|
| **Lift success** | **90.8%** | 79.2% | 79.4% | 83.4% |
| **Unsafe rate** | **80.2%** | 64.6% | 74.5% | 68.3% |
| Physics (% all eps) | 32.8% | 17.5% | 36.6% | 16.1% |
| **Object OOB (% all eps)** | 17.5% | 9.4% | 20.1% | **35.0%** |
| **Collision (% all eps)** | **25.8%** | 8.5% | 11.3% | 15.3% |
| Palm flipped (% all eps) | 4.1% | 5.9% | 6.6% | 1.8% |

**Eval JSON:** `eval_results/eval_metrics_20260416_011300.json`

**Key finding: the dip is narrow — recovery is sharp at t=2.2.**

Lift jumps from 79.2% (t=2.0) to 90.8% (t=2.2) — an 11.6pp increase from a 0.2 threshold change. The dip region is narrower than initially suspected: it spans roughly t∈[1.0, 2.0] (β≈0.5–0.8) and recovers sharply by t=2.2.

But the recovery comes with a tradeoff:
- **Lift recovers** to 90.8% (BC-comparable)
- **Unsafe rate worsens** from 64.6% → 80.2% — student is more aggressive without enough teacher correction
- **Collision dominates** (25.8% of all eps) — t=2.2 student grabs aggressively, hits things

This suggests the 2.0 → 2.2 transition is a regime shift: at t=2.0 the teacher intervenes enough to keep the student conservative (low lift, low unsafe). At t=2.2 the threshold relaxes just enough that the student becomes confident-but-aggressive — high lift but high collision rate. The dense sweep (14c/h/i and beyond) will tell us if there's a sweet spot in between or if it's a hard regime boundary.

### run14c — SafeDagger threshold=2.4 (between late p80 and p95)

Slightly less aggressive than 2.2 — threshold falls between late p80 (2.21) and late p95 (2.53). Expected β decays moderately across training.

**Run directory:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_15-17-52-23/`

**Status (2026-04-16):** ✅ complete.

**Standalone eval (640 episodes = 20 rollouts × 32 envs):**

| Metric | run14c (t=2.4) | run14b (t=2.2) | run10b (t=2.0) | run14a DAgger (∞) |
|---|---|---|---|---|
| **Lift success** | **87.3%** | 90.8% | 79.2% | 83.4% |
| **Unsafe rate** | **86.7%** | 80.2% | 64.6% | 68.3% |
| Physics (% all eps) | 17.3% | 32.8% | 17.5% | 16.1% |
| Object OOB (% all eps) | 20.1% | 17.5% | 9.4% | 35.0% |
| Collision (% all eps) | 11.7% | 25.8% | 8.5% | 15.3% |
| **Palm flipped (% all eps)** | **37.5%** | 4.1% | 5.9% | 1.8% |

**Eval JSON:** `eval_results/eval_metrics_20260416_011406.json`

**Key finding: palm-flip emerges as the dominant failure mode at t=2.4 — anomalous and threshold-specific.**

In every other run we've measured, palm flipping is a marginal failure mode (1-7% of all episodes). At t=2.4 it explodes to **37.5% of all episodes** — by far the dominant failure. This is a regime-specific anomaly worth investigating further.

**Hypothesis for the palm-flip anomaly:** At t=2.4, the threshold sits in a narrow band where the teacher overrides occasionally during early grasp formation but not consistently. The student commits to an off-axis grasp orientation (palm rotated wrong) and proceeds to lift — by the time the orientation error compounds to detectable levels, the student has already lifted enough that the palm-flipped fault triggers. At lower thresholds (2.0, 2.2) the teacher catches this earlier; at higher thresholds (DAgger), the student presumably learns to correct palm orientation independently.

**Worth verifying:** run14h (t=2.6) and run14i (t=2.8) — does palm-flip persist into the 2.5–3.0 range, or is it specifically a t=2.4 quirk?

**Re-eval (2026-04-16, JSON `20260416_183138`) confirms palm-flip anomaly is NOT noise:**

| Metric | Eval 1 (01:14) | Eval 2 (18:31) | Diff |
|--------|---------------|---------------|------|
| Lift | 87.3% | 84.5% | -2.8pp |
| Unsafe | 86.7% | 87.3% | +0.6pp |
| **Palm flip (% unsafe)** | **37.5%** | **35.8%** | **-1.7pp** |

Palm-flip stays 35%+ of unsafe across both evals. This establishes that t=2.4 learned a specific palm-orientation failure mode, distinct from adjacent thresholds (2.2 physics/collision dominant, 2.6 physics dominant). **Threshold choice produces qualitatively different policies, not just different success rates.**

Also provides a rough eval noise floor: ~3pp lift, ~1pp unsafe between identical re-evals. Differences smaller than this in the sweep cannot be distinguished from eval noise without more seeds.

**Lift curve update:** With t=2.4 added, the lift curve is now oscillating, not monotonic or U-shaped:
```
β:     1.0    0.95   0.8    0.5    0.3    0.2    0
t:     BC     0.5    1.0    2.0    2.2    2.4    ∞
lift:  87.8 → 90.9 → 79.4 → 79.2 → 90.8 → 87.3 → 83.4
       peak   peak   dip    dip    peak   dip2   recover
```

Different thresholds produce qualitatively different policies — the relationship between threshold and lift is not smooth.

### run14h — SafeDagger threshold=2.6 (≈ late p95, densification)

Fills the 2.4–3.0 gap. Threshold ≈ late p95 (2.53) → catches the worst 5% of late-training envs, decays to near-zero intervention late.

**Run directory:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_15-17-59-06/`

**Status (2026-04-16):** ✅ complete.

**Standalone eval (640 episodes = 20 rollouts × 32 envs):**

| Metric | run14h (t=2.6) | run14c (t=2.4) | run14b (t=2.2) |
|---|---|---|---|
| **Lift success** | **88.8%** | 87.3% | 90.8% |
| **Unsafe rate** | **86.7%** | 86.7% | 80.2% |
| Physics (% all eps) | 51.7% | 17.3% | 32.8% |
| Object OOB (% all eps) | 18.0% | 20.1% | 17.5% |
| Collision (% all eps) | 12.5% | 11.7% | 25.8% |
| **Palm flipped (% all eps)** | **4.5%** | **37.5%** | 4.1% |

**Eval JSON:** `eval_results/eval_metrics_20260416_011458.json`

**Key finding: palm-flip anomaly was specific to t=2.4.**

At t=2.6, palm-flip drops back to 3.9% (matching every run except t=2.4). This confirms the t=2.4 palm-flip dominance was a regime-specific artifact, not a general trend across the 2.4-2.6 range. **The threshold sweep reveals discrete policy regimes, not a smooth gradient** — within a 0.2 threshold step the dominant failure mode flipped from palm flip back to physics.

The lift / unsafe values at t=2.6 (88.8% / 86.7%) are very close to t=2.4 (87.3% / 86.7%) — both are in a "high lift but high unsafe" regime where the student is competent at grasping but lacks safety margin. The β≈0.15-0.2 region appears to be a relatively flat plateau in lift, with physics instability rising back into dominance.

### run14d,e,i–m — Remaining sweep runs (2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0)

Launched in batches as GPU capacity allows. All use identical settings to run14b (32 envs, visdex_top8, 10s eps, ADR off). Same launch template, only `--unsafe_l2_threshold` differs.

**Launch template:**
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=<N> /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold <T> \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

| Run | Threshold | Status | Run directory |
|-----|-----------|--------|---------------|
| run14i | 2.8 | ✅ **82.0% / 87.0%** (eval JSON `20260416_102642`) | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_15-20-50-53/` |
| run14d | 3.0 | ✅ **🎯 97.0% / 75.3%** (eval JSON `20260416_102648`) — best of sweep | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-01-46-05/` |
| run14j | 3.2 | ✅ **89.5% / 84.1%** (eval JSON `20260416_102808`) | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-01-46-49/` |
| run14k | 3.4 | ✅ **🎯 95.3% / 76.7%** (eval JSON `20260416_102733`) — high-lift region | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-01-47-45/` |
| run14e | 3.6 | ✅ **90.6% / 82.0%** (eval JSON `20260416_175222`) | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-11-25-07/` |
| run14l | 3.8 | ✅ **85.9% / 71.6%** (eval JSON `20260416_175236`) — drops toward DAgger | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-11-26-21/` |
| run14k-seed2 | **3.4** (second seed, originally mislogged as 4.0) | *done*, started 11:27 | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-11-27-33/` |
| run14m | 4.0 | *running*, started 17:40 | `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_16-17-40-20/` |

### Step 3 — analysis

**Primary plot:** lift success and unsafe episode rate vs. threshold, with BC (run9c, β=1) and DAgger (run14a, β=0) as continuous-axis endpoints. Reveals the shape of the tradeoff curve — is there a clear optimum, or monotonic gradient from BC to DAgger?

**Secondary plots:**
- Observed training β per threshold vs. training iteration — validates (or disputes) our β predictions
- Failure-mode breakdown per threshold — does the BC-physics / DAgger-OOB tradeoff hold continuously?
- Winning threshold's observed early β vs. SafeDagger paper's 20–30% target — post-hoc comparison

**Thesis story:**
*"We avoided the circular reasoning of prior threshold calibrations by deriving the sweep range from the uncorrected DAgger L2 distribution (run14a). The DAgger baseline gives an objective scale on which to pick meaningful threshold values but does not prescribe an optimum — the SafeDagger paper's 20–30% early-intervention recipe is itself a rough target that doesn't translate cleanly to a single numeric threshold (temporal ambiguity in 'early', distributional ambiguity between per-step mean and per-env L2). We therefore sweep four thresholds ({2.2, 2.5, 3.0, 3.6}) evenly distributed within the meaningful range, validate empirically, and report where the winning threshold falls relative to DAgger percentiles and the paper's recipe as a post-hoc observation."*

## run12 — Teacher v2 distillation (2026-04-13)

**Motivation:** All previous distillation runs used Teacher 11 (85.8% lift, ADR 14, unrealistic starting limits). Teacher v2 (79.1% lift, ADR 13, hardware-realistic constraints from step 0) may produce a student that transfers better to real hardware despite lower sim performance. This run isolates the teacher variable — identical distillation config to run10b (best student, 79.2% lift).

**Settings:** Same as run10b: SafeDagger + L2, 32 envs, threshold=2.0, visdex_top8, 10s episodes, no ADR, arm randomization OFF.

```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/12_teacher_v2_hw_realistic_adr13_04-06_12-49-15/nn/dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 2.0 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directory:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_13-11-07-04/`

**Status (2026-04-13):** *Running* on GPU 0.

**Key question:** Does Teacher v2's ~7pp lift gap propagate linearly to the student, or does the more realistic teacher produce a proportionally better student for sim2real?

## run9c — Pure Behavior Cloning ablation (2026-04-13)

**Role in thesis:** Third baseline method in the distillation comparison (SafeDagger/run9a vs DAgger/run9b vs BC/run9c). Forms the minimal imitation baseline — no online distribution correction.

**Motivation:** All previous distillation runs use online imitation learning (DAgger/SafeDagger) where the student's actions influence the state distribution. Pure BC is the simplest baseline — teacher always drives the environment, student only learns the observation→action mapping from teacher-generated trajectories. This ablation quantifies how much DAgger's distribution correction contributes vs. the teacher supervision signal alone.

**Settings:** Same as run10b config (32 envs, ADR 0, visdex_top8, 10s episodes) with `--bc` flag. Teacher always steps the environment; student computes forward passes and loss but never acts. Note: uses 32 envs vs run9a/9b's 24 envs — direct comparison to run10b (SafeDagger 32 envs: 79.2% lift) is cleanest, but the run9a/9b pairing remains valid since method is the primary variable.

**Key differences from DAgger/SafeDagger:**
- `step_student_actions=False` — teacher actions are always sent to `env.step()`
- `disable_unsafe_override=True` — no SafeDagger intervention (irrelevant since teacher always steps)
- `beta=1.0` throughout (teacher acts in all envs)
- Training-time `lift_success`/`unsafe_rate` reflect teacher performance, not student — use `eval_student.py` for true student metrics
- All other TensorBoard metrics identical (imitation_loss, l2_loss_mean, per-object metrics, termination breakdown)

**Run 9c — Pure BC (32 envs, ADR 0):**
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --bc \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directory:** `runs/dextrah-fr3-agilehand-bc-stereo-transformer_13-18-09-23/`

**Standalone eval (640 episodes = 20 rollouts × 32 envs, 2026-04-13):**

| Metric | run9c BC | run9a SafeD | run9b DAgger | run10b SafeD (32 envs) | Teacher 11 (fair TOP8) |
|---|---|---|---|---|---|
| **Lift success** | **87.8%** | 72.7% | 71.7% | 79.2% | **99.0%** |
| **Unsafe rate** | **90.8%** | 62.7% | 58.3% | 64.6% | 23.1% |
| Physics (% all eps) | 49.0% | 32.4% | 25.8% | 27.1% | 11.0% |
| Object OOB (% all eps) | 21.1% | 10.3% | 18.7% | 14.6% | 9.4% |
| Collision (% all eps) | 19.3% | 15.9% | 11.0% | 13.1% | 1.7% |
| Palm flipped (% all eps) | 1.3% | 4.1% | 2.8% | 9.2% | 0.6% |

**Eval JSON:** `eval_results/eval_metrics_20260413_233515.json`

**Key findings:**
- **BC achieves highest lift of any student** (87.8%) — surpasses SafeDagger (run10b) by 8.6pp, trails Teacher 11 (fair TOP8: 99.0%) by 11.2pp
- **BUT unsafe rate is catastrophically high** (90.8% total / 41.7% real) — 26pp worse than run10b SafeDagger, ~3.4× teacher's real unsafe (12.2%)
- Student learns the teacher's *grasping* well but not the *safety behavior* — during training, teacher always keeps trajectories in-distribution, so student never learns to recover from drift
- Classic BC failure mode: high in-distribution accuracy, poor out-of-distribution robustness (covariate shift)
- Physics instability dominates failures (49% of all episodes) — student's aggressive actions when slightly off-distribution cause sim instability
- **Lift numbers are misleading** — many lifts succeed briefly before the episode terminates unsafely

**Interpretation for thesis:** BC achieves high apparent success but cannot handle out-of-distribution states. DAgger's online correction (9a/9b) sacrifices peak lift performance for robustness. SafeDagger with more envs (run10b) hits the sweet spot. The 26pp unsafe gap between BC (9c) and SafeDagger (10b) quantifies the empirical value of online distribution correction for this task. This trio (9a/9b/9c) forms the clean ablation: all use identical env, teacher, loss, iterations, objects — only the stepping strategy differs.

**Checkpoint sweep — overfitting analysis (640 eps / ckpt, 2026-04-14):**

| Iter | Lift | Unsafe | Physics | Collision | Obj OOB | Palm flip |
|---|---|---|---|---|---|---|
| 10k  | 71.9% | 95.3% | 45.9% | 13.4% | 12.5% | **23.4%** |
| 25k  | 86.9% | 86.6% | 39.4% | 13.9% | 23.4% | 9.8% |
| **50k**  | **95.6%** | 91.4% | 52.3% | 19.4% | 16.6% | 3.1% |
| **75k**  | 95.2% | 87.7% | 51.9% | 14.4% | 14.8% | 6.6% |
| 100k | 85.6% | 91.7% | 51.6% | 20.6% | 18.0% | 1.6% |

**Eval JSONs:** `eval_results/eval_metrics_20260414_{015240,022529,025853,033230,040714}.json` (iters 10k, 25k, 50k, 75k, 100k)

**Overfitting confirmed:**
- **Lift peaks at 50k (95.6%) and degrades to 85.6% at 100k** — a 10pp drop, classic overfitting signature
- **75k (95.2%) plateaus with 50k** — no additional learning, just noise around the peak
- **Peak BC (50k) trails Teacher 11** (95.6% vs fair TOP8 99.0%) — the earlier claim that BC exceeds teacher was based on the wrong all-13 baseline (85.8%). With the correct fair baseline, BC peak is 3.4pp below teacher

**Safety does not improve with training:**
- **Unsafe rate flat across all iters** (86-95%) — BC cannot learn safety regardless of training time
- **Physics instability rises** (45.9% → 51.6%) — late BC is more aggressive, causes more sim-level instabilities
- **Palm-flip drops monotonically** (23.4% → 1.6%) — the *only* thing BC robustly learns is hand orientation, but this gain is offset by increased physics failures

**Refined thesis narrative:** *"The checkpoint sweep reveals that BC peaks in lift success at 50k iterations (95.6%), before degrading to 85.6% at 100k — a 10pp drop consistent with overfitting. At its peak, BC's lift (95.6%) is 3.4pp below Teacher 11 on the fair TOP8 baseline (99.0%), and its unsafe rate remains catastrophically high across all iterations (87–95% total, 41.7% real unsafe at 100k), compared to SafeDagger's 62.7% (run9a) and 64.6% (run10b) total unsafe. This decouples two phenomena: BC learns in-distribution action prediction efficiently, but cannot acquire the corrective behavior needed for episode-level safety. More training makes BC more aggressive without making it safer, eventually harming in-distribution performance as well. This motivates reporting the 100k-iter BC checkpoint as the fair thesis comparison (no cherry-picking)."*

**Sweep command (reproducible):**
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/distillation_new
RUN=runs/dextrah-fr3-agilehand-bc-stereo-transformer_13-18-09-23
for iters in 10000 25000 50000 75000 100000; do
  CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python eval_student.py \
    --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
    --checkpoint $RUN/nn/dextrah_student_${iters}_iters.pth \
    --num_episodes 20 \
    env.distillation=True env.simulate_stereo=True \
    env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
    env.teacher_objects_dir=multi_objects/visdex_selected \
    env.distillation_episode_length_s=10.0
done
```

## run11a/11b — SafeDagger vs DAgger at ADR 5 (2026-04-06)

**Motivation:** All previous distillation runs used `enable_adr=False` (ADR 0). The teacher was trained to ADR 14, so it performs well at ADR 5. Running at ADR 5 adds moderate physics randomization (friction, mass, stiffness) during distillation — student sees diverse physical conditions while teacher still provides good supervision. This should improve sim2real robustness without degrading teacher quality.

**Settings:** Same as run10b (32 envs, best config) but with `enable_adr=True starting_adr_increments=5`.

**Run 11a — SafeDagger + ADR 5:**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 2.0 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=True env.starting_adr_increments=5 env.disable_arm_randomization=True
```

**Run 11b — DAgger + ADR 5:**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 32 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 \
  --vanilla_dagger \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=True env.starting_adr_increments=5 env.disable_arm_randomization=True
```

**Run directories:**
- run11a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_06-14-00-34/`
- run11b (DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_06-14-01-16/`

**Status (2026-04-06):** Both running. SafeDagger (11a) started 14:00 on GPU 0, DAgger (11b) started 14:01 on GPU 1.

**Standalone eval (640 episodes = 20 rollouts × 32 envs, 2026-04-07):**

| Metric | run11a SafeD+ADR5 | run11b DAgger+ADR5 | run10b (no ADR) | Teacher 11 (fair TOP8) |
|---|---|---|---|---|
| **Lift success** | 63.9% | 48.8% | **79.2%** | **99.0%** |
| **Unsafe rate** | 73.3% | 73.0% | 64.6% | 23.1% |
| Physics (% all eps) | 39.2% | 19.9% | 27.1% | 11.0% |
| Collision (% all eps) | 19.4% | 25.9% | 13.1% | 1.7% |
| Object OOB (% all eps) | 12.7% | 23.3% | 14.6% | 9.4% |
| Palm flipped (% all eps) | 2.1% | 3.9% | 9.2% | 0.6% |

**Eval JSONs:**
- run11a: `eval_results/eval_metrics_20260407_084953.json`
- run11b: `eval_results/eval_metrics_20260407_084926.json`

**Key findings:**
- **ADR 5 hurts both methods at 100k iters** — SafeDagger -15.3pp (79.2→63.9%), DAgger -22.9pp (71.7→48.8%)
- **SafeDagger more robust to physics randomization** — drops less than DAgger, smaller relative degradation
- DAgger's collision rate jumped (12.5→25.9%) — student can't compensate for novel physics
- Both methods need longer training to learn ADR-randomized dynamics
- ADR during distillation may help sim2real but trades off sim eval performance

## run10a/10b — Ablation: arm randomization + 32 envs (2026-04-05)

**Motivation:** run9a/9b reached 72.7%/71.7% lift (85% of teacher). Two ablations to push further:
- **run10a**: Enable visual randomization (`disable_arm_randomization=False`) — randomizes robot arm textures, table textures at each reset. May hurt training lift slightly but improves sim2real robustness.
- **run10b**: Increase to 32 envs (4 envs/object, up from 3) — cleaner gradients. May hang on RTX 4090 (previous finding: 32 hangs, 24 works). Both use SafeDagger to isolate each variable against run9a baseline.

**Baseline (run9a):** 72.7% lift, 62.7% unsafe, 24 envs, arm randomization OFF.

**Run 10a — SafeDagger + arm randomization (GPU 0, 24 envs):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 ... --num_envs 24 ... env.disable_arm_randomization=False
```

**Run 10b — SafeDagger + 32 envs (GPU 1):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 ... --num_envs 32 ... env.disable_arm_randomization=True
```

**All other settings identical to run9a:** threshold=2.0, visdex_top8, teacher_onehot_size=13, teacher_objects_dir=visdex_selected, 10s episodes, L2 loss, ADR off.

**Run directories:**
- run10a (arm randomization): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_05-15-21-00/`
- run10b (32 envs): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_05-15-21-49/`

**Status (2026-04-05):** Both running. run10a started 15:21 on GPU 0, run10b started 15:21 on GPU 1. 32 envs did NOT hang.

**Standalone eval (480 episodes, 2026-04-05):**

| Metric | run10a (arm rand) | run10b (32 envs) | run9a (baseline) | Teacher 11 (fair TOP8) |
|---|---|---|---|---|
| **Lift success** | 62.3% | **79.2%** | 72.7% | **99.0%** |
| **Unsafe rate** | 72.9% | 64.6% | 62.7% | 23.1% |
| Physics (% all eps) | 36.7% | 27.1% | 32.5% | 11.0% |
| Collision (% all eps) | 18.5% | 13.1% | 16.2% | 1.7% |
| Object OOB (% all eps) | 15.8% | 14.6% | 11.5% | 9.4% |
| Palm flipped (% all eps) | 1.9% | 9.2% | 2.5% | 0.6% |

**Eval JSONs:**
- run10a: `eval_results/eval_metrics_20260405_224028.json`
- run10b: `eval_results/eval_metrics_20260405_224012.json`

**Key findings:**
- **32 envs = new best student: 79.2% lift (92.3% of teacher ceiling)**
- More envs/object (4 vs 3) provides cleaner gradients → better policy
- Arm randomization hurts training lift (-10pp) — visual diversity adds learning difficulty without matching eval conditions
- 32 envs confirmed working on RTX 4090 (no hang) — updates previous finding

## run9a/9b — SafeDagger vs DAgger with FIXED one-hot index mapping (2026-04-04)

**Motivation: CRITICAL BUG FIX.** All previous runs using `visdex_top8` (runs 7-8) had **wrong teacher one-hot indices**. When distilling with a subset of objects, `multi_object_idx_onehot` used sequential indices (0-7) instead of the teacher's original indices (from 13-object training). This meant the teacher received the wrong object identity for 7 out of 8 objects:

| Object | Teacher index (correct) | Distillation index (run7-8, wrong) |
|---|---|---|
| basketball_shoe | 0 | 0 ✅ |
| closed_fist | 2 | 1 ❌ (teacher thought: chicken_head_in_car) |
| elephant_toy | 3 | 2 ❌ (teacher thought: closed_fist) |
| mario | 5 | 3 ❌ (teacher thought: elephant_toy) |
| milk_pot | 6 | 4 ❌ (teacher thought: homer) |
| teddy_bear | 8 | 5 ❌ (teacher thought: mario) |
| toy_bagger | 9 | 6 ❌ (teacher thought: milk_pot) |
| tutle_candle_holder | 12 | 7 ❌ (teacher thought: plane) |

**Evidence:** basketball_shoe (index 0, the only correctly mapped object) consistently had the best per-object lift across all run7-8 results for both methods.

**Fix:** Added `teacher_objects_dir` config — specifies the teacher's training object directory. The env now maps each object name to its correct index in the teacher's sorted object list. One-hot vectors use the teacher indices, not sequential indices.

**All other settings identical to run8:** threshold=2.0, top8, 10s episodes, L2 loss, AverageMeter logging.

**Run 9a — SafeDagger:**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 2.0 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run 9b — DAgger:**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 \
  --vanilla_dagger \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.teacher_objects_dir=multi_objects/visdex_selected \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directories:**
- run9a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_04-14-07-25/`
- run9b (DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_04-14-08-19/`

**Status (2026-04-04):** Both running. SafeDagger (9a) started 14:07 on GPU 0, DAgger (9b) started 14:08 on GPU 1.

**Expected:** Significant improvement across all objects (especially non-basketball_shoe). Teacher now provides correct actions for each object.

**Standalone eval (480 episodes, 2026-04-04):**

| Metric | run9a SafeDagger+L2 | run9b DAgger+L2 | run8a (broken) | run8b (broken) | Teacher 11 (fair TOP8) |
|---|---|---|---|---|---|
| **Lift success** | **72.7%** | **71.7%** | 41.7% | 55.2% | **99.0%** |
| **Unsafe rate** | 62.7% | **58.3%** | 70.0% | 42.3% | 23.1% |
| Physics instab. (% all eps) | 32.5% | 25.6% | 44.1% | 20.8% | 11.0% |
| Harmful collision (% all eps) | 16.2% | 12.5% | 11.5% | 11.7% | 1.7% |
| Object OOB (% all eps) | 11.5% | 19.2% | 9.4% | 9.8% | 9.4% |
| Palm flipped (% all eps) | 2.5% | 1.1% | 5.0% | 0.0% | 0.6% |

**Eval JSONs:**
- run9a: `eval_results/eval_metrics_20260404_201637.json`
- run9b: `eval_results/eval_metrics_20260404_201628.json`

**Key findings:**
- **One-hot fix produced massive improvement:** SafeDagger +31pp (41.7→72.7%), DAgger +16.5pp (55.2→71.7%)
- **SafeDagger and DAgger now comparable in lift** (72.7% vs 71.7%) — first time SafeDagger matches DAgger
- **Best student results yet:** 72.7% lift = 84.7% of teacher ceiling
- SafeDagger unsafe rate still higher (62.7% vs 58.3%) but gap narrowed significantly
- Confirms the one-hot index bug was the dominant issue in runs 7-8

#### Train-vs-eval UER gap (SafeDagger 9a, BC 9c)

A reader of the training plots will notice that **SafeDagger's training-time real UER hovers around 10-15%, while its eval real UER is 30.2%** (62.7% total - 32.5% physics). The 2× gap is expected and stems from three effects, in order of magnitude:

1. **Teacher safety net masks unsafe terminations.** In `distillation_safedagger.py:611-616`, whenever per-env L2 > `unsafe_l2_threshold` (=2.0), the teacher's actions overwrite the student's before the env step. The env continues safely → no unsafe termination is recorded. The training-time UER counts only episodes that actually terminated unsafely, so every teacher rescue subtracts from the numerator. `beta` (intervention rate) shows how often this fires — typically ~30% at training start, ~5-15% at convergence.
2. **Cleaner state distribution during training.** Even in envs the student is currently acting in, the env's history was partly shaped by past teacher rescues. The student acts from in-distribution states (close to teacher's trajectory). At eval, the student acts in **every** env from the very first step → small errors compound → out-of-distribution states the student has never seen. This is the classic DAgger covariate-shift argument.
3. **Windowed AverageMeter.** `game_unsafe_terminated` keeps only the last `games_to_track` episodes, so the metric reflects recent behavior. As the student improves, the window flushes early bad episodes. Eval averages 480 fresh episodes uniformly.

Sanity-check the asymmetry across methods:
- **DAgger** (no rescue, vanilla): training real UER ≈ eval real UER (~25-32%). Run9b: training ~20-25%, eval 32.7%. Small gap, explained by #2 alone.
- **SafeDagger** (rescue active): training ~10-15%, eval 30.2%. ~2× gap, mostly #1.
- **BC** (teacher 100%, run9c): training UER ≈ teacher's UER inside env. Eval UER 90.8%. Largest gap, driven entirely by #1+#2 — student never acts in env during training.

The eval/training UER ratio rank order — **BC >> SafeDagger > DAgger** — directly mirrors how much teacher steering happens during training. For sim2real-relevant comparisons, only the eval numbers should be trusted.

#### Auxiliary loss comparison (2026-04-14)

Plot: [aux_loss_object_pos.png](../Report/figures/plots/run9/aux_loss_object_pos.png)

The student network has an auxiliary head that regresses object position from stereo images. The vision backbone and regression target are identical across all three methods, so comparing `aux_loss_object_pos` isolates the effect of the training **state distribution** on vision-feature learning.

| Method | Early (first 10k iters) | Late (last 10k iters) |
|---|---:|---:|
| SafeDagger (9a) | 0.021 | **0.011** |
| DAgger (9b) | 0.035 | **0.018** |
| BC (9c) | 0.021 | **0.010** |

**Observations:**
- **SafeDagger and BC converge to nearly identical aux loss** (0.011 vs 0.010). Their training envs are both teacher-dominated (BC 100%, SafeD ~85% after rescues), so the vision backbone sees similar state distributions.
- **DAgger's aux loss is ~1.7× higher** (0.018) throughout. Student-driven envs reach more off-distribution / unusual camera viewpoints that the object-pose regressor struggles with. This is a **clean measurement of covariate shift's impact on representation learning**, independent of the policy output.
- All three methods' imitation losses (`l2_loss_mean`) follow the same ordering: BC 0.83 < SafeD 1.05 < DAgger 2.05. Teacher-dominated envs give the student easier targets to match; student-driven envs require the student to match teacher actions in states the teacher would never have reached.
- The aux loss gap (1.7×) is much smaller than the imitation loss gap (2×), suggesting the vision backbone is relatively distribution-robust — the covariate shift cost is concentrated in the action-matching head, not the representation.

## run8a/8b — SafeDagger vs DAgger with calibrated threshold + episode-level lift (2026-04-03)

**Motivation:** Consolidating all fixes from runs 6-7. Key improvements over run7:

1. **L2 threshold calibrated to 2.0** — Based on original SafeDagger paper principle: set threshold so ~20-30% of envs are unsafe at start, let it decay naturally. Analysis of run7 L2 distribution: early p80=2.6, late p80=1.7. Threshold=2.0 gives ~30% initial intervention → ~5% at convergence. Previous: 0.665 (beta=70%, behavior cloning) and 3.0 (beta=5%, essentially DAgger).
2. **Episode length doubled to 10s** — `distillation_episode_length_s=10.0` (was 5.0 = 150 steps). Matches teacher training episode length (300 steps). Gives student more time to approach, grasp, and lift each object.
3. **Episode-level lift tracking** — `train/avg/lift_success` and `train/<object>/lift_success` via AverageMeter. Tracks "was object lifted at any point during episode" — directly comparable to eval. Previous per-step `in_success_region` / `per_object_lifted` didn't reflect standalone student capability.
4. **Global `lift_success` scalar** logged to TensorBoard each step.

**Unchanged from run7:** visdex_top8 (8 objects, 3 envs/obj), L2 loss for both methods, teacher 11, `teacher_onehot_size=13`.

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43`

**Run 8a — SafeDagger (L2, threshold=2.0, 10s episodes):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 2.0 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run 8b — DAgger (L2, no override, 10s episodes):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 \
  --vanilla_dagger \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 env.teacher_onehot_size=13 \
  env.distillation_episode_length_s=10.0 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directories:**
- run8a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_03-20-30-06/`
- run8b (DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_03-20-33-10/`

**Status (2026-04-03):** Both running. SafeDagger (8a) started 20:30 on GPU 0, DAgger (8b) started 20:33 on GPU 1.

**Threshold calibration summary:**

| Run | Threshold | Beta (start) | Beta (end) | Behavior |
|---|---|---|---|---|
| run6a | 0.665 | ~95% | ~70% | Behavior cloning |
| run7a | 3.0 | ~30% | ~5% | Essentially DAgger |
| **run8a** | **2.0** | **~30%** | **~5-10%** | **Calibrated SafeDagger** |
| Paper (tg2) | 0.5 | ~80% | ~20% | Paper's SafeDagger |

**Results:** *pending*

## run7a/7b — SafeDagger vs DAgger with corrected threshold + top 8 objects (2026-04-03)

**Motivation:** Three critical fixes from run6 analysis:
1. **L2 threshold too low** — threshold 0.665 produced beta=70% = behavior cloning. DAgger L2 mean is ~4.4, so threshold must be much higher for student to act. Set to **3.0** (teacher only intervenes when student is dramatically off).
2. **Loss function corrected** — previous DAgger runs used KL loss (`--vanilla_dagger` set `imitation_loss_type: "kl"`). Paper uses weighted L2 for both methods. Fixed: both SafeDagger and DAgger now use L2.
3. **Reduced object set** — `visdex_top8` (8 best-performing objects) instead of `visdex_selected` (13). Gives 3 envs/object (up from 1.8) for cleaner gradients.

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43` — trained on 13 objects, compatible with 8-object subset.

**Objects (visdex_top8):** basketball_shoe, closed_fist, mario, milk_pot, plane, teddy_bear, toy_cow, train

**Run 7a — SafeDagger (L2 loss + teacher override, threshold=3.0):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 --unsafe_l2_threshold 3.0 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run 7b — DAgger (L2 loss, no teacher override):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 \
  --vanilla_dagger \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_top8 \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run directories:**
- run7a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_03-14-53-40/`
- run7b (DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_03-14-54-52/`

**Status (2026-04-03):** Both running. SafeDagger (7a) started 14:53 on GPU 0, DAgger (7b) started 14:54 on GPU 1.

**What changed from run6:**
- L2 threshold: 0.665 → **3.0** (expect beta ~20-30% instead of ~70%)
- Loss: DAgger now L2 (was KL in runs 2a-5b)
- Objects: 13 → **8** (visdex_top8, 3 envs/obj)
- `--unsafe_l2_threshold` CLI flag added

**What to watch:**
- Beta decay — should drop much faster with threshold=3.0
- DAgger L2 vs SafeDagger with same loss function — first true apples-to-apples comparison
- Per-object lift/unsafe with better gradient signal (3 envs/obj)

**Results:** *pending*

## run6a/6b — SafeDagger vs DAgger with episode-level AverageMeter logging (2026-04-03)

**Motivation:** run5 showed that per-step termination rates (~0.1%) are not comparable to eval episode-level unsafe rates (~50-75%). Adopted upstream tg2_inspirehand AverageMeter pattern for episode-level tracking. Now `train/avg/unsafe_episode_rate` directly matches eval's `unsafe_episode_rate`.

**Changes from run5:**
- `classify_out_of_reach_reasons()` from `eval_utils.py` classifies termination reasons per step (matching eval)
- `AverageMeter` rolling window (100 episodes) tracks episode-level: `train/avg/unsafe_episode_rate`, `train/avg/unsafe_reason_prop/<reason>`
- Per-object episode-level: `train/<object>/unsafe_episode_rate`, `train/<object>/unsafe_reason_prop/<reason>`
- Per-env `current_unsafe_terminated` and `current_unsafe_reason_idx` accumulate within episodes, reset on done
- `per_object_lifted/<name>` tracks `lift_success` (object above table, less strict than `in_success_region`)
- ResNet18 backbone confirmed finetuning during distillation (ImageNet pretrained, gradients enabled)
- **L2 threshold scaled by action dim**: `0.5 × sqrt(23/13) = 0.665` (was 0.5). Matches effective per-joint difficulty of upstream tg2_inspirehand (13 actions). Should reduce beta from ~75% toward ~20-40%.

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43`

**Run directories:**
- run6a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_03-13-04-24/`
- run6b (Vanilla DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_03-13-06-54/`

**Status (2026-04-03):** Both running. SafeDagger (6a) started 13:04 on GPU 0, DAgger (6b) started 13:06 on GPU 1.

**Commands:** Same as run5a/5b.

**Standalone eval (480 episodes, 2026-04-03):**

| Metric | run7a SafeDagger+L2 | run7b DAgger+L2 | Teacher 11 (fair TOP8) |
|---|---|---|---|
| **Lift success** | 27.9% | **56.9%** | **99.0%** |
| **Unsafe rate** | 79.0% | **59.4%** | 23.1% |
| Physics instability (% all eps) | 52.1% | 39.2% | 11.0% |
| Harmful collision (% all eps) | 11.6% | 8.7% | 1.7% |
| Object out of bound (% all eps) | 14.9% | 11.2% | 9.4% |
| Palm flipped (% all eps) | 0.3% | 0.2% | 0.6% |

**Eval JSONs:**
- run7a: `eval_results/eval_metrics_20260403_192337.json`
- run7b: `eval_results/eval_metrics_20260403_192212.json`

**Key findings:**
- DAgger (L2 loss) outperforms SafeDagger: 56.9% vs 27.9% lift — consistent with all prior runs
- SafeDagger with threshold=3.0 still underperforms — threshold is not the main issue
- First true apples-to-apples comparison (both L2 loss) confirms DAgger is the better method
- visdex_top8 (8 objects) didn't improve over visdex_selected (13 objects) for DAgger (56.9% vs 52.1% run4b)
- Physics instability remains dominant failure mode for both

### SafeDagger beta analysis — why beta stays at ~75% vs upstream's ~20%

**Finding:** Both upstream (tg2_inspirehand) and fr3_agilehand use `unsafe_l2_threshold = 0.5`. But the L2 norm behaves differently due to action space size:

| Factor | tg2_inspirehand | fr3_agilehand |
|---|---|---|
| Action space | 13 (7 arm + 6 hand) | 23 (7 arm + 16 hand) |
| L2 threshold | 0.5 | 0.5 |
| L2 computation | `weighted_l2(mus, dim=-1)` — sums across ALL dims | same |
| Expected L2 at same per-joint error | lower (fewer dims) | ~√(23/13) ≈ 1.33× higher |
| Distillation envs | 48 (paper) | 24 |
| Envs per object | ~4-5 | ~1.8 |

The `weighted_l2` sums `(student - teacher)² * weight` across all action dimensions, then takes `sqrt`. With 23 dims vs 13, the L2 norm is inherently larger even if per-joint error is identical. The same 0.5 threshold is much harder to satisfy with 23 dims.

**Proposed fix:** Scale threshold by action dimensionality: `0.5 * sqrt(23/13) ≈ 0.67` or simply try `0.3` to match the effective difficulty. Alternatively, normalize L2 by `sqrt(num_actions)` before comparing to threshold.

**Additional concern:** 24 envs / 13 objects = 1.8 envs per object. Upstream likely uses 48 envs with fewer objects, giving 4-5× more gradient signal per object. This may slow learning independently of the threshold issue.

## run5a/5b — SafeDagger vs DAgger with per-object real termination logging (2026-04-02)

**Motivation:** run4a/4b logged `per_object_unsafe/<name>` which is L2-based divergence, not actual unsafe terminations. run5a/5b adds `per_object_term_real/<name>` (object OOB, hand too far, palm flipped) and `per_object_term_physics/<name>` (vel_explosion, robot_unstable) per object. Also includes fixed beta=0 for DAgger.

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43`

**Run directories:**
- run5a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_02-23-45-55/`
- run5b (Vanilla DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_02-23-41-36/`

**Status (2026-04-02):** Both running. DAgger (5b) started 23:41 on GPU 1, SafeDagger (5a) started 23:45 on GPU 0. Both 24 envs, 100k iters.

**New metrics logged:**
- `per_object_term_real/<name>` — real unsafe terminations per object (confirmed logging)
- `per_object_term_physics/<name>` — physics instabilities per object
- `beta` = 0 for DAgger (fixed), actual intervention rate for SafeDagger

**Commands:** Same as run4a/4b.

**Standalone eval (480 episodes, 2026-04-03):**

**Teacher 11 benchmark** (480 episodes, `eval_metrics_20260331_193003.json`): 85.8% lift, 23.1% unsafe. All student comparisons are relative to this teacher.

| Metric | run5a SafeDagger+L2 | run5b Vanilla DAgger+KL | Teacher 11 |
|---|---|---|---|
| **Lift success** | **61.3%** | 49.2% | 85.8% |
| **Unsafe rate** | 75.4% | 58.5% | 23.1% |
| Physics instability (% all eps) | 40.6% | 14.6% | 11.0% |
| Harmful collision (% all eps) | 13.8% | 9.8% | 1.7% |
| Object out of bound (% all eps) | 8.5% | 17.5% | 9.4% |
| Palm flipped (% all eps) | 12.5% | 16.7% | 0.6% |

*Note: failure breakdown scaled to % of all episodes (= reason_pct × unsafe_rate), not % of unsafe episodes.*

**Per-object standalone eval (480 episodes = 20 rollouts × 24 envs):**

| Object | run5a SafeD Lift | run5b DAgger Lift | run5a Unsafe | run5b Unsafe |
|---|---|---|---|---|
| basketball_shoe | 52.5% | 45.0% | 90.0% | 80.0% |
| chicken_head_in_car | 22.5% | 7.5% | 80.0% | 45.0% |
| closed_fist | **92.5%** | 85.0% | 65.0% | **35.0%** |
| elephant_toy | **87.5%** | 75.0% | **47.5%** | **45.0%** |
| homer | 42.5% | 35.0% | 92.5% | 62.5% |
| mario | 62.5% | 40.0% | 60.0% | 52.5% |
| milk_pot | 77.5% | 70.0% | 82.5% | 70.0% |
| plane | 45.0% | 22.5% | 85.0% | 67.5% |
| teddy_bear | 90.0% | **100.0%** | 52.5% | 55.0% |
| toy_bagger | 67.5% | 50.0% | 90.0% | 57.5% |
| toy_cow | 52.5% | 30.0% | 90.0% | 57.5% |
| train | 15.0% | 20.0% | 85.0% | 95.0% |
| tutle_candle_holder | 70.0% | 40.0% | 55.0% | 55.0% |

**Eval JSONs:**
- run5a: `eval_results/eval_metrics_20260403_105154.json`
- run5b: `eval_results/eval_metrics_20260403_105805.json`

**Key findings:**
- SafeDagger run5a achieves **61.3% lift — new best student** (surpassing run3a 59%)
- High variance between identical runs: SafeDagger 4a=36.9% vs 5a=61.3%, DAgger 4b=52.1% vs 5b=49.2%
- SafeDagger has higher lift but consistently higher unsafe rate (~75-80% vs DAgger ~45-59%)
- Palm_flipped emerged as significant failure mode in run5 (12.5-16.7% of all eps) — was <8% in run4
- DAgger achieves 100% lift on `teddy_bear` — best single-object result
- `train` and `chicken_head_in_car` remain hardest objects for both methods
- Excluding physics instability: SafeDagger real unsafe = 34.8%, DAgger real unsafe = 43.9%

**Metric insights from run5 analysis:**
- **`in_success_region` ≠ `lift_success`**: training-time `per_object_lift/<name>` logs `in_success_region` (object at goal, strict) — shows 0% for `train` and `chicken_head_in_car`. Eval `lift_success` (object above table, less strict) shows 15-20%. Added `per_object_lifted/<name>` for next run.
- **Per-step termination rate ≠ episode-level unsafe rate**: `termination/real_unsafe` during training is ~0.1% (instantaneous per-step), while eval shows 58-75% episode-level unsafe. This is because a termination flag is True for 1 step out of ~200 per episode. Added `episode/unsafe_rate` and `episode_per_object_unsafe/<name>` (cumulative episode-level) for next run — directly comparable to eval.
- **`out_of_reach_reason_pct` must be scaled**: values are % of unsafe episodes, not % of all episodes. Multiply by `unsafe_episode_rate` for absolute rates. All tables updated.

## run4a/4b — SafeDagger vs DAgger head-to-head with per-object logging (2026-04-02)

**Motivation:** Direct comparison of SafeDagger vs vanilla DAgger under identical conditions, with new per-object metrics. Prior runs used different iteration counts (run3a: 350k SafeDagger vs run3b: 100k DAgger) and lacked per-object breakdown during training. These runs add:
- `per_object_lift/<name>` and `per_object_unsafe/<name>` per training step
- `termination/real_unsafe` vs `termination/physics_instability` — separates sim artifacts from genuine failures
- Same object set as teacher training (`multi_objects/visdex_selected` — 13 objects)

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43` — sim2real curriculum, ADR 14

**Run 4a — SafeDagger (L2 loss + teacher override):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=0 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run 4b — Vanilla DAgger (KL loss, no teacher override):**
```bash
cd dextrah_lab/distillation_new
CUDA_VISIBLE_DEVICES=1 /home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 24 --enable_cameras --headless \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/11_multi_object_adr14_sim2real_03-30_17-41-43/nn/best_dextrah_tekken_lstm.pth \
  --max_iterations 100000 \
  --vanilla_dagger \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Run in parallel:** GPU 0 for SafeDagger (4a), GPU 1 for vanilla DAgger (4b). Two separate terminals.

**Status (2026-04-02):** Both running. Vanilla DAgger (4b) started first on GPU 1, SafeDagger (4a) started second on GPU 0. Both 24 envs.

**Run directories:**
- run4a (SafeDagger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_02-14-30-48/`
- run4b (Vanilla DAgger): `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_02-14-28-35/`

**Changes from run3a/3b:**
- **Per-object logging** added to `distillation_safedagger.py` — TensorBoard groups `per_object_lift/`, `per_object_unsafe/`
- **Termination breakdown** — `termination/real_unsafe` vs `termination/physics_instability`
- **Same object set** (`multi_objects/visdex_selected` = 13 objects) — consistent with teacher training
- **Both 100k iters** — matched iteration count for fair comparison
- **Both 24 envs, teacher 11** — identical conditions except distillation method

**What to watch:**
- Per-object lift curves — do some objects benefit more from SafeDagger's safety net?
- `termination/real_unsafe` vs `termination/physics_instability` — what fraction of failures are sim artifacts?
- `beta` decay in SafeDagger (run4a) — how fast does the student take over?
- Overall lift/unsafe at 100k — does run3b's finding hold (DAgger converges 3.5× faster)?

**Training completed (100k iters).** Raw metrics are noisy — check TensorBoard with smoothing (0.9+) for valid training curves. New per-object and termination breakdown metrics are logged.

**Standalone eval (480 episodes: 20 rollouts × 24 envs, 2026-04-02):**

| Metric | run4a SafeDagger+L2 | run4b Vanilla DAgger+KL | Teacher 11 |
|---|---|---|---|
| **Lift success** | 36.9% | **52.1%** | 85.8% |
| **Unsafe rate** | 79.6% | **44.8%** | 23.1% |
| Physics instability | 62.0% | 52.6% | 47.7% |
| Harmful collision | 19.6% | 16.3% | 7.2% |
| Object out of bound | 13.4% | 23.7% | 40.5% |
| Palm flipped | 5.0% | 7.4% | 2.7% |

**Eval JSONs:**
- run4a: `eval_results/eval_metrics_20260402_181427.json`
- run4b: `eval_results/eval_metrics_20260402_181413.json`

**Key findings:**
- Vanilla DAgger outperforms SafeDagger at 100k iters: 52.1% vs 36.9% lift, consistent with run3a/3b
- Vanilla DAgger achieves **44.8% unsafe rate** — best student unsafe rate so far (prev best: run3b 70.6%)
- SafeDagger's teacher intervention (beta=0.67) doesn't translate to standalone safety — 79.6% unsafe is worst of all run3/4 students
- Physics instability remains dominant failure mode for both (~52-62%), confirming it's a sim artifact not policy behavior

## run3a — SafeDagger + L2, run11 teacher, 24 envs (2026-03-31)

**Motivation:** New teacher (run11: ADR 14, sim2real curriculum, 33k epochs). Test with SafeDagger first, then vanilla DAgger for comparison. Also testing 24 envs (up from 16) for better per-object coverage.

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43` — multi-object, ADR 14, trained with sim2real effort/velocity limits

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand \
  --num_envs 24 \
  --enable_cameras \
  --headless \
  --teacher best_dextrah_tekken_lstm_run11.pth \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False \
  env.disable_arm_randomization=True
```

**Changes from run2a:**
- **New teacher** (run11 ADR 14 sim2real vs run10 ADR 19)
- **24 envs** (up from 16) — ~1.8 envs per object. 32 hangs, 24 works (~11GB of 24GB VRAM)
- SafeDagger (L2 loss) — baseline before vanilla DAgger comparison

**Training metrics at convergence (~125k iters):**

| Metric | run1c (teacher 10, 16 envs) | run3a (teacher 11, 24 envs) |
|---|---|---|
| Imitation Loss | 1.40 | **0.91** |
| Beta | 0.75 | **0.63** |
| In goal | 37.5% | **41.7%** |
| Sigma Loss | ~0.01 | **0.0001** |
| Avg ep step | 173 | **268** |
| obj_to_goal reward | 7.6 | **14.6** |
| Successes | 0/0 | **1/1** |

**Key observations:**
- Loss converged much lower (0.91 vs 1.40) — run11 teacher produces cleaner demonstrations
- Beta dropped to 0.63-0.68 (student acting in ~35% of envs) — faster handoff than run1c
- First ever recorded success (object held in goal for full timeout)
- Sigma loss near zero — student fully learned teacher's confidence structure
- Episodes lasting 268 steps (near 300 max) — far fewer crashes

**Standalone eval (480 episodes: 20 rollouts × 24 envs, 2026-03-31):**

| Metric | run2a (prev best, teacher 10) | run3a (teacher 11) | Teacher 10 |
|---|---|---|---|
| **Lift success** | 50.0% (80 eps) | **58.96%** (480 eps) | 96.25% |
| Unsafe rate | 75.0% | **72.7%** | 53.75% |
| Harmful collision | 45.0% | **19.5%** | 11.6% |
| Physics instability | 43.3% | 62.5% | 55.8% |
| Object out of bound | 11.7% | 17.8% | 32.6% |
| Palm flipped | 0% | 0.3% | 0% |

**Assessment:** New best student at 59% lift. The run11 teacher (sim2real effort/velocity limits) produces dramatically safer behavior — harmful collision dropped from 45% to 19.5% (approaching teacher's 11.6%). Physics instability dominates failures (62.5%) but is sim-only. Excluding physics instability, real-world-relevant unsafe rate is ~27%.

**Key insight:** Teacher quality matters more than distillation method. SafeDagger + L2 with a better teacher (run3a: 59%) outperforms vanilla DAgger + KL with a weaker teacher (run2a: 50%).

**Env count findings:**
- 32 envs with cameras hangs on RTX 4090 (loads 13.5GB but deadlocks — likely tiled renderer scheduling limit)
- 24 envs works reliably (~11GB of 24GB VRAM)
- 16 envs uses ~9.5GB

## run3b — Vanilla DAgger + KL, run11 teacher, 24 envs (2026-03-31)

**Motivation:** Compare vanilla DAgger + KL vs SafeDagger + L2 (run3a) using the same teacher 11.

**Teacher:** `11_multi_object_adr14_sim2real_03-30_17-41-43` — same as run3a

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand \
  --num_envs 24 \
  --enable_cameras \
  --headless \
  --vanilla_dagger \
  --teacher best_dextrah_tekken_lstm_run11.pth \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False \
  env.disable_arm_randomization=True
```

**Standalone eval (480 episodes, 2026-03-31):**

| Metric | run3a SafeDagger+L2 (~120k) | run3b Vanilla DAgger+KL (~105k) | Teacher 11 |
|---|---|---|---|
| **Lift success** | 58.96% | 57.71% | 85.83% |
| Unsafe rate | 72.7% | 70.6% | 23.1% |
| Harmful collision | 19.5% | 18.0% | 7.2% |
| Physics instability | 62.5% | 65.2% | 47.7% |
| Object out of bound | 17.8% | 16.8% | 40.5% |
| Palm flipped | 0.3% | 0% | 2.7% |

**Key finding: Both methods converge at similar speed and performance.**
Both methods reach ~58% lift with teacher 11 in ~100k iters. With teacher 11, the distillation method matters less than the teacher quality — both SafeDagger+L2 and vanilla DAgger+KL produce similar standalone performance.

**Updated conclusion:** Teacher quality is the dominant factor. run3a/3b (teacher 11, 59/58% lift) both outperform run2a (teacher 10, 50% lift) regardless of distillation method.

## Teacher checkpoints used

| Run | Teacher | Description |
|-----|---------|-------------|
| run1a | `09_again_multi_object_adr5_03-25_21-15-48` | Multi-object (13 visdex_selected), ADR 5, epoch 11124 |
| run1b | `10_multi_object_adr19_1024envs_03-26_00-21-45` | Multi-object (13 visdex_selected), ADR 19, 1024 envs, epoch 20000 |

## run1a — 2026-03-26

**Teacher:** ADR 5 multi-object teacher (09)

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 8 --enable_cameras \
  --teacher best_dextrah_tekken_lstm.pth \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Results (100k iterations):**

| Metric | Value |
|--------|-------|
| Imitation loss | 2.10 |
| Lifted | 62.5% |
| In goal | 37.5% |
| Total reward | 48.1 |
| Beta | 1.0 (teacher still dominant) |
| Successes | 0/0 (0%) |
| Contact reward | 14.0 |
| Lift reward | 20.5 |

**Notes:** First distillation attempt. Student learned to approach and lift but imitation loss remained high. Teacher was still providing all actions (beta=1.0).

## run1b — 2026-03-26

**Teacher:** ADR 19 multi-object teacher (10), 1024 envs, 20k epochs — much stronger teacher.

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 8 --enable_cameras \
  --teacher best_dextrah_tekken_lstm.pth \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Results (100k iterations):**

| Metric | Value |
|--------|-------|
| Imitation loss | 1.48 |
| Lifted | 87.5% |
| In goal | 62.5% |
| Total reward | 62.0 |
| Beta | 0.875 (student starting to take over) |
| Successes | 0/0 (0%) |
| Contact reward | 16.0 |
| Lift reward | 28.2 |

**Comparison vs run1a:** Stronger teacher (ADR 19 vs 5) → 40% relative improvement in lift rate (62.5→87.5%), 66% improvement in goal rate (37.5→62.5%), imitation loss dropped 30% (2.10→1.48), beta decreased from 1.0→0.875 meaning student is taking more control.

**Checkpoint:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_26-11-25-08/nn/dextrah_student_100000_iters.pth`

**Notes:** Significant improvement from stronger teacher. Student is learning but 100k iterations may not be enough — imitation loss still high. Next steps: try more iterations (200-500k), or evaluate per-object breakdown.

## Distillation loss functions explained

### Action imitation losses (`imitation_loss_type`)

**KL divergence (`"kl"`)** — recommended, used by DextrAH-RGB paper
- `KL(N_teacher || N_student)` over Gaussian action distributions
- Decomposes into **mu_loss** (action means, weighted by student variance) + **sigma_loss** (variance matching)
- Captures both "right action" and "right confidence" — student learns which DOFs need precision (finger joints during grasp) vs which are flexible (arm during approach)
- Paper found KL always outperforms L2 across all seeds

**L2 (`"l2"`)** — legacy, used in run1a-1c
- `weighted_l2(mu_student, mu_teacher, weights=1/sigma_teacher) + l2(sigma_student, sigma_teacher)`
- Only matches action means (weighted by teacher confidence) + variance. Simpler but misses distribution structure

**NLL (`"nll"`)** — samples teacher action, scores under student distribution
- `-log p_student(a_teacher)` — noisier due to single-sample dependence

**MSE (`"mse"`)** — both sample, compare directly
- Noisiest — two sources of sampling variance

### Auxiliary loss
- 3D object position prediction: `||x_obj_predicted - x_obj_actual||`
- Forces vision encoder to learn spatial awareness from stereo images
- Very small (~0.02) when well-learned — provides supervised signal for the backbone beyond action imitation

### Total loss
`total_loss = imitation_loss + aux_coeff * aux_loss`

### DAgger vs SafeDagger stepping

**SafeDagger** (default): per-env safety check `unsafe[i] = (l2_loss_per_env[i] > unsafe_l2_threshold)`. Unsafe envs get teacher actions for stepping, safe envs get student actions. Both train on teacher labels. Creates chicken-and-egg: student only practices where already good.

**Vanilla DAgger** (`--vanilla_dagger`): student always steps its own actions in all envs. Trains on teacher labels. Forces student to learn from its own mistakes — the core DAgger insight (distribution shift correction). Beta is logged but ignored.

### Key finding: KL + vanilla DAgger > L2 + SafeDagger
- run1c (SafeDagger + L2, 350k): 36% lift standalone
- run2a (vanilla DAgger + KL, 350k): 50% lift standalone
- KL captures uncertainty structure; vanilla DAgger forces the student to handle its own distribution

## Open question: ADR during distillation for sim2real

All runs so far use `env.enable_adr=False`. This means both teacher and student operate in idealized physics (no friction/stiffness/effort randomization).

**Why this is a problem for sim2real:**
- The student learns `image + proprio → action` in exactly one physics regime
- On real hardware, joint dynamics differ (friction, stiffness, backlash) — the student has never seen how environment *responses* change under different physics
- The teacher's actions are inherently robust (trained with ADR 19), but the student only learns the mapping in a fixed context — it can't correct for physics differences it hasn't experienced
- Proprio observations (joint velocities, positions under load) also differ with physics — student has no experience with these variations

**Why ADR is currently disabled:**
- Higher ADR degrades teacher performance → noisier demonstrations → harder to learn from
- tg2_inspirehand pipeline also uses `enable_adr=False` — this is the established pattern
- Visual augmentation (`data_aug`, `rgb_augs.py`) is handled separately from ADR and does help with visual sim2real

**Proposed approach: fixed low ADR during distillation**
- Set `env.enable_adr=True env.starting_adr_increments=5` — no curriculum, just fixed moderate randomization
- Teacher (ADR 19) still performs well at ADR 5 physics
- Student sees physics diversity without the teacher becoming a bad supervisor
- Compare sim2real transfer vs ADR 0 distillation

**Next runs to try:**
- run2a: vanilla DAgger (ADR 0) — compare distillation methods
- run2b: SafeDagger with fixed ADR 5 — test physics diversity during distillation

## run1c — 2026-03-27 ~17:30

**Teacher:** ADR 19 multi-object teacher (10) — same as run1b
**Student:** Fresh start (no resume from run1b — see note below)

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 16 --enable_cameras \
  --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/10_multi_object_adr19_1024envs_03-26_00-21-45/nn/best_dextrah_tekken_lstm.pth \
  -- \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False env.disable_arm_randomization=True
```

**CLI gotcha:** `--` separator is required. All `--flags` (task, num_envs, teacher, etc.) must go BEFORE `--`. All `env.*` Hydra overrides must go AFTER `--`. If `--teacher` comes after `--`, argparse doesn't see it and falls back to the default `pretrained_ckpts/` path. If `env.*` overrides come before `--`, Hydra doesn't receive them and `objects_dir` stays at `"replace_me"`.

**Changes from run1b:**
- **16 envs** (up from 8) — more object diversity per batch. 32 envs caused hang/stuck on this GPU.
- **350k iterations** (up from 100k) — `distillation_safedagger.py` default bumped to match `distillation_transformer.py`
- Fresh start instead of resume — `--network` path resolution was broken (resolved relative to repo root, not `distillation_new/`)

**Why fresh start instead of resume from run1b:**
- Attempted to resume with `--network` flag but path resolution joined relative paths with repo root instead of `distillation_new/`
- At 100k iters with beta=0.875, the student had barely started acting independently — not much value in the checkpoint
- Fresh start with 32 envs and 350k iters is a cleaner experiment

**What to watch:**
- Beta schedule — needs to drop well below 0.5 for student to be viable standalone
- Standalone eval (via `eval_student.py`) — run1b showed 0% lift when student acted solo despite 87.5% lifted during distillation (teacher was doing the work)

**Results (350k iterations):**

| Metric | run1b (100k, 8 envs) | run1c (350k, 16 envs) |
|---|---|---|
| Imitation Loss | 1.48 | 1.40 |
| Beta | 0.875 | 0.75 |
| Lifted | 62.5% | 62.5% |
| In goal | 37.5% | 37.5% |
| Total reward | 62.0 | 50.98 |
| Lift reward | 28.2 | 18.9 |
| Contact reward | 16.0 | 18.0 |
| Avg ep step | — | 173 |

**Checkpoint:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_27-19-13-17/nn/dextrah_student_350000_iters.pth`

**Standalone eval (2026-03-28):**
```bash
python eval_student.py --headless --task=dextrah_fr3_agilehand --num_envs 8 --enable_cameras --checkpoint dextrah_student_safedagger_stereo_transformer.pth --num_episodes 5 env.distillation=True env.simulate_stereo=True env.objects_dir=multi_objects/visdex_selected
```

| Metric | Value |
|---|---|
| Lift success | 45% |
| Unsafe episode rate | 67.5% |
| Harmful collision | 33.3% of failures |
| Object out of bound | 29.6% of failures |
| Hand too far | 0% |
| Palm flipped | 0% |
| Total episodes | 40 (8 envs × 5 rollouts) |

**Standalone eval — 16 envs, 80 episodes (2026-03-28):**

| Metric | 8 envs (40 eps) | 16 envs (80 eps) |
|---|---|---|
| Lift success | 45.0% | 36.25% |
| Unsafe rate | 67.5% | 67.5% |
| Harmful collision | 33.3% | 40.7% |
| Object out of bound | 29.6% | 33.3% |
| Palm flipped | 0% | 1.9% |
| Hand too far | 0% | 0% |

**Assessment:** 350k iters with 16 envs produces a student with ~36-45% solo lift — significant improvement over run1b (0% at 100k). Unsafe rate consistent at 67.5%. Main failure modes are harmful collisions (hand hitting table, 41%) and object knockoff (33%). Arm control is solid (no hand-too-far, minimal palm flip). More iterations or faster beta decay could help, but collision avoidance may need targeted work.

## run2a — Vanilla DAgger with KL loss (2026-03-28)

**Motivation:** DextrAH-RGB paper uses vanilla DAgger with KL loss, not SafeDagger with L2. Key differences:
- **KL divergence loss** — paper found KL always outperforms L2 across all seeds
- **No unsafe override** — student always steps its own actions (no teacher takeover for "unsafe" envs)
- SafeDagger's per-env safety gating creates chicken-and-egg: student only practices solo where already good

**Teacher:** ADR 19 multi-object teacher (10) — same as run1b/1c

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py --task=dextrah_fr3_agilehand --num_envs 16 --enable_cameras --vanilla_dagger --teacher /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/10_multi_object_adr19_1024envs_03-26_00-21-45/nn/best_dextrah_tekken_lstm.pth -- env.distillation=True env.simulate_stereo=True env.objects_dir=multi_objects/visdex_selected env.enable_adr=False env.disable_arm_randomization=True
```

**Changes from run1c:**
- `--vanilla_dagger` flag: sets KL loss + disables unsafe env override
- Everything else identical (16 envs, 350k iters, ADR disabled)

**Results (350k iterations, 2026-03-29):**

| Metric | run1c (SafeDagger + L2) | run2a (vanilla DAgger + KL) | Teacher |
|---|---|---|---|
| **Lift success** | 36.25% | **50.0%** | 96.25% |
| Unsafe rate | 67.5% | 75.0% | 53.75% |
| Harmful collision | 40.7% | 45.0% | 11.6% |
| Physics instability | n/a | 43.3% | 55.8% |
| Object out of bound | 33.3% | 11.7% | 32.6% |

**Checkpoint:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_28-15-55-43/nn/dextrah_student_350000_iters.pth`

**Assessment:** Vanilla DAgger + KL outperforms SafeDagger + L2: lift success 36→50%, object knockoff 33→12%. Student is now at 52% of teacher ceiling (50/96). Physics instability accounts for 43% of failures (sim-only, won't happen on hardware). Main real-world failure remains harmful collision (45% vs teacher's 12%).

**Stored checkpoint:** `stored_policies/fr3_agilehand/distillation/vanilla_dagger_kl_run2a_350k/`

## run2b — Vanilla DAgger + KL + data_aug, 700k iters (2026-03-29)

**Motivation:** run2a showed KL + vanilla DAgger improves lift 36→50%. KL loss and sigma loss were still decreasing at 350k — more iterations should help. Adding `--data_aug` for visual diversity (random backgrounds, color jitter) to improve generalization and sim2real readiness.

**Teacher:** ADR 19 multi-object teacher (10) — same as all prior runs

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand \
  --num_envs 16 \
  --enable_cameras \
  --vanilla_dagger \
  --data_aug \
  --max_iterations 700000 \
  --teacher best_dextrah_tekken_lstm.pth \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False \
  env.disable_arm_randomization=True
```

**Changes from run2a:**
- **700k iterations** (up from 350k) — KL loss still decreasing at 350k
- **`--data_aug`** enabled — visual augmentation for generalization

**Results (700k iterations, 2026-03-30):**

| Metric | run2a (350k) | run2b (700k) | Teacher |
|---|---|---|---|
| **Lift success** | 50.0% | **0.0%** | 96.25% |
| Unsafe rate | 75.0% | 100% | 53.75% |
| Harmful collision | 45.0% | 68.8% | 11.6% |
| Physics instability | 43.3% | 28.8% | 55.8% |
| Object out of bound | 11.7% | 2.5% | 32.6% |

**Checkpoint:** `runs/dextrah-fr3-agilehand-safedagger-stereo-transformer_29-08-52-01/nn/dextrah_student_700000_iters.pth`

**Critical finding: `--data_aug` is a NO-OP in the SafeDagger pipeline.**
`distillation_safedagger.py` never reads `config["student"]["data_aug"]`. Only the legacy `distillation.py` (vanilla DAgger) implements RGB augmentation. The env already does dome light randomization (30%) and has table/object texture randomization built-in — these are always active regardless of `--data_aug`.

**Therefore run2b was effectively: vanilla DAgger + KL + 700k iters (no extra augmentation).** The student collapsed between 350k-700k iterations — from 50% lift to 0%. All episodes terminate from hand_close (69%) or physics instability (29%). Avg episode step was 19-40 during eval (immediate crash).

**Possible causes for collapse:**
- Overfitting to training distribution — student memorizes trajectories instead of generalizing
- Learning rate too high (2e-4) for extended training — policy oscillates and destabilizes
- No learning rate schedule — constant LR over 700k iters without decay
- Catastrophic forgetting — later training on some objects destroys previously learned behavior on others

### What is data augmentation and why it matters for sim2real

**Data augmentation** modifies training images on-the-fly to increase visual diversity:
- **Color jitter**: random brightness, contrast, saturation shifts
- **Random backgrounds**: replace sim background with random photos
- **Motion blur**: simulate camera motion artifacts
- **Texture randomization**: vary object/table surface appearance
- **Dome light HDRI**: change scene lighting (already active in env at 30%)

**Why it helps sim2real:** The student needs to work with real camera images that differ from sim renders — different lighting, textures, reflections. Without augmentation, the student overfits to sim's specific visual appearance and fails on real images.

**Current state in this codebase:**
- `distillation.py` (legacy vanilla DAgger): implements `rgb_augs.py` augmentation pipeline
- `distillation_safedagger.py`: does NOT implement augmentation — `--data_aug` flag is ignored
- Env-level visual randomization (dome light, textures) is always active but is sim-render-only (not the same as post-render image augmentation)

## Fixes applied during experimentation

- `eval_student.py`: added `import dextrah_lab.tasks.fr3_agilehand.gym_setup` (was missing, caused eval to hang)
- `eval_student.py`: added `gym.wrappers.RecordVideo` wrapping for `--video` flag (was missing, no MP4 produced)
- `eval_student.py`: changed `env.env` → `env.unwrapped` (4 places) to support `RecordVideo` wrapper chain
- `eval_student.py`: relaxed `_reason_counts_checked` from hard crash to warning for unclassified unsafe episodes (fr3_agilehand env doesn't expose `last_*` termination reason attributes)
- `dextrah_fr3_agilehand_env.py`: added `self.last_*` termination reason masks (matching upstream `tg2_inspirehand` pattern) so `eval_utils.py` can classify unsafe episodes
- `distillation_safedagger.py`: bumped default `num_iters` 100k → 350k, added `max_iterations` param
- `run_distillation_safedagger_fr3_agilehand.py`: wired `--max_iterations` CLI flag through to `SafeDagger`
- `eval_teacher.py`: added `import dextrah_lab.tasks.fr3_agilehand.gym_setup`
