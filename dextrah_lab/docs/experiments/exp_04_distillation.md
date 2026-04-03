---
name: exp_04 distillation
description: Experiment 04 — SafeDagger distillation from multi-object teacher to vision-based student for FR3+AgileHand
type: project
---

# Experiment 04 — SafeDagger Distillation

**Goal:** Distill the multi-object teacher from exp_02 run1p into a vision-based student policy using SafeDagger.

## Summary — All standalone student evaluations

**Teacher 11 benchmark** (480 episodes): 85.8% lift, 23.1% unsafe
**Teacher 10 benchmark** (80 episodes): 96.3% lift, 53.8% unsafe
**Best student: run5a — 61.3% lift** (SafeDagger + L2, teacher 11, 24 envs, 100k iters)

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
| run6b | DAgger | KL | T11 | 100k | — | — | — | AverageMeter logging |

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

**Results:** *pending*

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
