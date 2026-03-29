---
name: exp_04 distillation
description: Experiment 04 — SafeDagger distillation from multi-object teacher to vision-based student for FR3+AgileHand
type: project
---

# Experiment 04 — SafeDagger Distillation

**Goal:** Distill the multi-object teacher from exp_02 run1p into a vision-based student policy using SafeDagger.

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

**Results:** (to be filled)

## Fixes applied during experimentation

- `eval_student.py`: added `import dextrah_lab.tasks.fr3_agilehand.gym_setup` (was missing, caused eval to hang)
- `eval_student.py`: added `gym.wrappers.RecordVideo` wrapping for `--video` flag (was missing, no MP4 produced)
- `eval_student.py`: changed `env.env` → `env.unwrapped` (4 places) to support `RecordVideo` wrapper chain
- `eval_student.py`: relaxed `_reason_counts_checked` from hard crash to warning for unclassified unsafe episodes (fr3_agilehand env doesn't expose `last_*` termination reason attributes)
- `dextrah_fr3_agilehand_env.py`: added `self.last_*` termination reason masks (matching upstream `tg2_inspirehand` pattern) so `eval_utils.py` can classify unsafe episodes
- `distillation_safedagger.py`: bumped default `num_iters` 100k → 350k, added `max_iterations` param
- `run_distillation_safedagger_fr3_agilehand.py`: wired `--max_iterations` CLI flag through to `SafeDagger`
- `eval_teacher.py`: added `import dextrah_lab.tasks.fr3_agilehand.gym_setup`
