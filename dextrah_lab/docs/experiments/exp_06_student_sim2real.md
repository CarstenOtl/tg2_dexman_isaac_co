---
name: exp_06 student sim2real
description: Experiment 06 — Improving student policy standalone performance and sim2real transfer for FR3+AgileHand
type: project
---

# Experiment 06 — Student Policy Improvement for Sim2Real

**Goal:** Get the SafeDagger student policy from ~36% standalone lift to sim2real-ready (>70% lift, <20% unsafe rate).

**Baseline (exp_04 run1c):** 36-45% lift success, 67.5% unsafe rate, beta=0.75, 350k iters, 16 envs, ADR disabled.

## Plan

### Phase 1 — Understand the ceiling
1. Eval the teacher standalone to know max achievable performance
2. Check per-object breakdown — identify which objects the student fails on
3. Inspect beta schedule — understand why it decays so slowly

### Phase 2 — Improve standalone student performance
4. Train longer (700k-1M iters) to let beta drop further
5. Enable `--data_aug` for visual diversity during distillation
6. Try lower learning rate (1e-4 vs current 2e-4) if loss plateau persists
7. Try faster beta decay if schedule is too conservative

### Phase 3 — Sim2real robustness
8. Distill with `env.enable_adr=True env.starting_adr_increments=5` — fixed physics randomization
9. Distill from sim2real teacher (exp_03 run1d) once available — teacher trained with effort/velocity limits
10. Add visual domain randomization (background textures, lighting, object textures)

## Runs

### run1 — Teacher eval baseline (2026-03-28)

**Purpose:** Establish teacher performance ceiling on the same object set.

**Command:**
```bash
cd dextrah_lab/rl_games
python eval_teacher.py --headless --task=dextrah_fr3_agilehand --num_envs 16 --eval_episodes 5 --checkpoint /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/stored_policies/fr3_agilehand/10_multi_object_adr19_1024envs_03-26_00-21-45/nn/best_dextrah_tekken_lstm.pth env.objects_dir=multi_objects/visdex_selected
```

**Results (2026-03-28):**

| Metric | Teacher | Student (exp_04 run1c) |
|---|---|---|
| Lift success | **96.25%** | 36.25% |
| Unsafe rate | 53.75% | 67.5% |
| Physics instability | 55.8% of failures | n/a (category added after student eval) |
| Object out of bound | 32.6% | 33.3% |
| Harmful collision | 11.6% | 40.7% |
| Palm flipped | 0% | 1.9% |
| Hand too far | 0% | 0% |
| Total episodes | 80 | 80 |

**Analysis:**
- Teacher ceiling is 96% lift — student has massive room to improve (36% → 96%)
- Teacher's 54% unsafe rate is dominated by `physics_instability` (56% of failures) — sim-only artifacts. Real-world-relevant unsafe rate is ~24%
- Student's main gap vs teacher: `harmful_collision` (41% vs 12%) — student crashes into table 3.5× more
- Object knockoff rates are similar (~33%) — inherent difficulty, not student-specific
- No palm flip or hand-too-far for either — arm control is solid in both
