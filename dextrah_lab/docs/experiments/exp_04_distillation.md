---
name: exp_04 distillation
description: Experiment 04 — SafeDagger distillation from multi-object teacher to vision-based student for FR3+AgileHand
type: project
---

# Experiment 04 — SafeDagger Distillation

**Goal:** Distill the multi-object teacher from exp_02 run1p into a vision-based student policy using SafeDagger.

**Teacher checkpoint:** `stored_policies/fr3_agilehand/09_again_multi_object_adr5_03-25_21-15-48/nn/best_dextrah_tekken_lstm.pth`
- Single multi-object teacher (13 visdex_selected objects), ADR 5, epoch 11124
- NOT per-object teachers — single teacher passed as `.pth` to `--teacher`

## run1a — 2026-03-26

**Command:**
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 8 --enable_cameras \
  --teacher ../stored_policies/fr3_agilehand/09_again_multi_object_adr5_03-25_21-15-48/nn/best_dextrah_tekken_lstm.pth \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/visdex_selected \
  env.enable_adr=False env.disable_arm_randomization=True
```

**Notes:** First distillation attempt with multi-object teacher. ADR disabled, arm randomization disabled.
