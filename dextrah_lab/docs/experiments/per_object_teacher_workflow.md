---
name: per_object_teacher_workflow
description: Multi-teacher distillation workflow for FR3+AgileHand — how to train per-object teachers and organize them for distillation
type: project
---

Train one teacher per object, then distill into a single student using multi-teacher routing.

**Why:** Per-object teachers (N=1) avoid obs-size mismatch during distillation. Multi-teacher routing in `distillation_safedagger.py` routes each env to the correct per-object teacher via `multi_object_idx`.

**How to apply:** When user asks about training teachers or starting distillation, refer to this workflow.

## Per-object training commands (from dextrah_lab/rl_games/)

**Plane** (already trained — best ckpt: `stored_policies/fr3_agilehand/08_ADR_7_single_object_plane_success_03_22_10_14_17/nn/last_dextrah_tekken_lstm_ep_16000_rew_15253.331.pth`):
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
  env.objects_dir=single_object_plane \
  env.use_cuda_graph=False
```

**Shoe** (`env.objects_dir=single_object_shoe` — assets dir created 2026-03-23):
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
  env.objects_dir=single_object_shoe \
  env.use_cuda_graph=False
```

**Female knight** (same as above but `env.objects_dir=single_object_female_knight`).

## Required directory structure for distillation

After training, copy best `.pth` for each object into:
```
stored_policies/fr3_agilehand/per_object_teachers/
├── plane/
│   └── last_dextrah_tekken_lstm_ep_XXXX_rew_XXXX.pth
├── shoe/
│   └── last_dextrah_tekken_lstm_ep_XXXX_rew_XXXX.pth
└── female_knight/
    └── last_dextrah_tekken_lstm_ep_XXXX_rew_XXXX.pth
```

Subfolder names **must match** the USD subfolder names in `assets/multi_objects/3/USD/`.

## Distillation command (from dextrah_lab/distillation_new/)

```bash
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 8 --enable_cameras \
  --teacher stored_policies/fr3_agilehand/per_object_teachers \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=multi_objects/3 \
  env.enable_adr=False env.disable_arm_randomization=True
```

## Single-object asset dirs (created 2026-03-23)

- `assets/single_object_shoe/USD/shoe` → symlink to `multi_objects/3/USD/shoe`
- `assets/single_object_female_knight/USD/female_knight` → symlink to `multi_objects/3/USD/female_knight`
- Both added to `valid_objects_dir` in `dextrah_fr3_agilehand_env_cfg.py`
- `test_object` already contains `plane` (used for the existing trained policy)
