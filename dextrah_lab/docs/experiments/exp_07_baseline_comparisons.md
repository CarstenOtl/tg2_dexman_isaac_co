---
name: exp_07 baseline comparisons
description: Experiment 07 — Train Kuka+Allegro and TG2+InspireHand teachers as baselines for comparison with FR3+AgileHand
type: project
---

# Experiment 07 — Baseline Comparisons

**Goal:** Train teacher policies for Kuka+Allegro and TG2+InspireHand to compare against FR3+AgileHand, then distill students for cross-robot comparison.

## Run 1 — Kuka+Allegro teacher training

**Date:** 2026-04-01

**Command:**
```bash
cd dextrah_lab/rl_games
python train.py \
  --headless \
  --task=Dextrah-Kuka-Allegro \
  --seed 42 \
  --num_envs 4096 \
  agent.params.config.minibatch_size=16384 \
  agent.params.config.central_value_config.minibatch_size=16384 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.wandb_activate=False \
  env.use_cuda_graph=True \
  env.objects_dir=visdex_objects \
  env.max_pose_angle=45.0
```

**Config notes:**
- Task: `Dextrah-Kuka-Allegro` — FGP/PCA-based hand control (11 actions: 6 palm pose + 5 PCA)
- 4096 envs, LSTM policy (default agent config: `rl_games_ppo_lstm_cfg.yaml`)
- CUDA graph enabled for speed
- `max_pose_angle=45.0` — palm orientation action range (required, default is placeholder -1)
- `objects_dir=visdex_objects` — full object set (required, default is placeholder "replace_me")

**Status:** complete

**Result:**
- Trained to epoch 6000
- Best reward: 1664.5 at ep 4200, declining to 1274.4 at ep 6000
- Checkpoint used: `logs/rl_games/dextrah_lstm/04-01_11-15-07/nn/last_dextrah_lstm_ep_6000_rew_1274.4115.pth`

### Teacher eval (ep 6000 checkpoint)

```bash
cd dextrah_lab/rl_games
python eval_teacher.py --headless --task=Dextrah-Kuka-Allegro --num_envs 64 \
  --eval_episodes 10 --objects_dir visdex_objects --max_pose_angle 45.0 \
  --checkpoint logs/rl_games/dextrah_lstm/04-01_11-15-07/nn/last_dextrah_lstm_ep_6000_rew_1274.4115.pth
```

| Metric | Value |
|---|---|
| Lift success | 99.84% |
| Unsafe episode rate | 25.0% |
| Total episodes | 640 |

Note: unsafe episodes are all unclassified — kuka_allegro env doesn't expose `last_*` termination masks.

## Run 2 — Kuka+Allegro vanilla DAgger distillation (L2 loss, 100k iters)

**Date:** 2026-04-01

**Command:**
```bash
cd dextrah_lab/distillation
export CUDA_VISIBLE_DEVICES=0,3
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  run_distillation.py \
  --headless --distributed --task=Dextrah-Kuka-Allegro --num_envs 48 --enable_cameras \
  --teacher logs/rl_games/dextrah_lstm/04-01_11-15-07/nn/last_dextrah_lstm_ep_6000_rew_1274.4115.pth \
  env.distillation=True env.simulate_stereo=True env.img_aug_type="rgb" env.aux_coeff=10. \
  env.objects_dir="visdex_objects" env.max_pose_angle=45.0 \
  env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
  env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
  env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
  env.use_cuda_graph=True
```

**Config notes:**
- Vanilla DAgger (`distillation.py`), L2 loss (default), stereo transformer student
- 48 envs across 2 GPUs (24 per GPU — tiled renderer limit)
- 100k iterations (default)
- Teacher: ep 6000 checkpoint (latest, not best)

**Result:**
- Final imitation loss: 2.25
- Final mean reward: 139.6
- Distillation in_success_region: 18.75% (includes teacher actions)
- Checkpoint: `distillation/runs/Dextrah-Kuka-Allegro_01-18-12-51/nn/dextrah_student_100000_iters.pth`

### Student eval (100k checkpoint)

```bash
cd dextrah_lab/distillation_new
python eval_student.py --task=Dextrah-Kuka-Allegro --num_envs 24 --enable_cameras --headless \
  --checkpoint ../distillation/runs/Dextrah-Kuka-Allegro_01-18-12-51/nn/dextrah_student_100000_iters.pth \
  --num_episodes 10 env.distillation=True env.simulate_stereo=True \
  env.objects_dir="visdex_objects" env.max_pose_angle=45.0
```

| Metric | Value |
|---|---|
| Lift success | 1.25% |
| Unsafe episode rate | 51.25% |
| Total episodes | 240 |

**Takeaway:** L2 loss at 100k iters produces near-zero standalone student performance despite 99.84% teacher. DextrAH-RGB paper finding: KL divergence significantly outperforms L2 for distillation.

### ADR mismatch analysis (2026-04-02)

**Critical finding:** The kuka_allegro env hardcodes `starting_adr_increments = num_adr_increments` (= 50) when `distillation=True` ([dextrah_kuka_allegro_env.py:150-152](../../tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env.py#L150-L152)):
```python
if self.cfg.distillation:
    self.cfg.starting_adr_increments = self.cfg.num_adr_increments  # = 50 (MAX)
```

The teacher was only trained to ~ADR 30 (epoch 6000, reward declining). During distillation the environment runs at ADR 50 — physics conditions the teacher has **never seen**. This creates a compounding problem:
1. **Teacher gives degraded demonstrations** — operating 20 ADR levels beyond its training distribution
2. **Student has maximum difficulty** — full randomization from step 0, no curriculum
3. **L2 loss on bad demos** — student learns to imitate a struggling teacher

This likely explains the 1.25% standalone lift despite a 99.84% teacher ceiling. The fr3_agilehand pipeline avoids this by setting `starting_adr_increments = 0` during distillation ([dextrah_fr3_agilehand_env.py:250-254](../../tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py#L250-L254)).

| Setting | kuka_allegro (Run 2) | fr3_agilehand (run3a/3b) |
|---|---|---|
| ADR during distillation | **50 (MAX)** | **0 (disabled)** |
| Teacher trained to ADR | ~30 | 14 |
| ADR gap | **20 levels beyond teacher** | 0 (within teacher range) |
| Standalone lift | 1.25% | 57-59% |

## Run 3 — Kuka+Allegro distillation with ADR 0 (planned)

**Date:** 2026-04-02 (planned)

**Motivation:** Test whether disabling ADR during distillation (matching fr3_agilehand approach) fixes the near-zero student performance. The teacher (99.84% lift) is strong — the bottleneck is likely the ADR mismatch, not the distillation method.

**Fix required:** Override the hardcoded ADR in `dextrah_kuka_allegro_env.py` by passing `env.starting_adr_increments=0` via CLI. Since the env sets `starting_adr_increments = num_adr_increments` *before* `set_num_increments()`, the CLI override may not take effect — need to verify or patch the env code.

**Option A — CLI override (if it works):**
```bash
cd dextrah_lab/distillation
export CUDA_VISIBLE_DEVICES=0,3
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  run_distillation.py \
  --headless --distributed --task=Dextrah-Kuka-Allegro --num_envs 48 --enable_cameras \
  --teacher logs/rl_games/dextrah_lstm/04-01_11-15-07/nn/last_dextrah_lstm_ep_6000_rew_1274.4115.pth \
  env.distillation=True env.simulate_stereo=True env.img_aug_type="rgb" env.aux_coeff=10. \
  env.objects_dir="visdex_objects" env.max_pose_angle=45.0 \
  env.starting_adr_increments=0 env.enable_adr=False \
  env.use_cuda_graph=True
```

**Option B — Patch the env code (APPLIED 2026-04-02):**
Removed the hardcoded `starting_adr_increments = num_adr_increments` override in `dextrah_kuka_allegro_env.py`. Distillation now defaults to ADR 0 (matching fr3_agilehand). CLI override `env.starting_adr_increments=N` is respected.

**Also use KL loss** (vanilla DAgger) instead of L2 — per DextrAH-RGB paper finding and fr3_agilehand run2a/3b results.

**Expected outcome:** Significant lift improvement (from 1.25% to potentially 50%+), matching the pattern seen in fr3_agilehand when teacher operates within its training distribution.

**Comparison matrix (planned):**

| Run | ADR | Loss | Method | Expected lift |
|---|---|---|---|---|
| Run 2 (done) | 50 (max) | L2 | Vanilla DAgger | 1.25% |
| Run 3 (planned) | 0 | L2 | Vanilla DAgger | ~30-50%? |
| Run 4 (future) | 0 | KL | Vanilla DAgger | ~50%+? |
