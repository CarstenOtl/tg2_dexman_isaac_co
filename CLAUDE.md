# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DextrAH (Dexterous Hand Manipulation) on Isaac Lab — a reinforcement learning framework for training hand-arm grasping policies in NVIDIA Isaac Sim. The two-stage pipeline trains a privileged teacher policy via PPO, then distills it into a vision-based student policy using DAGGER.

**Current focus**: FR3 + AgileHand robot configuration (branch `fr3_agilehand`).

## Key Dependencies

- Isaac Lab v2.2.1 (Isaac Sim underneath)
- FABRICS (geometric fabrics for arm control)
- rl-games (PPO training)
- PyTorch >= 2.4.0, warp-lang >= 1.5.0
- wandb (experiment tracking)

Install: `python -m pip install -e .` from repo root (within Isaac Lab conda env).

## Common Commands

All commands assume you're in the Isaac Lab conda environment.

### Teacher Training (privileged RL)
```bash
cd dextrah_lab/rl_games
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  train.py --headless --task=dextrah_fr3_agilehand --seed -1 --num_envs 1024 \
  agent.params.config.minibatch_size=4096 \
  agent.params.config.central_value_config.minibatch_size=4096 \
  agent.wandb_activate=False env.use_cuda_graph=False
```

### Replay Teacher Policy
```bash
cd dextrah_lab/rl_games
python play_test.py --task dextrah_fr3_agilehand --num_envs 4 \
  --objects_dir visdex_objects --checkpoint <path_to_checkpoint>
```

### Student Distillation (vision-based)
```bash
cd dextrah_lab/distillation
python -m torch.distributed.run --nproc_per_node=1 \
  run_distillation.py --headless --task=dextrah_fr3_agilehand \
  --num_envs 64 --enable_cameras --teacher <path_to_teacher> \
  env.distillation=True env.simulate_stereo=True
```

### Student Evaluation
```bash
cd dextrah_lab/distillation
python eval.py --task=dextrah_fr3_agilehand --num_envs 4 --enable_cameras \
  --checkpoint <path> --num_episodes 10 \
  env.distillation=True env.simulate_stereo=True
```

### Available Task IDs
- `dextrah_fr3_agilehand` — FR3 + AgileHand (active development)
- `Dextrah-Kuka-Allegro` — KUKA + Allegro hand
- `Dextrah-Kuka-Inspirehand` — KUKA + InspireHand
- `dextrah_tg2_inspirehand` — TG2 + InspireHand

### Debugging Hydra errors
Set `HYDRA_FULL_ERROR=1` before the command for full tracebacks.

## Architecture

### Task Structure (per robot configuration)

Each robot/hand combo lives in `dextrah_lab/tasks/<task_name>/` with this pattern:
- `dextrah_*_env.py` — Main environment class extending `DirectRLEnv`. Contains reward computation, sensor handling, episode resets, ADR curriculum, and physics management.
- `dextrah_*_env_cfg.py` — Config dataclass decorated with `@configclass` extending `DirectRLEnvCfg`. Defines scene, physics (PhysX), rewards, sensors (TiledCamera, ContactSensor), and ADR ranges.
- `dextrah_*_constants.py` — Hand PCA ranges, palm pose bounds, unit conversions.
- `dextrah_*_utils.py` — Tensor math helpers (quaternion ops, action scaling).
- `dextrah_adr.py` — Adaptive Domain Randomization curriculum manager.
- `gym_setup.py` — Gymnasium environment registration.
- `agents/` — RL-Games YAML configs for different network architectures (LSTM, feed-forward, transformer, CNN).
- `debug_tools/` — Ad-hoc diagnostic scripts for physics, rewards, and robot validation.

### Training Pipeline

1. **Teacher** (`dextrah_lab/rl_games/train.py`): PPO with LSTM/FF networks using privileged state (perfect object pose, contact forces). Uses `torch.distributed.run` for multi-GPU. Outputs checkpoints to `logs/rl_games/`.
2. **Student** (`dextrah_lab/distillation/run_distillation.py`): DAGGER behavioral cloning from teacher. Swaps privileged state for camera observations (RGB/depth, mono/stereo). Multiple encoder architectures in `a2c_*.py`, `mono_encoder.py`, `stereo_encoder.py`. Outputs to `runs/`.
3. **Evaluation** (`dextrah_lab/distillation/eval.py`): Runs student policy with optional data recording and video generation.

### Config Override Pattern

Configs are overridden via CLI using dot-notation (Hydra-style):
```
env.success_for_adr=0.4
env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]"
agent.params.config.learning_rate=0.0001
```

### Assets

Robot URDFs/USDs are in `dextrah_lab/assets/`. Training objects in `assets/visdex_objects/`. Visual textures (for distillation augmentation) must be downloaded separately from HuggingFace and placed in `dextrah_lab/assets/`.

Git LFS is used for `.pth` model weight files.

## Code Conventions

- Environment configs use Isaac Lab's `@configclass` decorator pattern
- GPU tensors throughout — all observation/action processing is batched on device
- ADR parameters are specified as `[min, max]` range lists in config
- `env.use_cuda_graph=True` speeds up training but may cause CUDA memory issues
- No formal test suite; validation is done via debug scripts in `debug_tools/`