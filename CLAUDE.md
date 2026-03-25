# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DextrAH (Dexterous Hand Manipulation) on Isaac Lab — a reinforcement learning framework for training hand-arm grasping policies in NVIDIA Isaac Sim. The two-stage pipeline trains a privileged teacher policy via PPO, then distills it into a vision-based student policy using SafeDagger.

**Current focus**: FR3 + AgileHand robot configuration (branch `fr3_agilehand`).

**Distillation strategy**: Train one **per-object teacher** (N=1 each), then use **multi-teacher distillation** to produce a single vision-based student that works across all objects. The `distillation_safedagger.py` script handles multi-teacher routing automatically when `--teacher` points to a directory.

## Key Dependencies

- Isaac Lab v2.2.1 (Isaac Sim underneath)
- Isaac Lab source at `/home/carsten.oertel/code/IsaacLab/` — grep here to verify API fields (e.g. `RigidBodyMaterialCfg`, `PhysxCfg`)
- FABRICS (geometric fabrics for arm control)
- rl-games (PPO training)
- PyTorch >= 2.4.0, warp-lang >= 1.5.0
- wandb (experiment tracking)

Install: `python -m pip install -e .` from repo root (within Isaac Lab conda env).

## Project Directory Structure

```
dextrah_lab/
├── assets/
│   ├── FrankaFR3/           — FR3 arm USD/URDF
│   ├── fr3_tekken_adof/     — FR3 + TekkenAdof (AgileHand) USD assets
│   ├── kuka_allegro/        — KUKA + Allegro assets
│   ├── kuka_inspirehand/    — KUKA + InspireHand assets
│   ├── tg2_inspirehand/     — TG2 + InspireHand assets
│   ├── tekken_adof/         — Standalone AgileHand URDF/USD
│   ├── multi_objects/       — Multi-object sets (subfolders: 3/, 14/)
│   ├── visdex_objects/      — Main training object set (USD per object)
│   ├── primitives/          — Geometric primitive objects
│   ├── background_imgs/     — RGB augmentation backgrounds
│   ├── object_textures/     — Object texture augmentation
│   ├── curated_table_textures/
│   └── dome_light_textures/
├── distillation/            — Legacy distillation scripts (DAGGER)
│   └── eval.py              — Basic student eval (success rate only)
├── distillation_new/        — Active distillation scripts (SafeDagger)
│   ├── run_distillation_safedagger.py  — Main distillation entry point
│   ├── distillation_safedagger.py      — Core SafeDagger logic + multi-teacher routing
│   ├── eval_student.py      — Rich student eval (lift+hold, unsafe rate, per-object metrics)
│   ├── eval_utils.py        — Shared metric helpers (imported by eval_student + eval_teacher)
│   └── runs/                — Distillation outputs
├── rl_games/                — Teacher training
│   ├── train.py             — PPO teacher training entry point
│   ├── play_test.py         — Teacher policy replay
│   ├── eval_teacher.py      — Teacher evaluation (single ckpt or per-object dir)
│   └── logs/                — Training checkpoints
├── stored_policies/         — Saved checkpoints by robot config
│   ├── fr3_agilehand/
│   ├── kuka_inspirehand/
│   └── tg2_inspirehand/
├── tasks/
│   ├── fr3_agilehand/       — FR3 + AgileHand (active development)
│   ├── tg2_inspirehand/     — TG2 + InspireHand
│   ├── dextrah_kuka_allegro/
│   ├── dextrah_kuka_inspirehand/
│   └── dextrah_kuka_inspirehand_v2/
├── deployment_tg2_inspirehand/ — Real-robot deployment scripts (TG2)
└── docs/
```

## Common Commands

All commands assume you're in the Isaac Lab conda environment and run from `dextrah_lab/rl_games/` (training) or `dextrah_lab/distillation_new/` (distillation).

### Teacher Training — development (livestream, small batch)
```bash
cd dextrah_lab/rl_games
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
  env.objects_dir=multi_objects/3 \
  env.use_cuda_graph=False
```

### Teacher Training — full speed (headless, large batch, resume from checkpoint)
```bash
cd dextrah_lab/rl_games
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 512 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=2048 \
  agent.params.config.central_value_config.minibatch_size=2048 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.learning_rate=0.0001 \
  agent.params.config.multi_gpu=False \
  agent.params.config.max_epochs=20000 \
  agent.wandb_activate=False \
  env.success_for_adr=0.4 \
  env.objects_dir=test_object \
  env.use_cuda_graph=False \
  --checkpoint logs/rl_games/dextrah_tekken_lstm/<run>/nn/dextrah_tekken_lstm.pth
```

### Evaluate Teacher Policy
```bash
cd dextrah_lab/rl_games
python eval_teacher.py --task dextrah_fr3_agilehand --eval_episodes 10 \
  --checkpoint <path>
# Or evaluate a whole per-object teacher directory:
python eval_teacher.py --task dextrah_fr3_agilehand --eval_episodes 10 \
  --teacher_policy_dir pretrained_ckpts/per_object_teachers
```

### Student Distillation — SafeDagger (single teacher)
```bash
cd dextrah_lab/distillation_new
python run_distillation_safedagger_fr3_agilehand.py \
  --task=dextrah_fr3_agilehand --num_envs 8 --enable_cameras \
  --teacher <path_to_checkpoint.pth> \
  env.distillation=True env.simulate_stereo=True \
  env.objects_dir=test_object \
  env.enable_adr=False env.disable_arm_randomization=True
```
For multi-teacher mode, pass a **directory** to `--teacher` (one subfolder per object name, each with a `.pth`).

> **Note**: Vanilla DAgger requires `torch.distributed.run` — see `notes.txt` for the full command.

### Student Evaluation (rich metrics)
```bash
cd dextrah_lab/distillation_new
python eval_student.py --task=dextrah_fr3_agilehand --num_envs 4 --enable_cameras \
  --checkpoint <path> --num_episodes 10 \
  env.distillation=True env.simulate_stereo=True
```
Reports: lift success (hold-gated), unsafe episode rate, failure reason breakdown, per-object metrics. Prefer over legacy `distillation/eval.py`.

### Available Task IDs
- `dextrah_fr3_agilehand` — FR3 + AgileHand (active development)
- `Dextrah-Kuka-Allegro` — KUKA + Allegro hand
- `Dextrah-Kuka-Inspirehand` — KUKA + InspireHand
- `dextrah_tg2_inspirehand` — TG2 + InspireHand

### Debugging Hydra errors
Set `HYDRA_FULL_ERROR=1` before the command for full tracebacks.

### Restoring files from a specific commit
`git checkout <commit> -- dextrah_lab/assets/fr3_tekken_adof/FR3_tekkenadof_left.usd` — works for any file path.

## Architecture

### Task Structure (per robot configuration)

Each robot/hand combo lives in `dextrah_lab/tasks/<task_name>/` with this pattern:
- `dextrah_*_env.py` — Main environment class extending `DirectRLEnv`. Contains reward computation, sensor handling, episode resets, ADR curriculum, and physics management.
- `dextrah_*_env_cfg.py` — Config dataclass decorated with `@configclass` extending `DirectRLEnvCfg`. Defines scene, physics (PhysX), rewards, sensors (TiledCamera, ContactSensor), and ADR ranges.
- `dextrah_*_constants.py` — **Only relevant for `kuka_allegro`** (FGP/PCA-based control). All other tasks use direct joint position control and do not use this file.
- `dextrah_*_utils.py` — Tensor math helpers (quaternion ops, action scaling).
- `dextrah_adr.py` — Adaptive Domain Randomization curriculum manager.
- `gym_setup.py` — Gymnasium environment registration.
- `agents/` — RL-Games YAML configs for different network architectures (LSTM, feed-forward, transformer, CNN).
- `debug_tools/` — Ad-hoc diagnostic scripts for physics, rewards, and robot validation.

**Action space**: `kuka_allegro` uses FGPs (Finger Grasp Primitives / PCA basis) for hand control. All other tasks (`fr3_agilehand`, `tg2_inspirehand`, `kuka_inspirehand`) use **direct joint position control** — actions map directly to target joint angles.

### Training Pipeline

1. **Teacher** (`dextrah_lab/rl_games/train.py`): PPO with LSTM/FF networks using privileged state (object pose, contact forces). Train one teacher per object (N=1). Checkpoints → `logs/rl_games/`.
2. **Student** (`dextrah_lab/distillation_new/run_distillation_safedagger.py`): SafeDagger behavioral cloning. Multi-teacher mode auto-activates when `--teacher` is a directory — each env is routed to the correct per-object teacher via `multi_object_idx`. Outputs → `runs/`.
3. **Evaluation** (`dextrah_lab/distillation_new/eval_student.py`): Rich metrics including hold-gated lift success, unsafe episode rate, and failure reason breakdown. `eval_teacher.py` for teacher-side benchmarking.

### Multi-Teacher Routing (distillation_safedagger.py)

- `_build_teacher_pool(ckpt_dir)`: loads one model per object from subdirectories
- `_get_actions_multi_teacher(obs)`: for each teacher model, selects envs where `multi_object_idx == obj_idx`, runs inference, writes results back
- RNN hidden states are maintained per teacher model independently
- Reads `env.object_names` and `env.multi_object_idx` from the environment

### Config Override Pattern

Configs are overridden via CLI using dot-notation (Hydra-style):
```
env.success_for_adr=0.4
env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]"
agent.params.config.learning_rate=0.0001
```

### Assets

Robot URDFs/USDs are in `dextrah_lab/assets/`. Training objects in `assets/visdex_objects/` (one subfolder per object, each containing `USD/` dir). Visual textures (for distillation augmentation) must be downloaded separately from HuggingFace and placed in `dextrah_lab/assets/`.

**Single-object training dirs**: `assets/test_object` = plane (existing). For other objects, create `assets/single_object_<name>/USD/<name>` as a symlink to `multi_objects/3/USD/<name>`, then add `"single_object_<name>"` to `valid_objects_dir` in `dextrah_fr3_agilehand_env_cfg.py` — training will crash without this.

**Per-object teacher collection** (for multi-teacher distillation): copy best `.pth` for each object into `stored_policies/fr3_agilehand/per_object_teachers/<object_name>/`. Subfolder names must match `multi_objects/3/USD/` subdirectory names exactly.

Git LFS is used for `.pth` model weight files.

**USD physics bake-in**: `FR3_tekkenadof_left.usd` and `fr3.usd` bake in joint physics (damping, joint limits, collision meshes) that can override Python actuator config. When `vel_explosion` fires on every reset after a config change, restore the USD from the known-good commit.

**rl_games version**: Must use the isaac-sim fork, NOT pip 1.6.1. Install: `pip install git+https://github.com/isaac-sim/rl_games.git@6b3534f29568158e9e29ec8bf83cc88fce5f0cae`. Pinned requirements reference: `../requirements_common_chi_pinned.txt` (one level above repo root).

## Code Conventions

- Environment configs use Isaac Lab's `@configclass` decorator pattern
- GPU tensors throughout — all observation/action processing is batched on device
- ADR parameters are specified as `[min, max]` range lists in config
- `env.use_cuda_graph=True` speeds up training but may cause CUDA memory issues
- No formal test suite; validation is done via debug scripts in `debug_tools/`

### Observation Space (fr3_agilehand)

| Obs | Size | Contents |
|---|---|---|
| Teacher (policy) | `teacher_base + 1` | dof_pos/vel, hand_pos/vel, obj_pos/rot/goal/scale, actions, `teacher_onehot=[1]` |
| Student | `student_base` | dof_pos/vel, hand_pos/vel, obj_goal, actions — no object state, no one-hot |
| Critic | `critic_base + 1` | Full state including forces, torques, obj_vel, `teacher_onehot=[1]` |

- `teacher_onehot` is a fixed `ones(num_envs, 1)` — size-1 placeholder so per-object teachers (trained N=1) stay compatible during multi-object distillation
- `multi_object_idx_onehot` (N-dimensional) is still computed but NOT used in obs — only `multi_object_idx` is used by distillation for teacher routing

### Physics Material (fr3_agilehand)

- `RigidBodyMaterialCfg` supports `friction_combine_mode` and `restitution_combine_mode`: `"average" | "min" | "multiply" | "max"` (verified in Isaac Lab source)
- Global sim material uses `friction_combine_mode="max"` and `restitution_combine_mode="max"` — rubber fingertip friction dominates contact
- Fingertip: static=2.0, dynamic=1.5 at episode start; Object: static=1.0, dynamic=1.0 (ADR widens ranges during training)
- `hand_action_rate_penalty_scale=2.5` — finger joints penalized 2.5× more than arm for jerky actions

### Joint Init Positions (fr3_agilehand)

- Joint init positions (`revolute_thumb_rot`, finger joints) are set in `env_cfg.py` `init_state.joint_pos`, NOT in `fr3_tekken_left.py` — `env_cfg.py` values are applied at episode reset and override the asset file defaults.
- `revolute_thumb_rot` init at -0.3491 rad (-20°, joint min) pre-rotates the thumb away from the object approach path, preventing collision knock-back during approach.

### Early Termination Penalty (fr3_agilehand)

- `out_of_reach` (not `time_out`) is the done flag for premature episode endings. Set `early_termination_penalty: float` in `env_cfg.py`; env reads it via `getattr(self.cfg, "early_termination_penalty", 0.0)` and applies a flat penalty tensor in `_get_rewards` after `_get_dones` sets `self._early_terminated`.
