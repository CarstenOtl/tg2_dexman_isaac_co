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
├── deployment_fr3_agilehand/  — Real-robot deployment scripts (FR3 + AgileHand)
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

### Teacher Training — full speed (headless, large batch)
```bash
cd dextrah_lab/rl_games
python train.py --headless --task=dextrah_fr3_agilehand --seed 42 \
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

### Teacher Training — multi-GPU (only if single GPU lacks VRAM)
```bash
export CUDA_VISIBLE_DEVICES=0,3  # select GPUs
python -m torch.distributed.run --nproc_per_node=2 \
  train.py --headless --distributed --task=dextrah_fr3_agilehand --seed 42 \
  --num_envs 2048 \
  agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 \
  ...
```
**Note:** Multi-GPU is ~2x slower per epoch than single GPU due to gradient sync overhead. Prefer single GPU with 1024 envs.

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

### Per-Object Teacher Training (automated)
```bash
cd dextrah_lab/rl_games
bash train_multi_objects_fr3_agilehand.sh
```
Loops over all objects in `multi_objects/14/USD/`, trains one teacher per object sequentially. Override defaults via env vars: `MULTI_OBJECTS_ROOT`, `MAX_EPOCHS`, `NUM_ENVS`, etc. Logs failed/succeeded objects in summary. Checkpoints land in `logs/rl_games/dextrah_tekken_lstm/{timestamp}_{obj_name}/nn/`.

### Student Evaluation (rich metrics)
```bash
cd dextrah_lab/distillation_new
python eval_student.py --task=dextrah_fr3_agilehand --num_envs 4 --enable_cameras --headless \
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

### SafeDagger Distillation Details
- Default iterations: 350k (`distillation_safedagger.py`), overridable via `--max_iterations` on launch script
- Resume student from checkpoint: `--network <path_to_student.pth>` on the `run_distillation_safedagger*.py` script
- Distillation metrics (lifted %, in_goal %) include teacher actions — student standalone performance is lower. Eval with `eval_student.py` to see true solo performance.
- `env.env` breaks when `RecordVideo` wrapper is active — always use `env.unwrapped` to access the Isaac env
- **Loss functions**: KL divergence (`"kl"`) outperforms L2 for distillation (DextrAH-RGB paper finding). Set via `imitation_loss_type` in dagger_config or `--vanilla_dagger` flag (which sets KL + disables unsafe override).
- **Vanilla DAgger vs SafeDagger**: `--vanilla_dagger` = KL loss + student always steps (no teacher override). SafeDagger (default) = L2 loss + teacher overrides unsafe envs. Vanilla DAgger produces better standalone students (50% vs 36% lift at 350k iters).
- **Noisy losses with vanilla DAgger + data_aug are expected** — no teacher safety net + visual augmentation variance. Check trends with TensorBoard smoothing (0.9+), not per-step values.

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

### Test Repo (`code/test/tg2_dexman_isaac_co`)

- Separate clone with editable install in `dextrah_test` conda env. Use `conda activate dextrah_test`.
- **Asset dirs must be COPIED, not symlinked** — Isaac Sim USD resolver doesn't follow symlinks. Use `cp -r`, not `ln -s`.
- Missing assets that may need copying from main repo: `test_object/`, `multi_objects/visdex_selected/`, `dome_light_textures/`, `background_imgs/`, `object_textures/`, `curated_table_textures/`.
- **`pip install -e . --no-deps`** to switch editable install without touching Isaac Sim dependencies.

### Distillation Gotchas (fr3_agilehand)

- **`--enable_cameras` is always required** for distillation and student eval — cameras are in the scene config, omitting it causes `RuntimeError` or silent hang.
- **`--headless` recommended** for eval scripts on remote machines — prevents GLFW display issues.
- **`pretrained_ckpts/` directory**: distillation scripts resolve relative `--teacher` paths via `os.path.join(parent_path, "pretrained_ckpts", teacher_arg)`. The dir must exist at repo root.
- **`eval_student.py` requires gym import for each task** — if eval hangs silently after scene creation, check that `import dextrah_lab.tasks.<task>.gym_setup` is present.
- **`eval.py` (legacy) hardcodes `dextrah_kuka_allegro/agents`** path for student config — works because YAMLs are identical across tasks, but prefer `eval_student.py`.
- **First camera run is slow** (10-20 min shader compilation) — subsequent runs use cached shaders.
- **SafeDagger beta** is NOT a fixed schedule — it's the fraction of envs where per-env L2 loss exceeds `unsafe_l2_threshold` (default 0.5). Teacher takes over only in those envs.

## Experiment Logs

Training experiment logs are tracked in `dextrah_lab/docs/experiments/`:
- `exp_02_runs.md` — Multi-object reward tuning, reward shaping, physics stability
- `exp_03_sim2real.md` — Sim2real curriculum: hardware actuator gains, thumb velocity, effort limits
- `exp_04_distillation.md` — SafeDagger distillation from multi-object teacher
- `per_object_teacher_workflow.md` — Per-object teacher training and directory structure

## Isaac Lab API Gotchas

- **PhysX view CPU tensors**: `root_physx_view.get_dof_max_forces()` etc. return CPU tensors. Index with `env_ids.cpu()` before `.to(device)`. `write_joint_*_to_sim()` expects shape `(len(env_ids), num_dofs)`, not `(num_envs, num_dofs)`.
- **ADR state is NOT saved in .pth checkpoints** — restored from `cfg.starting_adr_increments` (default 0). Set `env.starting_adr_increments=N` via CLI when resuming.
- **EventTerm field names in EventCfg must exactly match keys in `adr_cfg_dict`** — the ADR system looks up terms by name via `event_manager.get_term_cfg(term_name)`.
- **No Isaac Lab API for dynamic `max_depenetration_velocity`** — set at USD spawn time only. `write_joint_*_to_sim` exists for: stiffness, damping, effort_limit, velocity_limit, position_limit, armature, friction.
- **LSTM hidden states are NOT saved in checkpoints** — reset to zero on load. Resuming LSTM policies from checkpoints at high ADR levels causes degradation because the LSTM can't rebuild temporal context fast enough. Prefer training from scratch over checkpoint resume for LSTM policies.
- **`min_steps_for_dr_change` counts reset-batch calls, not epochs** — with 1024 envs, `_reset_idx` is called multiple times per epoch. `5 * episode_steps = 3000` ≈ 300-600 epochs per ADR step. Scale accordingly.
- **Finger gain ADR ranges** — proven sim2real range is `(0.5, 2.0)` (matching kuka_allegro/tg2_inspirehand). More aggressive ranges like `(0.1, 1.0)` cause vel_explode and training instability without sim2real benefit, since the real hardware PID handles position tracking independently.

## Code Conventions

### Entry-point script registration
Every script that accepts `--task` must import the task's `gym_setup` module (e.g. `import dextrah_lab.tasks.fr3_agilehand.gym_setup`). When adding a new task, check ALL entry points: `train.py`, `play_test.py`, `eval_teacher.py`, and every `run_distillation*.py` / `eval_student.py`.

### Video recording
`--video` flag (via `gym.wrappers.RecordVideo`) exists in: `train.py`, `eval_student.py`, all `run_distillation*.py`, legacy `eval.py`. NOT in: `eval_teacher.py`, `play_test.py` — use `--livestream 2` + external screen capture for those.

### Running headless
`eval_student.py` and other scripts with `--enable_cameras` will hang on machines without a display unless `--headless` is passed. GLFW initialization warnings are the symptom. Always use `--headless` when running remotely/SSH.

### CLI arg ordering for Hydra overrides
`env.*` overrides (e.g. `env.distillation=True`) go as bare positional args — `parse_known_args()` routes them to Hydra automatically. All `--flags` must come before `env.*` args. Do NOT use `--` separator with `eval_student.py` — bash interprets remaining args as separate shell commands.
- `eval_teacher.py` uses `parse_args()` (NOT `parse_known_args()`), so `env.*` Hydra overrides don't work. Use `--objects_dir` flag instead.
- `--teacher` flag in distillation scripts resolves relative paths as `<repo_root>/pretrained_ckpts/<value>`. Use absolute paths to skip this. The `--teacher` flag must come BEFORE the `--` separator, otherwise argparse doesn't see it.

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
- `hand_action_rate_penalty_scale=1.5` — finger joints penalized 1.5× more than arm for jerky actions

### Joint Init Positions (fr3_agilehand)

- Joint init positions (`revolute_thumb_rot`, finger joints) are set in `env_cfg.py` `init_state.joint_pos`, NOT in `fr3_tekken_left.py` — `env_cfg.py` values are applied at episode reset and override the asset file defaults.
- `revolute_thumb_rot` init at -0.3491 rad (-20°, joint min) pre-rotates the thumb away from the object approach path, preventing collision knock-back during approach.

### Early Termination Penalty (fr3_agilehand)

- `out_of_reach` (not `time_out`) is the done flag for premature episode endings. Set `early_termination_penalty: float` in `env_cfg.py`; env reads it via `getattr(self.cfg, "early_termination_penalty", 0.0)` and applies a flat penalty tensor in `_get_rewards` after `_get_dones` sets `self._early_terminated`.

### Termination Reason Masks (fr3_agilehand)
- `_get_dones()` must expose `self.last_*` boolean masks (e.g. `last_hand_too_far`, `last_palm_flipped`, `last_object_outside_upper_x`, etc.) for `eval_utils.py` to classify unsafe episode reasons. Pre-allocate in `__init__`, use `.copy_()` in `_get_dones()` — matches upstream `tg2_inspirehand` pattern.
- Without these, `eval_student.py` crashes with `RuntimeError: unclassified unsafe episodes` because the fallback recomputation in `eval_utils.py` references `tg2_inspirehand`-specific attributes.
- Eval categories: `object_out_of_bound`, `hand_too_far`, `harmful_collision`, `palm_flipped`, `physics_instability` (sim-only: `robot_unstable` + `vel_explosion`). Defined in `eval_utils.py:UNSAFE_REASON_NAMES`.

### Camera Config (fr3_agilehand distillation)

- Stereo cameras defined in `env_cfg.py` lines 352-424 as `TiledCameraCfg`
- Training resolution: 320×240 (`img_width=320`, `img_height=240`)
- FOV: ~48° horizontal (focal_length=23.59mm, horizontal_aperture=21.02mm)
- Stereo baseline: 55mm
- Camera pose from real-world 4×4 tf matrix (left); right is offset from left
- Camera randomization: ±3° rotation, ±0.03m position (for sim2real robustness)
- Cameras only activate when `env.distillation=True`
- At deployment: capture at native resolution (e.g., 1080p), resize to 320×240 before inference

### Deployment (FR3 + AgileHand)

- Deployment scripts in `dextrah_lab/deployment_fr3_agilehand/`
- ROS1 Noetic Docker container, stereo camera capture, policy inference node
- Inference pipeline: stereo images → resize to 320×240 → normalize → student policy → joint commands
- FR3 controlled via `franka_ros` / FCI (not bodyctrl_msgs like TG2)
- `policy_inference_stereo.py`: template inference node — proprio integration is TODO
- Camera intrinsics must match training config; calibrate real cameras and update `focal_length_val` and `horizontal_aperture` in env_cfg
