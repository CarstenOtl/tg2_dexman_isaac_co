# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DextrAH (Dexterous Hand Manipulation) on Isaac Lab — a reinforcement learning framework for training hand-arm grasping policies in NVIDIA Isaac Sim. The two-stage pipeline trains a privileged teacher policy via PPO, then distills it into a vision-based student policy using SafeDagger.

**Current focus**: FR3 + AgileHand robot configuration (branch `fr3_agilehand`).

**Active branches:**
- `fr3_agilehand` — distillation experiments (student training from Teacher 11)
- `fr3_agilehand_teacher_v2` — Teacher v2 training with hardware-realistic starting limits (see exp_03 run2a)

**Distillation strategy**: Train one **per-object teacher** (N=1 each), then use **multi-teacher distillation** to produce a single vision-based student that works across all objects. The `distillation_safedagger.py` script handles multi-teacher routing automatically when `--teacher` points to a directory.

## Key Dependencies

- Isaac Lab v2.2.1 (Isaac Sim underneath)
- Isaac Lab source at `/home/carsten.oertel/code/IsaacLab/` — grep here to verify API fields (e.g. `RigidBodyMaterialCfg`, `PhysxCfg`)
- FABRICS (geometric fabrics for arm control)
- rl-games (PPO training)
- PyTorch >= 2.4.0, warp-lang >= 1.5.0
- wandb (experiment tracking)

Install: `python -m pip install -e .` from repo root (within Isaac Lab conda env).

**Conda env**: `dextrah_clean` at `/home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python` — used for training, distillation, eval, and plotting.

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
    ├── experiments/         — Experiment log markdown files
    ├── export_runs.py       — Export TensorBoard runs + eval JSONs to CSV
    └── scripts/
        ├── plot_style.py    — Thesis plotting style (colors, fonts, save_fig)
        └── plot_results.py  — Generate all thesis figures from experiment data
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

**Multi-object training needs ≥64 envs** (ideally 1024) — with 16 envs across 13 objects, each object gets ~1 env per batch, causing oscillating gradients and unstable learning. Use `num_envs ≥ 64 * num_objects` for stable training.

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

**Eval metric interpretation:**
- `lift_success` and `unsafe_episode_rate` are NOT mutually exclusive — an episode can lift the object AND terminate unsafely (e.g., lifted then knocked out of bounds). They don't sum to 1.
- `out_of_reach_reason_pct` values are percentages **within unsafe episodes** (sum to ~100%), NOT of all episodes. To get % of all episodes, multiply by `unsafe_episode_rate`. Example: physics_instability=47.7% with unsafe_rate=23.1% → 11.0% of all episodes.
- Always scale failure breakdowns when plotting or comparing across runs with different unsafe rates.

**Teacher 11 benchmark** (480 episodes, `eval_metrics_20260331_193003.json`): 85.8% lift, 23.1% unsafe. All fr3_agilehand student comparisons are relative to this teacher.

**Teacher v2** (branch `fr3_agilehand_teacher_v2`, exp_03 run2a): *training in progress*. Hardware-realistic starting limits (90/20 Nm arm effort, 15 deg/s thumb velocity, 60 stiffness thumb_rot, soft_joint_pos_limit=0.8, arm init randomization ±0.2 rad). Goal: push past ADR 14 ceiling by reducing curriculum gap.

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

### FABRICS vs Direct Joint Control
- `kuka_allegro` uses FABRICS (geometric fabrics) as intermediate controller — smooths policy outputs into physically consistent trajectories. The FABRICS controller is deterministic with no internal randomization; all noise comes from ADR at reset.
- `fr3_agilehand`, `tg2_inspirehand`, `kuka_inspirehand` use direct joint position control — no smoothing layer. More sensitive to spawn noise since the policy must handle recovery trajectories itself.
- When comparing ADR parameters across tasks, account for FABRICS' smoothing effect — the same `robot_spawn.joint_pos_noise` is much harder to handle without it.

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
- **`--data_aug` is a NO-OP in `distillation_safedagger.py`** — the flag is passed in config but never read. Only legacy `distillation.py` implements RGB augmentation. Env-level visual randomization (dome light 30%, textures) is always active regardless.
- **Student collapses beyond ~350k iters** with vanilla DAgger + KL at lr=2e-4 (50% lift at 350k → 0% at 700k). Needs LR decay or early stopping. Save checkpoints frequently to find the peak.
- **Per-object metrics** are logged to TensorBoard during distillation: `per_object_lift/<name>`, `per_object_unsafe/<name>`. Requires `multi_object_idx` and `object_names` on the env (all multi-object tasks have these).
- **Termination breakdown** is logged separately: `termination/real_unsafe` (object OOB, hand too far, palm flipped) vs `termination/physics_instability` (vel_explosion + robot_unstable). Use `real_unsafe` for sim2real-relevant comparisons — physics instabilities are simulation artifacts.
- **`beta` metric** = fraction of envs where student L2 loss exceeds `unsafe_l2_threshold` (0.5). For SafeDagger, teacher overrides in those envs. For vanilla DAgger, beta stays ~1.0 (no override, but check still runs). Effectively the training-time unsafe rate.
- **Run directories are timestamp-unique** — experiment name includes `datetime.now().strftime("_%d-%H-%M-%S")`, so parallel runs won't overwrite each other unless started in the same second.
- **Noisy per-object metrics** — with 24 envs and 13 objects, each object gets ~2 envs. Per-step data is binary (0/1). Use EMA smoothing α=0.999 (matching TensorBoard 0.999) for readable per-object plots, not rolling mean.
- **`per_object_unsafe/<name>` is L2-based**, not actual terminations. For real unsafe episodes (object OOB, hand too far, palm flipped), use `per_object_term_real/<name>` (added 2026-04-02, requires re-run to populate).
- **`in_success_region` vs `lift_success`** — `in_success_region` = object at goal position (strict). `lift_success` = object above table (less strict). Per-object training plots logged `in_success_region` which shows zero for hard objects that get lifted but never reach goal. `self.lift_success` per-env tensor logged as `per_object_lifted/<name>` (added 2026-04-03, requires re-run).
- **Per-step vs episode-level termination rates** — `termination/real_unsafe` is a per-step instantaneous rate (~0.1%), NOT comparable to eval's episode-level unsafe rate (~50-75%). Episode-level metrics (`episode/unsafe_rate`, `episode_per_object_unsafe/<name>`) added 2026-04-03 — these accumulate episode outcomes and match eval numbers.

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

**Object subsets for limited envs**: `multi_objects/visdex_top8/` contains the 8 best-performing objects from Teacher 11 eval (≥60% lift): toy_cow, mario, teddy_bear, train, plane, basketball_shoe, closed_fist, milk_pot. Use with 24 envs for 3 envs/object. A single teacher trained on all 13 objects works with any subset — `teacher_onehot` is a size-1 placeholder, not N-dimensional, so obs space is independent of object count.

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
- **`--headless` is always required** for distillation and eval on remote machines — prevents GLFW display hang.
- **Max camera envs on RTX 4090:** 24 works (~11GB), 32 hangs (deadlocks in tiled renderer). 16 is safe (~9.5GB).
- **`pretrained_ckpts/` directory**: distillation scripts resolve relative `--teacher` paths via `os.path.join(parent_path, "pretrained_ckpts", teacher_arg)`. The dir must exist at repo root.
- **`eval_student.py` requires gym import for each task** — if eval hangs silently after scene creation, check that `import dextrah_lab.tasks.<task>.gym_setup` is present.
- **`eval.py` (legacy) hardcodes `dextrah_kuka_allegro/agents`** path for student config — works because YAMLs are identical across tasks, but prefer `eval_student.py`.
- **First camera run is slow** (10-20 min shader compilation) — subsequent runs use cached shaders.
- **SafeDagger beta** is NOT a fixed schedule — it's the fraction of envs where per-env L2 loss exceeds `unsafe_l2_threshold` (default 0.5). Teacher takes over only in those envs.
- **Multi-GPU distillation is NOT supported** — `distillation_safedagger.py` doesn't implement `torch.distributed.run`. `CUDA_VISIBLE_DEVICES=0,1` alone does NOT split workload. Instead, run two separate distillation runs on different GPUs in parallel (e.g., SafeDagger on GPU 0, DAgger on GPU 1).
- **Always use `multi_objects/visdex_selected`** for distillation — matches the teacher's training set.
- **`--teacher` with absolute path bypasses `pretrained_ckpts/` resolution** — the code checks `os.path.isabs()` first.

### Distillation Gotchas (kuka_allegro)

- **ADR during distillation was hardcoded to max** — `dextrah_kuka_allegro_env.py` previously forced `starting_adr_increments = num_adr_increments` (50) when `distillation=True`. Patched 2026-04-02 to default to 0 (matching fr3_agilehand). CLI override `env.starting_adr_increments=N` now works.
- **Kuka+Allegro distillation uses legacy `distillation.py`** (vanilla DAgger, `torch.distributed.run`) — NOT `distillation_safedagger.py`. A new `run_distillation_safedagger_kuka_allegro.py` was added for SafeDagger pipeline support.

### Sim2Real Insights (fr3_agilehand)

- **Joint stiffness/damping don't directly map to real hardware** — the policy outputs target positions, and real hardware has its own PID controller. What matters for sim2real is effort limits (achievable force) and velocity limits (achievable speed per step), not sim PD gains.
- **Verify joint tracking in Isaac gain tuner** before training — default gains at ADR 0 must show clean tracking. Poor tracking (oscillation, amplitude mismatch) causes vel_explode at higher ADR.
- **Thumb rot velocity is the key sim2real constraint** — real hardware limit is 8-12 deg/s. ADR curriculum ramps from 10 rad/s → 0.14 rad/s (8 deg/s).
- **Contact reward easily dominates lift** — with 5+ fingertips, contact compounds. Keep `hand_object_contact_weight` ≤ 3.0 and use `lift_sharpness` ≤ 2.0 so lift gradient is discoverable from table height.

## Experiment Logs

Training experiment logs are tracked in `dextrah_lab/docs/experiments/`. **Convention:** Log new distillation runs in `exp_04_distillation.md` with: full bash command, GPU assignment, env count, objects_dir, run directory, and what metrics are being tracked. Mark as `*running*` in summary table until results are in.
- `exp_02_runs.md` — Multi-object reward tuning, reward shaping, physics stability
- `exp_03_sim2real.md` — Sim2real curriculum: hardware actuator gains, thumb velocity, effort limits, Teacher v2 (run2a)
- `exp_04_distillation.md` — SafeDagger distillation from multi-object teacher
- `per_object_teacher_workflow.md` — Per-object teacher training and directory structure
- `exp_07_baseline_comparisons.md` — Kuka+Allegro and TG2+InspireHand teacher/distillation baselines

### Thesis Plotting & Data Export

**Export TensorBoard data to CSV** (prerequisite for plotting from real run data):
```bash
cd dextrah_lab/docs
python export_runs.py [--output_dir exports/]
```
Exports teacher training curves (from `stored_policies/fr3_agilehand/`) and distillation curves (from `distillation_new/runs/`) to long-form CSV (`step, metric, value`). Also copies eval JSON files and produces `eval_summary.csv`. Requires `tbparse` and `pandas`.

**Generate thesis figures**:
```bash
cd dextrah_lab/docs/scripts
python plot_results.py [--show]
```
Generates all thesis plots to `dextrah_lab/docs/Report/figures/plots/` (PDF + PNG). Available plots:
- Hardcoded data: `all_runs_gsr`, `failure_breakdown`, `teacher_comparison`, `teacher_training_curve`, `adr_ranges`, `beta_decay`, `physics_challenges_timeline`
- CSV-based (from `export_runs.py` output): `distillation_lift_success_100k`, `distillation_unsafe_rate_100k`, `distillation_comparison_3panel`, `per_object_lift_comparison`, `per_object_lift_head_to_head`, `per_object_unsafe_real_comparison` — read exported CSVs and eval JSONs from `docs/exports/`

**Plot style** (`plot_style.py`): thesis-consistent style — Charter/serif font, golden-ratio aspect, 300 DPI export, academic color palette. Import `apply_style()` before creating figures, `save_fig(fig, name)` to save.

**Environment:** Always run plotting scripts with `dextrah_clean` conda env (`/home/carsten.oertel/bin/yes/envs/dextrah_clean/bin/python`). System python has numpy 2.x / matplotlib 1.x incompatibility.

**CSV step-to-iteration conversion:** Exported CSV `step` = `iteration × num_envs`. For 16-env runs divide by 16, for 24-env runs divide by 24. `_load_distillation_metric()` in `plot_results.py` handles this automatically.

## Isaac Lab API Gotchas

- **PhysX view CPU tensors**: `root_physx_view.get_dof_max_forces()` etc. return CPU tensors. Index with `env_ids.cpu()` before `.to(device)`. `write_joint_*_to_sim()` expects shape `(len(env_ids), num_dofs)`, not `(num_envs, num_dofs)`.
- **ADR state is NOT saved in .pth checkpoints** — restored from `cfg.starting_adr_increments` (default 0). Set `env.starting_adr_increments=N` via CLI when resuming.
- **EventTerm field names in EventCfg must exactly match keys in `adr_cfg_dict`** — the ADR system looks up terms by name via `event_manager.get_term_cfg(term_name)`.
- **No Isaac Lab API for dynamic `max_depenetration_velocity`** — set at USD spawn time only. `write_joint_*_to_sim` exists for: stiffness, damping, effort_limit, velocity_limit, position_limit, armature, friction.
- **LSTM hidden states are NOT saved in checkpoints** — reset to zero on load. Resuming LSTM policies from checkpoints at high ADR levels causes degradation because the LSTM can't rebuild temporal context fast enough. Prefer training from scratch over checkpoint resume for LSTM policies.
- **`min_steps_for_dr_change` counts reset-batch calls, not epochs** — with 1024 envs, `_reset_idx` is called multiple times per epoch. `5 * episode_steps = 3000` ≈ 300-600 epochs per ADR step. Scale accordingly.
- **Finger gain ADR ranges** — proven sim2real range is `(0.5, 2.0)` (matching kuka_allegro/tg2_inspirehand). More aggressive ranges like `(0.1, 1.0)` cause vel_explode and training instability without sim2real benefit, since the real hardware PID handles position tracking independently.

### ADR Plateau Debugging (fr3_agilehand)
- **ADR 13 wall pattern**: If policy plateaus at a specific ADR level, check (1) reward curriculum eroding signal — rewards declining *within* a fixed ADR level means reward shaping issue, not randomization; (2) `robot_spawn.joint_pos_noise` compounding with arm EventTerm randomization — fr3_agilehand has both, kuka_allegro only has spawn noise; (3) hand_to_object_distance increasing = arm can't reach object = spawn noise too aggressive.
- **`robot_spawn.joint_pos_noise` for fr3_agilehand** should be (0, 0.35) matching kuka_allegro — the ±0.2 rad arm init EventTerm already provides diversity from step 0. Original (0, 0.8) caused 4.5× more randomization than kuka_allegro at same ADR level.
- **`lift_sharpness` affects ADR progression**: 2.0 (flat) → policy hovers near table for easy reward, never reaches goal → `in_success_region` stays low. 4.0+ saturates lift reward faster, forcing goal reward to dominate. kuka_allegro uses 8.5.
- **Object set composition affects `in_success_region` average**: 13 curated objects (visdex_selected) have higher difficulty than 152 ShapeNet objects (visdex_objects). Hard objects drag average below `success_for_adr` threshold (0.4).
- **Reward curriculum can erode within a fixed ADR level** — `lift_weight` decay + `object_to_goal_sharpness` increase + `finger_curl_reg` increase compound to reduce total reward signal. If metrics decline at fixed ADR, the reward shaping is the issue.
- **Eval at ADR 0 reveals reward-driven forgetting** — `eval_teacher.py` runs at ADR 0 (no randomization). If a later checkpoint trained at ADR N performs *worse* at lifting than an earlier checkpoint pre-ADR-N, the policy is forgetting learned behavior under reward decay (not failing due to harder conditions). This was observed in run2g: ep 6500 (ADR 13 settled) had 75.9% lift vs ep 2500 (pre-ADR-13) at 79.1%.
- **ADR 13 wall is structural for fr3_agilehand Teacher v2** — invariant across reward tuning, spawn noise reduction, object set changes (13/152), arm gain tightening, finger gain tightening. Best Teacher v2 result: 79.1% lift / 25.3% unsafe (run2g ep 2500), ~7pp lift gap to Teacher 11 (85.8%). Hardware-realistic actuator constraints from step 0 pay a real cost.
- **`eval_teacher.py` runs at ADR 0** — no `--starting_adr_increments` flag. To eval at a higher ADR level, modify the env config or set via Hydra (which `eval_teacher.py` doesn't fully support — use `train.py` with `--checkpoint` and a tiny step count instead).

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
- `thumb_rot_init` EventTerm randomizes thumb rotation between -20° and 0° at reset via `mdp.reset_joints_by_offset` — critical for preventing the policy from learning to wedge the thumb between object and fingers.
- Changing finger stiffness/damping changes what the **same action** produces physically. Higher stiffness = fingers reach targets faster = actions that worked with sluggish fingers cause overshoot/curling with responsive ones. Requires retraining from scratch.

### Early Termination Penalty (fr3_agilehand)

- `out_of_reach` (not `time_out`) is the done flag for premature episode endings. Set `early_termination_penalty: float` in `env_cfg.py`; env reads it via `getattr(self.cfg, "early_termination_penalty", 0.0)` and applies a flat penalty tensor in `_get_rewards` after `_get_dones` sets `self._early_terminated`.
- `_penalty_terminated` is a subset of `_early_terminated` — only includes intentional bad behavior (object OOB, hand OOB, palm flip). Excludes physics artifacts (vel_explosion, robot_unstable, hand_too_close, arm_table_contact) which should not be penalized.
- `_apply_actuator_curriculum()` writes effort limits and velocity limits to sim on every reset + after ADR steps. Uses `root_physx_view` (CPU tensors — index with `env_ids.cpu()`). Covers params NOT handled by EventTerms: `thumb_rot_vel_limit`, `arm_14_effort_limit`, `arm_57_effort_limit`.

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
