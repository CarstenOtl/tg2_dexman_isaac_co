---
name: exp_05 eval and data collection
description: Experiment 05 — Evaluation tooling, video recording, and data collection fixes for FR3+AgileHand
type: project
---

# Experiment 05 — Evaluation & Data Collection

**Goal:** Get evaluation scripts (`eval_student.py`, `eval_teacher.py`) and video/data recording working for the `fr3_agilehand` task.

## Task registration fixes — 2026-03-27

**Problem:** `eval_teacher.py` and `eval_student.py` silently hung or failed when run with `--task dextrah_fr3_agilehand` because they didn't import the task's gym_setup module.

**Fix:** Added `import dextrah_lab.tasks.fr3_agilehand.gym_setup` to both scripts. Every entry-point script that accepts `--task` must register the task — see CLAUDE.md "Entry-point script registration" for the full list.

## Bug fix: `last_*` termination masks for eval_student.py — 2026-03-27

**Problem:** `eval_student.py` crashed with `RuntimeError: found N unclassified unsafe episodes` when running on `dextrah_fr3_agilehand`.

**Root cause:** The eval pipeline needs to classify *why* each early termination happened (object out of bounds, hand too far, harmful collision, palm flipped). It does this by reading `self.last_*` boolean mask attributes from the env after each `env.step()`. The upstream `tg2_inspirehand` env already exposes these, but `fr3_agilehand` didn't — the termination reasons were computed as local variables in `_get_dones()` and discarded after the function returned.

Without `last_*`, `eval_utils.py` falls back to recomputing reasons from env state, but that fallback references `ov_env.middle_link_0_body_idx` (a `tg2_inspirehand` attribute) — `fr3_agilehand` uses `hand_workspace_body_idx` instead, so the fallback silently fails, all reason masks stay zero, and `_reason_counts_checked()` raises because unsafe episodes can't be classified.

**Fix:** Added `self.last_*` tensors to `fr3_agilehand`, matching the upstream `tg2_inspirehand` pattern:
1. Pre-allocate 9 boolean tensors in `__init__` (one per termination condition)
2. `.copy_()` each local mask into the corresponding `self.last_*` at the end of `_get_dones()`

### How the termination classification pipeline works

**Step 1 — `_get_dones()` in the env (every sim step)**

Computes ~9 boolean termination conditions as local variables:
- `object_outside_upper_x`, `object_outside_lower_x`, `object_outside_upper_y`, `object_outside_lower_y`, `object_too_low` — object left the workspace
- `hand_too_far` — reference hand body outside the bounding box above the table
- `hand_too_close` — any fingertip/palm point within 1cm of table surface
- `palm_flipped` — palm direction deviated past `palm_flip_cos_thresh`
- `arm_table_contact_mask` — arm link physically touching the table

These are OR'd into `out_of_reach` (combined early termination signal). The `last_*` `.copy_()` calls persist each per-reason mask on `self` before returning.

**Step 2 — `classify_out_of_reach_reasons()` in `eval_utils.py` (called by eval after each step)**

After `eval_env.step()`, the evaluator calls this function. It:
1. Calls `_get_raw_out_of_reach_reason_masks(ov_env)` which checks `hasattr(ov_env, "last_hand_too_far")` etc.
2. If found, reads them directly (fast/reliable path).
3. If NOT found, falls back to recomputing from env state — this is where `fr3_agilehand` was failing.
4. Groups raw masks into 4 categories:
   - `object_out_of_bound` = object_outside_upper_x | lower_x | upper_y | lower_y | too_low
   - `hand_too_far` = hand outside workspace bounding box
   - `harmful_collision` = hand_too_close | arm_table_contact
   - `palm_flipped` = palm orientation exceeded threshold
5. Returns a per-env `reason_idx` tensor mapping each env to its category.

**Step 3 — `_reason_counts_checked()` in `eval_student.py` (end of each rollout)**

Counts how many unsafe episodes fall into each category. If any unsafe episode has `reason_idx == -1` (unclassified), it raises `RuntimeError` — fail-fast mode. Only the first reason per env is recorded (line 430-431: assigns only if `unsafe_reason_idx < 0`).

**Step 4 — Reporting**

Counts are aggregated across rollouts and reported as percentages of total unsafe episodes, both globally and per-object.

## Video recording support — 2026-03-27

Scripts with `--video` flag (via `gym.wrappers.RecordVideo`):
- `train.py`, `eval_student.py`, all `run_distillation*.py`, legacy `eval.py`

Scripts WITHOUT `--video`:
- `eval_teacher.py`, `play_test.py` — use `--livestream 2` + external screen capture

Legacy `eval.py` also supports `--record_data --create_video` for per-environment camera-view MP4s (left/right stereo frames from the onboard cameras) via the `DataRecorder` class.

## `eval_student.py` RecordVideo wrapper fix — 2026-03-27

**Problem:** `eval_student.py` had a `--video` flag that set `render_mode="rgb_array"` but never actually wrapped the env with `gym.wrappers.RecordVideo` — no MP4 was produced.

**Fix:** Added the `RecordVideo` wrapper after `gym.make()`, matching the `train.py` pattern. Videos saved to `<checkpoint_dir>/videos/eval_student/`.

**Gotcha:** Wrapping with `RecordVideo` adds a layer to the env chain. `env.env` then points to `RecordVideo`, not the Isaac env. Changed all `env.env` references to `env.unwrapped` (4 places) so the evaluator accesses the actual Isaac env regardless of how many wrappers are active.

## SafeDagger iteration count fix — 2026-03-27

**Problem:** `distillation_safedagger.py` hardcoded `num_iters = 100_000`, but `distillation_transformer.py` (vanilla DAgger with transformer) uses 350k. At 100k with beta=0.875, the student barely starts acting independently.

**Context — why 100k isn't enough:**
- SafeDagger beta schedule controls teacher/student action mix
- At 100k iters, beta was still 0.875 (teacher providing 87.5% of actions)
- Distillation metrics (87.5% lifted, 62.5% in goal) were mostly the *teacher* performing — standalone student eval showed 0% lift
- For sim2real, the student runs **100% solo** (no teacher) — it must perform well standalone in sim first

**Fix:**
- `distillation_safedagger.py`: default bumped 100k → 350k, accepts `max_iterations` param
- `run_distillation_safedagger_fr3_agilehand.py`: passes `--max_iterations` CLI flag through to `SafeDagger`

| Distillation class | `num_iters` | Used by |
|---|---|---|
| `distillation.py` (vanilla DAgger, CNN) | 100k | kuka_allegro, tg2_inspirehand |
| `distillation_transformer.py` (vanilla DAgger, transformer) | 350k | tg2_inspirehand stereo |
| `distillation_safedagger.py` (SafeDagger) | 350k (was 100k) | fr3_agilehand |

## `physics_instability` eval category — 2026-03-28

**Problem:** `eval_teacher.py` crashed with `RuntimeError: found N unclassified unsafe episodes` even after adding `last_*` masks. 5 of 10 unsafe episodes were unclassified.

**Root cause:** `robot_unstable` (NaN/Inf in joint state) and `vel_explosion` (finger velocities > threshold) are part of the `out_of_reach` termination signal but were not mapped to any eval category. The existing categories (`object_out_of_bound`, `hand_too_far`, `harmful_collision`, `palm_flipped`) didn't cover sim-only physics artifacts.

**Fix:** Added `physics_instability` as a new category in `eval_utils.py:UNSAFE_REASON_NAMES`. These are sim-only terminations that can't happen on real hardware (real controller has firmware safety limits).

**Changes:**
- `eval_utils.py`: added `"physics_instability"` to `UNSAFE_REASON_NAMES`, added raw mask extraction for `last_robot_unstable` + `last_vel_explosion`, added fallback recomputation
- `dextrah_fr3_agilehand_env.py`: added `self.last_robot_unstable` and `self.last_vel_explosion` tensors (init + `.copy_()`)

**Eval category summary:**

| Category | Termination conditions | Sim2real relevant? |
|---|---|---|
| `object_out_of_bound` | object_outside_x/y, object_too_low | Yes — object knocked off workspace |
| `hand_too_far` | hand body outside workspace box | Yes — arm positioning failure |
| `harmful_collision` | hand_too_close + arm_table_contact | Yes — collision with table |
| `palm_flipped` | palm orientation exceeded threshold | Yes — grasp approach failure |
| `physics_instability` | robot_unstable (NaN/Inf) + vel_explosion | **No** — sim-only artifact |

## `eval_teacher.py` `--objects_dir` flag — 2026-03-28

**Problem:** `eval_teacher.py` uses `parse_args()` (not `parse_known_args()`), so `env.objects_dir=...` Hydra overrides are rejected as unrecognized arguments. The default `objects_dir="replace_me"` always fails validation.

**Fix:** Added `--objects_dir` CLI flag to `eval_teacher.py`, wired into `_run_eval_for_checkpoint(objects_dir_override=...)` for single-checkpoint mode.

**Usage:**
```bash
python eval_teacher.py --headless --task=dextrah_fr3_agilehand --num_envs 16 --eval_episodes 5 --objects_dir multi_objects/visdex_selected --checkpoint <path>
```
