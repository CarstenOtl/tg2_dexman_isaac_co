#!/usr/bin/env python3
"""Export TensorBoard runs and eval results to CSV for thesis plotting.

Usage:
    python dextrah_lab/docs/export_runs.py [--output_dir exports/]

Exports:
    - Teacher training curves (stored_policies) → CSV per run
    - Distillation training curves → CSV per run
    - Eval results (JSON) → copied + combined CSV summary
    - manifest.json with run metadata
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from tbparse import SummaryReader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Run definitions ──────────────────────────────────────────────────────────

TEACHER_RUNS = {
    "teacher_10_multi_adr19": (
        "10_multi_object_adr19_1024envs_03-26_00-21-45",
        "Multi-object, 1024 envs, ADR 19 (exp_02 best)",
    ),
    "teacher_11_sim2real_adr14": (
        "11_multi_object_adr14_sim2real_03-30_17-41-43",
        "Multi-object, sim2real curriculum, ADR 14 (exp_03 run1m, best teacher)",
    ),
}

DISTILLATION_RUNS = {
    "student_run1c_safedagger_l2_teacher10": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_27-19-13-17",
        "SafeDagger + L2, teacher 10, 16 envs, 350k iters",
    ),
    "student_run2a_vanilla_kl_teacher10": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_28-15-55-43",
        "Vanilla DAgger + KL, teacher 10, 16 envs, 350k iters",
    ),
    "student_run2b_vanilla_kl_700k_teacher10": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_29-08-52-01",
        "Vanilla DAgger + KL, teacher 10, 700k iters (collapsed)",
    ),
    "student_run3a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_31-13-33-09",
        "SafeDagger + L2, teacher 11, 24 envs, 350k iters (BEST student)",
    ),
    "student_run3b_vanilla_kl_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_31-19-48-40",
        "Vanilla DAgger + KL, teacher 11, 24 envs, 100k iters",
    ),
    "student_run4a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_02-14-30-48",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, per-object logging",
    ),
    "student_run4b_vanilla_kl_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_02-14-28-35",
        "Vanilla DAgger + KL, teacher 11, 24 envs, 100k iters, per-object logging",
    ),
    "student_run5a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_02-23-45-55",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, per-object real termination",
    ),
    "student_run5b_vanilla_kl_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_02-23-41-36",
        "Vanilla DAgger + KL, teacher 11, 24 envs, 100k iters, per-object real termination",
    ),
    "student_kuka_allegro_vanilla_kl": (
        "dextrah-kuka-allegro-safedagger-stereo-transformer_02-21-52-43",
        "Vanilla DAgger + KL, kuka_allegro, 24 envs, 100k iters, ADR 0",
    ),
    "student_run6a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_03-13-04-24",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, scaled L2 threshold, AverageMeter",
    ),
    "student_run6b_dagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_03-13-06-54",
        "DAgger + L2, teacher 11, 24 envs, 100k iters, AverageMeter",
    ),
    "student_run7a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_03-14-53-40",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, threshold=3.0, visdex_top8",
    ),
    "student_run7b_dagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_03-14-54-52",
        "DAgger + L2, teacher 11, 24 envs, 100k iters, visdex_top8",
    ),
    "student_run8a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_03-20-30-06",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, threshold=2.0, top8, 10s eps",
    ),
    "student_run8b_dagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_03-20-33-10",
        "DAgger + L2, teacher 11, 24 envs, 100k iters, top8, 10s eps",
    ),
    "student_run9a_safedagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_04-14-07-25",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, FIXED one-hot mapping",
    ),
    "student_run9b_dagger_l2_teacher11": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_04-14-08-19",
        "DAgger + L2, teacher 11, 24 envs, 100k iters, FIXED one-hot mapping",
    ),
    "student_run10a_safedagger_arm_rand": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_05-15-21-00",
        "SafeDagger + L2, teacher 11, 24 envs, 100k iters, arm randomization ON",
    ),
    "student_run10b_safedagger_32envs": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_05-15-21-49",
        "SafeDagger + L2, teacher 11, 32 envs, 100k iters, 4 envs/obj",
    ),
    "student_run11a_safedagger_adr5": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_06-14-00-34",
        "SafeDagger + L2, teacher 11, 32 envs, ADR 5",
    ),
    "student_run11b_dagger_adr5": (
        "dextrah-fr3-agilehand-safedagger-stereo-transformer_06-14-01-16",
        "DAgger + L2, teacher 11, 32 envs, ADR 5",
    ),
}


def export_tb_run(event_dir: Path, output_csv: Path, run_name: str) -> dict:
    """Export a single TensorBoard run to CSV. Returns metadata dict."""
    reader = SummaryReader(str(event_dir))
    df = reader.scalars

    if df.empty:
        print(f"  WARNING: No scalar data in {event_dir}")
        return {"run_name": run_name, "rows": 0, "tags": []}

    tags = sorted(df["tag"].unique().tolist())

    # Long-form CSV: step, metric, value
    df = df.rename(columns={"tag": "metric"})
    df = df[["step", "metric", "value"]]
    df = df.sort_values(["metric", "step"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "run_name": run_name,
        "rows": len(df),
        "tags": tags,
        "step_range": [int(df["step"].min()), int(df["step"].max())],
    }


def export_eval_results(output_dir: Path) -> list:
    """Copy and index all eval JSON files."""
    eval_dir = output_dir / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    sources = []

    # Student evals
    student_eval_dir = REPO_ROOT / "dextrah_lab" / "distillation_new" / "eval_results"
    if student_eval_dir.exists():
        for f in sorted(student_eval_dir.glob("eval_metrics_*.json")):
            sources.append(("student", f))

    # Teacher evals
    teacher_log_dir = REPO_ROOT / "dextrah_lab" / "rl_games" / "logs"
    if teacher_log_dir.exists():
        for f in sorted(teacher_log_dir.rglob("eval_metrics_*.json")):
            sources.append(("teacher", f))

    results = []
    for eval_type, src in sources:
        dst = eval_dir / f"{eval_type}_{src.name}"
        shutil.copy2(src, dst)

        with open(src) as fh:
            data = json.load(fh)

        metrics = data.get("metrics", {})
        summary = {
            "file": dst.name,
            "type": eval_type,
            "task": data.get("task", ""),
            "checkpoint": data.get("checkpoint", ""),
            "eval_episodes": data.get("eval_episodes", 0),
            "total_episodes": data.get("total_episodes", 0),
            "lift_success": metrics.get("eval/lift_success", None),
            "unsafe_episode_rate": metrics.get("eval/unsafe_episode_rate", None),
            "avg_reward": metrics.get("eval/avg_reward", None),
        }

        per_obj = data.get("per_object_metrics", {})
        if per_obj:
            summary["per_object"] = {
                obj: {
                    "lift_success": m.get("lift_success"),
                    "unsafe_episode_rate": m.get("unsafe_episode_rate"),
                }
                for obj, m in per_obj.items()
            }

        reason_prop = metrics.get("eval/unsafe_reason_prop", {})
        if reason_prop:
            summary["unsafe_reasons"] = reason_prop

        results.append(summary)
        print(f"  {dst.name}: lift={summary['lift_success']}, unsafe={summary['unsafe_episode_rate']}")

    if results:
        summary_rows = []
        for r in results:
            row = {k: v for k, v in r.items() if k not in ("per_object", "unsafe_reasons")}
            for reason, prop in r.get("unsafe_reasons", {}).items():
                row[f"unsafe_{reason}"] = prop
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(eval_dir / "eval_summary.csv", index=False)

    return results


def main():
    parser = argparse.ArgumentParser(description="Export training runs for thesis plots")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "dextrah_lab" / "docs" / "exports"),
        help="Output directory for exported data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to: {output_dir}\n")

    manifest = {"teacher_runs": {}, "distillation_runs": {}, "eval_results": []}

    # ── Export teacher runs ──────────────────────────────────────────────────
    print("=== Teacher Training Runs ===")
    stored = REPO_ROOT / "dextrah_lab" / "stored_policies" / "fr3_agilehand"
    for run_name, (dirname, description) in TEACHER_RUNS.items():
        event_dir = stored / dirname / "summaries"
        if not event_dir.exists():
            print(f"  SKIP {run_name}: {event_dir} not found")
            continue
        csv_path = output_dir / "teacher" / f"{run_name}.csv"
        print(f"  {run_name} ({description})...")
        meta = export_tb_run(event_dir, csv_path, run_name)
        meta["description"] = description
        meta["source_dir"] = dirname
        manifest["teacher_runs"][run_name] = meta
        print(f"    -> {meta['rows']} rows, steps {meta.get('step_range', [])}")

    # ── Export distillation runs ─────────────────────────────────────────────
    print("\n=== Distillation Runs ===")
    runs_dir = REPO_ROOT / "dextrah_lab" / "distillation_new" / "runs"
    for run_name, (dirname, description) in DISTILLATION_RUNS.items():
        event_dir = runs_dir / dirname / "summaries"
        if not event_dir.exists():
            print(f"  SKIP {run_name}: {event_dir} not found")
            continue
        csv_path = output_dir / "distillation" / f"{run_name}.csv"
        print(f"  {run_name} ({description})...")
        meta = export_tb_run(event_dir, csv_path, run_name)
        meta["description"] = description
        meta["source_dir"] = dirname
        manifest["distillation_runs"][run_name] = meta
        print(f"    -> {meta['rows']} rows, steps {meta.get('step_range', [])}")

    # ── Export eval results ──────────────────────────────────────────────────
    print("\n=== Eval Results ===")
    manifest["eval_results"] = export_eval_results(output_dir)

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to: {manifest_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_teacher = len(manifest["teacher_runs"])
    n_distill = len(manifest["distillation_runs"])
    n_eval = len(manifest["eval_results"])
    print(f"\nDone! Exported {n_teacher} teacher runs, {n_distill} distillation runs, {n_eval} eval results.")
    print(f"Output: {output_dir}/")
    print(f"  teacher/          — {n_teacher} CSVs (long-form: step, metric, value)")
    print(f"  distillation/     — {n_distill} CSVs (same format)")
    print(f"  eval_results/     — {n_eval} JSONs + eval_summary.csv")
    print(f"  manifest.json     — run metadata & available metrics")


if __name__ == "__main__":
    main()
