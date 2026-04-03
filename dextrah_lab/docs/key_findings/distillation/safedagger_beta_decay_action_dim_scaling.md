# Key Finding: SafeDagger Beta Stays High (~75%) Due to L2 Threshold Not Scaling with Action Dimensionality

**Date:** 2026-04-03
**Branch:** fr3_agilehand
**Context:** SafeDagger intervention rate (beta) in fr3_agilehand stays at ~70-75% through 100k iterations, while upstream tg2_inspirehand (DexSafeDagger paper) shows beta dropping to ~20%. Both use the same `unsafe_l2_threshold = 0.5`.

---

## Root Cause: L2 Norm Scales with sqrt(num_actions)

The unsafe check computes `weighted_l2(student_mus, teacher_mus, weights)` across ALL action dimensions:

```python
def weighted_l2(model, target, weights):
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5
```

This is a sum over `dim=-1` (all actions). With more dimensions, the L2 norm is inherently larger even if per-joint error is identical. The expected L2 scales as `sqrt(num_actions)` for equal per-joint error.

| Factor | tg2_inspirehand (upstream) | fr3_agilehand |
|---|---|---|
| Action space | 13 (7 arm + 6 hand) | 23 (7 arm + 16 hand) |
| L2 threshold | 0.5 | 0.5 (same) |
| Expected L2 at same per-joint error | baseline | ~1.33x higher (sqrt(23/13)) |
| Beta at 100k iters | ~20% (paper) | ~70-75% |

The student's per-joint accuracy may be comparable between robots, but the aggregate L2 is higher for fr3_agilehand simply because there are more joints contributing to the sum.

## Fix: Scale Threshold by Action Dimensionality

```python
reference_actions = 13  # tg2_inspirehand (upstream baseline)
scale = math.sqrt(num_actions / reference_actions)
unsafe_l2_threshold = base_threshold * scale
# fr3_agilehand (23 actions): 0.5 * sqrt(23/13) = 0.665
```

This normalizes the per-joint error budget to match the upstream baseline. Applied in run6 (2026-04-03).

## Additional Contributing Factor: Env Count per Object

| | tg2_inspirehand | fr3_agilehand |
|---|---|---|
| Distillation envs | 48 (paper) | 24 |
| Objects | ~10 | 13 |
| Envs per object | ~5 | ~1.8 |

With only ~2 envs per object, gradient signal is noisy. The student may learn some objects well but struggle on others, keeping the average L2 high across the batch. More envs per object would give cleaner gradients and faster L2 reduction.

## Impact on Distillation Quality

High beta means the teacher acts in ~75% of envs. The student only practices independently in 25% of envs — creating a chicken-and-egg problem where the student can't improve because it rarely acts.

This explains the SafeDagger vs DAgger results:
- SafeDagger (beta ~75%): student barely practices solo, but training lift is high (teacher helps)
- DAgger (beta = 0): student always acts, lower training lift but better standalone eval

The standalone eval confirms this: DAgger consistently outperforms SafeDagger when the student acts alone (52% vs 37% lift in run4, 49% vs 61% in run5 — high variance suggests other factors too).

## Recommendations

1. **Scale L2 threshold** by `sqrt(num_actions / reference)` — applied in run6
2. **Increase envs** to 48 if GPU memory allows (single GPU limit is 24 with cameras)
3. **Consider per-joint-group thresholds**: arm joints (7) vs finger joints (16) may need different thresholds since arm tracking is easier than finger manipulation
4. **Try adaptive threshold**: start high and decay, similar to learning rate scheduling

---

## Key Takeaway for Thesis

> SafeDagger's intervention rate is controlled by an L2 threshold that doesn't account for action dimensionality. The fr3_agilehand's 23-action space produces ~1.33x higher L2 norms than tg2_inspirehand's 13-action space at equivalent per-joint accuracy. Using the same threshold (0.5) makes SafeDagger overly conservative for higher-dimensional robots, keeping the teacher active in ~75% of environments and preventing the student from learning through its own experience. Scaling the threshold by `sqrt(num_actions)` restores the effective per-joint error budget to match the upstream baseline.
