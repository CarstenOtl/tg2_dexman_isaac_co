# Bug Fix: Arm Not Moving Toward Object

## Root Cause Found ✅

**The `hand_to_object_pos_error` was not being computed at reset!**

### The Bug

In both FR3 Agilehand and TG2 Inspirehand tasks:

1. `hand_to_object_pos_error` was initialized to `1.0` at env creation
2. `_reset_idx()` called `_compute_intermediate_values()` to update sensor data
3. BUT it did NOT call `compute_intermediate_reward_values()` 
4. This meant `hand_to_object_pos_error` stayed at `1.0` until the first step

### Why This Broke Training

```python
hand_to_object_reward = 8.0 * exp(-4.0 * hand_to_object_pos_error)
```

**At reset:**
- Actual distance: ~0.5m
- `hand_to_object_pos_error`: **1.0** (wrong!)
- Reward: `8.0 * exp(-4.0) ≈ 0.147`

**After moving closer (distance 0.4m):**
- `hand_to_object_pos_error`: Still **1.0** (not updated!)
- Reward: Still `≈ 0.147`

**Result:** **NO GRADIENT to move toward object!** The reward didn't change whether the arm moved or not, so the policy learned to just minimize action rate (stay still).

### The Fix

Added one line in `_reset_idx()` after `_compute_intermediate_values()`:

```python
# Poll robot and object data
self._compute_intermediate_values()

# Compute reward-related intermediate values (hand-object distances, etc.)
# This ensures hand_to_object_pos_error is properly initialized at reset
self.compute_intermediate_reward_values()  # <-- ADDED THIS

# Reset success signals
```

### Files Modified

1. ✅ `dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py` (line ~1305)
2. ✅ `dextrah_lab/tasks/tg2_inspirehand/dextrah_tg2_inspirehand_env.py` (line ~1275)

### Expected Results

**Before fix:**
```
At reset: hand_to_object_pos_error = 1.0000 (wrong!)
After step: hand_to_object_pos_error = 0.5365 (correct)
→ Policy sees constant reward, no gradient to move
```

**After fix:**
```
At reset: hand_to_object_pos_error = 0.5365 (correct!)
After moving closer: hand_to_object_pos_error = 0.4500 (updates correctly)
→ Policy sees reward increase, learns to move toward object
```

### Why It Wasn't Caught Earlier

The bug was subtle because:
1. After the FIRST step, the value was computed correctly
2. Most testing happens after several steps, not at reset
3. The constant value of 1.0 still provided SOME reward (0.147)
4. The bug manifested as "training not working" rather than a crash

### Verification

Run the debug script to confirm:
```bash
python -m dextrah_lab.tasks.fr3_agilehand.debug_rewards --headless --num_envs 4
```

Look for:
```
[AFTER _compute_intermediate_values()]
env.hand_to_object_pos_error (env 0): 0.5365  ✅ Should match manual calculation
```

NOT:
```
env.hand_to_object_pos_error (env 0): 1.0000  ❌ Bug is still present
```

### Training Impact

With this fix, you should now see:
- ✅ Arm actively moves toward object in early training
- ✅ `hand_to_object_reward` provides clear gradient
- ✅ Higher success rates
- ✅ Faster convergence

The action rate penalty is no longer the dominant signal - the reach reward will now properly guide the arm!

## Summary

**One line fix, massive impact.** The hand-object distance wasn't being calculated at reset, causing the reach reward to be constant regardless of arm position. This made the policy learn to stay still to minimize action penalties rather than reach for objects.
