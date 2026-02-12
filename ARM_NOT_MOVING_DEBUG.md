# Debug Guide: Arm Not Moving Toward Object

## Problem Summary

Training is only minimizing action rate penalty → arms stay completely still.
The hand-to-object reward should provide gradient for arm movement, but it's not working.

## Possible Causes

1. **Hand body indices not found correctly** (nested hand structure issue)
2. **Hand-object distance calculation returning wrong values**
3. **Reward weights too low relative to penalties**
4. **Object spawning out of reach**
5. **Initial robot pose too far from object**

## Diagnostic Steps

### Step 1: Run the Debug Script

I've created a debug script that will print all the relevant information:

```bash
cd ~/code/tg2_dexman_isaac_co
python -m dextrah_lab.tasks.fr3_agilehand.debug_rewards --headless --num_envs 4
```

**What to check in the output:**

1. **Body Name Mapping**
   - Are ALL hand_body_names found? 
   - Are ALL hand_object_distance_body_names found?
   - If ANY say "NOT FOUND!", that's the problem

2. **Initial Distances**
   - What is the hand-to-object distance at reset?
   - Should be < 1.0m typically for reasonable gradient
   - If > 1.5m, object might be spawning too far

3. **Reward Values**
   - Is `hand_to_object_reward` > 0.001 initially?
   - If it's ~0.0, the distance is too large (exponential decay)
   - Compare to `action_rate_penalty` magnitude

### Step 2: Check if Bodies Are Found

The issue might be that body names aren't being found due to the nested hand structure.

From the check_articulation output, the bodies are:
```
[ 9] base_link
[10] Index_MCP_Pitch
[11] Middle_MCP_Pitch
[12] Pinky_MCP_Pitch
[13] Ring_MCP_Pitch
[14] Thumb_rot
[15] Index_MCP_Yaw
[16] Middle_MCP_Yaw
[17] Pinky_MCP_Yaw
[18] Ring_MCP_Yaw
[19] Thumb_MCP_Pitch
[20] Index_Middle_Phalanx
[21] Middle_Middle_Phalanx
[22] Pinky_Middle_Phalanx
[23] Ring_Middle_Phalanx
[24] Thumb_MCP_Yaw
[25] Index_Distal_Phalanx
[26] Middle_Distal_Phalanx
[27] Pinky_Distal_Phalanx
[28] Ring_Distal_Phalanx
[29] Thumb_Middle_Phalanx
[30] Thumb_Distal_Phalanx
```

Your config uses:
```python
hand_body_names = [
    "base_link",           # [9] ✅
    "Index_Distal_Phalanx",  # [25] ✅
    "Middle_Distal_Phalanx", # [26] ✅
    "Ring_Distal_Phalanx",   # [28] ✅
    "Pinky_Distal_Phalanx",  # [27] ✅
    "Thumb_Distal_Phalanx",  # [30] ✅
]
```

These should all be found. But if body lookup is failing silently, rewards would be zero.

### Step 3: Compare Reward Magnitudes

In early training, you should see (approximately):

| Reward Component | Expected Range | Purpose |
|-----------------|----------------|---------|
| `hand_to_object_reward` | 0.01 - 8.0 | **Drives arm toward object** |
| `action_rate_penalty` | -0.01 to 0.0 | Smoothness |
| `joint_vel_penalty` | -0.01 to 0.0 | Smoothness |
| `finger_curl_reg` | -1.0 to 0.0 | Hand pose |

**If `hand_to_object_reward` is ~0.0**, the arm won't move because there's no gradient.

**If `action_rate_penalty` dominates**, policy learns to do nothing.

### Step 4: Check Robot/Object Spawn Positions

Add this debug print to your env after reset:

```python
# In dextrah_fr3_agilehand_env.py, after reset_idx():
print(f"Robot base pos: {self.robot.data.root_pos_w[0]}")
print(f"Object pos: {self.object.data.root_pos_w[0]}")
print(f"Hand workspace body pos: {self.robot.data.body_pos_w[0, self.hand_workspace_body_idx]}")
print(f"Distance hand->object: {self.hand_to_object_pos_error[0]}")
```

**Expected:**
- Object should be within 0.3-0.8m of hand
- If > 1.0m, robot can't reach it easily

## Potential Fixes

### Fix 1: Increase Reward Weight

If hand_to_object_reward is non-zero but small:

```python
# In dextrah_fr3_agilehand_env_cfg.py
hand_to_object_weight = 20.  # Was 8.0, try higher
hand_to_object_sharpness = 2.  # Was 4.0, try lower (wider gradient)
```

### Fix 2: Reduce Action Penalty Temporarily

For debugging, try disabling action penalties:

```python
action_rate_penalty_weight = 0.0  # Was 0.005
joint_velocity_penalty_weight = 0.0  # Check what it is
```

Train for 100-500 iterations. If arm starts moving, then the balance was wrong.

### Fix 3: Add Explicit Reach Reward

You could add a simple distance-based reward:

```python
# In compute_rewards():
reach_reward = -0.1 * hand_to_object_pos_error  # Linear penalty for distance
```

This gives a constant gradient regardless of distance (unlike exponential).

### Fix 4: Check ADR

Automatic Domain Randomization might be making the task too hard initially:

```python
# In dextrah_fr3_agilehand_env_cfg.py, check:
num_adr_increments = 0  # Disable ADR for debugging
```

## What the debug_rewards.py Script Will Show

```
================================================================================
BODY NAME MAPPING
================================================================================

Total bodies: 31

Configured hand_body_names:
  'base_link' -> body index 9
  'Index_Distal_Phalanx' -> body index 25
  'Middle_Distal_Phalanx' -> body index 26
  'Ring_Distal_Phalanx' -> body index 28
  'Pinky_Distal_Phalanx' -> body index 27
  'Thumb_Distal_Phalanx' -> body index 30

Configured hand_object_distance_body_names:
  ... (same as above)

Palm body: 'base_link' -> index 9
Workspace body: 'base_link' -> index 9

================================================================================
INITIAL STATE
================================================================================

Hand bodies indices: [9, 25, 26, 27, 28, 30]
Hand-object distance bodies indices: [9, 25, 26, 27, 28, 30]

Object position (env 0): [x, y, z]
Palm position (env 0): [x, y, z]

Hand-object distance bodies positions (env 0):
  [9] base_link                     pos=[x,y,z] dist=0.XXXX
  [25] Index_Distal_Phalanx         pos=[x,y,z] dist=0.XXXX
  ...

Hand-to-object error (env 0): 0.XXXX

================================================================================
REWARD COMPONENTS (initial state)
================================================================================

hand_to_object_reward (env 0): X.XXXXXX
  weight: 8.0
  sharpness: 4.0
  distance: 0.XXXX
  expected reward: X.XXXXXX

...
```

## Expected Results

### If working correctly:
- ✅ All body names found
- ✅ Hand-to-object distance: 0.3-0.8m
- ✅ hand_to_object_reward: 0.1-8.0
- ✅ Reward >> penalties in magnitude

### If broken:
- ❌ Body names "NOT FOUND!"
- ❌ Hand-to-object distance: > 1.5m
- ❌ hand_to_object_reward: < 0.001
- ❌ Penalties dominate rewards

## Next Steps

1. Run `debug_rewards.py` and share the output
2. Check if body names are found
3. Check if distances are reasonable
4. Adjust reward weights based on findings

Let me know what the debug script shows!
