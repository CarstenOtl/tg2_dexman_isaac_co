# Summary of Hand Body Names Configuration Changes

## Changes Made to FR3 Agile Hand Task

### Updated Configuration (`dextrah_fr3_agilehand_env_cfg.py`)

**Before:**
- Used distal phalanx bodies directly: `Index_Distal_Phalanx`, `Middle_Distal_Phalanx`, etc.
- Had conflicting/duplicate `hand_object_distance_body_names` definitions
- Missing clear documentation

**After:**
- Uses dedicated fingertip coordinate frames: `Index_Tip`, `Middle_Tip`, etc.
- Clear separation between `hand_body_names` and `hand_object_distance_body_names`
- Comprehensive inline documentation

### Configuration Details

```python
# hand_body_names: Used for observations (fingertip positions/velocities)
hand_body_names = [
    "base_link",     # Palm
    "Index_Tip",     # Fingertip coordinate frames
    "Middle_Tip",
    "Ring_Tip",
    "Pinky_Tip",
    "Thumb_Tip",
]

# hand_object_distance_body_names: Used for hand-to-object distance reward
hand_object_distance_body_names = [
    "base_link",     # Palm center
    "Index_Tip",     # Use tip frames for better distance calculation
    "Middle_Tip",
    "Ring_Tip",
    "Pinky_Tip",
    "Thumb_Tip",
]
```

## How These Are Used (Matching TG2 Implementation)

### 1. **hand_body_names** - Proprioceptive Observations

**Purpose:** Provides the policy with spatial awareness of hand configuration

**Usage in `dextrah_fr3_agilehand_env.py`:**

```python
# Line 134-137: Initialize body indices
self.hand_bodies = list()
for body_name in self.cfg.hand_body_names:
    self.hand_bodies.append(self.robot.body_names.index(body_name))

# Line 1683-1684: Get positions for observations
self.hand_pos = self.robot.data.body_pos_w[:, self.hand_bodies]
self.hand_vel = self.robot.data.body_vel_w[:, self.hand_bodies]

# Used in observation buffer
# - Teacher obs: self.hand_pos.view(num_envs, num_hand_bodies * 3)
# - Teacher obs: self.hand_vel.view(num_envs, num_hand_bodies * 6)
```

**Policy receives:**
- 18 values: 6 bodies × 3 (x,y,z positions)
- 36 values: 6 bodies × 6 (linear + angular velocities)

### 2. **hand_object_distance_body_names** - Approach Reward

**Purpose:** Drives the hand to approach the object during pre-grasp phase

**Usage in `dextrah_fr3_agilehand_env.py`:**

```python
# Line 144-148: Initialize distance body indices
self.hand_object_distance_bodies = []
for body_name in self.cfg.hand_object_distance_body_names:
    self.hand_object_distance_bodies.append(self.robot.body_names.index(body_name))

# Line 1686-1689: Get positions for distance calculation
self.hand_object_distance_pos = self.robot.data.body_pos_w[:, self.hand_object_distance_bodies]
self.hand_object_distance_pos -= self.scene.env_origins.repeat(...)

# Line 1788-1790: Compute hand-to-object distance (MAX distance across all points)
self.hand_to_object_pos_error = (
    torch.norm(self.hand_object_distance_pos - self.object_pos[:, None, :], dim=-1).max(dim=-1).values
)

# Used in reward function (line ~947):
hand_to_object_reward = hand_to_object_weight * torch.exp(-hand_to_object_sharpness * hand_to_object_pos_error)
```

**Key insight:** Using `.max(dim=-1)` means the reward encourages **all** fingertips to get close to the object, not just the nearest one.

## Benefits of Using Tip Frames

### 1. **Precise Distance Calculation**
- Tip frames are exactly at fingertip contact points
- More accurate than using center-of-mass of distal phalanx bodies
- Better reward shaping for approach phase

### 2. **Better Observations**
- Policy gets exact fingertip locations
- Matches real-world sensor placement (tactile sensors at tips)
- Easier sim-to-real transfer

### 3. **Consistent with TG2 Implementation**
- Same design pattern as TG2 Inspirehand task
- Proven to work well for dexterous manipulation
- Easier to transfer hyperparameters and insights

## Comparison: TG2 vs FR3

| Aspect | TG2 Inspirehand | FR3 Agile Hand |
|--------|----------------|----------------|
| **hand_body_names** | `["palm", "index_tip", "middle_tip", ...]` | `["base_link", "Index_Tip", "Middle_Tip", ...]` |
| **hand_object_distance_body_names** | `["palm_center", "index_tip", ...]` | `["base_link", "Index_Tip", ...]` |
| **Number of bodies** | 6 (palm + 5 fingertips) | 6 (palm + 5 fingertips) |
| **Observation size** | 6×3 + 6×6 = 54 values | 6×3 + 6×6 = 54 values |
| **Distance metric** | Max distance across tips | Max distance across tips |

## Next Steps

1. **Test the configuration:**
   ```bash
   ./isaaclab.sh -p scripts/train.py --task=DextrahFR3Agilehand --headless --num_envs=64
   ```

2. **Verify tip frames are recognized:**
   - Check that `Index_Tip`, `Middle_Tip`, etc. appear in `robot.body_names`
   - Verify positions update correctly during simulation

3. **Monitor hand-to-object reward:**
   - Should see gradual approach during training
   - Distance should decrease as policy learns

4. **If tips not found:**
   - First remove lowercase duplicates: `./isaaclab.sh -p dextrah_lab/assets/fr3_tekken_adof/remove_fingertip_frames_isaacsim.py --headless`
   - Check USD has `Index_Tip` (capital letters) as children of distal phalanx links
   - Verify tip frames are Xforms, not RigidBodies

## Troubleshooting

### Error: "base_link not found in body_names"
- Check your USD hierarchy
- Verify the palm body is named `base_link` in the tekken_left_adof subtree

### Error: "Index_Tip not found in body_names"
- Run: `./isaaclab.sh -p dextrah_lab/assets/fr3_tekken_adof/remove_fingertip_frames_isaacsim.py --headless`
- Verify tip frames exist in USD and are Xforms (not RigidBodies)
- Check they are children of the distal phalanx links

### Observation size mismatch
- Ensure 6 bodies in both `hand_body_names` and `hand_object_distance_body_names`
- Check observation buffer size calculations in the environment

## Files Modified

1. `dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py`
   - Updated `hand_body_names` to use tip frames
   - Updated `hand_object_distance_body_names` to use tip frames
   - Added documentation

2. No changes needed to `dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env.py`
   - Already had correct implementation matching TG2
   - Uses `hand_object_distance_bodies` for distance calculation
   - Uses `hand_bodies` for observations
