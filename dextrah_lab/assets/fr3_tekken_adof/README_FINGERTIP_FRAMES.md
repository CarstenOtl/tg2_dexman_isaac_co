# Fingertip Frame Setup Guide

## Overview
These scripts add coordinate frames at the fingertips that inherit the parent link's orientation.
You only need to tune the position offsets (X, Y, Z) in the local frame.

## Files Created
1. `add_fingertip_frames.py` - Adds tip frames to the USD
2. `remove_fingertip_frames.py` - Removes tip frames (for re-tuning)

## Step-by-Step Process

### 1. Restore Clean USD (if corrupted)
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co
git checkout dextrah_lab/assets/fr3_tekken_adof/FR3_tekkenadof_left.usd
```

### 2. Run the Script
```bash
cd dextrah_lab/assets/fr3_tekken_adof
python add_fingertip_frames.py
```

### 3. Verify in Isaac Sim
- Open the USD file in Isaac Sim
- Select the robot in the stage hierarchy
- Manually actuate finger joints
- Check that tip frames:
  - Move with the fingers ✓
  - Rotate with the fingers ✓
  - Are positioned at the fingertip contact points ✓

### 4. Adjust Offsets (if needed)
If the tips aren't at the right location:

1. **Remove the tips:**
   ```bash
   python remove_fingertip_frames.py
   ```

2. **Edit `add_fingertip_frames.py`** and adjust the offsets:
   ```python
   tip_configs = [
       ("Thumb_Distal_Phalanx", "thumb_tip", (0.030, 0.0, 0.0)),  # Changed X from 0.025 to 0.030
       # ... adjust others as needed
   ]
   ```

3. **Re-run:**
   ```bash
   python add_fingertip_frames.py
   ```

4. **Repeat** until satisfied

### 5. Update Task Configuration
Once the tips are correctly positioned, update your env config:

```python
# In dextrah_fr3_agilehand_env_cfg.py

hand_body_names = [
    "base_link",           # Palm
    "thumb_tip",           # NEW - fingertip frames
    "index_tip",           # NEW
    "middle_tip",          # NEW
    "ring_tip",            # NEW
    "pinky_tip",           # NEW
]

hand_object_distance_body_names = hand_body_names
```

## Understanding the Coordinate System

The offsets are in the **local frame** of each distal phalanx:
- **X-axis**: Usually along the finger length (toward tip)
- **Y-axis**: Usually finger width direction
- **Z-axis**: Usually finger thickness direction

**Example:**
```python
("Index_Distal_Phalanx", "index_tip", (0.020, 0.0, 0.0))
```
This places the tip 2cm along the X-axis from the distal phalanx origin.

## Key Features

✅ **Orientation inheritance**: Tips automatically rotate with parent fingers  
✅ **Simple position tuning**: Only need to adjust (X, Y, Z) offsets  
✅ **No physics**: Tips are pure coordinate frames, no joints or rigid bodies  
✅ **Reversible**: Easy to remove and recreate with new offsets  

## Troubleshooting

### "Invalid PhysX transform detected"
- Your USD is corrupted
- Restore from git: `git checkout FR3_tekkenadof_left.usd`
- DO NOT manually edit physics attributes in the USD

### Tips don't move with fingers
- Make sure tips are **children** of the distal phalanx prims
- Check the USD hierarchy in Isaac Sim's stage tree
- Tips should appear under their parent link

### Tips already exist warning
- Run `python remove_fingertip_frames.py` first
- Then run `add_fingertip_frames.py` again

### Can't find parent links
- Check the `base_path` variable in the script
- Verify distal phalanx names match your USD
- Use Isaac Sim's stage tree to find the correct names

## Testing in Your Task

After adding tips, test them:

```python
# In your task/environment
print("All body names:", self.robot.body_names)
# Should include: thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip

# Check positions
tip_idx = self.robot.body_names.index("index_tip")
tip_pos = self.robot.data.body_pos_w[:, tip_idx]
print(f"Index tip position: {tip_pos}")
```
