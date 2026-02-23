# Setting Up Rigid-Body Tip Frames - Complete Guide

## Overview
This guide creates proper rigid-body fingertip frames that will appear in Isaac Sim's `body_names`, just like the TG2 Inspirehand implementation.

## Why This Approach?

**Simple Xforms (what you had before):**
- ❌ Don't appear in `body_names`
- ❌ Can't be tracked by articulation
- ❌ Not useful for observations/rewards

**Rigid Bodies with Fixed Joints (what we're creating):**
- ✅ Appear in `body_names` 
- ✅ Tracked by articulation
- ✅ Can be used for observations/rewards
- ✅ Massless (don't affect dynamics)
- ✅ Move perfectly with parent

## Step-by-Step Instructions

### Step 1: Backup Your USD
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/assets/fr3_tekken_adof
cp FR3_tekkenadof_left.usd FR3_tekkenadof_left.usd.backup
```

### Step 2: Run the Creation Script
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co
./isaaclab.sh -p dextrah_lab/assets/fr3_tekken_adof/create_rigid_tip_frames.py --headless
```

**Expected output:**
```
1. Cleaning up existing tip frames...
   Removed existing: /fr3_tekken_left/tekken_left_adof/Index_Distal_Phalanx/Index_Tip
   ...

2. Creating rigid-body tip frames...
   ✓ Created: Thumb_Tip
      Rigid body: /fr3_tekken_left/tekken_left_adof/Thumb_Distal_Phalanx/Thumb_Tip
      Fixed joint: /fr3_tekken_left/tekken_left_adof/fixed_thumb_tip
      Offset: (0.025, 0.0, 0.0)
   ...

SUMMARY: 5/5 tip frames created
✅ USD saved successfully
```

### Step 3: Verify the Tips Are Recognized
```bash
./isaaclab.sh -p dextrah_lab/tasks/fr3_agilehand/print_body_names.py --headless
```

**Look for:**
```
[XX] Thumb_Tip ← TIP FRAME
[XX] Index_Tip ← TIP FRAME
[XX] Middle_Tip ← TIP FRAME
[XX] Ring_Tip ← TIP FRAME
[XX] Pinky_Tip ← TIP FRAME
```

### Step 4: Update Your Config

Once verified, update `dextrah_fr3_agilehand_env_cfg.py`:

```python
hand_body_names = [
    "base_link",
    "Thumb_Tip",      # Now these will work!
    "Index_Tip",
    "Middle_Tip",
    "Ring_Tip",
    "Pinky_Tip",
]

hand_object_distance_body_names = hand_body_names
```

### Step 5: Test Training
```bash
./isaaclab.sh -p dextrah_lab/rl_games/train.py --task=DextrahFR3Agilehand --num_envs=64 --headless
```

Should start without errors!

## What the Script Does

### For Each Fingertip:

1. **Creates Rigid Body** (massless, 1 gram)
   - Applied `PhysicsRigidBodyAPI`
   - Set mass = 0.001 kg (negligible)
   - Position offset from parent

2. **Creates Fixed Joint**
   - Connects tip to parent distal phalanx
   - Zero relative motion (rigidly fixed)
   - Proper local pose alignment

3. **Result:**
   - Tip appears in `body_names` ✓
   - Moves perfectly with parent ✓
   - Can query position/velocity ✓
   - Doesn't affect dynamics ✓

## Technical Details

### Joint Configuration
```python
# Joint connects parent to child
Body0: Index_Distal_Phalanx (parent)
Body1: Index_Tip (child)

# Local poses ensure alignment
LocalPos0: (0.02, 0, 0)  # Offset in parent frame
LocalPos1: (0, 0, 0)     # Origin in child frame
LocalRot0/1: Identity    # No rotation offset
```

### Why This Works
- The fixed joint constrains all 6 DOF (translation + rotation)
- The tip has 0.001 kg mass (negligible for dynamics)
- LocalPos0 matches the tip's xformOp:translate
- No "disjointed body transforms" because poses are aligned

## Troubleshooting

### Error: "disjointed body transforms"
- LocalPos0 doesn't match tip's translate offset
- Solution: Adjust offsets in the script to match your geometry

### Tips don't appear in body_names
- PhysicsRigidBodyAPI not applied correctly
- Solution: Check USD with Isaac Sim's property inspector

### Tips are in wrong location
- Offset values are incorrect
- Solution: Open USD in Isaac Sim, measure actual distances, update offsets

### Joint warnings on load
- Parent body doesn't exist
- Solution: Check distal phalanx names match exactly

## Adjusting Tip Positions

If tips aren't at the right location, edit `create_rigid_tip_frames.py`:

```python
tip_configs = [
    ("Thumb_Distal_Phalanx", "Thumb_Tip", (0.030, 0.0, 0.0)),  # Changed X from 0.025 to 0.030
    ("Index_Distal_Phalanx", "Index_Tip", (0.025, 0.0, 0.0)),  # Changed X from 0.020 to 0.025
    # ... adjust others
]
```

Then re-run the script (it automatically removes old tips first).

## Comparison: Before vs After

| Aspect | Before (Xforms) | After (Rigid Bodies) |
|--------|----------------|---------------------|
| **In body_names** | ❌ No | ✅ Yes |
| **Trackable** | ❌ No | ✅ Yes |
| **For observations** | ❌ Can't use | ✅ Can use |
| **Mass** | N/A | 0.001 kg (negligible) |
| **Joint** | None | Fixed (0 DOF) |
| **Physics impact** | None | None (massless) |

## Files Created

1. `create_rigid_tip_frames.py` - Main creation script
2. `print_body_names.py` - Verification helper
3. `verify_hand_bodies.py` - Full config verification

## If Something Goes Wrong

Restore backup:
```bash
cd /home/carsten.oertel/code/tg2_dexman_isaac_co/dextrah_lab/assets/fr3_tekken_adof
cp FR3_tekkenadof_left.usd.backup FR3_tekkenadof_left.usd
```

Or restore from git:
```bash
git checkout FR3_tekkenadof_left.usd
```
