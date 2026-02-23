#!/usr/bin/env python3
"""
Safe script to add fingertip coordinate frames to FR3 Tekken USD.
The tip frames inherit the parent distal phalanx orientation automatically.
You only need to tune the Y and Z position offsets.

Run this ONCE on a clean USD file.

Usage:
    python add_fingertip_frames.py
"""
from pxr import Usd, UsdGeom, Gf
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
usd_path = os.path.join(script_dir, "FR3_tekkenadof_left.usd")

print(f"Opening USD file: {usd_path}")

# Open the USD stage
stage = Usd.Stage.Open(usd_path)

if not stage:
    print(f"ERROR: Could not open {usd_path}")
    sys.exit(1)

# Define the base path for your hand in the USD hierarchy
base_path = "/fr3_tekken_left/tekken_left_adof"

# ============================================================================
# CONFIGURE TIP OFFSETS HERE
# ============================================================================
# Each entry: (distal_phalanx_name, tip_name, (x_offset, y_offset, z_offset))
#
# The offsets are in the LOCAL FRAME of the parent distal phalanx.
# The tip will INHERIT the parent's rotation, so you only need to position it.
#
# Coordinate system (adjust based on your finger geometry):
#   X: typically along the finger length (toward tip)
#   Y: typically finger width direction
#   Z: typically finger thickness direction
#
# Start with small offsets and adjust based on visual inspection in Isaac Sim.
# Units are in METERS.
# ============================================================================

tip_configs = [
    # (Parent link name, Tip frame name, (X offset, Y offset, Z offset))
    ("Thumb_Distal_Phalanx", "thumb_tip", (0.025, 0.0, 0.0)),   # 2.5cm along X
    ("Index_Distal_Phalanx", "index_tip", (0.020, 0.0, 0.0)),   # 2.0cm along X
    ("Middle_Distal_Phalanx", "middle_tip", (0.020, 0.0, 0.0)), # 2.0cm along X
    ("Ring_Distal_Phalanx", "ring_tip", (0.020, 0.0, 0.0)),     # 2.0cm along X
    ("Pinky_Distal_Phalanx", "pinky_tip", (0.020, 0.0, 0.0)),   # 2.0cm along X
]

print("\n" + "=" * 70)
print("ADDING FINGERTIP COORDINATE FRAMES")
print("=" * 70)

success_count = 0
skip_count = 0
error_count = 0

for distal_name, tip_name, offset in tip_configs:
    # Path to the distal phalanx (parent)
    distal_path = f"{base_path}/{distal_name}"
    distal_prim = stage.GetPrimAtPath(distal_path)
    
    if not distal_prim.IsValid():
        print(f"❌ ERROR: Parent link '{distal_name}' not found at {distal_path}")
        error_count += 1
        continue
    
    # Path for the new tip frame (as child of distal phalanx)
    tip_path = f"{distal_path}/{tip_name}"
    
    # Check if tip already exists
    existing_tip = stage.GetPrimAtPath(tip_path)
    if existing_tip.IsValid():
        print(f"⚠️  SKIP: '{tip_name}' already exists at {tip_path}")
        skip_count += 1
        continue
    
    # Create the tip as a simple Xform (coordinate frame only)
    # By default, Xform children inherit parent rotation
    tip_xform = UsdGeom.Xform.Define(stage, tip_path)
    
    # Add ONLY a translation operation
    # This means the tip inherits the parent's rotation but is offset in position
    xform_op = tip_xform.AddTranslateOp()
    xform_op.Set(Gf.Vec3d(*offset))
    
    # Verify the transform op order (should just be translate)
    tip_xform.SetXformOpOrder([xform_op])
    
    print(f"✓ Created: {tip_name}")
    print(f"    Parent: {distal_name}")
    print(f"    Local offset (m): X={offset[0]:.4f}, Y={offset[1]:.4f}, Z={offset[2]:.4f}")
    print(f"    Full path: {tip_path}")
    print()
    
    success_count += 1

print("=" * 70)
print(f"SUMMARY: {success_count} created, {skip_count} skipped, {error_count} errors")
print("=" * 70)

if success_count > 0:
    # Save the modified USD
    stage.Save()
    print(f"\n✅ USD saved successfully to:\n   {usd_path}")
    
    print("\n" + "!" * 70)
    print("NEXT STEPS:")
    print("!" * 70)
    print("1. Open the USD in Isaac Sim")
    print("2. Visually inspect the tip frame positions")
    print("3. If tips are not at the fingertip contact points:")
    print("   - Adjust the offsets in this script")
    print("   - Delete the tip prims from the USD (or restore from backup)")
    print("   - Run this script again")
    print("4. Once satisfied, update your env config to use the new tip names")
    print()
    print("To verify in Isaac Sim:")
    print("  - Select the robot in the stage tree")
    print("  - Manually move finger joints")
    print("  - The tip frames should move and rotate with the fingers")
    print("!" * 70)
elif skip_count > 0:
    print("\n⚠️  All tips already exist. No changes made to USD.")
    print("   To recreate tips, first delete them from the USD or restore from backup.")
else:
    print("\n❌ No tips were created. Check the errors above.")
    sys.exit(1)
