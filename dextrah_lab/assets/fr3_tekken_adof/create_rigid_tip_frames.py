#!/usr/bin/env python3
"""
Create proper rigid-body fingertip frames with fixed joints.
These will be massless bodies that appear in the articulation's body_names.

Usage:
    ./isaaclab.sh -p dextrah_lab/assets/fr3_tekken_adof/create_rigid_tip_frames.py --headless
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create rigid-body tip frames with fixed joints")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# Get USD path
script_dir = os.path.dirname(os.path.abspath(__file__))
usd_path = os.path.join(script_dir, "FR3_tekkenadof_left.usd")

print(f"\nOpening USD file: {usd_path}")
stage = Usd.Stage.Open(usd_path)

if not stage:
    print(f"ERROR: Could not open {usd_path}")
    simulation_app.close()
    sys.exit(1)

# Base path
base_path = "/fr3_tekken_left/tekken_left_adof"

# ============================================================================
# CONFIGURATION
# ============================================================================
# Format: (parent_link_name, tip_name, local_offset)
# Offsets are in the LOCAL FRAME of the parent distal phalanx
tip_configs = [
    ("Thumb_Distal_Phalanx", "Thumb_Tip", (0.025, 0.0, 0.0)),
    ("Index_Distal_Phalanx", "Index_Tip", (0.020, 0.0, 0.0)),
    ("Middle_Distal_Phalanx", "Middle_Tip", (0.020, 0.0, 0.0)),
    ("Ring_Distal_Phalanx", "Ring_Tip", (0.020, 0.0, 0.0)),
    ("Pinky_Distal_Phalanx", "Pinky_Tip", (0.020, 0.0, 0.0)),
]

print("\n" + "=" * 70)
print("CREATING RIGID-BODY TIP FRAMES WITH FIXED JOINTS")
print("=" * 70)

# First, remove any existing tip frames (Xforms or old rigid bodies)
print("\n1. Cleaning up existing tip frames...")
for distal_name, tip_name, _ in tip_configs:
    # Check both locations: as child and as sibling
    tip_path_child = f"{base_path}/{distal_name}/{tip_name}"
    tip_path_sibling = f"{base_path}/{tip_name}"
    
    if stage.GetPrimAtPath(tip_path_child).IsValid():
        stage.RemovePrim(tip_path_child)
        print(f"   Removed existing (child): {tip_path_child}")
    
    if stage.GetPrimAtPath(tip_path_sibling).IsValid():
        stage.RemovePrim(tip_path_sibling)
        print(f"   Removed existing (sibling): {tip_path_sibling}")

# Also remove old fixed joint definitions if they exist
joints_path = f"{base_path}/joints"
if stage.GetPrimAtPath(joints_path).IsValid():
    joints_prim = stage.GetPrimAtPath(joints_path)
    for child in joints_prim.GetChildren():
        if "tip" in child.GetName().lower():
            stage.RemovePrim(child.GetPath())
            print(f"   Removed old joint: {child.GetPath()}")

print("\n2. Creating rigid-body tip frames...")
success_count = 0

for distal_name, tip_name, offset in tip_configs:
    distal_path = f"{base_path}/{distal_name}"
    distal_prim = stage.GetPrimAtPath(distal_path)
    
    if not distal_prim.IsValid():
        print(f"   ❌ ERROR: Parent '{distal_name}' not found")
        continue
    
    # ========================================================================
    # STEP 1: Create tip at articulation root level (sibling to parent)
    # ========================================================================
    # CRITICAL: Tip must be at same level as distal phalanx, not a child!
    tip_path = f"{base_path}/{tip_name}"  # Sibling, not child
    tip_xform = UsdGeom.Xform.Define(stage, tip_path)
    
    # Don't set local offset here - the fixed joint will handle positioning
    # The tip will be positioned relative to parent via the joint
    # Leave tip at origin for now
    
    # ========================================================================
    # STEP 2: Apply PhysicsRigidBodyAPI to make it a rigid body
    # ========================================================================
    tip_prim = stage.GetPrimAtPath(tip_path)
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(tip_prim)
    
    # Set very small mass (almost massless, won't affect dynamics)
    mass_api = UsdPhysics.MassAPI.Apply(tip_prim)
    mass_api.GetMassAttr().Set(0.001)  # 1 gram
    
    # Disable gravity (parent's gravity setting will dominate anyway)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    
    # ========================================================================
    # STEP 3: Create fixed joint connecting tip to parent
    # ========================================================================
    # Joint must be at the articulation root level, not inside the body
    joint_name = f"fixed_{tip_name.lower()}"
    joint_path = f"{base_path}/{joint_name}"
    
    # Create FixedJoint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    
    # Set the two bodies connected by this joint
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(distal_path)])  # Parent
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(tip_path)])     # Child (tip)
    
    # CRITICAL: Set the local poses
    # LocalPos0 is the attachment point in the parent's frame (at the fingertip)
    # LocalPos1 is the attachment point in the child's frame (at origin)
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))  # Fingertip location in parent frame
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))  # Tip body origin
    
    # Rotation is identity (tips inherit parent rotation)
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    
    # Enable the joint
    fixed_joint.CreateJointEnabledAttr(True)
    
    print(f"   ✓ Created: {tip_name}")
    print(f"      Rigid body: {tip_path}")
    print(f"      Fixed joint: {joint_path}")
    print(f"      Offset: {offset}")
    
    success_count += 1

print("\n" + "=" * 70)
print(f"SUMMARY: {success_count}/{len(tip_configs)} tip frames created")
print("=" * 70)

if success_count > 0:
    stage.Save()
    print(f"\n✅ USD saved successfully to:\n   {usd_path}")
    
    print("\n" + "!" * 70)
    print("NEXT STEPS:")
    print("!" * 70)
    print("1. Test loading the robot in Isaac Sim")
    print("2. Run the verification script:")
    print("   ./isaaclab.sh -p dextrah_lab/tasks/fr3_agilehand/verify_hand_bodies.py --headless")
    print("3. Check for physics warnings - should be clean now")
    print("4. If successful, update your env config to use tip names:")
    print("   hand_body_names = ['base_link', 'Thumb_Tip', 'Index_Tip', ...]")
    print("!" * 70)
else:
    print("\n❌ No tip frames were created. Check errors above.")
    simulation_app.close()
    sys.exit(1)

simulation_app.close()
