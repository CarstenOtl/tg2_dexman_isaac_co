#!/usr/bin/env python3
"""
Inspect USD articulation structure to find issues.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect USD structure")
parser.add_argument("--headless", action="store_true", help="Run headless")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import sys
import os
from pathlib import Path
from pxr import Usd, UsdPhysics, PhysxSchema
import omni.usd

# Find the workspace root (where dextrah_lab is)
current_file = Path(__file__).resolve()
workspace_root = current_file.parents[3]  # Go up to workspace root
usd_path = workspace_root / "dextrah_lab" / "assets" / "fr3_tekken_adof" / "FR3_tekkenadof_left.usd"

print(f"Loading USD from: {usd_path}")
if not usd_path.exists():
    print(f"ERROR: USD file not found at {usd_path}")
    simulation_app.close()
    sys.exit(1)

stage = Usd.Stage.Open(str(usd_path))

if not stage:
    print(f"Failed to open {usd_path}")
    sys.exit(1)

print("=" * 80)
print(f"USD: {usd_path}")
print("=" * 80)

# Find all prims with articulation root API
print("\nARTICULATION ROOTS:")
for prim in stage.Traverse():
    articulation = UsdPhysics.ArticulationRootAPI(prim)
    if articulation:
        print(f"  {prim.GetPath()} ({prim.GetTypeName()})")

# Find the main robot prim
robot_paths = ["/Robot", "/fr3", "/World/Robot"]
robot_prim = None
for path in robot_paths:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        robot_prim = prim
        print(f"\nFound robot at: {path}")
        break

if not robot_prim:
    print("\nSearching for any Xform with children...")
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Xform" and len(list(prim.GetChildren())) > 5:
            robot_prim = prim
            print(f"Found potential robot at: {prim.GetPath()}")
            break

if not robot_prim:
    print("ERROR: Could not find robot prim")
    sys.exit(1)

print(f"\n{'=' * 80}")
print(f"ROBOT STRUCTURE: {robot_prim.GetPath()}")
print(f"{'=' * 80}")

# Check root prim properties
articulation_api = UsdPhysics.ArticulationRootAPI(robot_prim)
if articulation_api:
    print("✅ Has ArticulationRootAPI")
else:
    print("❌ Missing ArticulationRootAPI")

rigidbody_api = UsdPhysics.RigidBodyAPI(robot_prim)
if rigidbody_api:
    print("✅ Has RigidBodyAPI")
    if rigidbody_api.GetKinematicEnabledAttr():
        is_kinematic = rigidbody_api.GetKinematicEnabledAttr().Get()
        if is_kinematic:
            print(f"  ❌ ROOT IS KINEMATIC! This would freeze the entire robot!")
        else:
            print(f"  ✅ Root is dynamic")

# Find root_joint and check its configuration
print(f"\n{'=' * 80}")
print("CHECKING ROOT_JOINT")
print(f"{'=' * 80}")

root_joint = None
for child in robot_prim.GetAllChildren():
    if "root" in child.GetName().lower() and "joint" in child.GetName().lower():
        root_joint = child
        break

if root_joint:
    print(f"Found: {root_joint.GetPath()}")
    print(f"Type: {root_joint.GetTypeName()}")
    
    joint_api = UsdPhysics.Joint(root_joint)
    if joint_api:
        # Get body0 and body1
        body0_rel = joint_api.GetBody0Rel()
        body1_rel = joint_api.GetBody1Rel()
        
        body0_targets = body0_rel.GetTargets()
        body1_targets = body1_rel.GetTargets()
        
        print(f"  Body0 (parent): {body0_targets if body0_targets else 'NONE (world)'}")
        print(f"  Body1 (child):  {body1_targets if body1_targets else 'NONE'}")
        
        if not body0_targets:
            print("  ✅ Body0 is empty (fixed to world)")
        
        # Check if joint is enabled
        if joint_api.GetJointEnabledAttr():
            enabled = joint_api.GetJointEnabledAttr().Get()
            print(f"  Enabled: {enabled}")
else:
    print("❌ root_joint not found")

# Check first arm joint
print(f"\n{'=' * 80}")
print("CHECKING FIRST ARM JOINT (fr3_joint1)")
print(f"{'=' * 80}")

fr3_joint1 = None
for child in robot_prim.GetAllChildren():
    if child.GetName() == "fr3_joint1":
        fr3_joint1 = child
        break

if fr3_joint1:
    print(f"Found: {fr3_joint1.GetPath()}")
    print(f"Type: {fr3_joint1.GetTypeName()}")
    
    joint_api = UsdPhysics.Joint(fr3_joint1)
    if joint_api:
        body0_rel = joint_api.GetBody0Rel()
        body1_rel = joint_api.GetBody1Rel()
        
        body0_targets = body0_rel.GetTargets()
        body1_targets = body1_rel.GetTargets()
        
        print(f"  Body0 (parent): {body0_targets}")
        print(f"  Body1 (child):  {body1_targets}")
        
        # Check joint drive
        drive_api = UsdPhysics.DriveAPI(fr3_joint1, "angular")
        if drive_api:
            print("  ✅ Has DriveAPI")
            
            # Check drive type
            drive_type = drive_api.GetTypeAttr()
            if drive_type:
                print(f"     Type: {drive_type.Get()}")
            
            # Check stiffness and damping
            stiffness = drive_api.GetStiffnessAttr()
            damping = drive_api.GetDampingAttr()
            max_force = drive_api.GetMaxForceAttr()
            
            if stiffness:
                print(f"     Stiffness: {stiffness.Get()}")
            if damping:
                print(f"     Damping: {damping.Get()}")
            if max_force:
                print(f"     MaxForce: {max_force.Get()}")
        else:
            print("  ❌ Missing DriveAPI")
else:
    print("❌ fr3_joint1 not found")

# List all links and their kinematic status
print(f"\n{'=' * 80}")
print("ALL LINKS")
print(f"{'=' * 80}")

for child in robot_prim.GetChildren():
    if "link" in child.GetName().lower():
        rigidbody_api = UsdPhysics.RigidBodyAPI(child)
        is_kinematic = None
        if rigidbody_api and rigidbody_api.GetKinematicEnabledAttr():
            is_kinematic = rigidbody_api.GetKinematicEnabledAttr().Get()
        
        if is_kinematic:
            print(f"❌ {child.GetName():30s} KINEMATIC")
        elif is_kinematic is False:
            print(f"✅ {child.GetName():30s} dynamic")
        else:
            print(f"⚠️  {child.GetName():30s} (kinematic not set)")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("If any links are KINEMATIC (❌), they need to be set to dynamic.")
print("If the root prim is KINEMATIC, the entire robot will be frozen.")
print("The root_joint should connect to world (Body0=None) and fr3_link0 (Body1).")

simulation_app.close()
