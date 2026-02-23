#!/usr/bin/env python3
"""
Compare joint drive properties between FR3 and TG2 USDs.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare USD joint properties")
parser.add_argument("--headless", action="store_true", help="Run headless")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import sys
from pathlib import Path
from pxr import Usd, UsdPhysics, PhysxSchema

# Find the workspace root
current_file = Path(__file__).resolve()
workspace_root = current_file.parents[3]

fr3_usd = workspace_root / "dextrah_lab" / "assets" / "fr3_tekken_adof" / "FR3_tekkenadof_left.usd"
tg2_usd = workspace_root / "dextrah_lab" / "assets" / "tg2_inspirehand" / "tg2_inspirehand_no_leg.usd"

print("=" * 80)
print("COMPARING JOINT DRIVE PROPERTIES")
print("=" * 80)

def inspect_joints(usd_path, robot_name):
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open {usd_path}")
        return
    
    print(f"\n{robot_name}: {usd_path.name}")
    print("-" * 80)
    
    # Find all revolute joints
    joint_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            joint_name = prim.GetName()
            
            # Skip hand joints for FR3, focus on arm
            if "revolute_" in joint_name:
                continue
            
            joint_count += 1
            if joint_count > 3:  # Only show first 3 arm joints
                continue
            
            print(f"\n  Joint: {joint_name}")
            print(f"    Path: {prim.GetPath()}")
            
            # Check drive API
            drive_api = UsdPhysics.DriveAPI(prim, "angular")
            if drive_api:
                drive_type = drive_api.GetTypeAttr()
                stiffness = drive_api.GetStiffnessAttr()
                damping = drive_api.GetDampingAttr()
                max_force = drive_api.GetMaxForceAttr()
                target_position = drive_api.GetTargetPositionAttr()
                target_velocity = drive_api.GetTargetVelocityAttr()
                
                print(f"    DriveAPI: Yes")
                if drive_type and drive_type.Get():
                    print(f"      Type: {drive_type.Get()}")
                if stiffness and stiffness.Get() is not None:
                    print(f"      Stiffness: {stiffness.Get()}")
                if damping and damping.Get() is not None:
                    print(f"      Damping: {damping.Get()}")
                if max_force and max_force.Get() is not None:
                    print(f"      MaxForce: {max_force.Get()}")
                if target_position and target_position.Get() is not None:
                    print(f"      TargetPosition: {target_position.Get()}")
                if target_velocity and target_velocity.Get() is not None:
                    print(f"      TargetVelocity: {target_velocity.Get()}")
            else:
                print(f"    ❌ DriveAPI: MISSING")
            
            # Check PhysX joint API
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            if physx_joint:
                if physx_joint.GetArmatureAttr():
                    armature = physx_joint.GetArmatureAttr().Get()
                    if armature:
                        print(f"      Armature: {armature}")
            
            # Check joint limits
            joint_api = UsdPhysics.Joint(prim)
            if joint_api:
                # For revolute joint, check lower/upper limits
                lower_limit_attr = prim.GetAttribute("physics:lowerLimit")
                upper_limit_attr = prim.GetAttribute("physics:upperLimit")
                
                if lower_limit_attr and upper_limit_attr:
                    lower = lower_limit_attr.Get()
                    upper = upper_limit_attr.Get()
                    print(f"      Limits: [{lower}, {upper}]")

inspect_joints(fr3_usd, "FR3")
inspect_joints(tg2_usd, "TG2")

print("\n" + "=" * 80)
print("CHECKING LINK MASSES")
print("=" * 80)

def check_masses(usd_path, robot_name):
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return
    
    print(f"\n{robot_name}:")
    print("-" * 80)
    
    link_count = 0
    for prim in stage.Traverse():
        if "link" in prim.GetName().lower() and prim.GetTypeName() == "Xform":
            link_count += 1
            if link_count > 3:  # Only show first 3 links
                continue
            
            mass_api = UsdPhysics.MassAPI(prim)
            if mass_api:
                mass_attr = mass_api.GetMassAttr()
                if mass_attr and mass_attr.Get() is not None:
                    print(f"  {prim.GetName():20s} Mass: {mass_attr.Get()} kg")

check_masses(fr3_usd, "FR3")
check_masses(tg2_usd, "TG2")

simulation_app.close()
