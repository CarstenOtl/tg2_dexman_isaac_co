#!/usr/bin/env python3
"""
Fix FR3 USD file to have correct default joint positions and base link mass.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fix FR3 USD")
parser.add_argument("--headless", action="store_true", help="Run headless")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

from pathlib import Path
from pxr import Usd, UsdPhysics

# Find USD file
current_file = Path(__file__).resolve()
workspace_root = current_file.parents[3]
usd_path = workspace_root / "dextrah_lab" / "assets" / "fr3_tekken_adof" / "FR3_tekkenadof_left.usd"
backup_path = usd_path.with_suffix('.usd.backup')

print(f"Loading USD: {usd_path}")
stage = Usd.Stage.Open(str(usd_path))

if not stage:
    print("Failed to open USD")
    simulation_app.close()
    exit(1)

# Backup original
if not backup_path.exists():
    print(f"Creating backup: {backup_path}")
    import shutil
    shutil.copy(usd_path, backup_path)

print("\n" + "=" * 80)
print("FIXING USD FILE")
print("=" * 80)

# Fix base link mass
print("\n1. Setting fr3_link0 mass to 2.0 kg...")
for prim in stage.Traverse():
    if prim.GetName() == "fr3_link0":
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(2.0)
        print(f"   ✅ Set {prim.GetPath()} mass to 2.0 kg")
        break

# Set default joint positions to match init_state
print("\n2. Setting default joint positions...")
joint_defaults = {
    "fr3_joint1": 0.0,
    "fr3_joint2": 0.0,
    "fr3_joint3": 0.0,
    "fr3_joint4": -0.9599,  # -55 degrees
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.7453,   # 100 degrees
    "fr3_joint7": 0.0,
}

for prim in stage.Traverse():
    joint_name = prim.GetName()
    if joint_name in joint_defaults:
        drive_api = UsdPhysics.DriveAPI(prim, "angular")
        if drive_api:
            target_pos = joint_defaults[joint_name]
            drive_api.GetTargetPositionAttr().Set(target_pos)
            print(f"   ✅ Set {joint_name} default position to {target_pos:.4f}")

# Save
print("\n3. Saving USD...")
stage.Save()
print(f"   ✅ Saved to {usd_path}")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
print(f"\nBackup saved to: {backup_path}")
print("\nNow test the robot again:")
print("  python -m dextrah_lab.tasks.fr3_agilehand.check_robot_physics --headless --num_envs 1")

simulation_app.close()
