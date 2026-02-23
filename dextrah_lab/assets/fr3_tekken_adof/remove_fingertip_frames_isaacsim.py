#!/usr/bin/env python3
"""
Helper script to remove tip frames from USD.
Runs as an Isaac Sim application to access the pxr USD library.

Usage:
    ./isaaclab.sh -p dextrah_lab/assets/fr3_tekken_adof/remove_fingertip_frames_isaacsim.py
"""

import argparse
import sys
import os

# Setup Isaac Sim
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Remove fingertip frames from FR3 Tekken USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app (headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import USD libraries
from pxr import Usd

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
usd_path = os.path.join(script_dir, "FR3_tekkenadof_left.usd")

# Define the base path for your hand
base_path = "/fr3_tekken_left/tekken_left_adof"

# Tip frame names to remove (lowercase versions - the duplicates)
tip_names = [
    ("Thumb_Distal_Phalanx", "thumb_tip"),   # Remove lowercase duplicate
    ("Index_Distal_Phalanx", "index_tip"),   # Remove lowercase duplicate
    ("Middle_Distal_Phalanx", "middle_tip"), # Remove lowercase duplicate
    ("Ring_Distal_Phalanx", "ring_tip"),     # Remove lowercase duplicate
    ("Pinky_Distal_Phalanx", "pinky_tip"),   # Remove lowercase duplicate
]

def main():
    print(f"\nOpening USD file: {usd_path}")
    
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    
    if not stage:
        print(f"ERROR: Could not open {usd_path}")
        simulation_app.close()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("REMOVING FINGERTIP COORDINATE FRAMES")
    print("=" * 70)
    
    removed_count = 0
    not_found_count = 0
    
    for distal_name, tip_name in tip_names:
        tip_path = f"{base_path}/{distal_name}/{tip_name}"
        tip_prim = stage.GetPrimAtPath(tip_path)
        
        if tip_prim.IsValid():
            stage.RemovePrim(tip_path)
            print(f"✓ Removed: {tip_name} from {tip_path}")
            removed_count += 1
        else:
            print(f"⚠️  Not found: {tip_name} at {tip_path}")
            not_found_count += 1
    
    print("=" * 70)
    print(f"SUMMARY: {removed_count} removed, {not_found_count} not found")
    print("=" * 70)
    
    if removed_count > 0:
        stage.Save()
        print(f"\n✅ USD saved successfully to:\n   {usd_path}")
        print("\nYou can now run add_fingertip_frames_isaacsim.py again with adjusted offsets.")
    else:
        print("\n⚠️  No tips were found to remove.")
    
    # Close Isaac Sim
    simulation_app.close()

if __name__ == "__main__":
    main()
