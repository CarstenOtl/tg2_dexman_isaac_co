#!/usr/bin/env python3
"""
Helper script to remove tip frames from USD.
Use this if you need to adjust offsets and recreate the tips.

Usage:
    python remove_fingertip_frames.py
"""
from pxr import Usd
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

# Define the base path for your hand
base_path = "/fr3_tekken_left/tekken_left_adof"

# Tip frame names to remove
tip_names = [
    ("Thumb_Distal_Phalanx", "thumb_tip"),
    ("Index_Distal_Phalanx", "index_tip"),
    ("Middle_Distal_Phalanx", "middle_tip"),
    ("Ring_Distal_Phalanx", "ring_tip"),
    ("Pinky_Distal_Phalanx", "pinky_tip"),
]

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
    print("\nYou can now run add_fingertip_frames.py again with adjusted offsets.")
else:
    print("\n⚠️  No tips were found to remove.")
