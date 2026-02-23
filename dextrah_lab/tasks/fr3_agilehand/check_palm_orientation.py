#!/usr/bin/env python3
"""Check palm orientation at spawn."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import torch
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

cfg = DextrahFR3AgilehandEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
cfg.objects_dir = "test_object"

env = DextrahFR3AgilehandEnv(cfg, render_mode=None)
env.reset()

print("=" * 80)
print("PALM ORIENTATION CHECK")
print("=" * 80)

# Compute palm direction
env._compute_intermediate_values()

palm_dir = env.palm_direction_vec[0]
target_dir = env._palm_dir_target_world[0]

print(f"\nPalm down axis (local): {env._palm_down_local_axis.squeeze().cpu().tolist()}")
print(f"Palm direction (world): {palm_dir.cpu().tolist()}")
print(f"Target direction (world -Z): {target_dir.cpu().tolist()}")

cos_angle = torch.sum(palm_dir * target_dir).item()
angle_deg = torch.acos(torch.clamp(torch.tensor(cos_angle), -1, 1)).item() * 57.3

print(f"\nCosine of angle: {cos_angle:.4f}")
print(f"Angle from target: {angle_deg:.1f} degrees")
print(f"Palm flip threshold: cos > {env.cfg.palm_flip_cos_thresh}")

if cos_angle < env.cfg.palm_flip_cos_thresh:
    print(f"\n❌ PALM IS FLIPPED! (cos={cos_angle:.4f} < {env.cfg.palm_flip_cos_thresh})")
    print("   Robot will reset immediately every step!")
    print("\nFIX: Need to either:")
    print("   1. Rotate robot base spawn orientation")
    print("   2. Adjust palm_down_local_axis")
    print("   3. Increase palm_flip_cos_thresh (temporary)")
else:
    print(f"\n✅ Palm orientation is OK")

env.close()
simulation_app.close()
