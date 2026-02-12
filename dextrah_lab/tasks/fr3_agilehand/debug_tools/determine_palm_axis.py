#!/usr/bin/env python3
"""
Simple guide to determine the correct palm_down_local_axis.

This will spawn the robot and print clear instructions on how to visually
determine which local axis corresponds to the palm-down direction.

Usage:
    python -m dextrah_lab.tasks.fr3_agilehand.determine_palm_axis
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

def quaternion_to_matrix(quat):
    """Convert quaternion (w,x,y,z) to rotation matrix."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    ])
    return R

def main():
    # Create environment
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    
    # Set required objects directory
    cfg.objects_dir = "test_object"
    
    env = DextrahFR3AgilehandEnv(cfg)
    env.reset()
    
    # Get palm body
    palm_body_idx = env.robot.body_names.index(cfg.palm_body_name)
    
    print("\n" + "="*80)
    print("HOW TO DETERMINE THE CORRECT palm_down_local_axis")
    print("="*80)
    
    print(f"\n1. VISUAL INSPECTION:")
    print(f"   Look at your robot in the simulator (re-run without --headless)")
    print(f"   The palm should be facing DOWNWARD toward the table")
    print(f"   Identify which direction the PALM SURFACE points")
    print(f"   (Not the fingers, but the palm itself)")
    
    print(f"\n2. UNDERSTAND LOCAL AXES:")
    print(f"   In the palm body's local coordinate frame:")
    print(f"   - +X points in one direction from the palm center")
    print(f"   - +Y points in another direction")
    print(f"   - +Z points in the third direction")
    print(f"   One of these should point in the direction the palm faces")
    
    print(f"\n3. TEST EACH AXIS:")
    print(f"   The table below shows which world direction each local axis points to.")
    print(f"   Find the axis that points DOWNWARD (world -Z direction)")
    
    # Step simulation to get current state
    for _ in range(10):
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    
    # Get palm orientation
    palm_quat = env.robot.data.body_quat_w[0, palm_body_idx]  # (w,x,y,z)
    palm_pos = env.robot.data.body_pos_w[0, palm_body_idx]
    palm_rot = quaternion_to_matrix(palm_quat)
    
    print(f"\n" + "="*80)
    print(f"PALM ORIENTATION ANALYSIS")
    print(f"="*80)
    print(f"\nPalm body: {cfg.palm_body_name}")
    print(f"Palm position: {palm_pos.cpu().numpy()}")
    print(f"Palm quaternion (w,x,y,z): {palm_quat.cpu().numpy()}")
    
    # Test all cardinal directions
    cardinal_axes = {
        "+X": torch.tensor([1.0, 0.0, 0.0], device=env.device),
        "-X": torch.tensor([-1.0, 0.0, 0.0], device=env.device),
        "+Y": torch.tensor([0.0, 1.0, 0.0], device=env.device),
        "-Y": torch.tensor([0.0, -1.0, 0.0], device=env.device),
        "+Z": torch.tensor([0.0, 0.0, 1.0], device=env.device),
        "-Z": torch.tensor([0.0, 0.0, -1.0], device=env.device),
    }
    
    target_down = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    
    results = []
    for name, local_vec in cardinal_axes.items():
        # Transform to world frame
        world_vec = torch.matmul(palm_rot, local_vec)
        
        # Compute alignment with downward
        cos_angle = torch.dot(world_vec, target_down).item()
        angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        results.append((name, world_vec, cos_angle, angle_deg))
    
    # Sort by alignment (best first)
    results.sort(key=lambda x: -x[2])
    
    print(f"\n{'='*80}")
    print(f"LOCAL AXIS → WORLD DIRECTION MAPPING")
    print(f"{'='*80}")
    print(f"{'Local Axis':<12} {'World Direction':<30} {'Alignment to Down':<20} {'Angle'}")
    print(f"{'-'*80}")
    
    for name, world_vec, cos_angle, angle_deg in results:
        world_str = f"({world_vec[0]:+.3f}, {world_vec[1]:+.3f}, {world_vec[2]:+.3f})"
        alignment_str = f"cos={cos_angle:+.4f}"
        
        # Highlight the best match
        marker = "✅ BEST" if cos_angle == results[0][2] else ""
        if cos_angle > 0.866:  # < 30° off
            marker = "✅ GOOD"
        elif cos_angle > 0.5:  # < 60° off
            marker = "⚠️  OK"
        elif cos_angle > 0:  # < 90° off
            marker = "⚠️  POOR"
        else:
            marker = "❌ BAD"
        
        print(f"{name:<12} {world_str:<30} {alignment_str:<20} {angle_deg:>5.1f}°  {marker}")
    
    # Recommendation
    best_name = results[0][0]
    best_cos = results[0][2]
    best_angle = results[0][3]
    best_vec = cardinal_axes[best_name].cpu().numpy()
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    print(f"\nBest match: {best_name} (angle offset: {best_angle:.1f}°)")
    print(f"\nIn your config file, set:")
    print(f"    palm_down_local_axis = ({best_vec[0]:+.1f}, {best_vec[1]:+.1f}, {best_vec[2]:+.1f})")
    
    if best_angle > 30:
        print(f"\n⚠️  WARNING: {best_angle:.1f}° is a significant offset!")
        print(f"   This suggests the hand may be mounted at an angle in the USD file.")
        print(f"\n   Possible solutions:")
        print(f"   1. Accept the offset and adjust palm_flip_cos_thresh accordingly")
        print(f"      Recommended: palm_flip_cos_thresh = {np.cos(np.radians(best_angle + 30)):.4f}")
        print(f"   2. Rotate the robot base in init_state to compensate")
        print(f"   3. Modify the USD file to fix the hand mounting angle")
    else:
        print(f"\n✅ Good alignment! You can use:")
        print(f"    palm_flip_cos_thresh = 0.0  # Triggers at 90° flip")
    
    print(f"\n{'='*80}")
    print(f"\n4. VERIFY:")
    print(f"   After updating palm_down_local_axis, run find_palm_orientation.py again")
    print(f"   to confirm the alignment improves.")
    print(f"{'='*80}\n")
    
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
