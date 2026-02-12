#!/usr/bin/env python3
"""
Find the correct palm orientation for FR3 robot.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import torch
import numpy as np
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

def main():
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"
    
    env = DextrahFR3AgilehandEnv(cfg, render_mode=None)
    env.reset()
    env._compute_intermediate_values()
    
    print("=" * 80)
    print("PALM ORIENTATION DIAGNOSTIC")
    print("=" * 80)
    
    # Get palm body info
    palm_body_idx = env.palm_body_idx
    palm_body_name = env.robot.body_names[palm_body_idx]
    
    print(f"\nPalm body: {palm_body_name} (index {palm_body_idx})")
    
    # Get palm pose
    # Converts quaternion → rotation matrix
    # This matrix tells us how the palm's local axes (X, Y, Z) 
    # are oriented in world space
    palm_pos = env.robot.data.body_pos_w[0, palm_body_idx]
    palm_quat = env.robot.data.body_quat_w[0, palm_body_idx]  # (w, x, y, z)
    
    print(f"\nPalm position (world): {palm_pos.cpu().numpy()}")
    print(f"Palm quaternion (w,x,y,z): {palm_quat.cpu().numpy()}")
    
    # Convert quaternion to rotation matrix manually
    # quat format: (w, x, y, z)
    w, x, y, z = palm_quat
    palm_rot_mat = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    ])  # 3x3 matrix
    
    print(f"\nPalm rotation matrix (columns are X, Y, Z axes in world frame):")
    print(palm_rot_mat.cpu().numpy())
    
    # Current palm local axis configuration
    palm_down_local = torch.tensor(cfg.palm_down_local_axis, device=env.device, dtype=torch.float)
    print(f"\nCurrent palm_down_local_axis: {palm_down_local.cpu().numpy()}")
    
    # IMPORTANT: Normalize the vector (user might provide non-unit vectors)
    palm_down_local_norm = torch.norm(palm_down_local)
    if palm_down_local_norm > 1e-6:
        palm_down_local = palm_down_local / palm_down_local_norm
        if abs(palm_down_local_norm - 1.0) > 0.01:
            print(f"⚠️  WARNING: palm_down_local_axis was not a unit vector (magnitude: {palm_down_local_norm:.4f})")
            print(f"   Normalized to: {palm_down_local.cpu().numpy()}")
    
    # Transform local axis to world frame
    palm_down_world = torch.matmul(palm_rot_mat, palm_down_local)
    print(f"Palm down direction (world): {palm_down_world.cpu().numpy()}")
    
    # Target direction (downward = -Z in world)
    target_world = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    print(f"Target direction (world -Z): {target_world.cpu().numpy()}")
    
    # Compute angle
    cos_angle = torch.sum(palm_down_world * target_world).item()
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    
    print(f"\nCurrent alignment:")
    print(f"  Cosine: {cos_angle:.4f}")
    print(f"  Angle: {angle_deg:.1f}°")
    print(f"  Threshold: {cfg.palm_flip_cos_thresh:.4f} (currently disabled at -1.0)")
    
    if cos_angle < 0.0:  # More than 90° off
        print(f"  Status: ❌ SEVERELY MISALIGNED (> 90° off)")
    elif cos_angle < 0.5:  # 60-90° off
        print(f"  Status: ⚠️  BADLY MISALIGNED (60-90° off)")
    elif cos_angle < 0.866:  # 30-60° off
        print(f"  Status: ⚠️  MISALIGNED (30-60° off)")
    else:
        print(f"  Status: ✅ OK (< 30° off)")
    
    # Find best local axis by trying all 6 cardinal directions
    print("\n" + "=" * 80)
    print("TESTING ALL CARDINAL DIRECTIONS")
    print("=" * 80)
    
    cardinal_axes = [
        ("+X", torch.tensor([1.0, 0.0, 0.0], device=env.device)),
        ("-X", torch.tensor([-1.0, 0.0, 0.0], device=env.device)),
        ("+Y", torch.tensor([0.0, 1.0, 0.0], device=env.device)),
        ("-Y", torch.tensor([0.0, -1.0, 0.0], device=env.device)),
        ("+Z", torch.tensor([0.0, 0.0, 1.0], device=env.device)),
        ("-Z", torch.tensor([0.0, 0.0, -1.0], device=env.device)),
    ]
    
    results = []
    for name, axis in cardinal_axes:
        world_dir = torch.matmul(palm_rot_mat, axis)
        cos = torch.sum(world_dir * target_world).item()
        angle = np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi
        results.append((name, axis, cos, angle))
    
    # Sort by cosine (best first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\nRanked by alignment with downward (-Z):")
    print(f"{'Axis':<6} {'Local Vector':<20} {'Cosine':>8} {'Angle':>8}")
    print("-" * 50)
    
    for i, (name, axis, cos, angle) in enumerate(results):
        marker = "✅" if i == 0 else "  "
        axis_str = f"({axis[0].item():+.1f}, {axis[1].item():+.1f}, {axis[2].item():+.1f})"
        print(f"{marker} {name:<6} {axis_str:<20} {cos:+8.4f} {angle:7.1f}°")
    
    # Recommendation
    best_name, best_axis, best_cos, best_angle = results[0]
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if best_cos > 0.95:  # < ~18° off
        print(f"\n✅ Best axis is {best_name} with {best_angle:.1f}° misalignment")
        print(f"\nUpdate your config:")
        print(f"```python")
        print(f"# dextrah_lab/tasks/fr3_agilehand/dextrah_fr3_agilehand_env_cfg.py")
        print(f"palm_down_local_axis = ({best_axis[0].item():+.1f}, {best_axis[1].item():+.1f}, {best_axis[2].item():+.1f})  # {best_name}")
        print(f"```")
        print(f"\nThen restore the threshold:")
        print(f"```python")
        print(f"palm_flip_cos_thresh = 0.0  # Restore from -1.0")
        print(f"```")
    else:
        print(f"\n⚠️  No cardinal axis aligns well (best is {best_angle:.1f}° off)")
        print(f"\nThis suggests the hand might be rotated oddly in the robot model.")
        print(f"\nOptions:")
        print(f"1. Use best available: {best_name} (but expect some false positives)")
        print(f"2. Rotate robot base in init_state")
        print(f"3. Keep palm_flip_cos_thresh = -1.0 (disabled)")
        print(f"4. Check hand attachment in USD file")
    
    # Also check robot base rotation
    print("\n" + "=" * 80)
    print("ROBOT BASE ORIENTATION")
    print("=" * 80)
    
    base_quat = env.robot.data.root_quat_w[0]
    print(f"\nRobot base quaternion (w,x,y,z): {base_quat.cpu().numpy()}")
    
    # Check if it's identity (no rotation)
    if torch.allclose(base_quat, torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device), atol=0.01):
        print("Robot base: No rotation (identity)")
    else:
        print("Robot base: Has rotation applied")
        w, x, y, z = base_quat
        base_rot_mat = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
        ])
        print(f"Rotation matrix:\n{base_rot_mat.cpu().numpy()}")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
