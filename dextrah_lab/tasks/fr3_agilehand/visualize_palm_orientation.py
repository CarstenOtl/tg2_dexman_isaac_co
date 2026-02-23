#!/usr/bin/env python3
"""
Visualize palm orientation in GUI.
Shows the robot so you can visually inspect which way the palm faces.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# Note: AppLauncher automatically handles --livestream flag (use --livestream 2 for native streaming)
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

# Check if running with livestream
args = parser.parse_args()
has_gui = not args.headless if hasattr(args, 'headless') else False
has_livestream = args.livestream > 0 if hasattr(args, 'livestream') else False

import torch
import numpy as np
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

def quat_to_rot_matrix(quat):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = quat
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    ])

def main():
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"
    
    # Run with or without GUI
    render_mode = "human" if (has_gui or has_livestream) else None
    env = DextrahFR3AgilehandEnv(cfg, render_mode=render_mode)
    env.reset()
    env._compute_intermediate_values()
    
    print("=" * 80)
    print("PALM ORIENTATION DIAGNOSTIC" + (" (GUI/LIVESTREAM MODE)" if render_mode else ""))
    print("=" * 80)
    if has_gui or has_livestream:
        print("\nGUI Controls:")
        print("  - Rotate view with mouse to see palm orientation")
        print("  - Look at the hand - note which way palm surface faces")
        if has_livestream:
            print(f"  - Livestream URL: http://localhost:8211/streaming/webrtc-client/?server=localhost")
        print("  - Press Ctrl+C in terminal when done")
        print("=" * 80)
    
    # Get palm info
    palm_body_idx = env.palm_body_idx
    palm_body_name = env.robot.body_names[palm_body_idx]
    palm_pos = env.robot.data.body_pos_w[0, palm_body_idx]
    palm_quat = env.robot.data.body_quat_w[0, palm_body_idx]
    palm_rot_mat = quat_to_rot_matrix(palm_quat)
    
    print(f"\nPalm body: {palm_body_name}")
    print(f"Position: {palm_pos.cpu().numpy()}")
    
    # Current configuration
    palm_down_local = torch.tensor(cfg.palm_down_local_axis, device=env.device, dtype=torch.float)
    palm_down_world = torch.matmul(palm_rot_mat, palm_down_local)
    
    # Target (world -Z)
    target_world = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    
    # Check all cardinal directions
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
        results.append((name, axis, world_dir, cos, angle))
    
    results.sort(key=lambda x: x[3], reverse=True)
    best_name, best_axis, best_world_dir, best_cos, best_angle = results[0]
    
    print(f"\nCurrent config: palm_down_local_axis = {palm_down_local.cpu().numpy()}")
    print(f"  → Points in world direction: {palm_down_world.cpu().numpy()}")
    print(f"  → Angle from down: {np.arccos(np.clip(torch.sum(palm_down_world * target_world).item(), -1, 1)) * 180 / np.pi:.1f}°")
    
    print(f"\nBest axis: {best_name}")
    print(f"  → local vector: {best_axis.cpu().numpy()}")
    print(f"  → Points in world direction: {best_world_dir.cpu().numpy()}")
    print(f"  → Angle from down: {best_angle:.1f}°")
    
    print("\n" + "=" * 80)
    print("ALL AXES RANKED")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Axis':<6} {'Local Vector':<20} {'Cosine':>8} {'Angle':>8}")
    print("-" * 60)
    
    for i, (name, axis, world_dir, cos, angle) in enumerate(results):
        marker = "✅" if i == 0 else f"{i+1}."
        axis_str = f"({axis[0].item():+.1f}, {axis[1].item():+.1f}, {axis[2].item():+.1f})"
        print(f"{marker:<6} {name:<6} {axis_str:<20} {cos:+8.4f} {angle:7.1f}°")
    
    # If GUI mode, just show the robot for visual inspection
    if has_gui or has_livestream:
        print("\n" + "=" * 80)
        print("GUI MODE - VISUAL INSPECTION")
        print("=" * 80)
        print("\nLook at the palm of the hand in the GUI.")
        print("Identify which local axis (+X, -X, +Y, -Y, +Z, -Z) points")
        print("toward the palm surface (should point DOWN in world space).")
        print("\nRotate the view to see the hand clearly.")
        print("Press Ctrl+C in this terminal when done inspecting.")
        print("=" * 80)
        
        try:
            step_count = 0
            while simulation_app.is_running():
                env.sim.step()
                env.scene.update(dt=env.cfg.sim_dt)
                step_count += 1
                
                if step_count % 60 == 0:  # Print reminder every 60 steps (~1 sec)
                    print(".", end="", flush=True)
                    
        except KeyboardInterrupt:
            print("\nClosing GUI...")
    
    env.close()
    simulation_app.close()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nUpdate your config to:")
    print(f"```python")
    print(f"palm_down_local_axis = ({best_axis[0].item():+.1f}, {best_axis[1].item():+.1f}, {best_axis[2].item():+.1f})  # {best_name}")
    print(f"```")

if __name__ == "__main__":
    main()
