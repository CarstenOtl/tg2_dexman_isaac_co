#!/usr/bin/env python3
"""
Visualize the palm's local coordinate axes in the simulator.

This tool draws RGB arrows showing the palm body's local X, Y, Z axes
in the simulator so you can visually identify which direction each points.

Red = +X local axis
Green = +Y local axis  
Blue = +Z local axis

Plus a YELLOW arrow showing your current palm_down_local_axis setting.

Usage:
    python -m dextrah_lab.tasks.fr3_agilehand.visualize_local_axes
    python -m dextrah_lab.tasks.fr3_agilehand.visualize_local_axes --livestream 1
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

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

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


def create_axis_marker(name: str, color: tuple, scale: float = 0.05) -> VisualizationMarkers:
    """Create a visualization marker for an axis (shown as a sphere at the end)."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/palm_axis_{name}",
        markers={
            f"sphere": sim_utils.SphereCfg(
                radius=scale,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def main():
    # Create environment
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    
    # Set required objects directory (using test_object as it's simple)
    cfg.objects_dir = "test_object"
    
    env = DextrahFR3AgilehandEnv(cfg)
    env.reset()
    
    # Get palm body
    palm_body_idx = env.robot.body_names.index(cfg.palm_body_name)
    
    print("\n" + "="*80)
    print("PALM LOCAL AXES VISUALIZER")
    print("="*80)
    print(f"\nPalm body: {cfg.palm_body_name}")
    print(f"Current palm_down_local_axis setting: {cfg.palm_down_local_axis}")
    
    # Normalize the configured axis
    palm_down_vec = np.array(cfg.palm_down_local_axis)
    palm_down_norm = np.linalg.norm(palm_down_vec)
    if palm_down_norm > 0:
        palm_down_vec_normalized = palm_down_vec / palm_down_norm
    else:
        palm_down_vec_normalized = palm_down_vec
    
    if abs(palm_down_norm - 1.0) > 0.01:
        print(f"\n⚠️  WARNING: palm_down_local_axis is NOT a unit vector!")
        print(f"   Current magnitude: {palm_down_norm:.4f}")
        print(f"   Normalized: ({palm_down_vec_normalized[0]:+.3f}, {palm_down_vec_normalized[1]:+.3f}, {palm_down_vec_normalized[2]:+.3f})")
        print(f"   Please use unit vectors like (+1,0,0) or (0,-1,0)")
    
    print("\n" + "="*80)
    print("VISUALIZATION KEY")
    print("="*80)
    print("The visualization will show colored spheres at the end of each axis:")
    print("  🔴 RED sphere    = Palm local +X axis (0.15m from palm center)")
    print("  🟢 GREEN sphere  = Palm local +Y axis (0.15m from palm center)")
    print("  🔵 BLUE sphere   = Palm local +Z axis (0.15m from palm center)")
    print("  🟡 YELLOW sphere = Your palm_down_local_axis setting (0.18m from palm center)")
    print("\nLook at which sphere is LOWEST (closest to the table).")
    print("That direction is your correct palm_down_local_axis!")
    print("="*80 + "\n")
    
    # Create visual markers for each axis
    sphere_scale = 0.03
    marker_x = create_axis_marker("x", (1.0, 0.0, 0.0), sphere_scale)  # Red
    marker_y = create_axis_marker("y", (0.0, 1.0, 0.0), sphere_scale)  # Green
    marker_z = create_axis_marker("z", (0.0, 0.0, 1.0), sphere_scale)  # Blue
    marker_config = create_axis_marker("config", (1.0, 1.0, 0.0), sphere_scale * 1.2)  # Yellow
    
    axes_info = [
        ("X", marker_x, np.array([1, 0, 0])),
        ("Y", marker_y, np.array([0, 1, 0])),
        ("Z", marker_z, np.array([0, 0, 1])),
        ("config", marker_config, palm_down_vec_normalized),
    ]
    
    # Simulation loop
    step_count = 0
    last_analysis = None
    
    while simulation_app.is_running():
        # Step with zero action to keep robot in place
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        
        # Update arrow visualizations every frame
        palm_pos = env.robot.data.body_pos_w[0, palm_body_idx]  # (3,)
        palm_quat = env.robot.data.body_quat_w[0, palm_body_idx]  # (4,) (w,x,y,z)
        
        # Convert quaternion to rotation matrix
        palm_rot = quaternion_to_matrix(palm_quat)  # (3, 3)
        
        # Transform local axes to world frame and visualize
        for name, marker, local_vec in axes_info:
            local_vec_torch = torch.tensor(local_vec, device=env.device, dtype=torch.float)
            world_vec = torch.matmul(palm_rot, local_vec_torch)  # (3,)
            
            # Create marker position: sphere at the end of the axis
            axis_length = 0.18 if name == "config" else 0.15
            sphere_pos = palm_pos + world_vec * axis_length
            
            # Marker expects shape (num_instances, 3)
            positions = sphere_pos.unsqueeze(0)  # (1, 3)
            marker.visualize(positions)
        
        # Print analysis every 100 steps
        if step_count % 100 == 0:
            target_down = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float)
            
            analyses = []
            for axis_name, axis_vec in [("X", torch.tensor([1.0, 0.0, 0.0])), 
                                         ("Y", torch.tensor([0.0, 1.0, 0.0])), 
                                         ("Z", torch.tensor([0.0, 0.0, 1.0]))]:
                axis_vec = axis_vec.to(env.device, dtype=torch.float)
                world_vec = torch.matmul(palm_rot, axis_vec)
                cos_angle = torch.dot(world_vec, target_down).item()
                angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                analyses.append((axis_name, cos_angle, angle_deg))
            
            # Also check current config
            config_local = torch.tensor(palm_down_vec_normalized, device=env.device, dtype=torch.float)
            config_world = torch.matmul(palm_rot, config_local)
            config_cos = torch.dot(config_world, target_down).item()
            config_angle = np.arccos(np.clip(config_cos, -1, 1)) * 180 / np.pi
            
            # Only print if changed significantly
            current_analysis = (config_cos, config_angle)
            if last_analysis is None or abs(current_analysis[0] - last_analysis[0]) > 0.01:
                print(f"\rStep {step_count}: ", end="")
                
                # Find best cardinal axis
                best = max(analyses, key=lambda x: x[1])
                print(f"Best cardinal: +{best[0]} ({best[2]:.1f}° off) | ", end="")
                print(f"Your config: {config_angle:.1f}° off (cos={config_cos:+.3f})", end="")
                
                if config_angle < 10:
                    print(" ✅")
                elif config_angle < 30:
                    print(" ⚠️ ")
                else:
                    print(" ❌")
                
                last_analysis = current_analysis
        
        step_count += 1
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
