#!/usr/bin/env python3
"""
Visual debugging tool to find the correct palm_down_local_axis.

This script will:
1. Spawn the robot with default pose
2. Draw visual markers showing all 6 cardinal directions from the palm
3. Highlight which direction best aligns with downward (-Z)
4. Let you interactively test different palm orientations

Usage:
    python -m dextrah_lab.tasks.fr3_agilehand.visual_palm_debug
"""

import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import (
    DextrahFR3AgilehandEnvCfg,
)


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to rotation matrix."""
    w, x, y, z = quat.unbind(-1)
    
    # Build rotation matrix
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=-1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=-1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=-1)
    ], dim=-2)
    
    return R


def run_visual_debug():
    """Run the visual palm orientation debugger."""
    
    # Create environment config
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    
    # Setup scene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Add robot to scene
    scene.articulations["robot"] = cfg.robot_cfg
    
    # Build scene
    sim = SimulationContext(cfg.sim)
    scene.build_scene()
    
    # Get robot and palm body
    robot = scene.articulations["robot"]
    palm_body_idx = robot.body_names.index(cfg.palm_body_name)
    
    print(f"\n{'='*80}")
    print(f"VISUAL PALM ORIENTATION DEBUG")
    print(f"{'='*80}")
    print(f"\nPalm body: {cfg.palm_body_name} (index {palm_body_idx})")
    print(f"Current config: palm_down_local_axis = {cfg.palm_down_local_axis}")
    
    # Create visualization markers for each cardinal direction
    # We'll draw 6 arrows from the palm: +X, -X, +Y, -Y, +Z, -Z
    cardinal_axes = {
        "+X": torch.tensor([1.0, 0.0, 0.0]),
        "-X": torch.tensor([-1.0, 0.0, 0.0]),
        "+Y": torch.tensor([0.0, 1.0, 0.0]),
        "-Y": torch.tensor([0.0, -1.0, 0.0]),
        "+Z": torch.tensor([0.0, 0.0, 1.0]),
        "-Z": torch.tensor([0.0, 0.0, -1.0]),
    }
    
    # Colors for each direction
    colors = {
        "+X": (1.0, 0.0, 0.0),  # Red
        "-X": (0.5, 0.0, 0.0),  # Dark red
        "+Y": (0.0, 1.0, 0.0),  # Green
        "-Y": (0.0, 0.5, 0.0),  # Dark green
        "+Z": (0.0, 0.0, 1.0),  # Blue
        "-Z": (0.0, 0.0, 0.5),  # Dark blue
    }
    
    # Create markers
    markers = {}
    for name in cardinal_axes.keys():
        marker_cfg = RAY_CASTER_MARKER_CFG.copy()
        marker_cfg.prim_path = f"/Visuals/palm_axis_{name}"
        marker_cfg.markers[f"arrow_{name}"].scale = (0.02, 0.02, 0.15)  # Arrow size
        markers[name] = VisualizationMarkers(marker_cfg)
    
    # Reset and simulate
    sim.reset()
    robot.reset()
    
    print(f"\n{'='*80}")
    print(f"INSTRUCTIONS")
    print(f"{'='*80}")
    print("Visual markers show all 6 cardinal directions from the palm:")
    print("  Red arrows: ±X axis")
    print("  Green arrows: ±Y axis")
    print("  Blue arrows: ±Z axis")
    print("\nThe arrow that points DOWNWARD (toward the table) is your palm_down_local_axis.")
    print(f"{'='*80}\n")
    
    step_count = 0
    target_down = torch.tensor([0.0, 0.0, -1.0], device=robot.device)
    best_alignment = None
    best_axis = None
    
    # Run for a while
    while simulation_app.is_running() and step_count < 10000:
        # Step simulation
        sim.step()
        
        # Update every 10 steps
        if step_count % 10 == 0:
            # Get palm pose
            palm_pos_w = robot.data.body_pos_w[:, palm_body_idx]  # (1, 3)
            palm_quat_w = robot.data.body_quat_w[:, palm_body_idx]  # (1, 4) in (w,x,y,z)
            
            # Convert quaternion to rotation matrix
            palm_rot = quaternion_to_matrix(palm_quat_w)  # (1, 3, 3)
            
            # Transform each cardinal axis to world frame and visualize
            alignments = {}
            for axis_name, local_vec in cardinal_axes.items():
                local_vec = local_vec.to(robot.device).unsqueeze(0).unsqueeze(-1)  # (1, 3, 1)
                world_vec = torch.matmul(palm_rot, local_vec).squeeze(-1)  # (1, 3)
                
                # Compute alignment with downward
                cos_angle = torch.sum(world_vec * target_down, dim=-1).item()
                alignments[axis_name] = cos_angle
                
                # Draw arrow from palm to direction
                arrow_length = 0.15
                end_pos = palm_pos_w + world_vec * arrow_length
                
                # Update marker
                positions = torch.cat([palm_pos_w, end_pos], dim=0).unsqueeze(0)  # (1, 2, 3)
                markers[axis_name].visualize(positions)
            
            # Find best alignment every 100 steps
            if step_count % 100 == 0:
                best_axis = max(alignments.items(), key=lambda x: x[1])
                print(f"\rStep {step_count}: Best alignment = {best_axis[0]} (cos={best_axis[1]:.4f}, angle={np.arccos(np.clip(best_axis[1], -1, 1))*180/np.pi:.1f}°)", end="")
        
        step_count += 1
    
    # Final recommendation
    if best_axis:
        axis_name, cos_val = best_axis
        angle_deg = np.arccos(np.clip(cos_val, -1, 1)) * 180 / np.pi
        
        print(f"\n\n{'='*80}")
        print(f"RECOMMENDATION")
        print(f"{'='*80}")
        print(f"\nBest palm_down_local_axis: {axis_name}")
        print(f"  Vector: {cardinal_axes[axis_name].cpu().numpy()}")
        print(f"  Alignment: {angle_deg:.1f}° off from downward")
        print(f"  Cosine: {cos_val:.4f}")
        
        local_vec_list = cardinal_axes[axis_name].cpu().numpy().tolist()
        print(f"\nUpdate your config:")
        print(f"  palm_down_local_axis = ({local_vec_list[0]:+.1f}, {local_vec_list[1]:+.1f}, {local_vec_list[2]:+.1f})")
        
        if angle_deg > 10:
            print(f"\n⚠️  Warning: {angle_deg:.1f}° is not ideal alignment.")
            print(f"   Consider checking the hand attachment in the USD file.")
            print(f"   You may need to rotate the robot base or hand in init_state.")
        print(f"{'='*80}")
    
    simulation_app.close()


if __name__ == "__main__":
    run_visual_debug()
