#!/usr/bin/env python3
"""
Visualize local coordinate frames for the palm and other bodies.

This script shows the local X, Y, Z axes for specified bodies to help
debug orientation issues.

Usage:
    python -m dextrah_lab.tasks.fr3_agilehand.show_local_frames
    python -m dextrah_lab.tasks.fr3_agilehand.show_local_frames --livestream 1
"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg


def main():
    # Create environment
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"
    
    env = DextrahFR3AgilehandEnv(cfg)
    env.reset()
    
    # Get body indices
    palm_body_idx = env.robot.body_names.index(cfg.palm_body_name)
    
    # Also show the end effector (fr3_link7) for comparison
    try:
        ee_body_idx = env.robot.body_names.index("fr3_link7")
        show_ee = True
    except ValueError:
        show_ee = False
        print("Note: fr3_link7 not found in robot bodies")
    
    print("\n" + "="*80)
    print("LOCAL COORDINATE FRAME VISUALIZER")
    print("="*80)
    print(f"\nShowing local coordinate frames for:")
    print(f"  - {cfg.palm_body_name} (palm)")
    if show_ee:
        print(f"  - fr3_link7 (FR3 end effector / flange)")
    
    print("\nCoordinate frame convention:")
    print("  🔴 RED axis   = Local +X direction")
    print("  🟢 GREEN axis = Local +Y direction")
    print("  🔵 BLUE axis  = Local +Z direction")
    
    print("\nThe palm should point DOWNWARD toward the table.")
    print("Observe which colored axis points down - that's your palm_down_local_axis!")
    print("="*80 + "\n")
    
    # Create frame markers
    palm_frame_cfg = FRAME_MARKER_CFG.copy()
    palm_frame_cfg.prim_path = "/Visuals/palm_frame"
    palm_frame_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)  # Make it visible
    palm_frame = VisualizationMarkers(palm_frame_cfg)
    
    if show_ee:
        ee_frame_cfg = FRAME_MARKER_CFG.copy()
        ee_frame_cfg.prim_path = "/Visuals/ee_frame"
        ee_frame_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
        ee_frame = VisualizationMarkers(ee_frame_cfg)
    
    # Simulation loop
    step_count = 0
    
    print("Simulation running... (Ctrl+C to stop)")
    print("Look at the coordinate frames in the visualizer.\n")
    
    while simulation_app.is_running():
        # Step with zero action to keep robot in place
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        
        # Update palm frame
        palm_pos = env.robot.data.body_pos_w[0, palm_body_idx].unsqueeze(0)  # (1, 3)
        palm_quat = env.robot.data.body_quat_w[0, palm_body_idx].unsqueeze(0)  # (1, 4) (w,x,y,z)
        
        # VisualizationMarkers expects orientations in (x,y,z,w) format
        palm_quat_xyzw = torch.cat([palm_quat[:, 1:], palm_quat[:, :1]], dim=-1)  # (1, 4)
        
        palm_frame.visualize(palm_pos, palm_quat_xyzw)
        
        # Update EE frame if available
        if show_ee:
            ee_pos = env.robot.data.body_pos_w[0, ee_body_idx].unsqueeze(0)
            ee_quat = env.robot.data.body_quat_w[0, ee_body_idx].unsqueeze(0)
            ee_quat_xyzw = torch.cat([ee_quat[:, 1:], ee_quat[:, :1]], dim=-1)
            ee_frame.visualize(ee_pos, ee_quat_xyzw)
        
        # Print status every 100 steps
        if step_count % 100 == 0:
            # Extract rotation info
            palm_quat_np = palm_quat[0].cpu().numpy()
            w, x, y, z = palm_quat_np
            
            print(f"\rStep {step_count}: Palm at {palm_pos[0].cpu().numpy()} | quat (w,x,y,z)=[{w:.3f}, {x:.3f}, {y:.3f}, {z:.3f}]", end="")
        
        step_count += 1
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
