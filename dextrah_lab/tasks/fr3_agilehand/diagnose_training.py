#!/usr/bin/env python3
"""
Diagnose why robot doesn't move during training.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose training issues")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import torch
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

def main():
    # Create environment
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = parser.parse_args().num_envs
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"  # Use single test object
    
    env = DextrahFR3AgilehandEnv(cfg, render_mode=None)
    
    print("=" * 80)
    print("TRAINING DIAGNOSTICS")
    print("=" * 80)
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"\nEnvironments: {env.num_envs}")
    print(f"Action space: {env.num_actions}")
    print(f"Actuated joints: {len(env.actuated_dof_indices)}")
    
    # Get initial state
    initial_arm_pos = env.robot.data.joint_pos[:, env.actuated_dof_indices[:7]].clone()
    
    print(f"\nInitial arm positions (env 0, first 7 joints):")
    for i in range(7):
        print(f"  [{i}] {initial_arm_pos[0, i].item():.4f} rad")
    
    # Create random actions (like policy would output)
    print("\n" + "=" * 80)
    print("TESTING WITH RANDOM ACTIONS")
    print("=" * 80)
    
    actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 0.5  # Actions in [0, 0.5]
    
    print(f"\nRandom actions (env 0, first 7 arm actions):")
    for i in range(7):
        print(f"  [{i}] {actions[0, i].item():.4f}")
    
    # Check joint limits
    print(f"\nJoint limits (first 7 arm joints):")
    for i in range(7):
        lower = env.robot_dof_lower_limits[0, i].item()
        upper = env.robot_dof_upper_limits[0, i].item()
        print(f"  [{i}] [{lower:.4f}, {upper:.4f}]")
    
    # Step the environment 50 times with these actions
    print(f"\nStepping environment 50 times with random actions...")
    
    for step in range(50):
        obs, rewards, dones, truncated, info = env.step(actions)
        
        if step % 10 == 0:
            current_arm_pos = env.robot.data.joint_pos[:, env.actuated_dof_indices[:7]]
            delta = (current_arm_pos - initial_arm_pos).abs().max().item()
            print(f"  Step {step:2d}: Max arm movement = {delta:.6f} rad, Reward = {rewards[0].item():.4f}")
    
    # Final check
    final_arm_pos = env.robot.data.joint_pos[:, env.actuated_dof_indices[:7]]
    total_delta = final_arm_pos - initial_arm_pos
    
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS (env 0)")
    print("=" * 80)
    
    print(f"\nArm joint changes:")
    for i in range(7):
        delta = total_delta[0, i].item()
        print(f"  [{i}] {delta:+.6f} rad ({delta * 57.3:+.3f} deg)")
    
    max_delta = total_delta.abs().max().item()
    print(f"\nMax movement: {max_delta:.6f} rad ({max_delta * 57.3:.3f} deg)")
    
    if max_delta > 0.01:
        print("\n✅ ROBOT MOVED during training simulation!")
    else:
        print("\n❌ ROBOT DID NOT MOVE during training simulation!")
        print("\nPossible causes:")
        print("  1. Actions are being scaled incorrectly")
        print("  2. PD gains too low for the robot's inertia")
        print("  3. Effort limits too low")
        print("  4. Action rate penalty too high (robot learns to not move)")
    
    # Check action targets
    print(f"\n" + "=" * 80)
    print("ACTION PROCESSING CHECK")
    print("=" * 80)
    
    # Get the joint position targets that were set
    current_targets = env.dof_pos_targets[:, env.actuated_dof_indices[:7]]
    print(f"\nCurrent joint position targets (env 0, first 7):")
    for i in range(7):
        target = current_targets[0, i].item()
        actual = final_arm_pos[0, i].item()
        error = target - actual
        print(f"  [{i}] Target: {target:.4f}, Actual: {actual:.4f}, Error: {error:.4f}")
    
    max_error = (current_targets - final_arm_pos).abs().max().item()
    print(f"\nMax tracking error: {max_error:.6f} rad")
    
    if max_error > 0.1:
        print("\n⚠️  Large tracking error! Robot cannot follow targets.")
        print("    This suggests PD gains or effort limits are too low.")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
