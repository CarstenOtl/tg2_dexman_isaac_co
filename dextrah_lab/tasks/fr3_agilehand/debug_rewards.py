"""Debug script to check if hand-object distances and rewards are computing correctly.

This will spawn the env and print out:
- Hand body indices
- Hand-object distance bodies
- Actual distance values
- Reward components
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

def main():
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"  # Use simple test object
    
    env = DextrahFR3AgilehandEnv(cfg=cfg, render_mode=None)
    
    print("=" * 80)
    print("BODY NAME MAPPING")
    print("=" * 80)
    print(f"\nTotal bodies: {len(env.robot.body_names)}")
    
    print(f"\nConfigured hand_body_names:")
    for name in cfg.hand_body_names:
        try:
            idx = env.robot.body_names.index(name)
            print(f"  '{name}' -> body index {idx}")
        except ValueError:
            print(f"  '{name}' -> NOT FOUND! ❌")
    
    print(f"\nConfigured hand_object_distance_body_names:")
    for name in cfg.hand_object_distance_body_names:
        try:
            idx = env.robot.body_names.index(name)
            print(f"  '{name}' -> body index {idx}")
        except ValueError:
            print(f"  '{name}' -> NOT FOUND! ❌")
    
    print(f"\nPalm body: '{cfg.palm_body_name}' -> index {env.palm_body_idx}")
    print(f"Workspace body: '{cfg.hand_workspace_body_name}' -> index {env.hand_workspace_body_idx}")
    
    print("\n" + "=" * 80)
    print("INITIAL STATE")
    print("=" * 80)
    
    obs, _ = env.reset()
    
    print(f"\nHand bodies indices: {env.hand_bodies}")
    print(f"Hand-object distance bodies indices: {env.hand_object_distance_bodies}")
    
    print(f"\nObject position (env 0): {env.object_pos[0]}")
    print(f"Palm position (env 0): {env.palm_pos[0]}")
    
    print(f"\nHand-object distance bodies positions (env 0):")
    for i, body_idx in enumerate(env.hand_object_distance_bodies):
        body_name = env.robot.body_names[body_idx]
        pos = env.hand_object_distance_pos[0, i]
        dist = torch.norm(pos - env.object_pos[0])
        print(f"  [{body_idx}] {body_name:30s} pos={pos.cpu().numpy()} dist={dist:.4f}")
    
    print(f"\nHand-to-object error (env 0): {env.hand_to_object_pos_error[0]:.4f}")
    
    print("\n" + "=" * 80)
    print("MANUAL DISTANCE CALCULATION")
    print("=" * 80)
    
    # Manually calculate hand-to-object distance
    manual_distances = []
    for i, body_idx in enumerate(env.hand_object_distance_bodies):
        body_name = env.robot.body_names[body_idx]
        pos = env.hand_object_distance_pos[0, i]
        dist = torch.norm(pos - env.object_pos[0])
        manual_distances.append(dist.item())
        print(f"  {body_name}: {dist:.4f}")
    
    manual_max_dist = max(manual_distances)
    print(f"\nManual max distance: {manual_max_dist:.4f}")
    
    # Compute intermediate values (distances, etc)
    env._compute_intermediate_values()
    
    print(f"\n[AFTER _compute_intermediate_values()]")
    print(f"env.hand_to_object_pos_error (env 0): {env.hand_to_object_pos_error[0]:.4f}")
    print(f"  ❌ Should be ~{manual_max_dist:.4f} but got {env.hand_to_object_pos_error[0]:.4f}")
    
    # Check the calculation step by step
    print(f"\n[DEBUG CALCULATION]")
    print(f"hand_object_distance_pos shape: {env.hand_object_distance_pos.shape}")
    print(f"object_pos shape: {env.object_pos.shape}")
    
    # Replicate the calculation from the env
    diff = env.hand_object_distance_pos - env.object_pos[:, None, :]
    print(f"diff shape: {diff.shape}")
    norms = torch.norm(diff, dim=-1)
    print(f"norms shape: {norms.shape}")
    print(f"norms (env 0): {norms[0]}")
    max_vals = norms.max(dim=-1).values
    print(f"max_vals shape: {max_vals.shape}")
    print(f"max_vals (env 0): {max_vals[0]:.4f}")
    
    # Compute rewards
    try:
        rewards = env._get_rewards()
        print(f"\n[REWARD COMPONENTS]")
        print(f"hand_to_object_reward (env 0): {env.hand_to_object_reward[0]:.6f}")
        print(f"Total reward (env 0): {rewards[0]:.6f}")
    except Exception as e:
        print(f"\n[ERROR computing rewards]: {e}")
    
    print("\n" + "=" * 80)
    print("STEP FORWARD WITH ACTIONS TOWARD OBJECT")
    print("=" * 80)
    
    # Take a few steps with actions that should move arm toward object
    # Positive actions on arm joints should move arm
    for step in range(5):
        # Create action that moves arm (first 7 DOF) positively
        action = torch.zeros((args_cli.num_envs, cfg.num_actions), device=env.device)
        action[:, :7] = 0.1  # Move arm joints
        
        obs, rewards, dones, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Hand-object distance: {env.hand_to_object_pos_error[0]:.4f}")
        print(f"  hand_to_object_reward: {env.hand_to_object_reward[0]:.6f}")
        print(f"  action_rate_penalty: {env.action_rate_penalty[0]:.6f}")
        print(f"  Total reward: {rewards[0]:.6f}")
        print(f"  Arm action norm: {torch.norm(action[0, :7]):.4f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
