"""Test if robot actually responds to actions.

This will:
1. Spawn the env
2. Send constant positive actions to arm
3. Check if joint positions actually change
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
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
    cfg.objects_dir = "test_object"
    
    env = DextrahFR3AgilehandEnv(cfg=cfg, render_mode=None)
    
    print("=" * 80)
    print("JOINT LIMITS")
    print("=" * 80)
    for i, name in enumerate(cfg.actuated_joint_names):
        lower = env.robot_dof_lower_limits[0, i].item()
        upper = env.robot_dof_upper_limits[0, i].item()
        range_deg = (upper - lower) * 57.3
        print(f"[{i:2d}] {name:30s} [{lower:+7.3f}, {upper:+7.3f}] = {range_deg:6.1f} deg range")
    
    print("\n" + "=" * 80)
    print("INITIAL JOINT POSITIONS")
    print("=" * 80)
    
    obs, _ = env.reset()
    initial_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices].clone()
    
    for i, name in enumerate(cfg.actuated_joint_names):
        pos = initial_pos[i].item()
        print(f"[{i:2d}] {name:30s} = {pos:+7.3f} rad ({pos*57.3:+7.1f} deg)")
    
    print("\n" + "=" * 80)
    print("APPLYING CONSTANT +0.5 ACTIONS TO ARM FOR 50 STEPS")
    print("=" * 80)
    
    # Apply constant positive action to arm joints
    for step in range(50):
        action = torch.zeros((args_cli.num_envs, cfg.num_actions), device=env.device)
        action[:, :7] = 0.5  # Positive action on arm joints (should move them toward upper limit)
        
        obs, rewards, dones, truncated, info = env.step(action)
        
        if step % 10 == 0:
            current_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices]
            delta = (current_pos - initial_pos)[:7]  # Only arm joints
            
            print(f"\nStep {step}:")
            print(f"  Arm joint deltas (rad): {delta.cpu().numpy()}")
            print(f"  Arm joint deltas (deg): {(delta * 57.3).cpu().numpy()}")
            print(f"  Max delta: {delta.abs().max().item():.4f} rad ({delta.abs().max().item()*57.3:.2f} deg)")
            print(f"  Reward: {rewards[0].item():.4f}")
            print(f"  hand_to_object_error: {env.hand_to_object_pos_error[0].item():.4f}")
    
    print("\n" + "=" * 80)
    print("FINAL JOINT POSITIONS")
    print("=" * 80)
    
    final_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices]
    
    for i, name in enumerate(cfg.actuated_joint_names):
        init = initial_pos[i].item()
        final = final_pos[i].item()
        delta = final - init
        print(f"[{i:2d}] {name:30s} init={init:+7.3f} final={final:+7.3f} delta={delta:+7.3f} ({delta*57.3:+7.1f} deg)")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    total_movement = (final_pos - initial_pos).abs().sum().item()
    arm_movement = (final_pos[:7] - initial_pos[:7]).abs().sum().item()
    hand_movement = (final_pos[7:] - initial_pos[7:]).abs().sum().item()
    
    print(f"\nTotal joint movement: {total_movement:.4f} rad ({total_movement*57.3:.1f} deg)")
    print(f"Arm movement:  {arm_movement:.4f} rad ({arm_movement*57.3:.1f} deg)")
    print(f"Hand movement: {hand_movement:.4f} rad ({hand_movement*57.3:.1f} deg)")
    
    if arm_movement < 0.01:
        print("\n❌ ARM DID NOT MOVE!")
        print("Possible causes:")
        print("  - PD gains too low (arm can't follow targets)")
        print("  - Effort limits too low (arm saturating)")
        print("  - Action scaling issue")
        print("  - Targets not being set correctly")
    else:
        print(f"\n✅ Arm moved {arm_movement*57.3:.1f} degrees total")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
