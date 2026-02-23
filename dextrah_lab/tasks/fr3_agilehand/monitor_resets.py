#!/usr/bin/env python3
"""
Monitor what causes episode resets during training.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Monitor reset triggers")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--steps", type=int, default=200, help="Number of steps to run")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import torch
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

def main():
    args = parser.parse_args()
    
    # Create environment
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"  # Use single test object
    
    env = DextrahFR3AgilehandEnv(cfg, render_mode=None)
    
    print("=" * 80)
    print("MONITORING EPISODE RESETS")
    print("=" * 80)
    print(f"\nRunning {args.steps} steps across {args.num_envs} environments...")
    print("Will report what causes resets.\n")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Track reset statistics
    reset_counts = {
        "object_out_of_bounds": 0,
        "hand_too_far": 0,
        "hand_too_close_to_table": 0,
        "arm_table_contact": 0,
        "palm_flipped": 0,
        "episode_timeout": 0,
        "total_resets": 0,
    }
    
    arm_movements = []
    
    # Run simulation
    for step in range(args.steps):
        # Random actions
        actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 0.5
        
        # Track arm position before step
        arm_pos_before = env.robot.data.joint_pos[:, env.actuated_dof_indices[:7]].clone()
        
        # Step
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # Track arm position after step
        arm_pos_after = env.robot.data.joint_pos[:, env.actuated_dof_indices[:7]]
        arm_delta = (arm_pos_after - arm_pos_before).abs().max(dim=1).values
        arm_movements.append(arm_delta.mean().item())
        
        # Check what caused resets
        if dones.any():
            reset_env_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            
            for env_id in reset_env_ids:
                env_id_item = env_id.item()
                reset_counts["total_resets"] += 1
                
                # Recompute termination conditions for this env
                env._compute_intermediate_values()
                
                # Object bounds
                obj_x = env.object_pos[env_id, 0].item()
                obj_y = env.object_pos[env_id, 1].item()
                obj_z = env.object_pos[env_id, 2].item()
                
                obj_x_bounds = [env.cfg.x_center - env.cfg.x_width/2, env.cfg.x_center + env.cfg.x_width/2]
                obj_y_bounds = [env.cfg.y_center - env.cfg.y_width/2, env.cfg.y_center + env.cfg.y_width/2]
                
                obj_out_x = obj_x < obj_x_bounds[0] or obj_x > obj_x_bounds[1]
                obj_out_y = obj_y < obj_y_bounds[0] or obj_y > obj_y_bounds[1]
                obj_out_z = obj_z < 0.2
                
                # Just use the masks that the environment already computed
                hand_too_far = False  # Simplified - check below
                hand_too_close = False  # Simplified - check below
                
                # Check if this specific env triggered these conditions
                # by looking at the computed masks
                if hasattr(env, 'arm_table_contact_mask'):
                    arm_contact = env.arm_table_contact_mask[env_id].item()
                else:
                    arm_contact = False
                
                # Palm flipped
                palm_flip_cos = torch.sum(env.palm_direction_vec[env_id] * env._palm_dir_target_world[0]).item()
                palm_flipped = palm_flip_cos < env.cfg.palm_flip_cos_thresh
                
                # Determine cause
                causes = []
                if obj_out_x or obj_out_y or obj_out_z:
                    reset_counts["object_out_of_bounds"] += 1
                    causes.append(f"object_out_of_bounds (x={obj_x:.2f}, y={obj_y:.2f}, z={obj_z:.2f})")
                if arm_contact:
                    reset_counts["arm_table_contact"] += 1
                    causes.append("arm_table_contact")
                if palm_flipped:
                    reset_counts["palm_flipped"] += 1
                    causes.append(f"palm_flipped (cos={palm_flip_cos:.3f})")
                
                # Note: hand_too_far and hand_too_close checks simplified out for now
                
                # Check if it's just episode timeout
                if env.episode_length_buf[env_id] >= env.max_episode_length - 1:
                    reset_counts["episode_timeout"] += 1
                    causes.append("episode_timeout")
                
                if step < 50 or len(reset_env_ids) <= 2:  # Print first 50 steps or if few resets
                    cause_str = ", ".join(causes) if causes else "unknown"
                    print(f"  Step {step:3d}, Env {env_id_item}: RESET - {cause_str}")
        
        # Print progress
        if step % 50 == 0 and step > 0:
            avg_arm_movement = sum(arm_movements[-50:]) / 50
            print(f"\n--- Step {step} ---")
            print(f"  Avg arm movement (last 50 steps): {avg_arm_movement:.6f} rad")
            print(f"  Total resets so far: {reset_counts['total_resets']}")
    
    # Final statistics
    print("\n" + "=" * 80)
    print("RESET STATISTICS")
    print("=" * 80)
    
    total = reset_counts["total_resets"]
    if total > 0:
        print(f"\nTotal resets: {total}")
        print(f"\nBreakdown:")
        print(f"  Object out of bounds:      {reset_counts['object_out_of_bounds']:4d} ({100*reset_counts['object_out_of_bounds']/total:.1f}%)")
        print(f"  Hand too far:              {reset_counts['hand_too_far']:4d} ({100*reset_counts['hand_too_far']/total:.1f}%)")
        print(f"  Hand too close to table:   {reset_counts['hand_too_close_to_table']:4d} ({100*reset_counts['hand_too_close_to_table']/total:.1f}%)")
        print(f"  Arm-table contact:         {reset_counts['arm_table_contact']:4d} ({100*reset_counts['arm_table_contact']/total:.1f}%)")
        print(f"  Palm flipped:              {reset_counts['palm_flipped']:4d} ({100*reset_counts['palm_flipped']/total:.1f}%)")
        print(f"  Episode timeout:           {reset_counts['episode_timeout']:4d} ({100*reset_counts['episode_timeout']/total:.1f}%)")
    else:
        print("\nNo resets occurred!")
    
    avg_movement = sum(arm_movements) / len(arm_movements)
    print(f"\nAverage arm movement per step: {avg_movement:.6f} rad ({avg_movement * 57.3:.3f} deg)")
    
    if avg_movement < 0.001:
        print("\n⚠️  ARM BARELY MOVING! Average movement < 0.001 rad")
    
    if total > args.num_envs * 2:  # More than 2 resets per env on average
        print(f"\n⚠️  HIGH RESET RATE! {total/(args.steps * args.num_envs):.1%} of env-steps end in reset")
        print("    This suggests the robot is frequently entering invalid states.")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
