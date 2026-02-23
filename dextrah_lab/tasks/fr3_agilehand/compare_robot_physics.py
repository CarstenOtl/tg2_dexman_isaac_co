#!/usr/bin/env python3
"""
Compare physics behavior between FR3 and TG2 robots.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare robot physics")
parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--robot", type=str, choices=["fr3", "tg2", "both"], default="both", 
                    help="Which robot to test: fr3, tg2, or both")
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

def test_robot(env_class, env_cfg_class, robot_name):
    """Test a robot's physics response."""
    print("\n" + "=" * 80)
    print(f"TESTING {robot_name}")
    print("=" * 80)
    
    # Create environment config
    print(f"Creating environment config...")
    cfg = env_cfg_class()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"
    
    # Create environment
    print(f"Loading environment (this may take 5-10 seconds)...")
    env = env_class(cfg, render_mode=None)
    print(f"Environment loaded!")
    
    print(f"\nRobot info:")
    print(f"  Bodies: {env.robot.num_bodies}")
    print(f"  Joints: {env.robot.num_joints}")
    print(f"  Actuators: {len(env.actuated_dof_indices)}")
    
    # Get initial arm joint positions (first 7 joints)
    num_arm_joints = min(7, len(env.actuated_dof_indices))
    initial_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices[:num_arm_joints]].clone()
    
    print(f"\nInitial arm positions ({num_arm_joints} joints):")
    for i in range(num_arm_joints):
        print(f"  [{i}] {initial_pos[i].item():.4f} rad ({initial_pos[i].item() * 57.3:.1f} deg)")
    
    # Create a target that's different from current position
    target_pos = initial_pos + 0.1  # Move 0.1 rad
    
    # Clamp to joint limits
    target_pos = torch.clamp(
        target_pos,
        env.robot_dof_lower_limits[0, :num_arm_joints],
        env.robot_dof_upper_limits[0, :num_arm_joints]
    )
    
    print(f"\nTarget arm positions:")
    for i in range(num_arm_joints):
        print(f"  [{i}] {target_pos[i].item():.4f} rad ({target_pos[i].item() * 57.3:.1f} deg)")
    
    # Set the target
    env.robot.set_joint_position_target(target_pos, joint_ids=env.actuated_dof_indices[:num_arm_joints])
    
    # Step simulation for 50 steps
    print("\nStepping simulation for 50 steps...")
    for i in range(50):
        env.sim.step()
        env.robot.update(dt=env.cfg.sim_dt)
    
    # Check final position
    final_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices[:num_arm_joints]]
    delta = final_pos - initial_pos
    
    print(f"\nFinal arm positions:")
    for i in range(num_arm_joints):
        print(f"  [{i}] {final_pos[i].item():.4f} rad (delta: {delta[i].item():.6f} rad)")
    
    max_delta = delta.abs().max().item()
    print(f"\nMax delta: {max_delta:.6f} rad ({max_delta * 57.3:.3f} deg)")
    
    if max_delta > 0.001:
        print(f"\n✅ {robot_name} MOVED! Physics is working.")
        result = "WORKING"
    else:
        print(f"\n❌ {robot_name} DID NOT MOVE! There's a physics issue.")
        result = "BROKEN"
    
    env.close()
    
    return {
        "name": robot_name,
        "max_delta_rad": max_delta,
        "max_delta_deg": max_delta * 57.3,
        "result": result,
    }

def main():
    results = []
    
    import sys
    robot_choice = args.robot
    
    print("\n" + "=" * 80)
    print("ROBOT PHYSICS COMPARISON TEST")
    print("=" * 80)
    
    robots_to_test = []
    if robot_choice in ["fr3", "both"]:
        robots_to_test.append("fr3")
    if robot_choice in ["tg2", "both"]:
        robots_to_test.append("tg2")
    
    print(f"\nTesting {len(robots_to_test)} robot(s): {', '.join(robots_to_test)}")
    print("Each robot takes ~5-10 seconds to load.")
    print("=" * 80)
    
    # Test FR3
    if "fr3" in robots_to_test:
        idx = robots_to_test.index("fr3") + 1
        total = len(robots_to_test)
        print(f"\n[{idx}/{total}] Preparing FR3 AgileHand test...")
        from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
        from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg
        
        fr3_result = test_robot(DextrahFR3AgilehandEnv, DextrahFR3AgilehandEnvCfg, "FR3 AgileHand")
        results.append(fr3_result)
    
    # Test TG2
    if "tg2" in robots_to_test:
        idx = robots_to_test.index("tg2") + 1
        total = len(robots_to_test)
        print(f"\n[{idx}/{total}] Preparing TG2 InspireHand test...")
        from dextrah_lab.tasks.tg2_inspirehand.dextrah_tg2_inspirehand_env import DextrahTG2InspirehandEnv
        from dextrah_lab.tasks.tg2_inspirehand.dextrah_tg2_inspirehand_env_cfg import DextrahTG2InspirehandEnvCfg
        
        tg2_result = test_robot(DextrahTG2InspirehandEnv, DextrahTG2InspirehandEnvCfg, "TG2 InspireHand")
        results.append(tg2_result)
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for r in results:
        status = "✅" if r["result"] == "WORKING" else "❌"
        print(f"\n{status} {r['name']:20s}")
        print(f"   Max movement: {r['max_delta_rad']:.6f} rad ({r['max_delta_deg']:.3f} deg)")
        print(f"   Status: {r['result']}")
    
    # Overall assessment
    all_working = all(r["result"] == "WORKING" for r in results)
    
    print("\n" + "=" * 80)
    if all_working:
        print("✅ ALL ROBOTS ARE WORKING!")
    else:
        print("⚠️  SOME ROBOTS HAVE ISSUES")
        broken = [r["name"] for r in results if r["result"] == "BROKEN"]
        print(f"   Broken: {', '.join(broken)}")
    print("=" * 80)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
