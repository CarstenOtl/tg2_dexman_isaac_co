#!/usr/bin/env python3
"""
Check robot physics properties to diagnose why it's not moving.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check robot physics properties")
parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import torch
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg
from pxr import Usd, UsdPhysics, PhysxSchema
import omni.usd

def main():
    # Create environment config
    cfg = DextrahFR3AgilehandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.objects_dir = "test_object"
    
    # Create environment
    print("=" * 80)
    print("SPAWNING ENVIRONMENT")
    print("=" * 80)
    env = DextrahFR3AgilehandEnv(cfg, render_mode=None)
    
    # Get USD stage
    stage = omni.usd.get_context().get_stage()
    
    print("\n" + "=" * 80)
    print("CHECKING ROBOT PHYSICS PROPERTIES")
    print("=" * 80)
    
    # Check robot articulation root
    robot_path = "/World/envs/env_0/Robot"
    robot_prim = stage.GetPrimAtPath(robot_path)
    
    if not robot_prim.IsValid():
        print(f"ERROR: Robot prim not found at {robot_path}")
        return
    
    print(f"\nRobot prim: {robot_path}")
    print(f"  Type: {robot_prim.GetTypeName()}")
    
    # Check articulation API
    articulation_api = UsdPhysics.ArticulationRootAPI(robot_prim)
    if articulation_api:
        print(f"  ArticulationRootAPI: Yes")
    
    # Check rigid body API    
    rigidbody_api = UsdPhysics.RigidBodyAPI(robot_prim)
    if rigidbody_api:
        print(f"  RigidBodyAPI: Yes")
        if rigidbody_api.GetKinematicEnabledAttr():
            is_kinematic = rigidbody_api.GetKinematicEnabledAttr().Get()
            print(f"  Kinematic: {is_kinematic}")
    
    # Check PhysX articulation API
    physx_articulation_api = PhysxSchema.PhysxArticulationAPI(robot_prim)
    if physx_articulation_api:
        print(f"  PhysxArticulationAPI: Yes")
        if physx_articulation_api.GetEnabledSelfCollisionsAttr():
            enabled_self_collisions = physx_articulation_api.GetEnabledSelfCollisionsAttr().Get()
            print(f"  EnabledSelfCollisions: {enabled_self_collisions}")
    
    # Check each link in the robot
    print("\n" + "-" * 80)
    print("CHECKING ROBOT LINKS")
    print("-" * 80)
    
    for child in robot_prim.GetChildren():
        child_path = child.GetPath()
        child_type = child.GetTypeName()
        
        # Only check Xform prims (links)
        if child_type != "Xform":
            continue
        
        rigidbody_api = UsdPhysics.RigidBodyAPI(child)
        is_kinematic = None
        if rigidbody_api and rigidbody_api.GetKinematicEnabledAttr():
            is_kinematic = rigidbody_api.GetKinematicEnabledAttr().Get()
        
        if is_kinematic is not None:
            status = "KINEMATIC" if is_kinematic else "dynamic"
            marker = "❌" if is_kinematic else "✅"
            print(f"{marker} {child.GetName():30s} {status}")
    
    # Check joint properties
    print("\n" + "-" * 80)
    print("CHECKING ROBOT JOINTS")
    print("-" * 80)
    
    for child in robot_prim.GetAllChildren():
        child_type = child.GetTypeName()
        
        if "Joint" in child_type:
            joint_name = child.GetName()
            joint_api = UsdPhysics.Joint(child)
            
            if joint_api:
                # Check if joint is enabled
                enabled = True
                if joint_api.GetJointEnabledAttr():
                    enabled = joint_api.GetJointEnabledAttr().Get()
                
                marker = "✅" if enabled else "❌"
                print(f"{marker} {joint_name:30s} Type: {child_type:30s} Enabled: {enabled}")
    
    # Check PhysX view info
    print("\n" + "-" * 80)
    print("CHECKING PHYSX VIEW")
    print("-" * 80)
    
    print(f"Number of bodies: {env.robot.num_bodies}")
    print(f"Number of joints: {env.robot.num_joints}")
    print(f"Number of actuators: {len(env.actuated_dof_indices)}")
    
    # Check if we can get joint states
    print("\n" + "-" * 80)
    print("CHECKING JOINT STATE ACCESS")
    print("-" * 80)
    
    joint_pos = env.robot.data.joint_pos
    joint_vel = env.robot.data.joint_vel
    
    print(f"Joint positions shape: {joint_pos.shape}")
    print(f"Joint velocities shape: {joint_vel.shape}")
    print(f"First 7 joints (arm) positions: {joint_pos[0, :7]}")
    
    # Try setting joint targets and see if anything happens
    print("\n" + "-" * 80)
    print("TESTING JOINT TARGET SETTING")
    print("-" * 80)
    
    # Get initial arm joint positions
    initial_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices[:7]].clone()
    print(f"Initial arm positions: {initial_pos}")
    
    # Create a target that's different from current position
    target_pos = initial_pos + 0.1  # Move 0.1 rad
    
    # Clamp to joint limits
    target_pos = torch.clamp(
        target_pos,
        env.robot_dof_lower_limits[0, :7],
        env.robot_dof_upper_limits[0, :7]
    )
    
    print(f"Target arm positions: {target_pos}")
    
    # Set the target
    full_target = env.robot.data.joint_pos[0].clone()
    full_target[env.actuated_dof_indices[:7]] = target_pos
    
    env.robot.set_joint_position_target(target_pos, joint_ids=env.actuated_dof_indices[:7])
    
    # Step simulation for 50 steps
    print("\nStepping simulation for 50 steps...")
    for i in range(50):
        env.sim.step()
        env.robot.update(dt=env.cfg.sim_dt)
    
    # Check final position
    final_pos = env.robot.data.joint_pos[0, env.actuated_dof_indices[:7]]
    delta = final_pos - initial_pos
    
    print(f"Final arm positions: {final_pos}")
    print(f"Delta: {delta}")
    print(f"Max delta: {delta.abs().max().item():.6f}")
    
    if delta.abs().max().item() > 0.001:
        print("\n✅ ROBOT MOVED! Physics is working.")
    else:
        print("\n❌ ROBOT DID NOT MOVE! There's a physics issue.")
        print("\nPossible causes:")
        print("  1. Base link is kinematic (check above)")
        print("  2. Joints are disabled")
        print("  3. Actuator effort limits too low")
        print("  4. PD gains too low")
        print("  5. Joint drive type incorrect")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
