"""Diagnostic script: verify FR3+Tekken articulation tree, joints, bodies, and actuators.

Boots Isaac Sim, spawns the robot, and prints everything needed to verify that
the nested tekken_left_adof prim structure is correctly wired up.

Usage:
    python -m dextrah_lab.tasks.fr3_agilehand.check_articulation --headless --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check FR3+Tekken articulation structure.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that need Kit running ────────────────────────────────────────────
import torch
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import build_simulation_context
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from dextrah_lab.assets.fr3_tekken_adof.fr3_tekken_left import FR3_TEK_LEFT_CONFIG
from dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg


def main():
    cfg = DextrahFR3AgilehandEnvCfg()

    @configclass
    class DiagSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot = FR3_TEK_LEFT_CONFIG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

    scene_cfg = DiagSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    sim_dt = 1.0 / 120.0

    with build_simulation_context(device="cuda:0", dt=sim_dt, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        sep = "=" * 80

        # ── 1. USD Prim Tree ────────────────────────────────────────────────
        print(f"\n{sep}")
        print("1. USD PRIM TREE (under /World/envs/env_0/Robot)")
        print(sep)
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if root.IsValid():
            def _print_tree(prim, depth=0, max_depth=6):
                indent = "  " * depth
                prim_type = prim.GetTypeName()
                marker = ""
                if "Joint" in prim_type:
                    marker = " <-- JOINT"
                elif "RigidBody" in str(prim.GetAppliedSchemas()):
                    marker = " <-- BODY"
                print(f"{indent}{prim.GetName()} ({prim_type}){marker}")
                if depth < max_depth:
                    for child in prim.GetChildren():
                        _print_tree(child, depth + 1, max_depth)
            _print_tree(root)
        else:
            print("  [ERROR] /World/envs/env_0/Robot not found!")

        # ── 2. PhysX Joint Names ────────────────────────────────────────────
        print(f"\n{sep}")
        print("2. JOINT NAMES (from PhysX articulation view)")
        print(sep)
        for i, name in enumerate(robot.joint_names):
            print(f"  [{i:2d}] {name}")
        print(f"  Total joints (DOFs): {robot.num_joints}")

        # ── 3. Body Names ──────────────────────────────────────────────────
        print(f"\n{sep}")
        print("3. BODY NAMES (from PhysX articulation view)")
        print(sep)
        for i, name in enumerate(robot.body_names):
            print(f"  [{i:2d}] {name}")
        print(f"  Total bodies: {robot.num_bodies}")

        # ── 4. Actuator Groups ──────────────────────────────────────────────
        print(f"\n{sep}")
        print("4. ACTUATOR GROUPS")
        print(sep)
        for act_name, act in robot.actuators.items():
            joint_names = getattr(act, "joint_names", [])
            joint_indices = getattr(act, "joint_indices", [])
            stiffness = getattr(act, "stiffness", None)
            damping = getattr(act, "damping", None)
            effort_limit = getattr(act, "effort_limit", None)
            vel_limit = getattr(act, "velocity_limit", None)
            print(f"\n  Actuator: '{act_name}'")
            print(f"    Matched joints ({len(joint_names)}): {list(joint_names)}")
            if hasattr(joint_indices, 'tolist'):
                print(f"    Joint indices: {joint_indices.tolist()}")
            if stiffness is not None and hasattr(stiffness, 'tolist'):
                print(f"    Stiffness: {stiffness[0].tolist()}")
            if damping is not None and hasattr(damping, 'tolist'):
                print(f"    Damping: {damping[0].tolist()}")
            if effort_limit is not None and hasattr(effort_limit, 'tolist'):
                print(f"    Effort limit: {effort_limit[0].tolist()}")
            if vel_limit is not None and hasattr(vel_limit, 'tolist'):
                print(f"    Velocity limit: {vel_limit[0].tolist()}")

        # ── 5. Config Cross-Check ──────────────────────────────────────────
        print(f"\n{sep}")
        print("5. CONFIG CROSS-CHECK")
        print(sep)

        # Check actuated joints
        print("\n  a) Actuated joint names from config vs PhysX:")
        all_actuated_ok = True
        for jname in cfg.actuated_joint_names:
            if jname in robot.joint_names:
                idx = robot.joint_names.index(jname)
                # Check if this joint is covered by any actuator
                covered = False
                for act_name, act in robot.actuators.items():
                    if jname in getattr(act, "joint_names", []):
                        covered = True
                        break
                status = "OK" if covered else "NOT ACTUATED!"
                if not covered:
                    all_actuated_ok = False
                print(f"    {jname} -> index {idx} [{status}]")
            else:
                print(f"    {jname} -> NOT FOUND IN PHYSX JOINTS!")
                all_actuated_ok = False
        print(f"  => All actuated joints OK: {all_actuated_ok}")

        # Check body names
        print("\n  b) Hand body names from config vs PhysX:")
        all_bodies_ok = True
        for bname in cfg.hand_body_names:
            if bname in robot.body_names:
                idx = robot.body_names.index(bname)
                print(f"    {bname} -> body index {idx}")
            else:
                print(f"    {bname} -> NOT FOUND IN PHYSX BODIES!")
                all_bodies_ok = False
        print(f"  => All hand bodies OK: {all_bodies_ok}")

        # Check palm body
        print(f"\n  c) Palm body '{cfg.palm_body_name}':")
        if cfg.palm_body_name in robot.body_names:
            idx = robot.body_names.index(cfg.palm_body_name)
            print(f"    -> body index {idx} [OK]")
        else:
            print(f"    -> NOT FOUND!")

        # Check workspace body
        print(f"\n  d) Workspace body '{cfg.hand_workspace_body_name}':")
        if cfg.hand_workspace_body_name in robot.body_names:
            idx = robot.body_names.index(cfg.hand_workspace_body_name)
            print(f"    -> body index {idx} [OK]")
        else:
            print(f"    -> NOT FOUND!")

        # ── 6. Unactuated joints ────────────────────────────────────────────
        print(f"\n{sep}")
        print("6. UNACTUATED / UNCOVERED JOINTS")
        print(sep)
        all_actuator_joints = set()
        for act_name, act in robot.actuators.items():
            for jn in getattr(act, "joint_names", []):
                all_actuator_joints.add(jn)
        for i, jname in enumerate(robot.joint_names):
            if jname not in all_actuator_joints:
                print(f"  [{i:2d}] {jname}  <-- NO ACTUATOR ASSIGNED")

        # ── 7. Joint limits ─────────────────────────────────────────────────
        print(f"\n{sep}")
        print("7. JOINT LIMITS (actuated joints)")
        print(sep)
        limits = robot.root_physx_view.get_dof_limits()[0].cpu()
        for jname in cfg.actuated_joint_names:
            idx = robot.joint_names.index(jname)
            lo, hi = limits[idx, 0].item(), limits[idx, 1].item()
            span = hi - lo
            print(f"  {jname:40s}  [{lo:+8.4f}, {hi:+8.4f}]  span={span:.4f} rad ({span*57.3:.1f} deg)")

        # ── 8. Gravity compensation check ───────────────────────────────────
        print(f"\n{sep}")
        print("8. GRAVITY TORQUE CHECK (zero-action hold)")
        print(sep)
        # Step a few times with zero action to let robot settle, then read torques
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.joint_vel)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        act_indices = [robot.joint_names.index(n) for n in cfg.actuated_joint_names]
        robot.set_joint_position_target(joint_pos[:, act_indices], joint_ids=act_indices)
        for _ in range(10):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=sim_dt)

        measured_torque = robot.root_physx_view.get_dof_projected_joint_forces()[0].cpu()
        for jname in cfg.actuated_joint_names:
            idx = robot.joint_names.index(jname)
            torque = measured_torque[idx].item()
            # Get effort limit for this joint
            elim = None
            for act_name, act in robot.actuators.items():
                if jname in getattr(act, "joint_names", []):
                    elim_tensor = getattr(act, "effort_limit", None)
                    if elim_tensor is not None:
                        j_idx_in_act = list(act.joint_names).index(jname)
                        elim = elim_tensor[0, j_idx_in_act].item()
                    break
            elim_str = f"{elim:.1f}" if elim else "?"
            pct = abs(torque / elim * 100) if elim and elim > 0 else 0
            warning = " ** SATURATED!" if pct > 90 else (" * HIGH" if pct > 60 else "")
            print(f"  {jname:40s}  torque={torque:+8.3f} Nm  effort_limit={elim_str} Nm  ({pct:.0f}%){warning}")

        print(f"\n{sep}")
        print("DIAGNOSTIC COMPLETE")
        print(sep)

    simulation_app.close()


if __name__ == "__main__":
    main()
