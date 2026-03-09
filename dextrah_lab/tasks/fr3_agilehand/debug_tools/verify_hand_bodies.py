#!/usr/bin/env python3
"""
Quick verification script to check if hand body names are correctly configured.
Tests that all specified body names exist in the robot articulation.

Usage:
    ./isaaclab.sh -p dextrah_lab/tasks/fr3_agilehand/verify_hand_bodies.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify hand body names configuration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that need Kit running ────────────────────────────────────────────
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
    class VerifySceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot = FR3_TEK_LEFT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene_cfg = VerifySceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    sim_dt = 1.0 / 120.0

    with build_simulation_context(device="cuda:0", dt=sim_dt, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        sep = "=" * 80

        print(f"\n{sep}")
        print("HAND BODY NAMES VERIFICATION")
        print(sep)

        # Check hand_body_names
        print("\n1. Checking hand_body_names:")
        print(f"   Config specifies: {cfg.hand_body_names}")
        all_found = True
        for body_name in cfg.hand_body_names:
            if body_name in robot.body_names:
                idx = robot.body_names.index(body_name)
                print(f"   ✓ {body_name:<30} -> body index {idx}")
            else:
                print(f"   ✗ {body_name:<30} -> NOT FOUND!")
                all_found = False
        
        if all_found:
            print(f"\n   ✅ All hand_body_names found! ({len(cfg.hand_body_names)} bodies)")
        else:
            print(f"\n   ❌ Some hand_body_names are missing!")

        # Check hand_object_distance_body_names
        print(f"\n2. Checking hand_object_distance_body_names:")
        print(f"   Config specifies: {cfg.hand_object_distance_body_names}")
        all_found = True
        for body_name in cfg.hand_object_distance_body_names:
            if body_name in robot.body_names:
                idx = robot.body_names.index(body_name)
                print(f"   ✓ {body_name:<30} -> body index {idx}")
            else:
                print(f"   ✗ {body_name:<30} -> NOT FOUND!")
                all_found = False
        
        if all_found:
            print(f"\n   ✅ All hand_object_distance_body_names found! ({len(cfg.hand_object_distance_body_names)} bodies)")
        else:
            print(f"\n   ❌ Some hand_object_distance_body_names are missing!")

        # Show all available body names for reference
        print(f"\n3. All available body names in robot ({robot.num_bodies} total):")
        for i, name in enumerate(robot.body_names):
            marker = ""
            if name in cfg.hand_body_names:
                marker = " ← in hand_body_names"
            elif "Tip" in name or "tip" in name:
                marker = " ← TIP FRAME (unused)"
            print(f"   [{i:2d}] {name}{marker}")

        print(f"\n{sep}")
        print("VERIFICATION COMPLETE")
        print(sep)

    simulation_app.close()


if __name__ == "__main__":
    main()
