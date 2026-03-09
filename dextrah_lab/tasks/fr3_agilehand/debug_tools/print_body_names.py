#!/usr/bin/env python3
"""Quick script to print all body names from the FR3 robot."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import build_simulation_context
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from dextrah_lab.assets.fr3_tekken_adof.fr3_tekken_left import FR3_TEK_LEFT_CONFIG

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = FR3_TEK_LEFT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

with build_simulation_context(device="cuda:0", dt=1/120, add_ground_plane=False, add_lighting=False) as sim:
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    robot = scene.articulations["robot"]
    
    print("\n" + "="*80)
    print("ALL BODY NAMES IN ROBOT")
    print("="*80)
    for i, name in enumerate(robot.body_names):
        marker = ""
        if "Tip" in name or "tip" in name:
            marker = " ← TIP FRAME"
        elif "base_link" in name:
            marker = " ← PALM"
        print(f"[{i:2d}] {name}{marker}")
    
    print(f"\nTotal bodies: {robot.num_bodies}")
    print("="*80)

simulation_app.close()
