"""Dummy grasp sanity check for the FR3 Tekken ADoF.

Boot Isaac Sim/Kit first via AppLauncher so carb/USD are available.
Spawns the arm+hand with ground plane and dome light, then holds a nominal pose.
"""

import argparse

from isaaclab.app import AppLauncher


def main():
    """Spawn the robot in a simple scene and hold a nominal joint configuration."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        "--print-joints",
        action="store_true",
        help="Print joint positions/velocities during the run.",
    )
    parser.add_argument(
        "--plot-joints",
        action="store_true",
        help="Plot joint positions during the run.",
    )
    args_cli = parser.parse_args()

    # Launch Kit before importing anything that needs carb/USD.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import build_simulation_context
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from dextrah_lab.assets.fr3_tekken_adof.fr3_tekken_left import FR3_TEK_LEFT_CONFIG

    plot_enabled = getattr(args_cli, "plot_joints", False)
    if plot_enabled:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider
        except Exception:
            plt = None
            Slider = None
    else:
        plt = None
        Slider = None

    # Nominal joint angles for a reachable tabletop pose; fingers open.
    init_joint_pos = {
        "fr3_joint1": 0.0,
        "fr3_joint2": 0.0,
        "fr3_joint3": 0.0,
        "fr3_joint4": -0.4,
        "fr3_joint5": 0.0,
        "fr3_joint6": 0.6,
        "fr3_joint7": 0.0,
        "revolute_thumb_rot": 0.0,
        "revolute_thumb_mcp_pitch": 0.1,
        "revolute_thumb_mcp_yaw": 0.0,
        "revolute_thumb_pip": 0.1,
        "revolute_index_mcp_pitch": 0.1,
        "revolute_index_mcp_yaw": 0.0,
        "revolute_index_pip": 0.1,
        "revolute_middle_mcp_pitch": 0.1,
        "revolute_middle_mcp_yaw": 0.0,
        "revolute_middle_pip": 0.1,
        "revolute_ring_mcp_pitch": 0.1,
        "revolute_ring_mcp_yaw": 0.0,
        "revolute_ring_pip": 0.1,
        "revolute_pinky_mcp_pitch": 0.1,
        "revolute_pinky_mcp_yaw": 0.0,
        "revolute_pinky_pip": 0.1,
    }
    nominal_joint_pos = init_joint_pos
    report_every_s = 1.0  # print joint states at this interval
    plot_every_s = 0.01  # update plot at this interval
    subplot_height = 0.12  # subplot height in figure coordinates
    subplot_gap = 0.02  # vertical gap between subplots
    table_top_z = 0.30  # meters above ground
    table_thickness = 0.05
    table_size = (0.6, 0.6, table_thickness)
    cube_size = 0.04
    table_pos_xy = (0.45, 0.0)

    @configclass
    class DummyGraspSceneCfg(InteractiveSceneCfg):
        """Simple scene with ground, lighting, and the robot."""

        # Ground plane and lighting
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(0.9, 0.9, 0.9)),
        )

        # Table with top at 0.30 m and a cube on top.
        table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.CuboidCfg(
                size=table_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(table_pos_xy[0], table_pos_xy[1], table_top_z - table_thickness / 2.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(
                size=(cube_size, cube_size, cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(density=300.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, 0.0, table_top_z + cube_size / 2.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Robot at nominal pose
        robot = FR3_TEK_LEFT_CONFIG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FR3_TEK_LEFT_CONFIG.init_state.replace(
                pos=(0.0, 0.0, 0.25),
                # Rotate base 180 deg around Z to face the table.
                rot=(0.0, 0.0, 1.0, 0.0),
                joint_pos=init_joint_pos,
            ),
        )

    # Build the scene
    scene_cfg = DummyGraspSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    device = getattr(args_cli, "device", "cuda:0")
    sim_dt = 1.0 / 120.0
    # Spin up simulation context before building the scene so SimulationContext.instance() is valid.
    with build_simulation_context(device=device, dt=sim_dt, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)

        # Start physics (this triggers asset initialization/actuators), then reset scene.
        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        num_joints = len(robot.joint_names)
        actuated_joint_names = []
        actuators = getattr(robot, "actuators", None)
        if actuators:
            for actuator in actuators.values():
                names = getattr(actuator, "joint_names", None)
                if names:
                    actuated_joint_names.extend(list(names))
        if not actuated_joint_names:
            cfg_actuators = getattr(FR3_TEK_LEFT_CONFIG, "actuators", {})
            for actuator_cfg in cfg_actuators.values():
                names = getattr(actuator_cfg, "joint_names_expr", None)
                if names:
                    actuated_joint_names.extend(list(names))
        seen = set()
        actuated_joint_names = [
            name for name in actuated_joint_names
            if name in robot.joint_names and not (name in seen or seen.add(name))
        ]
        actuated_dof_indices = [robot.joint_names.index(name) for name in actuated_joint_names]
        if not actuated_dof_indices:
            print("[WARN] No actuated joints found; defaulting to all joints.")
            actuated_dof_indices = list(range(num_joints))
        joint_pos = robot.data.joint_pos.clone()
        for idx, name in enumerate(robot.joint_names):
            if name in nominal_joint_pos:
                joint_pos[:, idx] = nominal_joint_pos[name]
        joint_vel = torch.zeros_like(robot.data.joint_vel)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos[:, actuated_dof_indices], joint_ids=actuated_dof_indices)

        name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
        nominal_tensor = joint_pos[0].clone()

        approach_pose = {
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.9,
            "fr3_joint3": 0.0,
            "fr3_joint4": -1.6,
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.2,
            "fr3_joint7": 0.0,
        }
        close_pose = {
            "revolute_thumb_mcp_pitch": 0.6,
            "revolute_thumb_pip": 1.0,
            "revolute_index_mcp_pitch": 0.8,
            "revolute_index_pip": 1.0,
            "revolute_middle_mcp_pitch": 0.8,
            "revolute_middle_pip": 1.0,
            "revolute_ring_mcp_pitch": 0.8,
            "revolute_ring_pip": 1.0,
            "revolute_pinky_mcp_pitch": 0.8,
            "revolute_pinky_pip": 1.0,
        }

        def lerp_pose(base, pose_a, pose_b, alpha):
            pos = base.clone()
            joint_names = set(pose_a.keys()) | set(pose_b.keys())
            for name in joint_names:
                idx = name_to_idx.get(name)
                if idx is None:
                    continue
                a = pose_a.get(name, base[idx].item())
                b = pose_b.get(name, base[idx].item())
                pos[idx] = (1.0 - alpha) * a + alpha * b
            return pos

        print("[INFO] Dummy grasp scene running. Close the window to exit.")
        t = 0.0
        step_count = 0
        report_every_steps = max(1, int(report_every_s / sim_dt))
        plot_every_steps = max(1, int(plot_every_s / sim_dt))
        num_joints = len(robot.joint_names)
        plot_times = []
        plot_desired = [[] for _ in range(num_joints)]
        plot_observed = [[] for _ in range(num_joints)]
        plot_fig = None
        plot_axes = []
        plot_lines_obs = []
        plot_lines_des = []
        scroll_slider = None
        if not plot_enabled:
            pass
        elif plt is None:
            print("[WARN] matplotlib unavailable; skipping plot rendering.")
        else:
            plt.ion()
            plot_fig = plt.figure(figsize=(10, 6))
            plot_left = 0.08
            plot_width = 0.82
            viewport_bottom = 0.06
            viewport_top_margin = 0.08
            viewport_height = 1.0 - viewport_bottom - viewport_top_margin
            content_height = num_joints * (subplot_height + subplot_gap) - subplot_gap
            max_scroll = max(0.0, content_height - viewport_height)
            scroll_val = 0.0
            for name in robot.joint_names:
                ax = plot_fig.add_axes([plot_left, viewport_bottom, plot_width, subplot_height])
                color = ax._get_lines.get_next_color()
                line_obs, = ax.plot([], [], color=color)
                line_des, = ax.plot([], [], color=color, linestyle="--", alpha=0.6)
                plot_axes.append(ax)
                plot_lines_obs.append(line_obs)
                plot_lines_des.append(line_des)
                ax.set_ylabel("rad")
                ax.set_title(f"{name} (solid=observed, dashed=desired)", fontsize=8, pad=2)
                ax.grid(True, alpha=0.3)
            plot_axes[-1].set_xlabel("time (s)")

            def layout_axes(scroll_val: float) -> None:
                scroll_val = max(0.0, min(float(scroll_val), max_scroll))
                for i, ax in enumerate(plot_axes):
                    y_top = content_height - i * (subplot_height + subplot_gap)
                    y0 = viewport_bottom + y_top - subplot_height - scroll_val
                    ax.set_position([plot_left, y0, plot_width, subplot_height])
                    visible = (y0 + subplot_height) >= viewport_bottom and y0 <= (viewport_bottom + viewport_height)
                    ax.set_visible(visible)
                plot_fig.canvas.draw_idle()

            if max_scroll > 0.0:
                if Slider is None:
                    print("[WARN] matplotlib slider unavailable; use mouse wheel to scroll.")

                    def on_scroll(event) -> None:
                        nonlocal scroll_val
                        step = subplot_height + subplot_gap
                        delta = -step if event.button == "up" else step
                        scroll_val = min(max(scroll_val + delta, 0.0), max_scroll)
                        layout_axes(scroll_val)

                    plot_fig.canvas.mpl_connect("scroll_event", on_scroll)
                else:
                    scroll_ax = plot_fig.add_axes([0.93, viewport_bottom, 0.02, viewport_height])
                    scroll_slider = Slider(
                        scroll_ax,
                        "",
                        0.0,
                        max_scroll,
                        valinit=0.0,
                        orientation="vertical",
                    )
                    scroll_ax.set_xticks([])
                    scroll_ax.set_yticks([])
                    scroll_slider.on_changed(layout_axes)

                    def on_scroll(event) -> None:
                        step = subplot_height + subplot_gap
                        delta = -step if event.button == "up" else step
                        new_val = min(max(scroll_slider.val + delta, 0.0), max_scroll)
                        scroll_slider.set_val(new_val)

                    plot_fig.canvas.mpl_connect("scroll_event", on_scroll)

            layout_axes(0.0)
            plt.show(block=False)
        approach_duration = 2.0
        close_duration = 2.0
        while simulation_app.is_running():
            if t < approach_duration:
                alpha = t / approach_duration
                target = lerp_pose(nominal_tensor, nominal_joint_pos, approach_pose, alpha)
            elif t < approach_duration + close_duration:
                alpha = (t - approach_duration) / close_duration
                target = lerp_pose(nominal_tensor, approach_pose, close_pose, alpha)
            else:
                target = lerp_pose(nominal_tensor, approach_pose, close_pose, 1.0)

            joint_pos = joint_pos.clone()
            joint_pos[:, :] = target
            robot.set_joint_position_target(joint_pos[:, actuated_dof_indices], joint_ids=actuated_dof_indices)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)
            if getattr(args_cli, "print_joints", False) and step_count % report_every_steps == 0:
                joint_pos_report = robot.data.joint_pos[0].tolist()
                joint_vel_report = robot.data.joint_vel[0].tolist()
                print("[JOINT STATE]")
                for name, pos, vel in zip(robot.joint_names, joint_pos_report, joint_vel_report):
                    print(f"  {name}: pos={pos:.4f} rad, vel={vel:.4f} rad/s")
            if plot_fig is not None and step_count % plot_every_steps == 0:
                plot_times.append(t)
                desired_snapshot = joint_pos[0].tolist()
                observed_snapshot = robot.data.joint_pos[0].tolist()
                for idx in range(num_joints):
                    plot_desired[idx].append(desired_snapshot[idx])
                    plot_observed[idx].append(observed_snapshot[idx])
                    plot_lines_obs[idx].set_data(plot_times, plot_observed[idx])
                    plot_lines_des[idx].set_data(plot_times, plot_desired[idx])
                for ax in plot_axes:
                    ax.relim()
                    ax.autoscale_view()
                plot_fig.canvas.draw_idle()
                plot_fig.canvas.flush_events()
                plt.pause(0.001)
            t += sim_dt
            step_count += 1
        if plot_fig is not None:
            plt.close(plot_fig)


if __name__ == "__main__":
    main()
