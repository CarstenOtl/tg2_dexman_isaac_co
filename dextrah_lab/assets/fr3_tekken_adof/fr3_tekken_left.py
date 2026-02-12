# SPDX-License-Identifier: BSD-3-Clause
"""
Configuration for the FR3 Tekken ADoF left robot as an ArticulationCfg.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the robot USD (models live under omniisaacgymenvs/models)


# Resolve USD path relative to this file to avoid external package dependency.
ROBOT_USD_PATH = Path(__file__).resolve().parent / "FR3_tekkenadof_left.usd"


FR3_TEK_LEFT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        # usd_path=str(ROBOT_USD_PATH),
        activate_contact_sensors=True,  # enable contact sensors if any are defined, for dexsuite tasks
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.001,
            angular_damping=0.0,
            max_linear_velocity=500.0, # default 1000
            max_angular_velocity=500.0, # default 1000
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,  # Add sleep threshold like TG2
            stabilization_threshold=0.0005,
        ),
        # joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": 0.0,
            "fr3_joint3": 0.0,
            "fr3_joint4": -0.9599,  # -55 degrees
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.7453,  # 100 degrees
            "fr3_joint7": 0.0,
            ### ADOF joint limits
            # Thumb Rot: [-0.34 , 0.34] ==> [-20 , 20] deg
            # MCP Pitch: [0 , 1.22] ==> [0 , ~70 deg]
            # MCP Yaw: [-0.26 , 0.26] ==> [-15 deg , 15 deg]
            # PIP: [0 , 1.57] ==> [0 , 90 deg]
            "revolute_thumb_rot": 0.0,
            "revolute_thumb_mcp_pitch": 0.1,
            "revolute_thumb_mcp_yaw": 0.0,
            "revolute_thumb_pip": 0.1,
            # # "revolute_thumb_dip": 0.0,
            "revolute_index_mcp_pitch": 0.1,
            "revolute_index_mcp_yaw": 0.0,
            "revolute_index_pip": 0.1,
            # # "revolute_index_dip": 0.0,
            "revolute_middle_mcp_pitch": 0.1,
            "revolute_middle_mcp_yaw": 0.0,
            "revolute_middle_pip": 0.1,
            # # "revolute_middle_dip": 0.0,
            "revolute_ring_mcp_pitch": 0.1,
            "revolute_ring_mcp_yaw": 0.0,
            "revolute_ring_pip": 0.1,
            # # "revolute_ring_dip": 0.0,
            "revolute_pinky_mcp_pitch": 0.1,
            "revolute_pinky_mcp_yaw": 0.0,
            "revolute_pinky_pip": 0.1,
            # # "revolute_pinky_dip": 0.0,
        },
    ),
    actuators={
        "franka_arm": ImplicitActuatorCfg(
            joint_names_expr=[r"fr3_joint[1-7]"],
            effort_limit_sim=200.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,  # Reduced from 400 (closer to TG2's 10-60 range)
            damping=8.0,      # Reduced from 40 (closer to TG2's 1-3 range)
        ),
        "thumb_rot": ImplicitActuatorCfg(
            joint_names_expr=["revolute_thumb_rot"],
            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness=20.0,
            damping=2.0,
        ),
        "mcp_pitch": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_mcp_pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=15.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "mcp_yaw": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_mcp_yaw"],
            effort_limit_sim=10.0,
            velocity_limit_sim=15.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "pip": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_pip"],
            effort_limit_sim=10.0,
            velocity_limit_sim=15.0,
            stiffness=10.0,
            damping=1.0,
        ),
        #
        # "franka_tekken_actuators": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         r"fr3_joint(1|2|3|4|5|6|7)",
        #         r"revolute_thumb_rot",
        #         r"revolute_.*_mcp_yaw",
        #         r"revolute_.*_mcp_pitch",
        #         r"revolute_.*_pip",
        #     ],
        #     effort_limit_sim={
        #         "fr3_joint(1|2|3|4|5|6|7)": 600.,
        #         r"revolute_thumb_rot":100.0,
        #         r"revolute_.*_mcp_yaw":100.0,
        #         r"revolute_.*_mcp_pitch":100.0,
        #         r"revolute_.*_pip":100.0,
        #     },
        #     stiffness={
        #         "fr3_joint(1|2|3|4)": 6000.,
        #         "fr3_joint5": 3000.,
        #         "fr3_joint6": 3000.,
        #         "fr3_joint7": 3000.,
        #         r"revolute_thumb_rot":5.0,
        #         r"revolute_.*_mcp_yaw":1.5,
        #         r"revolute_.*_mcp_pitch":1.5,
        #         r"revolute_.*_pip":1.5,
        #     },
        #     damping={
        #         "fr3_joint(1|2|3|4)": 600.,
        #         "fr3_joint5": 300.,
        #         "fr3_joint6": 300.,
        #         "fr3_joint7": 300.,
        #         r"revolute_thumb_rot":1.5,
        #         r"revolute_.*_mcp_yaw":.5,
        #         r"revolute_.*_mcp_pitch":.5,
        #         r"revolute_.*_pip":1.5,
        #     },
        # ),
    },
    soft_joint_pos_limit_factor=0.9,
)

# Create a variant of the config with explicit actuators for stability
FR3_TEK_LEFT_STABLE = FR3_TEK_LEFT_CONFIG.replace(
    actuators={
        # Arm joints (Franka 7-DoF) – moderate PD gains
        "arm_pd": ImplicitActuatorCfg(
            joint_names_expr=[r"fr3_joint[1-7]"],  # adapt if your naming differs
            effort_limit_sim=120.0,
            velocity_limit_sim=5.0,
            stiffness=60000.0,
            damping=600.0,
        ),
        # Hand / Tekken joints – low stiffness, decent damping so they don't go crazy
        "hand_pd": ImplicitActuatorCfg(
            joint_names_expr=[
                r"revolute_thumb_rot",
                r"revolute_.*_mcp_pitch",
                r"revolute_.*_mcp_yaw",
                r"revolute_.*_pip",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=300.0,
            stiffness=50.0,
            damping=20.0,
            # friction=0.01,
            # armature=0.001,
        ),
    }
)

# __all__ = ["FR3_TEK_LEFT_CONFIG", "FR3_TEK_LEFT_STABLE", "ROBOT_USD_PATH"]
