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
            # Higher value so fingers cannot "phase through" objects: PhysX resolves
            # penetration faster, making invalid grasps (curl-through) impossible.
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # More iterations improve contact resolution for thin finger–object contacts.
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=6,
            stabilization_threshold=0.0005,
            # fixed_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
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
            joint_names_expr=[r"fr3_joint[1-4]"],
            effort_limit_sim= 87., # 200.0,
            # velocity_limit_sim=2.175,
            stiffness=100.0, # 400.0,
            damping=7.0, # 40.0,
        ),
        "franka_joints_ee": ImplicitActuatorCfg(
            joint_names_expr=[r"fr3_joint[5-7]"],
            effort_limit_sim=12.0, # 200.0,
            # velocity_limit_sim=2.175,
            stiffness=50.0, # 400.0,
            damping=2.0, # 40.0,
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
            stiffness=1.77531, #10.0,
            damping=0.5, # 1.0, # was 0.00095 (hardware ID) — too low for sim stability
        ),
        "mcp_yaw": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_mcp_yaw"],
            effort_limit_sim=10.0,
            velocity_limit_sim=15.0,
            stiffness=0.28467, #10.0,
            damping=0.2, # 1.0, # was 0.00038 (hardware ID) — too low for sim stability
        ),
        "pip": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_pip"],
            effort_limit_sim=10.0,
            velocity_limit_sim=15.0,
            stiffness=0.24299, # 10.0,
            damping=0.2, # 1.0, # was 0.0001 (hardware ID) — too low for sim stability
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
            joint_names_expr=[r"fr3_joint[1-4]"],  # joint_names_expr=[r"fr3_joint[1-7]"],adapt if your naming differs
            effort_limit_sim=87, #effort_limit_sim=120.0,
            # velocity_limit_sim=150.0,
            stiffness=100.0, #default 60000.0 
            damping=7.0, #default 600.0
        ),
        "arm_stable_pd": ImplicitActuatorCfg(
            joint_names_expr=[r"fr3_joint[5-7]"],  # adapt if your naming differs
            effort_limit_sim=12.0,
            # velocity_limit_sim=5.0,
            stiffness=50.0,
            damping=2.0,
        ),
        # Hand thumb joint should have different stiffness and damping than the other joints
        "hand_thumb_pd": ImplicitActuatorCfg(
            joint_names_expr=["revolute_thumb_rot"],
            # effort_limit_sim=30.0,
            # velocity_limit_sim=300.0,
            stiffness=100.0,
            damping=5.0,
        ),
        # Hand joints – low stiffness, decent damping so they don't go crazy
        "hand_pd": ImplicitActuatorCfg(
            joint_names_expr=[
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
