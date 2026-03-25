# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import numpy as np
import warp as wp
import math
from dextrah_lab.assets.fr3_tekken_adof.fr3_tekken_left import FR3_TEK_LEFT_CONFIG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

@configclass
class EventCfg:
    """Configuration for randomization."""

    # NOTE: the below ranges form the initial ranges for the parameters

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    # NOTE: no beginning randomization for these
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1., 1.),
            "damping_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # NOTE: no beginning randomization for this one
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0. , 0.),
             # NOTE: I don't really care about this one
#            "lower_limit_distribution_params": (0.00, 0.01),
#            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    # NOTE: no beginning randomization for these
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    
    # NOTE: no beginning randomization for this one
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # NOTE: I don't really care about this one
#    # -- scene
#    reset_gravity = EventTerm(
#        func=mdp.randomize_physics_scene_gravity,
#        mode="interval",
#        is_global_time=True,
#        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#        params={
#            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#            "operation": "add",
#            "distribution": "gaussian",
#        },
#    )

@configclass
class DextrahFR3AgilehandEnvCfg(DirectRLEnvCfg):
    # Placeholder for objects_dir which targets the directory of objects for training
    objects_dir = "replace_me"
    valid_objects_dir = ["visdex_objects",
                        "test_object",
                        "test_object_0",
                        "test_2",
                        "multi_objects/3",
                        "multi_objects/14",
                        "_single_object",
                        "playback",
                        "distill_multi_objects",
                        "single_object_shoe",
                        "single_object_female_knight",
                        ]

    # Toggle for using cuda graph
    use_cuda_graph = False

    # env
    sim_dt = 1/120.
    decimation = 2 # 60 Hz
    episode_length_s = 10. #10.0
    # Optional shorter episode length used only when distillation=True.
    distillation_episode_length_s = 5.0
    num_sim_steps_to_render=2 # renders every 4 sim steps, so 60 Hz
    num_actions = 23 # 1:1 joint position targets for 7 arm + 16 hand DOF
    success_timeout = 2.
    # num_observations = 94
    distillation = False
    num_student_observations = 0
    num_teacher_observations = 0
    num_observations = 0
    num_states = 0

    state_space = 0
    observation_space = 0
    action_space = 0

    asymmetric_obs = True
    obs_type = "full"
    simulate_stereo = False
    stereo_baseline = 55 / 1000

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.1,   # was 0.2 — catches slower collisions before bouncing
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_collision_stack_size= 2 ** 29
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = FR3_TEK_LEFT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),  # Raise robot to table height (matching TG2 config)
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "fr3_joint1": 0.3491,   # 20 degrees
                "fr3_joint2": 0.6109,   # 35 degrees
                "fr3_joint3": -0.8727,  # -50 degrees
                "fr3_joint4": -0.8727,  # -50 degrees
                "fr3_joint5": -0.3491,  # -20 degrees
                "fr3_joint6": 2.6180,   # 150 degrees
                "fr3_joint7": 0.0,      # 0 degrees
                "revolute_thumb_rot": -0.3491,  # -20 deg (joint min)
                "revolute_thumb_mcp_pitch": 0.1,
                "revolute_thumb_mcp_yaw": 0.0,
                "revolute_thumb_pip": 0.0,
                "revolute_index_mcp_pitch": 0.1,
                "revolute_index_mcp_yaw": 0.0,
                "revolute_index_pip": 0.0,
                "revolute_middle_mcp_pitch": 0.1,
                "revolute_middle_mcp_yaw": 0.0,
                "revolute_middle_pip": 0.0,
                "revolute_ring_mcp_pitch": 0.1,
                "revolute_ring_mcp_yaw": 0.0,
                "revolute_ring_pip": 0.0,
                "revolute_pinky_mcp_pitch": 0.1,
                "revolute_pinky_mcp_yaw": 0.0,
                "revolute_pinky_pip": 0.0,
            },
        )
    )
    actuated_joint_names = [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
        # Tekken ADoF hand joints (16 actuated, DIPs are mimic joints)
        "revolute_thumb_rot",
        "revolute_thumb_mcp_pitch",
        "revolute_thumb_mcp_yaw",
        "revolute_thumb_pip",
        "revolute_index_mcp_pitch",
        "revolute_index_mcp_yaw",
        "revolute_index_pip",
        "revolute_middle_mcp_pitch",
        "revolute_middle_mcp_yaw",
        "revolute_middle_pip",
        "revolute_ring_mcp_pitch",
        "revolute_ring_mcp_yaw",
        "revolute_ring_pip",
        "revolute_pinky_mcp_pitch",
        "revolute_pinky_mcp_yaw",
        "revolute_pinky_pip",
    ]
    
    ## Updated with actual body names from the FR3+Tekken USD file
    # hand_body_names: Used for observations (fingertip positions/velocities)
    # Now using rigid-body tip frames that appear in body_names
    hand_body_names = [
        "base_link",              # Palm
        "Thumb_Tip",              # Rigid-body tip frames
        "Index_Tip",
        "Middle_Tip",
        "Ring_Tip",
        "Pinky_Tip",
    ]
    
    # hand_object_distance_body_names: Used for hand-to-object distance reward
    # Uses the same tip frames for precise distance calculation
    hand_object_distance_body_names = hand_body_names

    # Palm body name for computing palm direction vectors
    palm_body_name = "base_link"
    
    # Hand workspace body name (typically the hand base or arm flange)
    hand_workspace_body_name = "base_link"

    # Optional URDF for forward-kinematics hand point taskmap.
    hand_points_urdf_path = None

    module_path = os.path.dirname(__file__)
    root_path = os.path.dirname(os.path.dirname(module_path))
    scene_objects_usd_path = os.path.join(root_path, "assets/scene_objects/")

    table_texture_dir = os.path.join(
        root_path, "assets", "curated_table_textures"
    )
    dome_light_dir = os.path.join(
        root_path, "assets", "dome_light_textures"
    )
    metropolis_asset_dir = os.path.join(
        root_path, "assets", "object_textures"
    )

    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=scene_objects_usd_path + "table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            # scale=(1.0, 1.5, 50.0),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.21 - 0.725 / 2,
                 0.668 - 1.16 / 2,
                 0.25 - 0.03 / 2),
            rot=(1.0, 0.0, 0.0, 0.0)),
    )
    table_size_x = 0.725
    table_size_y = 1.16
    table_size_z = 0.03
    # Extra tolerance for the palm bounding box (meters).
    # Increased to 0.35 to allow hand to reach objects spawning at X=-0.3
    hand_bbox_margin = 0.35

    # camera pose in the real world
    # tf = np.array([
    #     9.979802254757542679e-01, 5.805126464282436838e-02, -2.579767882449228097e-02, -6.452117743594977251e-01,
    #     2.867907587635045233e-02, -4.936231931159993508e-02, 9.983691061120923971e-01, -7.328016905360382749e-01,
    #     5.668315593050039097e-02, -9.970924792142141779e-01, -5.092747518000630136e-02, 4.559887081479024329e-01,
    #     0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00
    # ]).reshape(4,4)

    tf = np.array([
        7.416679444534866883e-02,-9.902696855667120213e-01,1.177507386359286923e-01,-7.236400044878017468e-01,
        -1.274026398887237732e-01,1.076995435286611930e-01,9.859864987275952508e-01,-6.886495877727516479e-01,
        -9.890742408692511090e-01,-8.812921292808308105e-02,-1.181752422362273985e-01,6.366771698474239516e-01,
        0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00
    ]).reshape(4,4)
    # camera pose in world frame
    ## left camera world pose
    # NOTE on rotation conventions (important for future tuning):
    # - The camera offsets below are used with `convention="ros"` in TiledCameraCfg.
    # - The UI "Orientation X/Y/Z" values are Euler XYZ in the UI camera frame, and
    #   they do NOT map 1:1 to the config quaternions.
    # ANYWAY:
    # The result we want is for the camera looking to the right and a little down at the table so the object will always be seen.
    camera_pos_left = tf[:3, 3].tolist()
    # camera_rot_left = [0.6887834, -0.7242703, -0.0299371, -0.0106609]
    camera_rot_left = [ 0.51567701, -0.52073085,  0.53658829,  0.41831759]
    # TODO: update these for your actual stereo right camera calibration
    # Convert to native Python floats to avoid OmegaConf errors
    camera_right_pos = [float(tf[0, 3] - 0.06169578743), float(tf[1, 3] - 0.00260621), float(tf[2, 3] + 0.0003994)]
    camera_right_rot = [0.51567701, -0.52073085, 0.53658829, 0.41831759]  # placeholder, same as left
    del tf # this is hacky but it needs to be done because omega conf doesn't support np.ndarray as a primitive
    camera_rand_rot_range = 3
    camera_rand_pos_range = 0.03

    # horizontal fov: 48, vertical fov is h:w ratio
    img_width = int(160 * 2)
    img_height = int(120 * 2)
    horizontal_aperture = 21.02
    focal_length_val = 23.59
    left_focal_length = focal_length_val
    right_focal_length = focal_length_val
    tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraLeft",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_pos_left, rot=camera_rot_left, convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=left_focal_length, focus_distance=400.0, horizontal_aperture=horizontal_aperture, clipping_range=(0.01, 2.)
        ),
        width=img_width,
        height=img_height,
    )
    tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraRight",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_right_pos, rot=camera_right_rot, convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=right_focal_length, focus_distance=400.0, horizontal_aperture=horizontal_aperture, clipping_range=(0.01, 2.)
        ),
        width=img_width,
        height=img_height,
    )

    fov = 2 * math.atan(horizontal_aperture / (2 * left_focal_length))
    focal_px = img_width * 0.5 / math.tan(fov / 2)
    a = focal_px
    b = img_width * 0.5
    c = focal_px
    d = img_height * 0.5
    intrinsic_matrix_left = [
        [a, 0., b],
        [0., c, d],
        [0., 0., 1.]
    ]
    intrinsic_matrix_right = [
        [a, 0., b],
        [0., c, d],
        [0., 0., 1.]
    ]

    # Contact sensor on hand links to fetch per-link contact forces
    palm_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tekken_left_adof/base_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Filter to only include object contacts (excludes robot self-collisions and table)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/object/.*/baseLink/",
        ],
    )

    # index0_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/index_link_0",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    index1_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tekken_left_adof/Index_Distal_Phalanx",
        update_period=0.0,
        history_length=6,
        debug_vis=True,  # Enable to see sensor location
        track_pose=True,  # Track sensor pose
        # Filter to only include object contacts (excludes robot self-collisions and table)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/object/.*/baseLink/",
        ],
    )

    # middle0_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/middle_link_0",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    middle1_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tekken_left_adof/Middle_Distal_Phalanx",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        # Filter to only include object contacts (excludes robot self-collisions and table)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/object/.*/baseLink/",
        ],
    )

    # ring0_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/ring_link_0",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    ring1_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tekken_left_adof/Ring_Distal_Phalanx",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        # Filter to only include object contacts (excludes robot self-collisions and table)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/object/.*/baseLink/",
        ],
    )
    
    # little0_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/little_link_0",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    little1_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tekken_left_adof/Pinky_Distal_Phalanx",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        # Filter to only include object contacts (excludes robot self-collisions and table)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/object/.*/baseLink/",
        ],
    )

    # thumb0_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/thumb_link_1",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    # thumb1_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/thumb_link_1",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    # thumb2_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/thumb_link_2",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    #     # Focus on contacts with the cube and table (add other objects as needed)
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/object/.*/baseLink/",
    #     ],
    # )

    thumb3_object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tekken_left_adof/Thumb_Distal_Phalanx",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        # Filter to only include object contacts (excludes robot self-collisions and table)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/object/.*/baseLink/",
        ],
    )

    # we add this to prevent the arm from colliding with the table
    fr3_link0_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link0",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box/",
        ],
    )

    fr3_link1_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link1",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box/",
        ],
    )

    fr3_link2_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link2",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box/",
        ],
    )

    fr3_link3_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link3",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box/",
        ],
    )

    fr3_link4_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link4",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box",
        ],
    )

    fr3_link5_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link5",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box/",
        ],
    )

    fr3_link6_table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fr3_link6",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # Focus on contacts with the cube and table (add other objects as needed)
        filter_prim_paths_expr=[
            "/World/envs/env_.*/table/box/",
        ],
    )

    pred_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )

    gt_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker_gt",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2., replicate_physics=False)

    # reward weights
    # phase 1: reaching
    hand_to_object_weight = 5. #default 1, prev 5
    hand_to_object_sharpness = 4. #default 10, increased from 4 to match TG2 - creates steeper gradient and urgency to approach
    
    palm_direction_alignment_weight = 0.5 # 2.0  # Increased from 0.1 - strongly encourage palm facing down
    in_grip_alignment_weight = 1. # 0.5
    
    palm_down_local_axis = (1.0, 0.0, 0.0) # x axis of agile-hand points in the direction of palm
    palm_finger_alignment_weight = 0.0  # Disabled - let robot find optimal approach direction
    palm_finger_local_axis = (0.0, -1.0, 0.0) # palm axis that points in the direction of the fingers in palm frame
    palm_finger_direction_target = (-1.0, -1.0, 0.0) # not used when weight=0
    
    palm_linear_velocity_penalty_weight = 0.005 # prev 0.005 -- removed to avoid "don't move" signal
    approach_speed_penalty_weight = 0.001        # prev 0.001 -- removed to avoid "don't move" signal
    
    action_rate_penalty_weight = 0.01         # prev 0.01 -- halved to allow exploration
    hand_action_rate_penalty_scale = 2.5       # prev 3.0

    joint_velocity_penalty_weight = 5e-4       # prev 5e-4 -- reduced to avoid freezing
    hand_joint_velocity_penalty_scale = 3.0    # prev 3.0

    # phase 2: contact
    hand_object_contact_weight = 3.0  # Increased to make contact more valuable than hovering
    good_grasp_weight = 5.0 # default 10.0 # too obsessed in finding a good contact, actually finds one
    finger_curl_reg_weight = -0.5    # penalization factor for finger curl
    finger_curl_reg_min = -3.0 # max penalty for finger curl
    finger_curl_reg_max = 0.0 # min penalty for finger curl 

    #phase 3: lifting
    object_to_goal_weight = 20 #default 5 
    in_success_region_at_rest_weight = 10. #default10
    lift_sharpness = 7.5 #default 8.5

    # extras
    episode_length_reward_weight = 0.005 # default 0.025   
    episode_length_gate_dist = 0.2 # default 0.3
    
    # Active reward terms summed into total_reward.
    active_reward_terms = [
        "action_rate_penalty",
        "hand_to_object",
        "finger_curl",
        "palm_align",
        "good_grasp",
        "episode_length",
        "object_to_goal",
        "lift",
    ]

    # Optional: print per-step reward breakdown for the first N steps (debugging aid).
    debug_reward_steps = -1  # Enable to see reward breakdown every step
    # How often to print the training status table (in env-steps; 16 = every rl_games epoch at horizon_length=16)
    debug_print_every_steps = 16
    # Terminate if palm flips beyond this cosine threshold relative to target (-Z).
    palm_flip_cos_thresh = -0.3  # Allows up to ~108 degrees deviation from downward
    early_termination_penalty: float = -5.0  # applied when episode ends early (not timeout)

    # Goal reaching parameters
    object_goal_tol = 0.1 # m
    success_for_adr = 0.4
    #min_steps_for_dr_change = 240 # number of steps
    min_steps_for_dr_change = 5 * int(episode_length_s / (decimation * sim_dt))


    # Lift criteria
    min_num_episode_steps = 60
    object_height_thresh = 0.15  # meters above the table top

    # Object spawning params
    x_center = -0.55
    x_width = 0.5#0.4
    y_center = 0.1
    y_width = 0.8 #0.5

    # DR Controls
    enable_adr = True
    num_adr_increments = 50
    starting_adr_increments = 0 # 0 for no DR up to num_adr_increments for max DR 

    # Default friction coefficients for all 28 joints (DOFs) in USD order.
    # NOTE: PhysX treats mimic/DIP joints as DOFs, so all 28 must be listed.
    # These are scaled multiplicatively by the robot_joint_friction EventTerm.
    #   0-6:   fr3_joint1..7
    #   7-10:  index/middle/pinky/ring _mcp_pitch
    #   11:    thumb_rot
    #   12-15: index/middle/pinky/ring _mcp_yaw
    #   16:    thumb_mcp_pitch
    #   17-20: index/middle/pinky/ring _pip
    #   21:    thumb_mcp_yaw
    #   22-25: index/middle/pinky/ring _dip  (mimic, still counted as DOF)
    #   26:    thumb_pip
    #   27:    thumb_dip  (mimic, still counted as DOF)
    starting_robot_dof_friction_coefficients = [
        1., 1., 1., 1., 1., 1., 1.,       # arm joints (7)
        0.01, 0.01, 0.01, 0.01,           # index/middle/pinky/ring mcp_pitch (4)
        0.01,                              # thumb_rot (1)
        0.01, 0.01, 0.01, 0.01,           # index/middle/pinky/ring mcp_yaw (4)
        0.01,                              # thumb_mcp_pitch (1)
        0.01, 0.01, 0.01, 0.01,           # index/middle/pinky/ring pip (4)
        0.01,                              # thumb_mcp_yaw (1)
        0.01, 0.01, 0.01, 0.01,           # index/middle/pinky/ring dip (4, mimic)
        0.01,                              # thumb_pip (1)
        0.01,                              # thumb_dip (1, mimic)
    ]

    # domain randomization config
    events: EventCfg = EventCfg()

    # These serve to set the maximum value ranges for the different physics parameters
    adr_cfg_dict = {
        "num_increments": num_adr_increments, # number of times you can change the parameter ranges
        "robot_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 2.),
            "damping_distribution_params": (0.5, 2.),
        },
        "robot_joint_friction": {
            "friction_distribution_params": (0., 5.),
        },
        "object_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.5, 3.),
        },
    }

    # Object disturbance wrench fixed params
    wrench_trigger_every = int(1. / (decimation * sim_dt)) # 1 sec
    torsional_radius = 0.01 # m
    hand_to_object_dist_threshold = .3 # m
    #wrench_prob_per_rollout = 0. # NOTE: currently not used

    # Object scaling
    object_scale_max = 1.75
    object_scale_min = 0.5
    deactivate_object_scaling = True

    # TODO: what is this?
    aux_coeff = 1.

    # Dictionary of custom parameters for ADR
    # NOTE: first number in range is the starting value, second number is terminal value
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0., 10.)
        },
        "object_spawn": {
            "x_width_spawn": (0., x_width),
            "y_width_spawn": (0., y_width),
            "rotation": (0., 1.)
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.03), # m
            "object_pos_bias": (0.0, 0.02), # m
            "object_rot_noise": (0.0, 0.1), # rad
            "object_rot_bias": (0.0, 0.08), # rad
        },
        "robot_spawn": {
            # TODO: Re-enable joint position noise after verifying open hand behavior
            # Original value was (0., 0.35) which adds ±20° randomization at reset
            "joint_pos_noise": (0., 0.8),  
            "joint_vel_noise": (0., 1.),
        },
        "robot_state_noise": {
            "robot_joint_pos_noise": (0.0, 0.08), # rad
            "robot_joint_pos_bias": (0.0, 0.08), # rad
            "robot_joint_vel_noise": (0.0, 0.18), # rad
            "robot_joint_vel_bias": (0.0, 0.08), # rad
        },
        "reward_weights": {
            "object_to_goal_sharpness": (-5., -10.),
            # "_weight": (5., 2.5) # default = (5,0)
            "lift_weight": (10., 5.),  # Increased from (20,20) for stronger lifting incentive
            "finger_curl_reg": (-0.1, -1),  # ADR: ramp up curl penalty to encourage better hand use
        },
        "pd_targets": {
            "velocity_target_factor": (1., 0.)
        },
        "observation_annealing": {
            "coefficient": (0., 0.)
        },
    }

    # Action space related parameters
    action_mode = "absolute"  # "absolute" or "delta"
    action_step = 0.005
    max_pose_angle = -1. # it is not used for now

    # depth randomization parameters
    img_aug_type = "rgb"
    aug_depth = True
    cam_matrix = wp.mat44f()
    cam_matrix[0,0] = 2.2460368
    cam_matrix[1, 1] = 2.9947157
    cam_matrix[2, 3] = -1.
    cam_matrix[3, 2] = 1.e-3
    d_min = 0.5
    d_max = 1.3
    depth_randomization_cfg_dict = {
        # Dropout and random noise blob parameters
        "pixel_dropout_and_randu": {
            "p_dropout": 0.0125 / 4,
            "p_randu": 0.0125 / 4,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Random stick parameters
        "sticks": {
            "p_stick":  0.001 / 4,
            "max_stick_len": 18.,
            "max_stick_width": 3.,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Correlated noise parameters
        "correlated_noise": {
            "sigma_s": 1./2,
            "sigma_d": 1./6,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Normal noise parameters
        "normal_noise": {
            "sigma_theta": 0.01,
            "cam_matrix": cam_matrix,
            "d_max": d_min,
            "d_min": d_max,
        }

    }

    # If enabled, the environment will terminate when time out.
    # This is used for data recording.
    disable_out_of_reach_done = False
    disable_dome_light_randomization = False
    disable_arm_randomization = False
