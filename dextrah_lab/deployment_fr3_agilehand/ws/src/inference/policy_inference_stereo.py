#!/usr/bin/env python3
"""Stereo vision student policy inference node for FR3 + AgileHand.

Subscribes to stereo image topics, runs the trained student policy,
and publishes joint position commands.

Usage:
    rosrun inference policy_inference_stereo.py \
        _checkpoint:=<path_to_student.pth> \
        _left_image_topic:=/stereo/left/image_raw \
        _right_image_topic:=/stereo/right/image_raw \
        _joint_command_topic:=/fr3_agilehand/joint_commands
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
import yaml

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

# Add project root to path for model imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rl_games.algos_torch.model_builder import ModelBuilder

# Training resolution — must match distillation config
TRAIN_WIDTH = 320
TRAIN_HEIGHT = 240

# FR3 + AgileHand joint names (23 DOF total)
JOINT_NAMES = [
    # FR3 arm (7 DOF)
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
    # AgileHand (16 DOF)
    "revolute_thumb_rot",
    "revolute_thumb_mcp_pitch", "revolute_thumb_mcp_yaw", "revolute_thumb_pip",
    "revolute_index_mcp_pitch", "revolute_index_mcp_yaw", "revolute_index_pip",
    "revolute_middle_mcp_pitch", "revolute_middle_mcp_yaw", "revolute_middle_pip",
    "revolute_ring_mcp_pitch", "revolute_ring_mcp_yaw", "revolute_ring_pip",
    "revolute_pinky_mcp_pitch", "revolute_pinky_mcp_yaw", "revolute_pinky_pip",
]


class StereoInferenceNode:
    def __init__(self):
        rospy.init_node("policy_inference_stereo", anonymous=True)

        # ROS params
        self.checkpoint_path = rospy.get_param("~checkpoint", None)
        if self.checkpoint_path is None:
            rospy.logfatal("No checkpoint specified. Use _checkpoint:=<path>")
            sys.exit(1)

        self.student_cfg_path = rospy.get_param(
            "~student_cfg",
            os.path.join(
                PROJECT_ROOT,
                "dextrah_lab/tasks/fr3_agilehand/agents/rl_games_ppo_stereo_transformer.yaml",
            ),
        )
        left_topic = rospy.get_param("~left_image_topic", "/stereo/left/image_raw")
        right_topic = rospy.get_param("~right_image_topic", "/stereo/right/image_raw")
        joint_cmd_topic = rospy.get_param("~joint_command_topic", "/fr3_agilehand/joint_commands")
        self.rate_hz = rospy.get_param("~rate", 30)

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")

        # Image buffers (protected by lock)
        self.lock = threading.Lock()
        self.left_img = None
        self.right_img = None
        self.bridge = CvBridge()

        # Load model
        self._load_model()

        # RNN state
        if self.is_rnn:
            self.hidden_states = tuple(
                s.to(self.device) for s in self.model.get_default_rnn_state()
            )

        # Previous actions (for autoregressive input)
        self.prev_actions = torch.zeros(
            (1, self.num_actions), dtype=torch.float32, device=self.device
        )

        # Subscribers
        rospy.Subscriber(left_topic, Image, self._left_cb, queue_size=1)
        rospy.Subscriber(right_topic, Image, self._right_cb, queue_size=1)

        # Publisher
        self.joint_pub = rospy.Publisher(joint_cmd_topic, JointState, queue_size=1)

        rospy.loginfo(
            f"Inference node ready. Subscribing to {left_topic}, {right_topic}. "
            f"Publishing to {joint_cmd_topic} at {self.rate_hz} Hz."
        )

    def _load_model(self):
        """Load the student policy network and checkpoint."""
        with open(self.student_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        params = cfg["params"]
        builder = ModelBuilder()
        network = builder.load(params)

        # TODO: these must match the env config used during training
        self.num_actions = 23  # 7 arm + 16 hand
        num_obs = params["network"].get("input_shape", [128])[0]  # fallback

        model_config = {
            "actions_num": self.num_actions,
            "input_shape": (num_obs,),
            "batch_size": 1,
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": params["config"].get("normalize_value", True),
            "normalize_input": params["config"].get("normalize_input", True),
        }

        self.model = network.build(model_config).to(self.device)
        self.model.eval()

        # Load checkpoint
        weights = torch.load(self.checkpoint_path, map_location=self.device)
        if "model" in weights:
            self.model.load_state_dict(weights["model"])
        else:
            self.model.load_state_dict(weights)

        if model_config["normalize_input"] and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

        self.is_rnn = self.model.is_rnn()
        self.is_aux = (
            hasattr(self.model.a2c_network, "is_aux") and self.model.a2c_network.is_aux
        )

        rospy.loginfo(
            f"Loaded checkpoint: {self.checkpoint_path} "
            f"(RNN={self.is_rnn}, AUX={self.is_aux}, actions={self.num_actions})"
        )

    def _left_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self.lock:
            self.left_img = img

    def _right_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self.lock:
            self.right_img = img

    def _preprocess_image(self, img):
        """Resize to training resolution and convert to [1, 3, H, W] float tensor."""
        img = cv2.resize(img, (TRAIN_WIDTH, TRAIN_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                left = self.left_img
                right = self.right_img

            if left is None or right is None:
                rate.sleep()
                continue

            # Preprocess
            left_tensor = self._preprocess_image(left)
            right_tensor = self._preprocess_image(right)

            # Build batch dict
            # NOTE: proprio observations must also be provided for full inference.
            # This requires subscribing to /franka_state_controller/joint_states
            # and constructing the observation vector to match training.
            # For now this is a template — proprio integration is TODO.
            batch_dict = {
                "is_train": False,
                "prev_actions": self.prev_actions,
                "img_left": left_tensor,
                "img_right": right_tensor,
                "finetune_backbone": False,
            }

            if self.is_rnn:
                batch_dict["rnn_states"] = self.hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None

            # Inference
            with torch.no_grad():
                res_dict = self.model(batch_dict)

            # Extract actions
            mus = res_dict["mus"]
            actions = torch.clamp(mus, -1.0, 1.0).squeeze(0).cpu().numpy()

            # Update RNN states
            if self.is_rnn:
                if self.is_aux:
                    self.hidden_states = tuple(
                        s.detach() for s in res_dict["rnn_states"][0]
                    )
                else:
                    self.hidden_states = tuple(
                        s.detach() for s in res_dict["rnn_states"]
                    )

            self.prev_actions = mus.detach()

            # Publish joint commands
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = JOINT_NAMES
            msg.position = actions.tolist()
            self.joint_pub.publish(msg)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = StereoInferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
