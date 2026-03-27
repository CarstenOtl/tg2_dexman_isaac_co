# FR3 AgileHand Deployment

## Overview

Real-robot deployment pipeline for FR3 + AgileHand (TekkenADoF). Adapted from `deployment_tg2_inspirehand/`.

**Pipeline**: Stereo cameras → ROS → image preprocessing → student policy inference → joint commands → FR3 + AgileHand

## Hardware

- **Arm**: Franka Research 3 (FR3), 7 DOF, controlled via `franka_ros` / FCI
- **Hand**: AgileHand (TekkenADoF), direct joint position control
- **Cameras**: Stereo pair (e.g., Intel RealSense D415 or OV9732 modules)
- **Training resolution**: 320×240 (images resized at inference time)

## Setup (Docker, ROS1)

Uses a ROS1 Noetic container (see `Dockerfile` + `docker-compose.yml`).

### Prerequisites

- Docker + Docker Compose v2 (`docker compose`)
- (Optional) NVIDIA Container Toolkit for GPU acceleration
- X11 enabled on the host for GUI tools (RViz, etc.)
- `franka_ros` installed for FR3 communication

## Quick Start

1. Build the Docker image:

```bash
cd dextrah_lab/deployment_fr3_agilehand
./scripts/build.sh
```

2. Start the container:

```bash
./scripts/run_with_robot.sh
```

3. Inside the container, build the catkin workspace:

```bash
cd /workspace/ws
catkin_make
source devel/setup.bash
```

4. Launch components (in separate terminals):

```bash
# Terminal 1: Stereo camera publisher
roslaunch stereo_camera stereo_ros_publisher.launch

# Terminal 2: Policy inference
rosrun inference policy_inference_stereo.py \
  _checkpoint:=<path_to_student.pth>

# Terminal 3: FR3 control (via franka_ros)
roslaunch franka_control franka_control.launch robot_ip:=<FR3_IP>
```

## Camera Calibration (Hand-Eye)

Estimates the camera-to-robot transform using an AprilTag and robot joint states.

### Usage

```bash
python calibration/camera_calibration.py \
  --camera left \
  --home-pose x y z yaw pitch roll \
  --target-pose x y z yaw pitch roll \
  --joint-state-topic /franka_state_controller/joint_states \
  --tag-frame-id tag36h11:0
```

### Outputs

- `robot_cam_<camera>_calibration.txt` (robot→camera 4×4 transform)

## Inference Pipeline

The student policy inference node (`ws/src/inference/policy_inference_stereo.py`):

1. Subscribes to `/stereo/left/image_raw` and `/stereo/right/image_raw`
2. Resizes images to 320×240 (training resolution)
3. Normalizes and converts to tensors
4. Runs student policy forward pass (stereo transformer)
5. Publishes joint commands to `/fr3_agilehand/joint_commands`

### Joint Mapping

| Index | Joint Name | DOF |
|-------|-----------|-----|
| 0-6 | `fr3_joint1` – `fr3_joint7` | FR3 arm (7) |
| 7 | `revolute_thumb_rot` | Thumb rotation |
| 8-10 | `revolute_thumb_mcp_pitch/yaw`, `revolute_thumb_pip` | Thumb (3) |
| 11-13 | `revolute_index_mcp_pitch/yaw`, `revolute_index_pip` | Index (3) |
| 14-16 | `revolute_middle_mcp_pitch/yaw`, `revolute_middle_pip` | Middle (3) |
| 17-19 | `revolute_ring_mcp_pitch/yaw`, `revolute_ring_pip` | Ring (3) |
| 20-22 | `revolute_pinky_mcp_pitch/yaw`, `revolute_pinky_pip` | Pinky (3) |

Total: 23 DOF (7 arm + 16 hand)

## Differences from TG2 Deployment

| Aspect | TG2 + InspireHand | FR3 + AgileHand |
|--------|-------------------|-----------------|
| Arm control | bodyctrl_msgs (motor IDs) | franka_ros / FCI |
| Hand DOF | 6 (InspireHand) | 16 (AgileHand) |
| Joint feedback | `/arm/status` (MotorStatusMsg) | `/franka_state_controller/joint_states` |
| URDF | `tg2_with_hands_no_legs.urdf` | `FR3_tekkenadof_left.usd` / URDF |
| Camera mount | Fixed on TG2 body | External mount (needs calibration) |
