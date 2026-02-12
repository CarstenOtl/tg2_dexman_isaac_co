# Debug Tools for FR3 Agile Hand

This directory contains diagnostic and debugging scripts for the FR3 Agile Hand task.

## Palm Orientation Tools

Tools to help determine and verify the correct `palm_down_local_axis` configuration:

- **`show_local_frames.py`** - Visualizes the palm's local coordinate frame (RGB axes) in the simulator
  ```bash
  python -m dextrah_lab.tasks.fr3_agilehand.debug_tools.show_local_frames --livestream 1
  ```

- **`find_palm_orientation.py`** - Analyzes palm orientation and recommends the best `palm_down_local_axis`
  ```bash
  python -m dextrah_lab.tasks.fr3_agilehand.debug_tools.find_palm_orientation --headless
  ```

- **`determine_palm_axis.py`** - Interactive tool to determine correct palm axis
  ```bash
  python -m dextrah_lab.tasks.fr3_agilehand.debug_tools.determine_palm_axis --headless
  ```

- **`check_palm_orientation.py`** - Quick check if palm orientation is correctly configured
  ```bash
  python -m dextrah_lab.tasks.fr3_agilehand.debug_tools.check_palm_orientation --headless
  ```

- **`visualize_local_axes.py`** - Shows colored spheres at the end of each local axis
  ```bash
  python -m dextrah_lab.tasks.fr3_agilehand.debug_tools.visualize_local_axes --livestream 1
  ```

- **`visualize_palm_orientation.py`** - Visualizes palm orientation during episodes
- **`visual_palm_debug.py`** - Visual debugging for palm orientation issues

## Robot & Physics Debugging

- **`check_articulation.py`** - Verifies robot articulation and joint configuration
- **`check_robot_physics.py`** - Checks robot physics properties
- **`compare_robot_physics.py`** - Compares physics settings between configurations
- **`compare_usd_joints.py`** - Compares joint configurations in USD files
- **`inspect_usd.py`** - Inspects USD file structure and properties

## Training & Reward Debugging

- **`debug_rewards.py`** - Monitors and debugs reward computations
- **`diagnose_training.py`** - Diagnoses training issues
- **`monitor_resets.py`** - Monitors episode resets and termination conditions

## Testing Scripts

- **`test_action_response.py`** - Tests robot response to actions
- **`test_dummy_grasp.py`** - Tests basic grasping functionality
- **`zero_agent.py`** - Tests environment with zero actions

## Utilities

- **`fix_fr3_usd.py`** - Utility to fix/modify FR3 USD files

## Usage Notes

All scripts can be run as Python modules from the repository root:

```bash
python -m dextrah_lab.tasks.fr3_agilehand.debug_tools.<script_name> [options]
```

Common options:
- `--headless` - Run without GUI
- `--livestream 1` - Enable WebRTC streaming for remote access
- `--num_envs N` - Set number of parallel environments
