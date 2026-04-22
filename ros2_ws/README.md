# ROS2 Workspace

This workspace contains the hardware-facing ROS2 package for this repo.

Current package:
- `robot_integration_ros`: MoveIt-based FR3 end-effector motion entrypoint for real-robot testing

## Build

Source ROS2 and your robot / MoveIt workspace first, then build this overlay:

```bash
source /opt/ros/<distro>/setup.bash
source /path/to/your/fr3_moveit_ws/install/setup.bash

cd /media/pdz/Elements1/robot_integration/ros2_ws
colcon build --packages-select robot_integration_ros --symlink-install
source install/setup.bash
```

## Run

Use two fresh terminals. In both terminals, use the same private ROS domain so you do not collide with other `move_group` instances on the network.

Terminal 1: launch the FR3 MoveIt stack

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1

ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=<robot_ip> use_fake_hardware:=false
```

Terminal 2: source this overlay and run the motion script

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash
source /media/pdz/Elements1/robot_integration/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
```

Optional check:

```bash
ros2 control list_controllers
```

You want `fr3_arm_controller` to be `active`.

Plan-only:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation
```

Execute on hardware:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation --execute
```

Notes:
- The package expects the FR3 MoveIt stack to already be running and exposing `/compute_ik`, `/plan_kinematic_path`, and `/execute_trajectory`.
- The package uses `fr3_arm` and `fr3_hand_tcp` by default.
- `--keep-current-orientation` reuses the current EE orientation from MoveIt FK.
- The default velocity and acceleration scaling are both `0.05` to keep motion conservative.
- `Ctrl-C` only sends a best-effort cancel to MoveIt. Do not rely on it as a safety stop; use the robot-side stop/pause for emergency interruption.
- The repo-root helper script `scripts/move_real_robot_ee.py` is a thin shim around this package so the CLI stays consistent.
