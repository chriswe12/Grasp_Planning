# KUKA Dual-Arm — Quick Startup Cheatsheet

(Full details: `KUKA_dual_arm_bringup_README.md`)

## Persistent-stack simulation

Terminal 1:

```bash
export ROS_DOMAIN_ID=1
./run_pipeline.sh --mode sim --robots both --bringup-only --rviz
```

Terminal 2:

```bash
export ROS_DOMAIN_ID=1
./run_pipeline.sh \
  --mode sim \
  --robots both \
  --pair-id p001_h0450_i0_0422 \
  --pickup-x 0.55 \
  --pickup-y 0.28
```

Add `--headless` to suppress the Isaac window. The task reuses the persistent
dual mock MoveIt stack, plans all holder and inserter phases, and executes the
saved waypoints in Isaac without stopping MoveIt. Use `--start-moveit` only
when a temporary stack owned by that task command is wanted. It uses
`DUAL_ROBOT_ROS_DOMAIN_ID` when set,
then an existing `ROS_DOMAIN_ID`, and finally domain `0`; it also applies
matching Fast DDS discovery settings.

To run mock planning while another operator uses domain `0` for the real
robots, keep both mock terminals on another domain:

```bash
# Terminal 1
export ROS_DOMAIN_ID=1
./run_pipeline.sh --mode sim --robots both --bringup-only --rviz

# Terminal 2
export ROS_DOMAIN_ID=1
./run_pipeline.sh \
  --mode sim \
  --robots both \
  --pair-id p001_h0450_i0_0422
```

Passing `--ros-domain-id 1` to both commands is equivalent and takes
precedence over the environment.

Reusing a matching stack is the default. If it is absent, the task fails fast;
it does not silently bring up or tear down robot control.

## Real holder/pickup vertical slice

### Gripper computer

Run:

```bash
cd ~/Workspaces/new_gripper/ros2
source servo_gripper/setup_gripper_env.sh
ros2 launch servo_gripper dual_grippers.launch.py
```

The verified physical mapping is:

- `lbr_one` / left: USB adapter `5B3D047592` (`ttyACM0`), servo ID 1
- `lbr_two` / right: USB adapter `5B3D044069` (`ttyACM1`), servo ID 1

The launch exposes `/left/gripper_controller/*` and
`/right/gripper_controller/*`, including `calibrate`, `open`, `close`, `stop`,
`position_command`, and `position`. With empty jaws, calibrate the two sides
separately before execution:

```bash
ros2 service call /left/gripper_controller/calibrate std_srvs/srv/Trigger "{}"
ros2 service call /right/gripper_controller/calibrate std_srvs/srv/Trigger "{}"
```

The real executor routes task roles through robot identity, so a swapped plan
still maps `lbr_one` to `/left` and `lbr_two` to `/right`.
MoveIt continuously republishes each gripper's last measured normalized
`position` as its passive finger joint. This remains correct after the gripper
finishes moving; before the first feedback sample the scene uses a warned,
fully-open fallback.

### Robot computer

After starting the `LBRServer` app on both SmartPADs, run the non-moving live
target preflight:

```bash
./run_pipeline.sh \
  --mode real \
  --robots both \
  --pair-id p001_h0450_i0_0422
```

The command reuses the separately started hardware MoveIt stack and builds a
fresh target task. Before either arm moves, the real executor plans the complete connected
sequence from the live shared joint state. Grasp approaches, pickup lift, and
the final pre-insertion descent are collision-aware straight Cartesian TCP
paths; other transfers are free-space MoveIt plans. Hardware execution uses
those exact preflight trajectories and aborts if either arm has drifted from a
saved segment start. It never replays mock or Isaac joint waypoints.

Ramp execution one stop point at a time. This first motion command moves only
the holder to pregrasp and does not actuate either gripper:

```bash
./run_pipeline.sh \
  --mode real \
  --robots both \
  --pair-id p001_h0450_i0_0422 \
  --execute \
  --allow-objectless-planning \
  --stop-after holder_pregrasp \
  --skip-grippers
```

After verifying the target and both gripper namespaces, continue through the
holder close:

```bash
./run_pipeline.sh \
  --mode real \
  --robots both \
  --pair-id p001_h0450_i0_0422 \
  --execute \
  --allow-objectless-planning \
  --stop-after holder_grasp
```

The complete current vertical slice stops with the incoming part at
pre-insertion while both grippers remain closed:

```bash
./run_pipeline.sh \
  --mode real \
  --robots both \
  --pair-id p001_h0450_i0_0422 \
  --execute \
  --allow-objectless-planning \
  --stop-after inserter_preinsertion
```

If an active role's namespaced gripper controller is not connected, the real
executor fails before either arm moves. A controller command failure also
aborts the run. The calibrated service contract is 7 mm closed, 74 mm open,
with `0.0=open` and `1.0=closed` on `position_command`.

Hardware execution retains a typed confirmation and defaults to 5% velocity
and acceleration scaling. Add `--rviz` when a live RViz view is useful. MoveIt
includes the table and both robots, but not the Fabrica object meshes. The
runtime table top is calibrated to `base_link z=-0.003 m`; its 50 mm collision
box is centered at `z=-0.028 m`. The
measured 840 mm base transform, physical assembly/pickup placement,
USB-to-robot gripper mapping, and clear approach paths must therefore be
confirmed at the cell before typing `yes`.

The gripper program still runs on its separate computer; one local robot-side
command covers MoveIt startup, target creation, live planning, and execution.

## Dual `GraspAssembly` action

The same `/grasp_assembly` interface used by the single-robot adapter now has a
dual mode for the validated holder/pickup-to-pre-insertion slice. The action
uses all five goal fields and accepts the current physical mapping only:

- `base_part_id: 2`, `holder_robot: left` -> `lbr_one`;
- `insertion_part_id: 0`, `inserter_robot: right` -> `lbr_two`.

Start dual MoveIt separately. For perception-in-the-loop Isaac on an isolated
domain:

```bash
# Terminal 1
export ROS_DOMAIN_ID=1
./run_pipeline.sh --mode sim --robots both --bringup-only --rviz

# Terminal 2; the perception publisher must also use domain 1
export ROS_DOMAIN_ID=1
./run_pipeline.sh --mode pitl --robots both --serve-action --headless
```

For hardware, use domain `0`, start both SmartPAD `LBRServer` apps and the
remote dual-gripper launcher, start dual MoveIt with `--mode hardware`, then
start the action adapter:

```bash
export ROS_DOMAIN_ID=0
./run_pipeline.sh --mode real --robots both --serve-action --execute \
  --allow-objectless-planning
```

The two server flags are persistent authorization for accepted action goals.
Dual real goals therefore do not show the standalone executor's typed prompt.
Do not leave an execution-enabled server running in an uncleared cell.

Send the currently supported goal from a third terminal on the same domain:

```bash
ros2 action send_goal --feedback \
  /grasp_assembly \
  fp_debug_msgs/action/GraspAssembly \
  "{assembly_name: plumbers_block, base_part_id: 2, insertion_part_id: 0, holder_robot: left, inserter_robot: right}"
```

The adapter waits for current `DebugPoseItem` messages for parts `2` and `0`,
derives the assembly/pickup frames, validates the selected-order step, and
invokes the same dual planner and PITL/real executor selected by
`run_pipeline.sh`. It returns the Isaac-measured pose in PITL and the
commanded pre-insertion source-frame pose in the current real slice. Actual
insertion, release, and retreat remain outside this action.

## 0. Pre-flight

- [ ] Both pendants set to **T1** mode.
- [ ] Right robot (`lbr_two`, 192.170.20.2) `LBRServer` app port = **30201**.
- [ ] Left robot (`lbr_one`, 192.170.10.2) `LBRServer` app port = **30200** (default).

| Arm   | Controller IP | ROS name  | FRI port |
|-------|----------------|-----------|----------|
| Left  | 192.170.10.2   | `lbr_one` | 30200    |
| Right | 192.170.20.2   | `lbr_two` | 30201    |

## Changes vs. base package (for porting to another machine)

Base repo: `lbr-stack/lbr_fri_ros2_stack`, branch `humble` (dual-arm demo added
upstream via PR #386, so most of `lbr_dual_arm` ships as-is). Only one item
below is an actual code diff to re-apply; the rest are preconditions to
verify match on the new machine.

| # | Item | File / location | Action needed on new machine |
|---|------|------------------|-------------------------------|
| 1 | Base joint offsets: `lbr_one_base_joint` xyz `0 0.5 0` → `0 -0.42 0`; `lbr_two_base_joint` xyz `0 -0.5 0` → `0 0.42 0` (swapped sign, magnitude = half of measured 0.84 m spacing) | `lbr_dual_arm_description/urdf/lbr_dual_arm.xacro` | **Apply this edit** — it is an uncommitted local diff vs. upstream (`git diff` confirms it's the only modified file in the tree). Not yet re-validated against the physical rig; confirm real X/Z offsets are 0 before trusting it as final. |
| 2 | `port_id: 30200` (lbr_one) / `port_id: 30201` (lbr_two) | `lbr_one_system_config.yaml` / `lbr_two_system_config.yaml` | No file change needed — already correct upstream. Just confirm these values match the pendant-side port config (item 4). |
| 3 | PC secondary IPs `192.170.10.1/24` + `192.170.20.1/24` on the FRI-facing NIC (`enp3s0` here) | OS network config, not in any repo file | Configure the equivalent secondary IPs on whatever interface name the new machine uses. |
| 4 | Right controller's `LBRServer` app port hardcoded 30200 → 30201 (`setPortOnRemote`-style value in `LBRServer.java`, Sunrise Workbench project), then re-synced to the controller | On the KUKA controller itself, not in this git repo | Only needed if these are new/re-flashed controllers — the physical controllers already carry this state if unchanged. Not editable from the pendant's smartHMI "Inputs" screen. |

## Command / goal frame

- **`base_link`** is the root of the URDF (no parent, no virtual joint) and is
  explicitly set as RViz's **Fixed Frame** and **Target Frame**
  (`lbr_dual_arm_moveit_config/config/moveit.rviz`). With no separate `world`
  frame declared, it is the de facto planning/reference frame for this setup.
- It sits at the midpoint between the two arms per the fix above:
  `lbr_one_link_0` at `y = -0.42`, `lbr_two_link_0` at `y = +0.42`, no
  rotation.
- SRDF planning groups (`lbr_dual_arm_moveit_config/config/lbr_dual_arm.srdf`):
  - `arm_one`: chain `lbr_one_link_0` → `lbr_one_link_ee`
  - `arm_two`: chain `lbr_two_link_0` → `lbr_two_link_ee`
  - `both_arms`: `arm_one` + `arm_two` combined
- **Send single-arm goals** relative to that arm's own `_link_0`, or via
  MoveIt's planning frame directly.
- **Send bimanual/coordinated goals (`both_arms` group)** in `base_link` —
  it's the only common ancestor frame linking the two chains, so relative
  poses between the arms must be expressed there.
