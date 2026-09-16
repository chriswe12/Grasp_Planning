# PDZ gripper ROS 2 description

Standalone visual and collision description for the belt-driven PDZ Slim
parallel gripper and its mounted Intel RealSense D405. The meshes were generated
from `PDZ_Gripper_Slim(3).STEP`, exported from SolidWorks 2022 using the
`Flange Mounting Point` output coordinate system.

This revision retains the Slim(2) D405 bracket and simplified servo motor, and
adds four heatsink solids: two at the camera and two at the motor. The STEP
contains 29 solids. Small solid-geometry revisions to the tensioner bracket
and camera bracket are also included; their poses and outer dimensions are
unchanged.

## Model conventions

- The base-link origin is the robot flange mounting point.
- `+Z` points outward from the robot flange through the gripper.
- `+X` is the finger opening direction.
- `pdz_gripper_left_finger_joint` is the independent joint.
- The right joint mimics the left joint because the GT2 belt moves both
  fingers symmetrically.
- Joint position `0 m` is the bare-finger closed pose; `0.032 m` is fully open.
- Bare-finger separation is twice the independent joint position, with a
  maximum separation of `0.064 m`.
- Trapezoidal TPU pads are parameterized from `0.005` to `0.014 m` thickness.
  Their mounting faces remain fixed as thickness changes.
- Pad thicknesses through 14 mm remain inside the finger recess and therefore
  do not change the finger joint limits or bare-finger opening range.
- With pad thickness `t` and finger-joint position `q`, the nominal pad-to-pad
  gap is `0.028 - 2*t + 2*q` metres. The 8 mm Slim pads therefore give a
  12 mm closed gap and 76 mm fully-open gap.
- `pdz_gripper_tcp` is centered between the Slim pads at their vertical
  midpoint, `[0, 0, 0.1355] m` in the flange frame.
- The D405 uses the standard `camera_link`, `camera_depth_frame`, and
  `camera_depth_optical_frame` names when `prefix` is empty.

The flexible GT2 belt supplied in the Slim STEP is included in the visual mesh
but intentionally excluded from collision geometry. Finger linear bearings are
visible but excluded from finger collision meshes to avoid permanent
self-collisions with the guide rods. The rigid heatsinks are included in both
visual and collision geometry. The pad meshes come from the 8 mm Slim reference
pads in the CAD assembly and are scaled only along their thickness axis while
keeping their mounting faces fixed.

Gripper inertial data and `ros2_control` interfaces are intentionally omitted.
The included upstream D405 description contains a nominal 72 g mass and marks
its inertia values as unreliable; do not use those values as measured load
data for robot dynamics. Add measured gripper inertials after the physical
assembly and actuator interface are finalized.

The bare-finger zero and 32 mm stroke are carried over from the previous CAD
revision because the fully-open finger X positions and pad geometry are
unchanged. Verify these against the assembled mechanism if the belt routing or
physical stops introduce a different mechanical limit.

## Build and display

Copy this package into a ROS 2 workspace under `src`, then run:

```bash
colcon build --packages-select pdz_gripper_description
source install/setup.bash
ros2 launch pdz_gripper_description display.launch.py pad_thickness:=0.008
```

Use the joint-state-publisher slider from `0` to `0.032 m` to close and open
the gripper.

`urdf/pdz_gripper.urdf` is an expanded standalone URDF for consumers that do
not use xacro. `urdf/pdz_gripper.urdf.xacro` produces the same standalone
model, while `urdf/pdz_gripper_macro.xacro` is intended for integration into a
larger robot description. The checked-in expanded URDF uses the default 8 mm
pads and includes nominal D405 frames. Generate a different fixed URDF with:

```bash
xacro urdf/pdz_gripper.urdf.xacro \
  use_pads:=true pad_thickness:=0.005 \
  -o urdf/pdz_gripper_5mm.urdf
```

Set `use_camera:=false` to omit the D405. For an offline model or RViz without
the camera driver, leave `camera_use_nominal_extrinsics:=true`. When the real
`realsense2_camera` node publishes its calibrated internal TFs, use
`camera_use_nominal_extrinsics:=false` to prevent duplicate publishers. The
URDF will still publish the fixed gripper-to-`camera_link` mounting transform.

See [D405_FRAME_REFERENCE.md](D405_FRAME_REFERENCE.md) for the CAD transform,
frame definitions, and verification details.

See [GEOMETRY_STATISTICS.md](GEOMETRY_STATISTICS.md) for overall dimensions,
CAD solid volume, convex-hull volume, definitions, and calculation settings.

See [COORDINATE_FRAME_AUDIT.md](COORDINATE_FRAME_AUDIT.md) for the frame audit
across every STEP export and the list of files that did not use the flange
output coordinate system.

## Include in another robot

Include the reusable macro and attach it to the robot's flange link:

```xml
<xacro:include filename="$(find pdz_gripper_description)/urdf/pdz_gripper_macro.xacro"/>

<xacro:pdz_gripper parent="lbr_link_ee" prefix="">
  <origin xyz="0 0 0" rpy="0 0 0"/>
</xacro:pdz_gripper>
```

To select a pad thickness or omit pads in a larger xacro:

```xml
<xacro:pdz_gripper
  parent="lbr_link_ee"
  prefix=""
  use_pads="true"
  pad_thickness="0.008"
  use_camera="true"
  camera_use_nominal_extrinsics="false">
  <origin xyz="0 0 0" rpy="0 0 0"/>
</xacro:pdz_gripper>
```

Supported pad thickness is 0.005-0.014 m. Values above 0.014 m are not modeled
because they would require a thickness-dependent closing limit.

The mounting origin should normally remain zero because the CAD meshes are
already expressed in the flange coordinate system.
