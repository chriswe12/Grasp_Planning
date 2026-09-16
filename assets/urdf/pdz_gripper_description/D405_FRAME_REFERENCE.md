# RealSense D405 frame reference

## Intended frame chain

With an empty prefix, the gripper macro creates this fixed mechanical chain:

```text
pdz_gripper_base_link
  -> camera_bottom_screw_frame
  -> camera_link
```

When `camera_use_nominal_extrinsics:=true`, the official RealSense description
also creates the nominal stream frames, including:

```text
camera_link
  -> camera_depth_frame
  -> camera_depth_optical_frame
```

`camera_depth_optical_frame` follows the optical convention: +X right in the
image, +Y down, and +Z forward into the scene.

## CAD placement

`PDZ_Gripper_Slim(3).STEP` retains `RealSense_D405` as a separate component. Its
front-plate mesh origin has this pose relative to `pdz_gripper_base_link`, which
is the SolidWorks `Flange Mounting Point` frame:

```text
translation [m]:  0, -0.048660254038, 0.066217967697
rotation:         -30 degrees about the camera-mesh X axis
```

The official RealSense macro is attached through the physical 1/4-20 bottom
screw frame. The corresponding fixed joint in the gripper frame is:

```text
xyz [m]:  0, -0.074171787517, 0.064030695532
rpy [rad]: pi, -1.047197551197, pi/2
```

This transform, combined with the internal D405 geometry from
`realsense2_description`, reproduces the supplied CAD camera bounding box to
within approximately 0.0018 mm.

## Depth origin

The D400-series datasheet defines the D405 depth X-Y origin as the centre of
the left imager. It is 9 mm from the centreline of the 1/4-20 mounting hole.
The depth-zero plane is 3.7 mm behind the front cover glass. The official ROS
description also accounts for the cover glass being 0.1 mm behind the front
aluminium plate.

For reference, the resulting nominal depth optical origin expressed directly
in `pdz_gripper_base_link` is approximately:

```text
xyz [m]: 0.009, -0.050560254038, 0.062927071163
rpy [rad]: 0.523598775598, 0, pi
```

Use the TF tree rather than copying this direct transform into a second static
publisher.

## Real camera operation

The robot URDF must always publish the fixed physical mounting transform from
the gripper to `camera_link`. Internal stream-to-stream extrinsics can come
from one of two sources:

- Offline/RViz/simulation: set `camera_use_nominal_extrinsics:=true`.
- Real `realsense2_camera` driver with TF enabled: set
  `camera_use_nominal_extrinsics:=false` and let the driver publish calibrated
  stream transforms.

Do not enable both nominal and driver-published internal transforms under the
same frame names.

## Sources

- [Intel RealSense D400 Series Product Family Datasheet](https://dev.realsenseai.com/download/42003/),
  revision 017, sections 4.8 and 4.8.1.
- [`realsense2_description/urdf/_d405.urdf.xacro`](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d405.urdf.xacro)
  from the official `realsense-ros` package.
- [RealSense SDK 2.0 projection and coordinate conventions](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0).
