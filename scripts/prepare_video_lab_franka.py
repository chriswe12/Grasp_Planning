#!/usr/bin/env python3
"""Compose the pencil-mark lab and NVIDIA Panda as a separate Isaac preview.

Run with a Python interpreter providing pxr (e.g. Blender's bundled Python).
Poses are calculated from the robot's authored joint frames, without simulation.
"""

import argparse
import json
import math
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/video_lab_franka")
p.add_argument("--base-position", type=float, nargs=3, default=(-0.48, 0.10, 0.0))
a = p.parse_args()
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
robot_asset = out / "robot/panda_instanceable.usd"
if not robot_asset.is_file():
    raise FileNotFoundError(robot_asset)
source = ROOT / "artifacts/video_lab_isaac/rl_metric/video_lab_pencil/isaac_preview.usda"
stage = Usd.Stage.CreateNew(str(out / "pencil_franka.usda"))
stage.GetRootLayer().subLayerPaths = [os.path.relpath(source, out)]
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, "Z")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
robot = UsdGeom.Xform.Define(stage, "/World/Franka")
robot.GetPrim().GetReferences().AddReference(os.path.relpath(robot_asset, out))
robot.AddTranslateOp().Set(Gf.Vec3d(*a.base_position))

# Same ready pose and open fingers as FrankaTrainingSceneCfg.
joints = {f"panda_joint{i + 1}": q for i, q in enumerate([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])}
joints.update(panda_finger_joint1=0.04, panda_finger_joint2=0.04)


def frame(pos, rot):
    m = Gf.Matrix4d(1.0)
    m.SetRotate(Gf.Quatd(rot))
    m.SetTranslateOnly(Gf.Vec3d(pos))
    return m


# All rigid link transforms are siblings beneath the articulation root.
transforms = {"/World/Franka/panda_link0": Gf.Matrix4d(1.0)}
pending = [UsdPhysics.Joint(p) for p in Usd.PrimRange(robot.GetPrim()) if p.IsA(UsdPhysics.Joint)]
while pending:
    progress = False
    for joint in list(pending):
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0:
            # Fixed-base anchor in world coordinates, matching the table mount.
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*a.base_position))
            pending.remove(joint)
            progress = True
            continue
        if str(body0[0]) not in transforms:
            continue
        name = joint.GetPrim().GetName()
        q = joints.get(name, 0.0)
        f0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        f1 = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        motion = Gf.Matrix4d(1.0)
        if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(joint.GetPrim())
            axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[j.GetAxisAttr().Get()]
            angle = math.degrees(q)
            assert j.GetLowerLimitAttr().Get() <= angle <= j.GetUpperLimitAttr().Get(), name
            motion.SetRotate(Gf.Rotation(axis, angle))
            UsdPhysics.DriveAPI(joint.GetPrim(), "angular").GetTargetPositionAttr().Set(angle)
        elif joint.GetPrim().IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.PrismaticJoint(joint.GetPrim())
            axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[j.GetAxisAttr().Get()]
            assert j.GetLowerLimitAttr().Get() - 1e-6 <= q <= j.GetUpperLimitAttr().Get() + 1e-6
            motion.SetTranslate(axis * q)
            UsdPhysics.DriveAPI(joint.GetPrim(), "linear").GetTargetPositionAttr().Set(q)
        child = str(body1[0])
        transforms[child] = f1.GetInverse() * motion * f0 * transforms[str(body0[0])]
        pending.remove(joint)
        progress = True
    if not progress:
        raise RuntimeError("Disconnected Panda joint graph")
for path, transform in transforms.items():
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp(opSuffix="previewPose").Set(transform)

# Disable the former cameras; the overview must include the arm's full height.
stage.GetPrimAtPath("/World/Cameras").SetActive(False)


def camera(name, eye, target, focal=40.0):
    cam = UsdGeom.Camera.Define(stage, "/World/Inspection/" + name)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(27.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 30.0))
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    return cam


camera("Overview", (1.65, -2.35, 1.65), (-0.04, 0.04, 0.24), 43.0)
camera("WorkArea", (0.94, -1.42, 1.07), (-0.05, 0.02, 0.23), 38.0)
camera("RobotSide", (-1.3, -1.85, 1.15), (-0.12, 0.04, 0.27), 40.0)

# Provisional wrist camera from the existing project profile (ROS -> USD optical).
profile = json.loads((ROOT / "configs/franka_zed_mini.json").read_text())
cam = UsdGeom.Camera.Define(stage, "/World/Franka/panda_hand/WristCamera")
q = Gf.Quatd(profile["quaternion_wxyz"][0], Gf.Vec3d(*profile["quaternion_wxyz"][1:]))
ros_to_usd = Gf.Matrix4d(1.0)
ros_to_usd.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0))
cam.AddTransformOp().Set(ros_to_usd * frame(profile["position_m"], q))
# Same optical field of view, 4:3 inspection output; no claim of sensor calibration.
cam.CreateFocalLengthAttr(3.06)
cam.CreateHorizontalApertureAttr(4.8)
cam.CreateVerticalApertureAttr(3.6)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

stage.GetRootLayer().Save()
hand_world = (
    UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Franka/panda_hand")).ExtractTranslation()
)
report = {
    "robot": "NVIDIA Isaac Lab Franka Emika Panda with Panda hand",
    "scene": str(source.relative_to(ROOT)),
    "base_position_m": a.base_position,
    "joint_positions_rad_or_m": joints,
    "hand_position_m": list(hand_world),
    "physics_stepped": False,
    "wrist_camera": "Provisional project mount, 4:3 inspection crop, not a calibrated training capture",
    "usd": "pencil_franka.usda",
    "source_scenes_modified": False,
}
(out / "scene_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
