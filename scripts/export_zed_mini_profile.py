#!/usr/bin/env python3
"""Read rectified LEFT intrinsics from a connected ZED Mini into a training profile.

Run with the ZED SDK's Python environment on the camera computer. Opening the
camera performs SDK self-calibration; this script does not move a robot.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_planning.rl.zed_mini import DEFAULT_ZED_PROFILE, load_zed_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_ZED_PROFILE)
    parser.add_argument("--resolution", choices=("VGA", "HD720", "HD1080"), default="VGA")
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=1.0)
    args = parser.parse_args()
    import pyzed.sl as sl

    profile = load_zed_profile(args.template)
    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = args.depth_min
    init.depth_maximum_distance = args.depth_max
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    try:
        info = camera.get_camera_information()
        if info.camera_model != sl.MODEL.ZED_M:
            raise ValueError(f"Connected camera is {info.camera_model}, not ZED Mini")
        config = info.camera_configuration
        calibration = config.calibration_parameters
        left = calibration.left_cam
        profile.update(
            serial_number=int(info.serial_number),
            calibration_status="zed_sdk_rectified_left",
            source_width=int(config.resolution.width),
            source_height=int(config.resolution.height),
            fx=float(left.fx),
            fy=float(left.fy),
            cx=float(left.cx),
            cy=float(left.cy),
            stereo_baseline_m=float(calibration.get_camera_baseline()),
            depth_min_m=args.depth_min,
            depth_max_m=args.depth_max,
            notes="Rectified LEFT SDK calibration. User mount/TCP retained from template; validate physically.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(profile, indent=2) + "\n")
        load_zed_profile(args.output)
        print(f"Saved {args.output}; rebuild goal catalogs before using this new camera profile.")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
