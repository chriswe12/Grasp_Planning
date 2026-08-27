"""Launch the canonical MuJoCo renderer for one MoveIt-selected live grasp."""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from grasp_planning.rl.d405_policy_runtime import D405RuntimeGoal
from grasp_planning.ros2.d405_visual_servo import D405VisualServoDeploymentConfig
from grasp_planning.subprocess_lifecycle import run_process_group


@dataclass(frozen=True)
class RenderedD405Goal:
    path: Path
    goal_id: str
    grasp_id: str
    part_id: str
    sha256: str


def _csv(values: Sequence[float]) -> str:
    return ",".join(f"{float(value):.17g}" for value in values)


def render_d405_goal_for_grasp(
    *,
    config_path: str | Path,
    stage2_bundle_path: str | Path,
    grasp_id: str,
    part_id: str,
    goal_joint_positions: Sequence[float],
    goal_tcp_position: Sequence[float],
    goal_tcp_orientation_xyzw: Sequence[float],
    approach_width_m: float,
    maximum_approach_width_m: float,
    output_path: str | Path,
    object_position: Sequence[float] | None = None,
    object_orientation_xyzw: Sequence[float] | None = None,
) -> RenderedD405Goal:
    """Render and validate one goal image after MoveIt has selected the grasp."""

    config = D405VisualServoDeploymentConfig.from_yaml(config_path)
    for label, path in (
        ("goal_renderer_launcher", config.goal_renderer_launcher),
        ("goal_renderer_script", config.goal_renderer_script),
        ("goal_renderer_robot_urdf", config.goal_renderer_robot_urdf),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    joints = tuple(float(value) for value in goal_joint_positions)
    position = tuple(float(value) for value in goal_tcp_position)
    orientation = tuple(float(value) for value in goal_tcp_orientation_xyzw)
    if len(joints) != 7 or len(position) != 3 or len(orientation) != 4:
        raise ValueError("Goal rendering requires 7 joints, a 3D TCP position, and an XYZW quaternion.")
    approach_width = float(approach_width_m)
    maximum_approach_width = float(maximum_approach_width_m)
    if not math.isfinite(maximum_approach_width) or maximum_approach_width <= 0.0:
        raise ValueError("Maximum approach width must be finite and positive.")
    if (
        not math.isfinite(approach_width)
        or approach_width <= 0.0
        or approach_width > maximum_approach_width + 1.0e-9
    ):
        raise ValueError(
            "Selected grasp does not fit the physical gripper approach aperture: "
            f"requested={approach_width:.6f} m maximum={maximum_approach_width:.6f} m."
        )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.goal_renderer_launcher),
        *shlex.split(config.goal_renderer_python_command),
        str(config.goal_renderer_script),
        "--input-json",
        str(Path(stage2_bundle_path).expanduser().resolve()),
        "--grasp-id",
        str(grasp_id),
        "--part-id",
        str(part_id),
        # Keep signed CSV vectors attached to their option.  argparse treats a
        # separate token beginning with '-' as another option, which broke
        # valid joint/TCP/quaternion values whose first component was negative.
        f"--goal-joint-positions={_csv(joints)}",
        f"--goal-tcp-position={_csv(position)}",
        f"--goal-tcp-orientation-xyzw={_csv(orientation)}",
        "--approach-width-m",
        f"{approach_width:.17g}",
        "--maximum-approach-width-m",
        f"{maximum_approach_width:.17g}",
        "--robot-urdf",
        str(config.goal_renderer_robot_urdf),
        "--camera-profile",
        str(config.expected_camera_profile),
        "--renderer-backend",
        str(config.goal_renderer_backend),
        "--output",
        str(output),
    ]
    if (object_position is None) != (object_orientation_xyzw is None):
        raise ValueError("Explicit object position and orientation must be provided together.")
    if object_position is not None and object_orientation_xyzw is not None:
        object_position_values = tuple(float(value) for value in object_position)
        object_orientation_values = tuple(float(value) for value in object_orientation_xyzw)
        if len(object_position_values) != 3 or len(object_orientation_values) != 4:
            raise ValueError("Explicit object pose requires a 3D position and an XYZW quaternion.")
        command.extend(
            (
                f"--object-position={_csv(object_position_values)}",
                f"--object-orientation-xyzw={_csv(object_orientation_values)}",
            )
        )
    returncode = run_process_group(
        command,
        cwd=Path(__file__).resolve().parents[2],
        timeout_s=float(config.goal_renderer_timeout_s),
    )
    if returncode != 0:
        raise RuntimeError(
            f"On-demand D405 goal rendering failed with exit code {returncode}: "
            f"{' '.join(shlex.quote(value) for value in command)}"
        )
    if not output.is_file():
        raise RuntimeError(
            "On-demand D405 goal renderer produced no goal artifact even though its launcher "
            f"returned success. The MuJoCo traceback above is the root failure; expected output: {output}"
        )

    try:
        rendered = D405RuntimeGoal(
            output,
            expected_camera_profile=config.expected_camera_profile,
            expected_observation_profile=config.expected_observation_profile,
        )
    except Exception as exc:
        raise RuntimeError(
            "On-demand D405 goal renderer wrote an artifact that failed strict validation. "
            f"The MuJoCo traceback above is the root failure; diagnostic artifact: {output}"
        ) from exc
    goal = rendered.load(expected_grasp_id=str(grasp_id), expected_part_id=str(part_id))
    return RenderedD405Goal(
        path=output,
        goal_id=goal.goal_id,
        grasp_id=goal.grasp_id,
        part_id=goal.part_id,
        sha256=rendered.sha256,
    )


__all__ = ["RenderedD405Goal", "render_d405_goal_for_grasp"]
