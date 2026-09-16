"""ROS2 action server that runs the real grasp pipeline for one insertion part."""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .dual_grasp_assembly_action_runner import (
    DEFAULT_DUAL_CONFIG,
    DUAL_MODES,
    DUAL_REAL_STOP_AFTER_CHOICES,
    DualPipelineRunner,
)

try:
    import rclpy
    from fp_debug_msgs.action import GraspAssembly
    from geometry_msgs.msg import Point
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from visualization_msgs.msg import Marker, MarkerArray
except Exception:  # pragma: no cover - exercised only without a sourced ROS2 overlay
    rclpy = None
    GraspAssembly = None
    Point = None
    ActionServer = None
    CancelResponse = None
    GoalResponse = None
    ReentrantCallbackGroup = None
    MultiThreadedExecutor = None
    DurabilityPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    Marker = None
    MarkerArray = None
    Node = object


DEFAULT_ACTION_NAME = "/grasp_assembly"
DEFAULT_CONFIG = Path("configs/grasp_pipeline_real_lbr_iiwa7.yaml")
ASSEMBLY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PREASSEMBLY_TRANSPORT_IMPLEMENTED = False
TRANSPORT_UNSUPPORTED_ERROR_CODE = "TRANSPORT_UNSUPPORTED"
TRANSPORT_UNSUPPORTED_MESSAGE = (
    "GraspAssembly success requires transport to the pre-assembly pose, but the current real pipeline "
    "only implements pickup and lift. No hardware motion was started."
)


@dataclass(frozen=True)
class SingleRobotGraspGoal:
    """The fields currently consumed from a GraspAssembly goal."""

    assembly_name: str
    insertion_part_id: int
    inserter_robot: str

    @classmethod
    def from_request(cls, request: Any) -> "SingleRobotGraspGoal":
        return cls(
            assembly_name=str(request.assembly_name).strip(),
            insertion_part_id=int(request.insertion_part_id),
            inserter_robot=str(request.inserter_robot).strip().lower(),
        )


@dataclass(frozen=True)
class PipelineOutcome:
    success: bool
    error_code: str
    message: str
    grasped_frame_id: str = ""
    grasped_position_xyz: tuple[float, float, float] | None = None
    grasped_orientation_xyzw: tuple[float, float, float, float] | None = None


def _find_repo_root(explicit_root: Path | None = None) -> Path:
    if explicit_root is not None:
        candidates = (Path(explicit_root).expanduser().resolve(),)
    else:
        source_path = Path(__file__).resolve()
        candidates = (Path.cwd().resolve(), source_path.parent, *source_path.parents)
    for candidate in candidates:
        if (candidate / "run_pipeline.sh").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the grasp-planning repo root containing run_pipeline.sh and configs/.")


def _resolve_config_path(repo_root: Path, config_path: Path) -> Path:
    resolved = Path(config_path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {resolved}")
    return resolved


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level YAML mapping in '{path}'.")
    return payload


def _target_mesh_relative_path(goal: SingleRobotGraspGoal) -> str:
    return f"obj/fabrica/{goal.assembly_name}/{goal.insertion_part_id}.obj"


def _target_mesh_path(repo_root: Path, goal: SingleRobotGraspGoal) -> Path:
    return repo_root / "assets" / _target_mesh_relative_path(goal)


def _action_config(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("grasp_assembly_action")
    return dict(raw) if isinstance(raw, dict) else {}


def _validate_goal(
    *,
    repo_root: Path,
    payload: dict[str, object],
    goal: SingleRobotGraspGoal,
) -> str | None:
    if not ASSEMBLY_NAME_PATTERN.fullmatch(goal.assembly_name):
        return "assembly_name must contain only letters, numbers, underscore, dot, or hyphen."
    if goal.insertion_part_id < 0:
        return "insertion_part_id must be non-negative."
    if goal.inserter_robot != "left":
        return "Only inserter_robot='left' is supported by the current single-robot server."
    mesh_path = _target_mesh_path(repo_root, goal)
    if not mesh_path.is_file():
        return f"Insertion-part mesh does not exist: {mesh_path}"
    ros2 = payload.get("ros2", {})
    if not isinstance(ros2, dict) or not str(ros2.get("pose_base_topic", "")).strip():
        return "The server pipeline config must define ros2.pose_base_topic."
    return None


def _goal_output_dir(
    *,
    payload: dict[str, object],
    goal: SingleRobotGraspGoal,
    goal_id: str,
) -> Path:
    configured_root = str(_action_config(payload).get("output_root", "artifacts/grasp_assembly_actions"))
    return Path(configured_root) / goal.assembly_name / f"part_{goal.insertion_part_id}" / goal_id


def _prepare_pipeline_payload(
    *,
    base_payload: dict[str, object],
    goal: SingleRobotGraspGoal,
    goal_id: str,
    skip_gripper: bool = False,
) -> tuple[dict[str, object], Path]:
    payload = copy.deepcopy(base_payload)
    output_dir = _goal_output_dir(payload=payload, goal=goal, goal_id=goal_id)

    geometry = dict(payload.get("geometry", {}))
    geometry["target_mesh_path"] = _target_mesh_relative_path(goal)
    geometry["assembly_glob"] = f"obj/fabrica/{goal.assembly_name}/*.obj"
    payload["geometry"] = geometry

    ros2 = dict(payload.get("ros2", {}))
    ros2.pop("object_id", None)
    ros2["assembly_name"] = goal.assembly_name
    ros2["part_id"] = goal.insertion_part_id
    payload["ros2"] = ros2

    artifacts = dict(payload.get("artifacts", {}))
    artifacts.update(
        {
            "stage1_json": str(output_dir / "stage1_assembly_grasps.json"),
            "stage1_html": str(output_dir / "stage1_assembly_grasps.html"),
            "stage2_json": str(output_dir / "stage2_ground_feasible.json"),
            "stage2_html": str(output_dir / "stage2_ground_feasible.html"),
            "part_frame_html": str(output_dir / "part_frame_debug.html"),
        }
    )
    payload["artifacts"] = artifacts

    real_execution = dict(payload.get("real_execution", {}))
    real_execution.update(
        {
            "enabled": True,
            "require_confirmation": False,
            "stop_after": "lift",
            "gripper_enabled": not skip_gripper,
            "attempt_artifact": str(output_dir / "real_robot_pick_attempt.json"),
        }
    )
    payload["real_execution"] = real_execution
    return payload, output_dir


def _error_code_from_output(output_lines: list[str]) -> str:
    text = "\n".join(output_lines).lower()
    if "timed out" in text and "object pose" in text:
        return "PERCEPTION_TIMEOUT"
    if "no feasible grasps" in text or "contains no feasible grasps" in text:
        return "PLANNING_FAILED"
    return "PIPELINE_FAILED"


def _normalized_error_code(status: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(status).upper()).strip("_")
    return normalized or fallback


def _outcome_from_attempt(
    *,
    return_code: int,
    attempt_path: Path,
    frame_id: str,
    output_lines: list[str],
    cancelled: bool,
) -> PipelineOutcome:
    if cancelled:
        return PipelineOutcome(False, "CANCELLED", "Grasp pipeline cancelled.")

    attempt: dict[str, object] | None = None
    if attempt_path.is_file():
        raw_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        if isinstance(raw_attempt, dict):
            attempt = raw_attempt

    if return_code != 0:
        if attempt is not None and isinstance(attempt.get("result"), dict):
            result = dict(attempt["result"])
            return PipelineOutcome(
                False,
                _normalized_error_code(str(result.get("status", "")), fallback="PIPELINE_FAILED"),
                str(result.get("message", "Pipeline execution failed.")),
            )
        message = next((line for line in reversed(output_lines) if line.strip()), "Pipeline execution failed.")
        return PipelineOutcome(False, _error_code_from_output(output_lines), message)

    if attempt is None or not isinstance(attempt.get("result"), dict):
        return PipelineOutcome(
            False, "RESULT_MISSING", f"Pipeline exited successfully but no result was found at {attempt_path}."
        )

    result = dict(attempt["result"])
    if not bool(result.get("success", False)):
        return PipelineOutcome(
            False,
            _normalized_error_code(str(result.get("status", "")), fallback="GRASP_FAILED"),
            str(result.get("message", "Grasp execution failed.")),
        )

    object_pose = attempt.get("object_pose_world")
    config = attempt.get("config")
    if not isinstance(object_pose, dict) or not isinstance(config, dict):
        return PipelineOutcome(False, "RESULT_POSE_MISSING", "Successful attempt did not record the object pose.")
    position = object_pose.get("position_world")
    orientation = object_pose.get("orientation_xyzw_world")
    if not isinstance(position, list | tuple) or len(position) != 3:
        return PipelineOutcome(False, "RESULT_POSE_INVALID", "Attempt recorded an invalid object position.")
    if not isinstance(orientation, list | tuple) or len(orientation) != 4:
        return PipelineOutcome(False, "RESULT_POSE_INVALID", "Attempt recorded an invalid object orientation.")

    return PipelineOutcome(
        False,
        TRANSPORT_UNSUPPORTED_ERROR_CODE,
        f"{str(result.get('message', 'Completed pickup and lift.'))} "
        "The insertion part was not transported to its pre-assembly pose, so the GraspAssembly contract "
        "was not completed.",
    )


class RealPipelineRunner:
    """Prepare and run one real-mode pipeline subprocess at a time."""

    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        allow_execution: bool,
        skip_gripper: bool = False,
    ) -> None:
        self.repo_root = _find_repo_root(repo_root)
        self.config_path = _resolve_config_path(self.repo_root, config_path)
        self.base_payload = _load_yaml_mapping(self.config_path)
        self.allow_execution = bool(allow_execution)
        self.skip_gripper = bool(skip_gripper)
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None

    @property
    def action_name(self) -> str:
        return str(_action_config(self.base_payload).get("action_name", DEFAULT_ACTION_NAME))

    def validate(self, request: Any) -> str | None:
        return _validate_goal(
            repo_root=self.repo_root,
            payload=self.base_payload,
            goal=SingleRobotGraspGoal.from_request(request),
        )

    def request_cancel(self) -> None:
        self._signal_active_process(signal.SIGINT)

    def _signal_active_process(self, requested_signal: signal.Signals) -> None:
        with self._process_lock:
            process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            return

    def run(
        self,
        *,
        request: Any,
        goal_id: str,
        cancel_requested: Callable[[], bool],
        publish_feedback: Callable[[str, float], None],
        publish_output: Callable[[str], None],
    ) -> PipelineOutcome:
        if not self.allow_execution:
            return PipelineOutcome(
                False,
                "EXECUTION_DISABLED",
                "Server was started without --execute; no hardware command was sent.",
            )
        if not PREASSEMBLY_TRANSPORT_IMPLEMENTED:
            return PipelineOutcome(
                False,
                TRANSPORT_UNSUPPORTED_ERROR_CODE,
                TRANSPORT_UNSUPPORTED_MESSAGE,
            )

        goal = SingleRobotGraspGoal.from_request(request)
        payload, relative_output_dir = _prepare_pipeline_payload(
            base_payload=self.base_payload,
            goal=goal,
            goal_id=goal_id,
            skip_gripper=self.skip_gripper,
        )
        output_dir = relative_output_dir if relative_output_dir.is_absolute() else self.repo_root / relative_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        request_config_path = output_dir / "request_config.yaml"
        request_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        attempt_path = Path(str(dict(payload["real_execution"])["attempt_artifact"]))
        if not attempt_path.is_absolute():
            attempt_path = self.repo_root / attempt_path

        command = [
            str(self.repo_root / "run_pipeline.sh"),
            "--workflow",
            "single-object",
            "--mode",
            "real",
            "--config",
            str(request_config_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._process_lock:
            self._active_process = process

        line_queue: queue.Queue[str | None] = queue.Queue()

        def _read_output() -> None:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line_queue.put(raw_line.rstrip())
            line_queue.put(None)

        reader = threading.Thread(target=_read_output, name="grasp-pipeline-output", daemon=True)
        reader.start()
        output_lines: list[str] = []
        reader_done = False
        cancel_sent_at: float | None = None
        publish_feedback("PLANNING", 0.0)
        try:
            while process.poll() is None or not reader_done:
                try:
                    line = line_queue.get(timeout=0.1)
                except queue.Empty:
                    line = ""
                if line is None:
                    reader_done = True
                elif line:
                    output_lines.append(line)
                    publish_output(line)
                    if "Stage 1 complete" in line:
                        publish_feedback("PLANNING", 0.5)
                    elif "Planning complete" in line:
                        publish_feedback("PLANNING", 1.0)
                    elif "Starting real-robot execution" in line:
                        publish_feedback("GRASPING_PART", 0.0)
                    elif "Real execution finished success=True" in line:
                        publish_feedback("GRASPING_PART", 1.0)

                if cancel_requested() and cancel_sent_at is None:
                    cancel_sent_at = time.monotonic()
                    self._signal_active_process(signal.SIGINT)
                if cancel_sent_at is not None:
                    elapsed = time.monotonic() - cancel_sent_at
                    if elapsed > 12.0 and process.poll() is None:
                        self._signal_active_process(signal.SIGKILL)
                    elif elapsed > 8.0 and process.poll() is None:
                        self._signal_active_process(signal.SIGTERM)

            return_code = process.wait()
        finally:
            with self._process_lock:
                self._active_process = None
            if process.stdout is not None:
                process.stdout.close()
            reader.join(timeout=1.0)

        frame_id = str(dict(payload.get("real_execution", {})).get("frame_id", ""))
        return _outcome_from_attempt(
            return_code=return_code,
            attempt_path=attempt_path,
            frame_id=frame_id,
            output_lines=output_lines,
            cancelled=cancel_sent_at is not None or cancel_requested(),
        )


def _set_result_pose(
    result: Any,
    *,
    frame_id: str,
    stamp: Any,
    position_xyz: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
) -> None:
    """Populate either deployed name of the GraspAssembly result pose."""

    if hasattr(result, "achieved_part_pose"):
        pose_stamped = result.achieved_part_pose
    elif hasattr(result, "grasped_part_pose"):
        pose_stamped = result.grasped_part_pose
    else:
        raise AttributeError("GraspAssembly.Result has neither achieved_part_pose nor grasped_part_pose.")
    pose_stamped.header.frame_id = str(frame_id)
    pose_stamped.header.stamp = stamp
    (
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
    ) = position_xyz
    (
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w,
    ) = orientation_xyzw


class GraspAssemblyActionServer(Node):
    """Adapter from fp_debug_msgs/GraspAssembly to one selected pipeline."""

    def __init__(self, runner: Any, *, action_name: str, node_name: str) -> None:
        if (
            rclpy is None
            or GraspAssembly is None
            or ActionServer is None
            or GoalResponse is None
            or CancelResponse is None
            or ReentrantCallbackGroup is None
        ):
            raise RuntimeError("ROS2 or fp_debug_msgs/GraspAssembly is unavailable; source the repo ROS2 overlay.")
        super().__init__(node_name)
        self._runner = runner
        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._debug_aabb_publisher = None
        if isinstance(runner, DualPipelineRunner):
            debug_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._debug_aabb_publisher = self.create_publisher(
                MarkerArray,
                runner.debug_aabb_topic,
                debug_qos,
            )
        self._action_server = ActionServer(
            self,
            GraspAssembly,
            action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        runner_description = str(getattr(runner, "description", "single-robot"))
        if isinstance(runner, DualPipelineRunner):
            if runner.mode == "real":
                mode = "ENABLED" if runner.allow_execution else "DISABLED (start with --execute)"
            else:
                mode = "ENABLED"
        elif not PREASSEMBLY_TRANSPORT_IMPLEMENTED:
            mode = "BLOCKED (pre-assembly transport is not implemented)"
        else:
            mode = "ENABLED" if runner.allow_execution else "DISABLED"
        gripper_mode = "SKIPPED" if runner.skip_gripper else "ENABLED"
        self.get_logger().warning(
            f"GraspAssembly server ready on '{action_name}'; "
            f"adapter={runner_description}; execution is {mode}; "
            f"gripper commands are {gripper_mode}."
        )

    def _publish_debug_aabbs(
        self,
        records: tuple[dict[str, object], ...],
    ) -> None:
        if self._debug_aabb_publisher is None:
            return
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        colors = {
            "base": (0.10, 0.85, 0.45),
            "incoming": (1.00, 0.48, 0.08),
        }
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        summaries = []
        for index, record in enumerate(records):
            role = str(record["role"])
            part_id = int(record["part_id"])
            frame_id = str(record["frame_id"])
            minimum = tuple(float(value) for value in record["minimum_world_m"])
            maximum = tuple(float(value) for value in record["maximum_world_m"])
            center = tuple(0.5 * (low + high) for low, high in zip(minimum, maximum, strict=True))
            size = tuple(max(high - low, 1e-6) for low, high in zip(minimum, maximum, strict=True))
            red, green, blue = colors.get(role, (0.20, 0.65, 1.00))

            fill = Marker()
            fill.header.frame_id = frame_id
            fill.header.stamp = stamp
            fill.ns = "perceived_part_aabb_fill"
            fill.id = index * 3
            fill.type = Marker.CUBE
            fill.action = Marker.ADD
            fill.pose.position.x, fill.pose.position.y, fill.pose.position.z = center
            fill.pose.orientation.w = 1.0
            fill.scale.x, fill.scale.y, fill.scale.z = size
            fill.color.r = red
            fill.color.g = green
            fill.color.b = blue
            fill.color.a = 0.12
            marker_array.markers.append(fill)

            corners = (
                (minimum[0], minimum[1], minimum[2]),
                (maximum[0], minimum[1], minimum[2]),
                (maximum[0], maximum[1], minimum[2]),
                (minimum[0], maximum[1], minimum[2]),
                (minimum[0], minimum[1], maximum[2]),
                (maximum[0], minimum[1], maximum[2]),
                (maximum[0], maximum[1], maximum[2]),
                (minimum[0], maximum[1], maximum[2]),
            )
            wire = Marker()
            wire.header.frame_id = frame_id
            wire.header.stamp = stamp
            wire.ns = "perceived_part_aabb_wire"
            wire.id = index * 3 + 1
            wire.type = Marker.LINE_LIST
            wire.action = Marker.ADD
            wire.pose.orientation.w = 1.0
            wire.scale.x = 0.003
            wire.color.r = red
            wire.color.g = green
            wire.color.b = blue
            wire.color.a = 1.0
            for first, second in edges:
                for corner_index in (first, second):
                    point = Point()
                    point.x, point.y, point.z = corners[corner_index]
                    wire.points.append(point)
            marker_array.markers.append(wire)

            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = stamp
            label.ns = "perceived_part_aabb_label"
            label.id = index * 3 + 2
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = center[0]
            label.pose.position.y = center[1]
            label.pose.position.z = maximum[2] + 0.025
            label.pose.orientation.w = 1.0
            label.scale.z = 0.025
            label.color.r = red
            label.color.g = green
            label.color.b = blue
            label.color.a = 1.0
            label.text = f"{role} part {part_id} AABB"
            marker_array.markers.append(label)
            summaries.append(f"{role}={part_id} center={center} size={size}")

        self._debug_aabb_publisher.publish(marker_array)
        self.get_logger().info("Published perceived-part collision AABBs: " + "; ".join(summaries))

    def _goal_callback(self, request: Any):
        error = self._runner.validate(request)
        if error is not None:
            self.get_logger().warning(f"Rejecting GraspAssembly goal: {error}")
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warning("Rejecting GraspAssembly goal because another goal is active.")
                return GoalResponse.REJECT
            self._goal_active = True
        if isinstance(self._runner, DualPipelineRunner):
            summary = (
                "Accepted dual-robot goal for "
                f"{request.assembly_name}: base={request.base_part_id} "
                f"holder={request.holder_robot}, "
                f"incoming={request.insertion_part_id} "
                f"inserter={request.inserter_robot}."
            )
        else:
            summary = (
                "Accepted single-robot goal for "
                f"{request.assembly_name}/{request.insertion_part_id}; "
                "holder/base fields are ignored for now."
            )
        self.get_logger().info(summary)
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle: Any):
        self.get_logger().warning("Cancellation requested for active GraspAssembly goal.")
        self._runner.request_cancel()
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle: Any):
        result = GraspAssembly.Result()

        def _publish_feedback(phase: str, progress: float) -> None:
            feedback = GraspAssembly.Feedback()
            feedback.phase = str(phase)
            feedback.progress = float(min(max(progress, 0.0), 1.0))
            goal_handle.publish_feedback(feedback)

        def _publish_output(line: str) -> None:
            prefix = "[WARNING] "
            if line.startswith(prefix):
                self.get_logger().warning(f"[real pipeline] {line.removeprefix(prefix)}")
            else:
                self.get_logger().info(f"[real pipeline] {line}")

        try:
            goal_id = "".join(f"{byte:02x}" for byte in goal_handle.goal_id.uuid)[:16] or "unknown_goal"
            debug_kwargs = {}
            if isinstance(self._runner, DualPipelineRunner):
                debug_kwargs["publish_debug_aabbs"] = self._publish_debug_aabbs
            outcome = self._runner.run(
                request=goal_handle.request,
                goal_id=goal_id,
                cancel_requested=lambda: bool(goal_handle.is_cancel_requested),
                publish_feedback=_publish_feedback,
                publish_output=_publish_output,
                **debug_kwargs,
            )
            result.success = bool(outcome.success)
            result.error_code = str(outcome.error_code)
            result.message = str(outcome.message)
            if outcome.grasped_position_xyz is not None and outcome.grasped_orientation_xyzw is not None:
                _set_result_pose(
                    result,
                    frame_id=str(outcome.grasped_frame_id),
                    stamp=self.get_clock().now().to_msg(),
                    position_xyz=outcome.grasped_position_xyz,
                    orientation_xyzw=outcome.grasped_orientation_xyzw,
                )

            if outcome.error_code == "CANCELLED" or goal_handle.is_cancel_requested:
                goal_handle.canceled()
            elif outcome.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        except Exception as exc:
            self.get_logger().error(f"GraspAssembly execution failed: {exc}")
            result.success = False
            result.error_code = "INTERNAL_ERROR"
            result.message = str(exc)
            goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False

    def destroy_node(self):
        self._runner.request_cancel()
        self._action_server.destroy()
        return super().destroy_node()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve GraspAssembly goals through the real grasp pipeline.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Grasp-planning repository root.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            f"Pipeline YAML. Defaults to the single-real config, or {DEFAULT_DUAL_CONFIG} when --dual-mode is selected."
        ),
    )
    parser.add_argument("--action-name", default=None, help=f"ROS action name; default is {DEFAULT_ACTION_NAME}.")
    parser.add_argument("--node-name", default="grasp_assembly_action_server", help="ROS node name.")
    parser.add_argument(
        "--dual-mode",
        choices=DUAL_MODES,
        default=None,
        help=(
            "Use the dual holder/inserter adapter: pitl runs mock MoveIt plus "
            "Isaac; real runs the guarded hardware vertical slice."
        ),
    )
    parser.add_argument(
        "--pair-id",
        default="",
        help="Optional fixed dual grasp-pair ID; empty selects the ranked fallback.",
    )
    parser.add_argument("--robots", choices=("left", "right", "both"), default="both")
    parser.add_argument("--single-role", choices=("holder", "inserter"), default="inserter")
    parser.add_argument("--stop-after", choices=DUAL_REAL_STOP_AFTER_CHOICES, default="")
    parser.add_argument("--policy", default="")
    parser.add_argument("--left-camera", choices=("realsense_1", "realsense_2"), default="realsense_1")
    parser.add_argument("--right-camera", choices=("realsense_1", "realsense_2"), default="realsense_2")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run dual PITL/real mode without the Isaac or live-planner browser view.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Authorize hardware execution. The single-robot adapter remains "
            "blocked before motion; dual real mode executes the guarded "
            "holder/inserter pre-insertion slice."
        ),
    )
    parser.add_argument(
        "--skip-gripper",
        action="store_true",
        help=(
            "Prepare future single-robot action execution without gripper "
            "commands. Dual mode rejects this option because both grasps are "
            "part of the validated slice."
        ),
    )
    parser.add_argument(
        "--allow-objectless-planning",
        action="store_true",
        help=(
            "Permit a legacy dual-real task without phase-aware part AABBs "
            "and an attached incoming-part collision body."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_argument_parser()
    args, ros_args = parser.parse_known_args(raw_argv)
    if rclpy is None or MultiThreadedExecutor is None:
        print("ROS2 dependencies are unavailable. Source ROS2 and ros2_ws/install/setup.bash.", file=sys.stderr)
        return 1

    try:
        repo_root = _find_repo_root(args.repo_root)
        if args.dual_mode is not None:
            if args.skip_gripper:
                parser.error("--skip-gripper is not compatible with a complete dual GraspAssembly action.")
            runner = DualPipelineRunner(
                repo_root=repo_root,
                config_path=args.config or DEFAULT_DUAL_CONFIG,
                mode=str(args.dual_mode),
                allow_execution=bool(args.execute),
                allow_objectless_planning=bool(args.allow_objectless_planning),
                headless=bool(args.headless),
                pair_id=str(args.pair_id),
                robots=str(args.robots),
                single_role=str(args.single_role),
                stop_after=str(args.stop_after),
                policy=str(args.policy),
                left_camera=str(args.left_camera),
                right_camera=str(args.right_camera),
            )
        else:
            if args.allow_objectless_planning:
                parser.error("--allow-objectless-planning applies only with --dual-mode.")
            runner = RealPipelineRunner(
                repo_root=repo_root,
                config_path=args.config or DEFAULT_CONFIG,
                allow_execution=bool(args.execute),
                skip_gripper=bool(args.skip_gripper),
            )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    action_name = str(args.action_name or runner.action_name)
    rclpy.init(args=ros_args)
    node = GraspAssemblyActionServer(runner, action_name=action_name, node_name=str(args.node_name))
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
        return 0
    except KeyboardInterrupt:
        runner.request_cancel()
        return 130
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = [
    "GraspAssemblyActionServer",
    "DualPipelineRunner",
    "PipelineOutcome",
    "RealPipelineRunner",
    "SingleRobotGraspGoal",
    "build_argument_parser",
    "main",
]
