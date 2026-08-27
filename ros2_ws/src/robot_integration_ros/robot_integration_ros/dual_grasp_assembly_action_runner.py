"""Goal-driven runner for the dual holder/inserter GraspAssembly slice."""

from __future__ import annotations

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

ASSEMBLY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_DUAL_CONFIG = Path("configs/dual_grasp_planning.yaml")
DUAL_MODES = ("pitl", "real")
DUAL_REAL_STOP_AFTER_CHOICES = (
    "holder_pregrasp",
    "holder_grasp",
    "inserter_pickup_pregrasp",
    "inserter_pickup_grasp",
    "inserter_pickup_lift",
    "inserter_above_preinsertion",
    "inserter_preinsertion",
)
DEFAULT_DUAL_REAL_STOP_AFTER = "inserter_preinsertion"


@dataclass(frozen=True)
class DualRobotGraspGoal:
    """All GraspAssembly goal fields used by the dual adapter."""

    assembly_name: str
    base_part_id: int
    insertion_part_id: int
    holder_robot: str
    inserter_robot: str

    @classmethod
    def from_request(cls, request: Any) -> "DualRobotGraspGoal":
        return cls(
            assembly_name=str(request.assembly_name).strip(),
            base_part_id=int(request.base_part_id),
            insertion_part_id=int(request.insertion_part_id),
            holder_robot=str(request.holder_robot).strip().lower(),
            inserter_robot=str(request.inserter_robot).strip().lower(),
        )


@dataclass(frozen=True)
class DualPipelineOutcome:
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
        candidates = (
            Path.cwd().resolve(),
            source_path.parent,
            *source_path.parents,
        )
    for candidate in candidates:
        if (candidate / "run_pipeline.sh").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the grasp-planning repo root containing run_pipeline.sh and configs/."
    )


def _resolve_config_path(repo_root: Path, config_path: Path) -> Path:
    resolved = Path(config_path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dual action config does not exist: {resolved}")
    return resolved


def _ensure_repo_import_path(repo_root: Path) -> None:
    """Expose the source-tree package to an installed ROS console script."""

    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _load_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level YAML mapping in '{path}'.")
    return payload


def _action_config(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("grasp_assembly_action")
    return dict(raw) if isinstance(raw, dict) else {}


def _vec3(raw: object, *, field_name: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in raw)  # type: ignore[arg-type]
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain three values.")
    return values


def _normalize_error_code(status: object, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(status).upper()).strip("_")
    return normalized or fallback


def _read_json_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def _selected_real_task(
    task: dict[str, object],
    attempt: dict[str, object],
) -> dict[str, object]:
    """Resolve the task candidate that the real executor actually selected."""

    raw_candidates = task.get("ranked_pair_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return task
    candidates = [dict(value) for value in raw_candidates if isinstance(value, dict)]
    if not candidates:
        raise ValueError("Dual task ranked_pair_candidates contains no candidate objects.")

    raw_selection = attempt.get("pair_selection")
    selection = dict(raw_selection) if isinstance(raw_selection, dict) else {}
    selected_candidate_id = str(
        selection.get(
            "selected_execution_candidate_id",
            attempt.get("execution_candidate_id", ""),
        )
    )
    selected_pair_id = str(
        selection.get(
            "selected_pair_id",
            attempt.get("pair_id", ""),
        )
    )
    if selected_candidate_id:
        matches = [
            candidate
            for candidate in candidates
            if str(candidate.get("execution_candidate_id", "")) == selected_candidate_id
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Dual task has multiple candidates with selected execution_candidate_id '{selected_candidate_id}'."
            )
    if selected_pair_id:
        matches = [candidate for candidate in candidates if str(candidate.get("pair_id", "")) == selected_pair_id]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "Selected pair_id is ambiguous across symmetry-transition candidates; "
                "the execution_candidate_id is required."
            )

    selected_rank = selection.get("selected_rank")
    if selected_rank is not None:
        rank = int(selected_rank)
        rank_matches = [
            candidate
            for index, candidate in enumerate(candidates, start=1)
            if int(candidate.get("candidate_rank", index)) == rank
        ]
        if len(rank_matches) == 1:
            return rank_matches[0]
    raise ValueError("Could not resolve the candidate selected by the dual real executor.")


class DualPipelineRunner:
    """Run one dual PITL/Isaac or real GraspAssembly goal at a time."""

    skip_gripper = False

    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path = DEFAULT_DUAL_CONFIG,
        mode: str,
        allow_execution: bool,
        allow_objectless_planning: bool = False,
        headless: bool = False,
        pair_id: str = "",
        robots: str = "both",
        single_role: str = "inserter",
        stop_after: str = "",
        policy: str = "",
        left_camera: str = "realsense_1",
        right_camera: str = "realsense_2",
    ) -> None:
        self.repo_root = _find_repo_root(repo_root)
        _ensure_repo_import_path(self.repo_root)
        self.config_path = _resolve_config_path(
            self.repo_root,
            config_path,
        )
        self.base_payload = _load_mapping(self.config_path)
        self.mode = str(mode)
        if self.mode not in DUAL_MODES:
            raise ValueError(f"Dual action mode must be one of {DUAL_MODES}; got {self.mode!r}.")
        self.allow_execution = bool(allow_execution)
        self.allow_objectless_planning = bool(allow_objectless_planning)
        self.headless = bool(headless)
        self.pair_id = str(pair_id)
        self.robots = str(robots)
        self.single_role = str(single_role)
        self.policy = str(policy).strip()
        self.left_camera = str(left_camera)
        self.right_camera = str(right_camera)
        if self.robots not in {"left", "right", "both"}:
            raise ValueError("robots must be left, right, or both.")
        if self.single_role not in {"holder", "inserter"}:
            raise ValueError("single_role must be holder or inserter.")
        self.stop_after = str(
            stop_after
            or _action_config(self.base_payload).get("stop_after", DEFAULT_DUAL_REAL_STOP_AFTER)
        )
        if self.stop_after not in DUAL_REAL_STOP_AFTER_CHOICES:
            raise ValueError(
                "grasp_assembly_action.stop_after must be one of "
                f"{DUAL_REAL_STOP_AFTER_CHOICES}; got {self.stop_after!r}."
            )
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None

    @property
    def action_name(self) -> str:
        return str(
            _action_config(self.base_payload).get(
                "action_name",
                "/grasp_assembly",
            )
        )

    @property
    def description(self) -> str:
        return f"dual-{self.mode}-{self.robots}"

    @property
    def debug_aabb_topic(self) -> str:
        return str(
            _action_config(self.base_payload).get(
                "debug_aabb_topic",
                "/grasp_assembly/debug_aabbs",
            )
        )

    def _artifact_dir(self, assembly_name: str) -> Path:
        action = _action_config(self.base_payload)
        configured = Path(
            str(
                action.get(
                    "artifact_root",
                    "artifacts/dual_grasp_planning",
                )
            )
        )
        if not configured.is_absolute():
            configured = self.repo_root / configured
        return configured / assembly_name

    def _sequence_and_step(
        self,
        goal: DualRobotGraspGoal,
    ) -> tuple[dict[str, object], dict[str, object]]:
        sequence_path = self._artifact_dir(goal.assembly_name) / ("assembly_sequence.json")
        if not sequence_path.is_file():
            raise ValueError(f"Dual planning artifacts are missing the assembly sequence: {sequence_path}")
        sequence = _read_json_mapping(sequence_path)
        if str(sequence.get("assembly", "")) != goal.assembly_name:
            raise ValueError(
                f"Sequence assembly {sequence.get('assembly')!r} does not match goal {goal.assembly_name!r}."
            )
        if str(sequence.get("base_part_id", "")) != str(goal.base_part_id):
            raise ValueError(
                f"Goal base_part_id={goal.base_part_id} does not match the "
                f"selected-order base {sequence.get('base_part_id')}."
            )
        steps = sequence.get("steps")
        if not isinstance(steps, list):
            raise ValueError("Assembly sequence is missing its steps list.")
        matching = [
            dict(step)
            for step in steps
            if isinstance(step, dict) and str(step.get("incoming_part_id", "")) == str(goal.insertion_part_id)
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Could not resolve exactly one selected-order step for insertion_part_id={goal.insertion_part_id}."
            )
        step = matching[0]
        if not bool(step.get("holder_base_available", False)):
            raise ValueError(
                f"Step {step.get('step_id')} does not yet have an assembled base available for the holder."
            )
        pair_path = self._artifact_dir(goal.assembly_name) / (f"dual_grasp_pairs_{step.get('step_id')}.json")
        if not pair_path.is_file():
            raise ValueError(f"Dual grasp-pair artifact does not exist: {pair_path}")
        return sequence, step

    def validate(self, request: Any) -> str | None:
        try:
            goal = DualRobotGraspGoal.from_request(request)
            if not ASSEMBLY_NAME_PATTERN.fullmatch(goal.assembly_name):
                return "assembly_name must contain only letters, numbers, underscore, dot, or hyphen."
            if goal.base_part_id < 0 or goal.insertion_part_id < 0:
                return "base_part_id and insertion_part_id must be non-negative."
            action = _action_config(self.base_payload)
            expected_holder = str(action.get("holder_robot", "left")).strip().lower()
            expected_inserter = str(action.get("inserter_robot", "right")).strip().lower()
            if goal.holder_robot != expected_holder or goal.inserter_robot != expected_inserter:
                return (
                    "The current validated dual slice requires "
                    f"holder_robot='{expected_holder}' and "
                    f"inserter_robot='{expected_inserter}'; got "
                    f"holder={goal.holder_robot!r}, "
                    f"inserter={goal.inserter_robot!r}."
                )
            if goal.holder_robot == goal.inserter_robot:
                return "holder_robot and inserter_robot must be different."
            for part_id in (goal.base_part_id, goal.insertion_part_id):
                mesh = self.repo_root / "assets" / "obj" / "fabrica" / goal.assembly_name / f"{part_id}.obj"
                if not mesh.is_file():
                    return f"Fabrica part mesh does not exist: {mesh}"
            self._sequence_and_step(goal)
            topic = str(action.get("pose_base_topic", "")).strip()
            if not topic:
                return "grasp_assembly_action.pose_base_topic must be configured for dual PITL/real execution."
        except (TypeError, ValueError) as exc:
            return str(exc)
        return None

    def request_cancel(self) -> None:
        self._signal_active_process(signal.SIGINT)

    def _signal_active_process(
        self,
        requested_signal: signal.Signals,
    ) -> None:
        with self._process_lock:
            process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            return

    def _output_dir(
        self,
        *,
        goal: DualRobotGraspGoal,
        goal_id: str,
    ) -> Path:
        action = _action_config(self.base_payload)
        root = Path(
            str(
                action.get(
                    "output_root",
                    "artifacts/dual_grasp_assembly_actions",
                )
            )
        )
        if not root.is_absolute():
            root = self.repo_root / root
        return root / goal.assembly_name / f"step_{goal.insertion_part_id}" / goal_id

    @staticmethod
    def _feedback_from_line(
        line: str,
    ) -> tuple[str, float] | None:
        if "Waiting for MoveIt" in line:
            return "PLANNING", 0.05
        if "IK preflight kept" in line or "preflight_holder" in line or "preflight_plan_holder" in line:
            return "PLANNING", 0.50
        if "Selected pair" in line or "Selected ranked pair" in line or "preflight_inserter_preinsertion" in line:
            return "PLANNING", 1.0
        if "holder_pregrasp:" in line:
            return "GRASPING_BASE", 0.35
        if "holder_grasp:" in line:
            return "GRASPING_BASE", 0.75
        if "close_holder_gripper:" in line or "holder_close" in line:
            return "GRASPING_BASE", 1.0
        if "inserter_pickup_pregrasp:" in line:
            return "GRASPING_PART", 0.25
        if "inserter_pickup_grasp:" in line:
            return "GRASPING_PART", 0.70
        if "close_inserter_gripper:" in line or "inserter_close" in line:
            return "GRASPING_PART", 1.0
        if "inserter_pickup_lift:" in line:
            return "TRANSPORTING", 0.25
        if "inserter_above_preinsertion:" in line:
            return "TRANSPORTING", 0.70
        if "inserter_preinsertion:" in line:
            return "TRANSPORTING", 1.0
        return None

    def _run_process(
        self,
        *,
        command: list[str],
        cancel_requested: Callable[[], bool],
        publish_feedback: Callable[[str, float], None],
        publish_output: Callable[[str], None],
    ) -> tuple[int, list[str], bool]:
        process_env = dict(os.environ)
        pythonpath_entries = [entry for entry in process_env.get("PYTHONPATH", "").split(os.pathsep) if entry]
        repo_path = str(self.repo_root)
        if repo_path not in pythonpath_entries:
            process_env["PYTHONPATH"] = os.pathsep.join([repo_path, *pythonpath_entries])
        process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=process_env,
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

        reader = threading.Thread(
            target=_read_output,
            name="dual-grasp-pipeline-output",
            daemon=True,
        )
        reader.start()
        output_lines: list[str] = []
        reader_done = False
        cancel_sent_at: float | None = None
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
                    feedback = self._feedback_from_line(line)
                    if feedback is not None:
                        publish_feedback(*feedback)

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
        return return_code, output_lines, cancel_sent_at is not None

    def _outcome_from_artifacts(
        self,
        *,
        return_code: int,
        task_path: Path,
        attempt_path: Path,
        output_lines: list[str],
        cancelled: bool,
    ) -> DualPipelineOutcome:
        if cancelled:
            return DualPipelineOutcome(
                False,
                "CANCELLED",
                "Dual grasp pipeline cancelled.",
            )
        attempt: dict[str, object] | None = None
        if attempt_path.is_file():
            attempt = _read_json_mapping(attempt_path)
        result = (
            dict(attempt.get("result", {}))
            if isinstance(attempt, dict) and isinstance(attempt.get("result"), dict)
            else {}
        )
        if return_code != 0 or not bool(result.get("success", False)):
            message = str(result.get("message", "")).strip()
            if not message:
                message = next(
                    (line for line in reversed(output_lines) if line.strip()),
                    "Dual grasp pipeline failed.",
                )
            return DualPipelineOutcome(
                False,
                _normalize_error_code(
                    result.get("status", ""),
                    fallback="PIPELINE_FAILED",
                ),
                message,
            )

        frame_id = str(_action_config(self.base_payload).get("frame_id", "base_link"))
        if self.mode == "pitl":
            final_pose = result.get("final_incoming_pose")
            if not isinstance(final_pose, dict):
                return DualPipelineOutcome(
                    False,
                    "RESULT_POSE_MISSING",
                    "Successful Isaac attempt did not record final incoming pose.",
                )
            position = _vec3(
                final_pose.get("position_world_m"),
                field_name="final_incoming_pose.position_world_m",
            )
            orientation_wxyz = tuple(float(value) for value in final_pose.get("orientation_wxyz_world", ()))
            if len(orientation_wxyz) != 4:
                return DualPipelineOutcome(
                    False,
                    "RESULT_POSE_INVALID",
                    "Isaac recorded an invalid final incoming orientation.",
                )
            orientation_xyzw = (
                orientation_wxyz[1],
                orientation_wxyz[2],
                orientation_wxyz[3],
                orientation_wxyz[0],
            )
            return DualPipelineOutcome(
                True,
                "",
                "Dual PITL/Isaac transport reached pre-insertion.",
                frame_id,
                position,
                orientation_xyzw,
            )

        last_completed_phase = str(result.get("last_completed_phase", "")).strip()
        if not last_completed_phase:
            status = str(result.get("status", "")).strip()
            if status.startswith("stopped_at_"):
                last_completed_phase = status.removeprefix("stopped_at_")
            elif status == "completed":
                last_completed_phase = "inserter_preinsertion"
        if last_completed_phase != "inserter_preinsertion":
            reached = last_completed_phase or "no motion phase"
            return DualPipelineOutcome(
                False,
                "PARTIAL_EXECUTION",
                (f"Dual real execution stopped after '{reached}', before the incoming part reached pre-insertion."),
            )

        task = _read_json_mapping(task_path)
        assert isinstance(attempt, dict)
        selected_task = _selected_real_task(task, attempt)
        objects = selected_task.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("Selected dual task candidate is missing objects.")
        incoming = objects.get("incoming")
        if not isinstance(incoming, dict):
            raise ValueError("Dual task is missing incoming object.")
        target_pose = incoming.get("preinsertion_source_pose_world")
        if not isinstance(target_pose, dict):
            raise ValueError("Dual task is missing the pre-insertion pose.")
        position = _vec3(
            target_pose.get("position_world_m"),
            field_name="preinsertion.position_world_m",
        )
        orientation = tuple(float(value) for value in target_pose.get("orientation_xyzw_world", ()))
        if len(orientation) != 4:
            raise ValueError("Dual task pre-insertion orientation is invalid.")
        return DualPipelineOutcome(
            True,
            "",
            (
                "Dual real transport reached the commanded pre-insertion "
                "target"
                + (f" using ranked pair {attempt.get('pair_id')}." if str(attempt.get("pair_id", "")) else ".")
                + " The returned pose is the commanded source-frame pose."
            ),
            frame_id,
            position,
            orientation,
        )

    def run(
        self,
        *,
        request: Any,
        goal_id: str,
        cancel_requested: Callable[[], bool],
        publish_feedback: Callable[[str, float], None],
        publish_output: Callable[[str], None],
        publish_debug_aabbs: (Callable[[tuple[dict[str, object], ...]], None] | None) = None,
    ) -> DualPipelineOutcome:
        if self.mode == "real" and not self.allow_execution:
            return DualPipelineOutcome(
                False,
                "EXECUTION_DISABLED",
                "Dual real server was started without --execute.",
            )
        goal = DualRobotGraspGoal.from_request(request)
        validation_error = self.validate(request)
        if validation_error is not None:
            return DualPipelineOutcome(
                False,
                "INVALID_GOAL",
                validation_error,
            )
        _, step = self._sequence_and_step(goal)
        step_id = str(step["step_id"])
        artifact_dir = self._artifact_dir(goal.assembly_name)
        action = _action_config(self.base_payload)
        topic = str(action["pose_base_topic"])
        message_type = str(
            action.get(
                "pose_message_type",
                "fp_debug_msgs/msg/DebugPoseItem",
            )
        )
        timeout_s = float(action.get("pose_timeout_s", 40.0))
        publish_feedback("WAITING_FOR_POSES", 0.0)

        from grasp_planning.grasping.world_constraints import ObjectWorldPose
        from grasp_planning.pipeline.dual_robot_simple_sim import (
            DEFAULT_FLOOR_Z_WORLD_M,
            resolve_planar_runtime_layout,
        )
        from grasp_planning.ros2.pose_listener import (
            wait_for_debug_pose_item_messages,
        )

        try:
            poses = wait_for_debug_pose_item_messages(
                topic_name=topic,
                message_type=message_type,
                assembly_name=goal.assembly_name,
                part_ids=(goal.base_part_id, goal.insertion_part_id),
                timeout_s=timeout_s,
                cancel_requested=cancel_requested,
            )
        except InterruptedError as exc:
            return DualPipelineOutcome(False, "CANCELLED", str(exc))
        except TimeoutError as exc:
            return DualPipelineOutcome(False, "PERCEPTION_TIMEOUT", str(exc))

        position_offset = _vec3(
            action.get("position_offset_m", (0.0, 0.0, 0.0)),
            field_name="grasp_assembly_action.position_offset_m",
        )

        def _offset_pose(pose: ObjectWorldPose) -> ObjectWorldPose:
            return ObjectWorldPose(
                position_world=tuple(
                    float(value) + float(offset)
                    for value, offset in zip(
                        pose.position_world,
                        position_offset,
                        strict=True,
                    )
                ),
                orientation_xyzw_world=pose.orientation_xyzw_world,
            )

        layout = resolve_planar_runtime_layout(
            artifact_dir=artifact_dir,
            step_id=step_id,
            base_source_pose_world=_offset_pose(poses[goal.base_part_id]),
            incoming_source_pose_world=_offset_pose(poses[goal.insertion_part_id]),
            maximum_assembly_tilt_deg=float(action.get("maximum_assembly_tilt_deg", 5.0)),
        )
        if publish_debug_aabbs is not None:
            part_ids = {
                "base": goal.base_part_id,
                "incoming": goal.insertion_part_id,
            }
            debug_aabbs = tuple(
                {
                    "frame_id": str(action.get("frame_id", "base_link")),
                    "role": bounds.role,
                    "part_id": int(part_ids[bounds.role]),
                    "minimum_world_m": list(bounds.minimum_world_m),
                    "maximum_world_m": list(bounds.maximum_world_m),
                }
                for bounds in layout.perceived_part_aabbs
            )
            try:
                publish_debug_aabbs(debug_aabbs)
            except Exception as exc:
                publish_output(f"[WARNING] Failed to publish perceived-part AABBs: {exc}")
        for warning in layout.warnings:
            publish_output(f"[WARNING] {warning}")
        output_dir = self._output_dir(goal=goal, goal_id=goal_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        task_path = output_dir / ("dual_real_task.json" if self.mode == "real" else "dual_pitl_plan.json")
        attempt_path = output_dir / (
            "dual_real_attempt.json" if self.mode == "real" else "dual_pitl_isaac_attempt.json"
        )
        command = [
            str(self.repo_root / "run_pipeline.sh"),
            "--mode",
            self.mode,
            "--robots",
            self.robots,
            "--role",
            self.single_role,
            "--reuse-moveit",
            "--artifact-dir",
            str(artifact_dir),
            "--step-id",
            step_id,
            "--assembly-x",
            str(layout.assembly_world.position_world_m[0]),
            "--assembly-y",
            str(layout.assembly_world.position_world_m[1]),
            "--assembly-z",
            str(layout.assembly_world.position_world_m[2]),
            "--assembly-yaw-deg",
            str(layout.assembly_world.yaw_deg),
            "--pickup-x",
            str(layout.pickup_source_world_xy[0]),
            "--pickup-y",
            str(layout.pickup_source_world_xy[1]),
            "--pickup-roll-deg",
            str(layout.pickup_orientation_rpy_deg[0]),
            "--pickup-pitch-deg",
            str(layout.pickup_orientation_rpy_deg[1]),
            "--pickup-yaw-deg",
            str(layout.pickup_orientation_rpy_deg[2]),
            "--floor-z",
            str(
                float(
                    action.get(
                        "floor_z_world_m",
                        DEFAULT_FLOOR_Z_WORLD_M,
                    )
                )
            ),
            "--attempt-output",
            str(attempt_path),
        ]
        if self.pair_id:
            command.extend(("--pair-id", self.pair_id))
        if self.policy:
            command.extend(
                (
                    "--policy",
                    self.policy,
                    "--left-camera",
                    self.left_camera,
                    "--right-camera",
                    self.right_camera,
                )
            )
        if self.mode == "pitl":
            command.extend(("--plan-output", str(task_path)))
            if self.headless or bool(action.get("headless", False)):
                command.append("--headless")
        else:
            if self.headless or bool(action.get("headless", False)):
                command.append("--no-planning-debug-gui")
            command.extend(
                (
                    "--task-output",
                    str(task_path),
                    "--execute",
                    "--stop-after",
                    self.stop_after,
                    "--yes",
                )
            )
            if self.allow_objectless_planning:
                command.append("--allow-objectless-planning")

        publish_feedback("PLANNING", 0.0)
        return_code, output_lines, cancelled = self._run_process(
            command=command,
            cancel_requested=cancel_requested,
            publish_feedback=publish_feedback,
            publish_output=publish_output,
        )
        return self._outcome_from_artifacts(
            return_code=return_code,
            task_path=task_path,
            attempt_path=attempt_path,
            output_lines=output_lines,
            cancelled=cancelled or cancel_requested(),
        )


__all__ = [
    "DEFAULT_DUAL_CONFIG",
    "DEFAULT_DUAL_REAL_STOP_AFTER",
    "DUAL_MODES",
    "DUAL_REAL_STOP_AFTER_CHOICES",
    "DualPipelineOutcome",
    "DualPipelineRunner",
    "DualRobotGraspGoal",
]
