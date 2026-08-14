"""No-hardware gripper stand-in for mock dual-arm runs.

The dual mock MoveIt stack (`dual_aligned_lbr_moveit.launch.py`) spawns no
gripper controller at all - only `joint_state_broadcaster` and the two arm
trajectory controllers - and nothing in that stack starts the separate
`/lbr_{one,two}/gripper_controller/{open,close,stop}` Trigger services
`TriggerServiceGripperClient` calls (those come from
`scripts/gripper_computer/dual_grippers.launch.py`, a different process
meant for real hardware). Using the real client against the mock stack hangs
until its timeout and then raises, aborting the run.

`MockGripperClient` implements the exact same interface without touching any
ROS service: it reports success immediately and best-effort mirrors the
requested opening onto the shared MoveIt-tracked robot state via
`MoveItPoseCommander.apply_robot_state_joint_positions`, so the requested
finger position is visible to anything reading the monitored planning scene.
"""

from __future__ import annotations

import time


class MockGripperClient:
    """Simulates gripper open/close with no gripper hardware or service."""

    def __init__(
        self,
        commander,
        *,
        finger_joint_name: str,
        grasp_settle_time_s: float,
        open_width_m: float = 0.08,
    ) -> None:
        self._commander = commander
        self._finger_joint_name = str(finger_joint_name)
        self._grasp_settle_time_s = float(grasp_settle_time_s)
        self._open_width_m = float(open_width_m)

    def wait_for_server(self, *, timeout_s: float) -> None:
        del timeout_s  # No server to wait for; this client never blocks here.

    def open(self, *, width: float) -> tuple[bool, str]:
        del width  # Matches TriggerServiceGripperClient: "open" is a fixed position.
        return self._set_finger_position(self._open_width_m, label="open")

    def close(self, *, width: float) -> tuple[bool, str]:
        ok, message = self._set_finger_position(float(width), label="close")
        if ok:
            time.sleep(max(self._grasp_settle_time_s, 0.0))
        return ok, message

    def stop(self) -> tuple[bool, str]:
        return True, "Mock gripper stop is a no-op (no gripper hardware/service in this mode)."

    def _set_finger_position(self, jaw_width_m: float, *, label: str) -> tuple[bool, str]:
        # left_finger_joint drives right_finger_joint via a 1:1 URDF mimic,
        # and jaw width is the sum of both fingers' travel from center.
        finger_position_m = max(float(jaw_width_m), 0.0) / 2.0
        ok, message = self._commander.apply_robot_state_joint_positions(
            {self._finger_joint_name: finger_position_m}
        )
        return ok, f"Mock gripper {label} (finger={finger_position_m:.4f} m): {message}"


__all__ = ["MockGripperClient"]
