# Franka Real-Policy Workbench

Run commands from the repository root. This standalone workflow uses a
hand-positioned FR3 with Panda hand, a ZED Mini, and saved Isaac goal images.
It does not run the stage-2 approach planner. See [training](training.md)
for catalog/checkpoint contracts and [deployment](deployment.md) for dependencies.

## Configure And Observe

```bash
./franka_policy.sh configure
```

Open <http://127.0.0.1:8765>, select part, resting orientation, and grasp,
then save the selection. Goal thumbnails are saved training images, not a
reconstruction of the real object pose. The selected catalog/checkpoint and
their hashes are saved in `configs/franka_real_selection.local.json`.
Do not assume a historical checkpoint or locally saved selection is available
in a new clone; select compatible assets explicitly.

The goal color controls can render a custom Isaac RGB reference while retaining
the original catalog depth channel. Renders are cached under
`artifacts/franka_goal_colors/`; modified or wrong-target renders block inference.
Custom goals are a deployment variation, not validated physical performance.

```bash
./franka_policy.sh stream
./franka_policy.sh run
```

These reuse the camera server, which stays in its original terminal. `run`
defaults to observe-only inference without robot commands. Stop the server
before using exclusive camera capture mode. Review live/goal image alignment,
depth, TCP, mount, and motion axes before considering physical operation.

## Connect Hardware

Enable FCI in [Franka Desk](https://192.168.1.200/desk/) and use separate terminals:

```bash
# Read-only model/connection probe, without ROS controllers
./franka_policy.sh probe

# Feedback without activating the arm command controller
./franka_policy.sh connect state_only:=true
```

Stop feedback-only launch before normal connection:

```bash
./franka_policy.sh connect
./franka_policy.sh run --execute
```

`connect` launches the compatible FR3 driver, gripper, Servo, and independent
watchdog. Do not run competing hardware-interface launches. It does not home
the gripper or send an approach trajectory. `run --execute` enables physical
buttons; it does not start motion. Hand-guide the stationary arm near the
intended grasp, confirm mount/axes, use the manual-open control to match the
goal, and explicitly confirm **Run policy**. Keep the physical stop accessible.
Stop/Escape ends the run. Closing is a separate action, not automatic policy completion.

The local compatible stack is built with `scripts/build_franka_policy_stack.sh`
under `.cache/franka_compatible`. Real motion additionally requires the verified
velocity-driver overlay built by `scripts/build_franka_velocity_driver.sh`
under `.cache/franka_velocity_driver`. The launcher rejects stale manifests
and duplicate launches through this entrypoint; it cannot exclude unrelated
external control software. Rebuild only while its driver library is not loaded.

Native joint-velocity tracking uses zero ROS PID gains and velocity feed-forward,
with the hardware-boundary 10 Hz filter and libfranka acceleration/jerk limiter.
Filtering adds stopping latency. Policy/Servo/hardware rates are 15/100/1000 Hz.
The watchdog and controller have independent 0.15 s command-expiry stops.
Successful feedback, fake-hardware tests, or a read-only probe do not establish
reliable realtime physical control.

## Safety And Pickup

Check the saved configuration before use. Conservative policy limits include
0.01 m/s, 0.06 rad/s, 150 mm travel, and 0.2 rad rotation. Fresh camera/state/TF,
starting velocity, aperture, workspace, force/torque, robot mode/collision flags,
and controller checks gate commands. Faults disable execution until reviewed.
The configurable external torque guard is a Cartesian norm, not a joint rating;
it does not replace hardware collision thresholds or physical contact review.

**Close + lift 20 cm** is an explicit physical command with no second
confirmation dialog. It stops policy dispatch, stops Servo, waits for three
stationary feedback samples, closes, and lifts using the same control instance.
Grasp action success and fresh matching nonempty finger aperture are required.
Stop/Escape takes priority over a queued pickup.

The lift is positive robot-base Z, not a collision-planned trajectory. Ensure
the path is clear. It targets 200 mm above the settled TCP with 2 mm tolerance,
at no more than 10 mm/s or the saved lower limit. The pickup-only envelope is
205 mm. Lateral drift over 5 mm, downward travel/overshoot over 3 mm, rotation
over 0.05 rad, aperture change over 3 mm, lack of 1 mm upward progress for
3 seconds, stale feedback, and existing fault/load gates stop it. Timeout is
distance/speed plus 10 seconds, capped at 40 seconds. Arrival adds one second
of recording. The gripper never opens automatically; support the part first.
Neither Grasp action success nor TCP travel proves physical grasp success.

## Camera And Recording

Use the camera profile embedded in the selected catalog/checkpoint, not dated
intrinsics copied from an experiment. Depth is aligned rectified-left optical Z
in metres, not robot-base height. A measured camera profile does not establish
the physical mount calibration. The read-only synchronized raw frame endpoint
is <http://127.0.0.1:8765/camera/snapshot.npz>.

Runs record under `artifacts/franka_real_runs/` unless explicitly disabled.
The default duration is unlimited; Stop, completion, and fault gates still apply.
A bounded 32-frame background queue spools raw frames; storage failure or queue
saturation stops control instead of silently dropping frames. NPZ/MP4 export
happens after control stops, with failed-export spools retained. JSON telemetry
separates limited commanded actions from measured motion. `applied_action`
means a command sent to Servo, not achieved movement.

**Good video - keep** archives a checksum-verified bundle under
`artifacts/franka_good_runs/`. Keep its shared `_assets/` directory when moving
the collection. Source recordings are not deleted. The rating describes video
usefulness, not verified grasp success.

## Connection Troubleshooting

Check `ip route get 192.168.1.200`, robot ping, Desk, and FCI separately.
The launcher fails a network preflight without changing network settings.
For this lab's known control-box wiring on `enp6s0`, a temporary address is:

```bash
nmcli device modify enp6s0 +ipv4.addresses 192.168.1.10/24
ip route get 192.168.1.200
ping -c 2 192.168.1.200
```

Use this only for that wiring/subnet; it is not portable new-machine setup.
The expected route is `dev enp6s0 src 192.168.1.10`. Profile reactivation/reboot
can remove the address. Remove it with
`nmcli device modify enp6s0 -ipv4.addresses 192.168.1.10/24` when needed.

The compatible stack pins libfranka 0.18.0, franka_ros2 v2.0.2, and
franka_description 1.0.1 for the observed protocol-10 robot. Older libfranka
0.13.2 cannot connect to that protocol. Verify versions against the actual
robot rather than assuming every installation is the same.

An octomap sensor warning does not mean the robot connection failed; the
ZED policy stream is separate from MoveIt's environment map. Startup waits
for arm/TCP/aperture feedback; runtime freshness gates remain active.

## Physical Validation Still Required

- Verify the actual hand/camera payload, mount, TCP, and motion axes.
- Verify realtime kernel/scheduling and tracking; successful networking or FCI is insufficient.
- Panda-trained visual/control behavior is not established on physical FR3.
- Without object-pose perception, policy completion is not independently verified.
- MoveIt self-collision checks do not add real table/part/environment geometry.

Historical motion diagnoses and checkpoint switches remain in Git history.
These instructions do not establish safe or reliable physical operation.
