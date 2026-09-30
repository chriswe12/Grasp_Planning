# Franka real-policy workbench

This standalone workflow uses a hand-positioned FR3 arm with Panda hand, a ZED Mini, and the Franka policy's saved Isaac goal images. It currently exposes the five Fabrica plumbers-block parts and their 151 validated catalog targets. It does not run the stage-2 approach planner.

## Active checkpoint on September 23

The saved real-robot selection now uses epoch 5000 from job 14789936, copied to `artifacts/franka_real_checkpoint_20260923/checkpoint/last_franka_zed_ep_5000_rew_28.035013.pth`. It uses the exact same v7 calibrated ZED catalog/contract as the previously selected epoch 2300. Only the checkpoint path and hash changed; the current selected grasp, camera/mount configuration, 15 Hz policy rate, motion limits, unlimited policy duration and recording behavior are preserved.

Verified strict GPU actor loading, exact contract agreement, and 30 live ZED observe-only inference steps over two seconds with video export. No robot connection or commands were issued. The diagnostic two-second limit does not change the saved unlimited duration. The ZED still reports skipped self-calibration due to low texture/occlusion; this checkpoint switch does not resolve the previously observed table-plane/mount discrepancy. Verification, video and the previous selection backup are in `artifacts/franka_real_checkpoint_20260923/`. Restart any existing workbench after changing checkpoints, then use `./franka_policy.sh run --execute`; expect Epoch 5000 in the UI.

## Active checkpoint on September 22

The local saved selection now uses update 2300 from job 14789936, with `isaac_rl/data/franka_clutter_v7_zed_calibrated/catalog.npz`. Checkpoint: `artifacts/franka_mixed_benchmark_20260922/checkpoint/nn/last_franka_zed_ep_2300_rew_15.141277.pth`. This pair uses measured ZED Mini intrinsics (fx/fy 338.6002, 672x376), corrected dense depth packing and mixed-color Isaac/MuJoCo goals. Selected hardware reference remains canonical Isaac blue (`goal_variant_index: 0`), with the same plumbers-block part 2 / orientation 002 / g0587 p0. Original selection backup and software verification are in `artifacts/franka_real_checkpoint_20260922/`. Motion limits, TCP and mount are retained; recording is enabled. Historical sections below describe earlier checkpoints.

Strict checkpoint/catalog/hash checks, goal inference, a two-second live ZED observe-only run (30 frames, recorded), and calibrated UI metadata passed. No physical motion was tested. The paired simulator benchmark reports 24.8/34.6/39.0 percent far/mid/near success; this is experimental, not hardware validation. Start the usual connection and `./franka_policy.sh run --execute`; the UI badge now derives the update number and camera intrinsics from the loaded selection rather than displaying the old constants.

## Launch

From this repository:

```bash
./franka_policy.sh configure
```

Open <http://127.0.0.1:8765>. Select the part, then its resting orientation, then a grasp. Left/right arrows cycle the current selection; Enter or down advances; up goes back; Escape stops a run. The grasp grid initially shows a diverse subset; the all-grasps option exposes every candidate for that orientation. Orientation diagrams are selection aids; the goal thumbnails are actual saved Isaac training images.

**Goal object color:** after choosing the grasp, select a color swatch or use the custom color picker, then click **Render color in Isaac**. The selected goal panel updates when the render completes; click **Save selection** to use that RGB-D image with the policy. **Original training goal** restores the exact catalog image. Selecting another grasp resets the color to the original, so choose the color after browsing. Grasp thumbnails remain the original catalog previews.

New colors use the existing local `isaac-lab-euler:2.3.2` Docker image and take approximately 1–2 minutes on this computer; matching target/color renders are cached. No image or system packages are downloaded. Rendering blocks policy/gripper actions and Stop cancels it. The owned render container exits afterward. Cached RGB-D, metadata and render logs live in `artifacts/franka_goal_colors/`; the saved configuration records the artifact path and hash. Missing, modified or wrong-target renders block inference. The original catalog/checkpoint are unchanged. Custom goal colors are an explicit deployment variation: training randomized live appearance while retaining canonical goal images, so policy performance with custom goals has not been validated.

Save the selection. `./franka_policy.sh stream` opens the same camera workbench. Repeated configure/stream/run commands reuse the existing server and camera. The server stays in its original terminal; leave that terminal running. Camera capture mode requires that server to be stopped first.

The **Observe policy** button runs live inference without robot commands. This is also the default for:

```bash
./franka_policy.sh run
```

For physical operation, enable FCI in Franka Desk at <https://192.168.1.200/desk/>. In another terminal:

```bash
cd /media/pdz/Elements1/Grasp_Planning_grasping_rl
./franka_policy.sh connect
```

This sources `/opt/ros/humble` and the isolated `.cache/franka_compatible/install` workspace, then launches the FR3 hardware interface, Panda gripper interface, MoveIt Servo, and an independent command watchdog. The incompatible `~/franka_ros2_ws` paths are removed from this launcher's environment; the old workspace and system installation are preserved. Do not run it alongside another hardware-interface launch for the same arm. It does not home the gripper or send an approach trajectory. The old custom impedance controller is not used: its activation contains a fixed pose target.

For a read-only connection/model check, without ROS controllers:

```bash
./franka_policy.sh probe
```

For ROS feedback without activating the arm command controller:

```bash
./franka_policy.sh connect state_only:=true
```

Stop this feedback-only launch before running the normal connect command.

After checking the camera mount, TCP and motion axes, hand-guide the stationary arm near the selected grasp, confirm the mount in the UI and save. Enable physical controls with:

```bash
./franka_policy.sh run --execute
```

This enables buttons; it does not start motion. Use the explicit manual-open button to match the selected goal's finger opening, then **Run policy** and its confirmation. Keep the physical stop accessible. The UI Stop button/Escape ends the run. Manual closing is a separate confirmed action after stopping, at 10 N and 0.02 m/s; there is no automatic close, lift, or grasp-success claim.

## One-click close and 20 cm pickup

During robot policy motion, click **Close + lift 20 cm** instead of Stop. There is no second confirmation dialog. The click queues a handoff in the same worker: no further policy command is intentionally dispatched after the request is observed, Servo is stopped, fresh feedback must show the arm stationary for three samples, then the gripper closes and the lift begins. The same ROS control instance is reused, avoiding competing command publishers. Duplicate clicks and requests during observation, closing, lifting, saving or other operations are rejected. Stop/Esc takes priority over a queued pickup. Physical execution mode, saved selection, mount/axis confirmation and all feedback/load gates remain required.

Closing uses the selected grasp width and configured manual force (currently 10 N). A successful Grasp action and fresh nonempty matching finger aperture are required before lift. The lift commands only positive robot-base Z velocity; with the base upright this is vertical, not hand-local Z. This is a direct Cartesian lift, not a collision-planned trajectory or a perception-based grasp check. Ensure the upward path is clear before clicking. **Manual close…** remains available without lift; clicking Close + lift while idle starts a new recorded pickup-only run.

The endpoint is 0.20 m above the settled starting TCP (2 mm arrival tolerance), inside the configured workspace. Speed is at most 10 mm/s or the saved lower limit, ramping up and slowing near the target. Only pickup has a separate 205 mm travel envelope; policy travel remains 150 mm, with no automatic time cutoff. Lateral drift over 5 mm, downward travel/overshoot over 3 mm, rotation over 0.05 rad, aperture change over 3 mm, no 1 mm upward progress for 3 seconds, stale feedback, controller faults and existing force/torque checks stop the action. Lift timeout is distance/speed plus 10 seconds, capped at 40 seconds. Payload compensation and contact thresholds are unchanged.

On arrival the arm stops and the same recording continues for **one second**, then the run ends and encodes one MP4. Stop/Esc can end it earlier. The gripper never opens automatically; support the part before opening it. No physical grasp success is inferred from Grasp success or TCP travel. The video identifies policy, settling, closing, lifting and post-lift phases; p(done) is shown as n/a during manual pickup. The combined JSON contains pickup outcome, configuration, start/last checked state, forces and commands; exact RGB-D data and video remain alongside it in `artifacts/franka_real_runs/`. The existing Good video — keep action archives this whole combined run and its artifacts.

Unlimited runs stream raw frames to a private local spool through a 32-frame background queue. Appending does not wait for disk I/O; queue saturation or a storage error stops the run instead of blocking robot control. RGB-D NPZ arrays are written incrementally and video is encoded after controller shutdown; the successful spool is then removed. Short explicitly timed runs can still use bounded RAM capture. Offline handoff/cancellation/recording tests do not validate physical pickup; no robot motion was issued by the agent.

## Raw depth inspection

While the workbench is running, `http://127.0.0.1:8765/camera/snapshot.npz` returns a fresh synchronized full-resolution RGB/metric-depth frame, frame sequence, capture monotonic timestamp and camera calibration. It is read-only, rejects stale frames and preserves float32 optical-Z metres and invalid pixels without policy resizing or JPEG quantization. Depth means camera-axis distance, not robot-base height. The September 23 colour-map example and plotting script are under `artifacts/franka_depth_20260923/`.

## Run recording

Every policy/observe run with a run-log path records automatically (`record_video: false` explicitly disables it). `max_duration_s: null` is the default and active setting: there is no automatic policy/observe time cutoff. Stop/Esc, policy completion, Close + lift, and motion/fault gates still end or transition the run. Positive finite durations remain available for explicitly timed smoke tests; the pickup motion timeout and one-second post-lift recording tail are unchanged.

For unlimited runs the 15 Hz loop enqueues camera RGB/depth references and an exact copy of the small policy RGB-D input. A background thread writes raw frames to a private spool in local temporary storage (`/tmp` on this host), keeping capture off the external project drive using a bounded 32-frame queue (roughly 65 MB for VGA, plus working arrays). The loop never waits for disk writes. Disk/storage failure or a full queue stops control with a recording error; there is no silent frame deletion or RAM growth proportional to video duration. Telemetry remains in the run log, while disk space grows with recording length. After control closes, RGB-D arrays are written incrementally to standard NPZ and video to a 15 fps H.264 MP4 in `artifacts/franka_real_runs/`. Encoding remains after motion and can take longer for long videos. Spools are removed after successful export and retained on export failure.

The video shows raw rectified ZED RGB, policy RGB, the selected goal, live/goal depth and frame-aligned telemetry: timestamp, p(done), maximum p(done), stable/completion streak, valid-depth coverage, TCP displacement and commanded speeds. The NPZ contains exact live/goal policy RGB-D, raw optical-Z depth, frame sequence and timestamps; camera calibration and actions remain in the matching JSON. Video timestamps are authoritative if cycle intervals vary; playback uses nominal 15 fps. **Watch last run** appears when the video is ready. Failed/no-frame starts produce no video. Recording failures are reported separately from the control stop reason. The installed imageio-ffmpeg binary is reused without downloading packages; `FRANKA_FFMPEG` can override its path.

## Keep useful videos

After a recording finishes, **Good video — keep** copies the run into `artifacts/franka_good_runs/<run_id>/`; **Not good** saves a negative rating without copying. The label describes operator-assessed video usefulness, not verified grasp success. The controls show the exact run ID and persist the last completed recording/rating across workbench restarts. They are disabled while a run or archival copy is active.

Each approved bundle has `video.mp4`, `frames.rgbd.npz`, `run.json` (actions, completion scores, feedback and outcome), `config.json`, `camera.json`, optional encoder log, a checksum manifest and the review label. The checkpoint, checkpoint contract, catalog and any custom goal are copied once under the collection's `_assets/` directory and referenced relatively by each manifest. Keep `_assets` with the collection when moving it. Source artifacts are never moved or deleted; repeated approval verifies/reuses the bundle. A later negative rating updates both labels and preserves the previous archive. Corrupted or changed artifacts fail the archive operation without marking the source good.

The operator-requested first entry is `20260922_114107` (25 seconds, 375 samples). Original recordings remain in `artifacts/franka_real_runs/` with separate `.review.json` labels.

## Connection troubleshooting

The September 21 robot reports arm protocol 10. The old libfranka 0.13.2 reports protocol 6 and cannot connect. The compatible local stack pins libfranka 0.18.0, franka_ros2 v2.0.2 and franka_description 1.0.1. `scripts/build_franka_policy_stack.sh` rebuilds these in `.cache/franka_compatible` from pinned official Git commits, using system ROS Humble and `/opt/openrobots` Pinocchio. It does not install into `/usr` or alter the old home workspace. Logs and verification are under `artifacts/franka_upgrade_20260921/`.

The read-only `probe` deliberately does not require realtime scheduling; it has no control-loop code. The host still has a generic kernel and limited realtime scheduling permissions, so successful feedback does not establish reliable 1 kHz physical control.

Upgrade verification on September 21: all seven selected ROS packages built, the hardware plugin linked to local `libfranka.so.0.18`, and the real feedback-only launch activated both state broadcasters with no claimed command interfaces. The policy feedback reader received 75 fresh samples at 15 Hz, including TCP TF and gripper aperture. A callback-draining fix prevents high-rate robot messages from starving TF updates. The isolated fake Servo test produced 101 moving trajectory messages and 48 zero-velocity timeout holds. Fourteen focused tests passed. Both test launches were stopped afterward. Physical motion remains untested.

If connection remains at `Connecting to robot`, check `ip route get 192.168.1.200`. On this computer the control-box cable uses `enp6s0`; the saved `kuka-koni` profile can leave it without a Franka-subnet address, sending robot traffic over Wi-Fi. For this known wiring, restore the temporary address without changing the saved KUKA profile:

```bash
nmcli device modify enp6s0 +ipv4.addresses 192.168.1.10/24
ip route get 192.168.1.200
ping -c 2 192.168.1.200
```

The route should say `dev enp6s0 src 192.168.1.10`. This address can disappear after reboot or profile reactivation. Remove it when needed with `nmcli device modify enp6s0 -ipv4.addresses 192.168.1.10/24`. The launcher now fails before starting ROS when its two-second ping preflight fails; it never changes network settings itself. `fake:=true` and launch-help commands skip that check.

`No 3D sensor plugin(s) defined for octomap updates` reports the absence of a MoveIt environment-map sensor, not a failed robot connection. The ZED policy stream is separate from MoveIt's octomap. FCI must still be enabled after network connectivity is restored.

`No fresh gripper aperture feedback` before the first policy step can also be a startup race: arm/TF data may arrive before the separate gripper publisher is discovered. Startup now waits up to eight seconds for fresh arm, TCP and aperture feedback together. The runtime aperture freshness limit remains 0.3 s and the width tolerance remains 3 mm. On September 21 this race was reproduced with a correctly set 48.046 mm opening; after the fix, twelve read-only startups passed. Restart only the policy web server to load this change.

## Camera and policy contract

The detected ZED Mini is serial 13829658, rectified left RGB 672×376 at 30 Hz, aligned optical-Z depth in metres, using NEURAL depth. Its measured intrinsics are fx=fy=338.6002, cx=332.0468, cy=184.4311. The training profile has different intrinsics (fx=428.4 at the source resolution). Live images are resampled by camera rays to the training intrinsics at 256×144, then packed using the existing training RGB-D preprocessing at 128×72. Invalid depth remains invalid.

By default, the goal comes directly from `goal_rgbd` in `isaac_rl/data/franka_clutter_v5_fast_fxaa/catalog.npz`; no MuJoCo goal rendering or replacement renderer is used. The default policy is epoch 2000 from job 14461123, with its matching contract sidecar. Custom-color goal RGB is rerendered with the same Isaac task, canonical goal lighting, object pose, hand joints, finger aperture and ZED image preprocessing. The original catalog depth channel is retained byte-for-byte: changing color must not introduce a new depth distribution. Fresh Isaac depth is stored separately as `rendered_depth` for diagnostics and is not fed to the actor. Canonical goals predate the live FXAA/background optimization, so the custom-goal renderer retains their original balanced render settings. Catalog/checkpoint hashes are bound into the saved selection at `configs/franka_real_selection.local.json`.

Inference/control runs at 15 Hz. Camera-frame actions are transformed to base-frame TCP twists using the configured training camera rotation and live FR3 TCP TF. MoveIt Servo uses FR3 kinematics. Conservative initial limits are 0.01 m/s, 0.06 rad/s, 15 cm displacement from the starting TCP position, 0.2 rad rotation, with no automatic time cutoff. Fresh camera, state and TF, starting velocity, finger aperture, workspace, external force and torque are checked. The separate watchdog holds measured joint position with zero requested velocity when policy commands expire after 0.15 seconds. These software checks do not replace the robot's physical stop.

## Verified on 2026-09-18

- Camera streaming, RGB-D preprocessing, strict actor checkpoint loading and the browser selection/save flow.
- Live observe-only run: 225 cycles in 15 seconds, median preprocessing/inference 3.17 ms, p95 4.30 ms; no robot commands.
- Isolated fake FR3 + MoveIt Servo command path and timeout hold: 48 zero-velocity hold samples after command expiry.
- Fourteen focused tests covering camera resampling, selection/configuration gates, state limits and HTTP controls.

Evidence is under `artifacts/franka_real_20260918/` and `artifacts/franka_real_runs/`. The saved example selection is part 3 / orientation 001 / grasp g1450 p1; choose the actual desired grasp before use.

## September 21 physical-motion diagnosis

The operator's run `artifacts/franka_real_runs/20260921_114839.json` sent 225 nonzero commands over 15 seconds and stopped on the time limit, not policy completion. Commanded translation reached the configured 0.01 m/s cap, but the operator reported barely any physical movement. The active effort trajectory controller has P gains `[600, 600, 600, 600, 250, 150, 50]` and D gains `[30, 30, 30, 30, 10, 10, 5]`. These observations do not establish where tracking is lost.

The original log recorded requested/limited commands, not measured motion. New runs now log measured TCP position/rotation and displacement, measured joints/velocities, Servo status, and controller reference/feedback/effort with its receipt age. The UI reports **COMMANDING** with measured TCP displacement. `applied_action` is retained for compatibility; it means the limited command sent to Servo, not achieved movement. Restart the policy web server to load this change; no controller gains or speed limits were changed. A passive three-minute trace after the test contained no new policy commands, so it cannot diagnose the moving interval. Evidence is under `artifacts/franka_motion_diagnosis_20260921/`.

## Native velocity control fix

The policy connection now uses `configs/franka_policy_controllers.yaml`: all seven joints claim **velocity**, with ROS PID gains zero and velocity feed-forward 1. Servo's velocity outputs are forwarded immediately, without rebuilding a spline from its short position targets. In this MoveIt version those velocities are derived from filtered positions; they are not themselves a low-pass-filtered velocity signal. Franka's native joint-velocity motion generator and internal joint-impedance controller own physical tracking; no robot-side impedance gains are changed. Policy inference remains 15 Hz, Servo 100 Hz, hardware 1 kHz.

The separate driver overlay at `.cache/franka_velocity_driver` applies a 10 Hz first-order velocity filter at the 1 kHz hardware boundary, followed by the existing libfranka joint-velocity acceleration/jerk limiter. Zero targets use the same bounded deceleration; filtering adds stopping latency (the offline 0.05 rad/s step test falls below 0.0002 rad/s within 100 ms). Physical behavior is not yet validated. Build it with `bash scripts/build_franka_velocity_driver.sh` after the compatible base stack. It leaves the original compatible driver library untouched and refuses to rebuild while its own library is loaded. The launcher requires this overlay for real motion and verifies a manifest against the built library and filter source hashes. A per-robot lock rejects duplicate launches through this entrypoint; it cannot lock out unrelated external control software. State-only and fake launches remain available independently.

The watchdog emits immediate zero velocity on policy expiry, using a zero trajectory timestamp so DDS latency cannot invalidate the stop. JTC also has an independent 0.15 s command timeout in case the watchdog or Servo stalls. Existing policy speed, force, workspace, aperture and travel gates, plus Servo collision/singularity checks, remain in place. New policy sessions reject an active effort controller and request a connection restart. Logs include the final commanded joint velocities as well as measured velocities.

To load the fix, stop the existing **connect** terminal and **policy web-server** terminal with Ctrl+C. Start these again in separate terminals:

```bash
./franka_policy.sh connect
./franka_policy.sh run --execute
```

Start the next bounded test manually from the browser. Do not increase limits for this test. Hardware tracking after this change still needs operator validation; simulated tracking and timeout tests do not establish physical performance. The generic-kernel realtime limitation remains.

The reproducible integration test uses `ROS_DOMAIN_ID=79 ./franka_policy.sh connect fake:=true` and, in another sourced terminal, `ROS_DOMAIN_ID=79 python3 scripts/smoke_franka_policy_control.py`. It verifies the loaded robot is GenericSystem before sending any commands, checks measured motion and normal stop, and freezes only the domain-79 watchdog to verify controller-level expiry. Artifacts are under `artifacts/franka_velocity_fix_20260921/`.

## Still unresolved for reliable physical operation

On September 21, run `20260921_133331` hit the 3 Nm external-torque check after three commanded steps. The operator noticed no contact. A subsequent passive check found stationary TCP feedback (under 10 micrometres per-axis range), zero commanded joint velocities, and estimated external torque 2.52–2.62 Nm at rest. The configured end-effector mass is 1.40 kg, COM (-10, 0, 70) mm relative to flange, with zero additional load. This must be verified against the actual hand/camera/mount; it is not proof of a payload error or absence of contact. Physical execution was disabled in the live workbench. At that time physical testing was paused for payload, cable and contact review. This dated 1.40 kg snapshot is not the current profile; see the September 22 update below.

New logs retain force/torque vectors and the last checked state even if that state triggers a stop; the earlier failing sample was not recorded. Error messages include measured norm and threshold. Force/torque/collision faults now disable execution in the workbench, preventing a simple retry from immediately starting another physical run. These diagnostic/UI changes require a policy-server restart; they do not establish hardware safety.

- FCI was enabled on September 21 and protocol-10 feedback/model access was verified after upgrading libfranka. The subsequent operator policy test had poor physical tracking; successful policy execution is not established. FCI must be re-enabled when the robot requires it.
- The real hand housing occupies substantially more of the image than the saved training goals. Validate camera extrinsics, TCP, axes and image alignment; the existing mount is an assumption, not a measured calibration. ZED reported that automatic self-calibration was skipped for the current low-texture/occluded view.
- Training used a Panda arm; this controller uses the real FR3 model with Panda hand. Visual-policy transfer and control response remain untested on hardware.
- Selecting a resting orientation and grasp supplies a goal image, not the current object pose. The side-by-side view compares live RGB with that exact goal; it is not an Isaac reconstruction of the current real pose. A same-pose reconstruction would additionally require the object pose and connected robot state.
- Without object-pose perception, policy completion is unverified. MoveIt self-collision/joint-limit checks do not supply collision geometry for the real table, part or surroundings. Start near the intended grasp with clearance and inspect an observe-only run first.

The original planning/execution paths and safe real-pipeline defaults remain separate from this standalone workflow.

## Rough-motion investigation and local fixes (September 21)

The 14:01 run was operator-stopped after 11 policy samples spanning 0.667 s. Policy intervals were 66.66–66.73 ms. The noise cause is not established: the 15 Hz log cannot resolve fast vibration. There was one arm controller in the launch logs. The generic-kernel/FIFO warning is a separate unresolved host timing limitation, not proof of the noise cause. No kernel/package/system changes were made or are required by this patch.

The previous 50 Hz joint-state aggregator has been replaced by an event-driven relay (`scripts/franka_policy_joint_states.py`) preserving the arm/hand source timestamps. It does not wait for a timer to publish feedback. Robot TF publication is now limited to 100 Hz instead of the default 20 Hz. Policy starts and command sends reject missing or duplicate publishers on the policy, Servo-output and controller-input command topics. Robot feedback logs additionally include robot-side desired velocities/accelerations and control-command success rate.

The compiled filter test (`scripts/test_franka_velocity_filter.cpp`) uses the exact header built into the hardware overlay, with no Robot/network object. It checks step overshoot, acceleration/jerk bounds, stop convergence and invalid input. The fake ROS integration remains separate: it tests command routing, fresh feedback and timeout stops, but does not execute the Franka hardware filter or validate physical vibration. All hardware motion limits remain unchanged.

### TCP freshness queue fix

After increasing TF publication to 100 Hz, the policy's default 100-message TF subscriber queue accumulated old transforms because the 15 Hz loop drains callbacks for only 4 ms. Read-only reproduction crossed 200 ms age within five ticks, then lagged around 900 ms. The policy now requests KEEP_LAST(1), volatile, best-effort dynamic TF; static TF remains transient-local. No timestamps are rewritten and the 200 ms maximum-age gate remains. Future timestamps beyond 50 ms are rejected as well. Errors and run feedback include the observed TF age. Three independent read-only sessions passed 135 samples (maximum observed age 81.6 ms); disconnecting only the diagnostic TF listener still correctly triggered the freshness stop. Restart the policy web process to load this fix; the connection launch does not need restarting for this subscriber-only change. These checks commanded no arm/gripper motion and do not validate the earlier physical noise issue.

## September 22 operator-requested torque guard adjustment

The operator requested more margin for the estimated external torque. Active selection and new-selection default now use `external_torque_stop_nm: 5.0` (previously 3.0), within the existing 5 Nm configuration ceiling. This is the norm of the estimated Cartesian torque at the stiffness frame, expressed in base axes, not a per-joint torque rating. The 10 N force guard, immediate Franka joint/Cartesian collision-flag check, robot-mode/reflex check, freshness, workspace, travel and controller gates remain active for policy and pickup. No hardware collision thresholds, payload model or torque bias were changed. A detected collision/reflex still stops the software; this cannot guarantee detection of every physical contact.

The preceding read-only September 22 check found about 2.14–2.29 Nm at an essentially stationary TCP, largest external joint residual at joint 2 (~−1.47 Nm), and total configured end-effector mass 0.730 kg with no extra load. The origin of the residual remains unproven. Regression tests confirm that the former 3.01 Nm trip sample now passes, torque above 5 Nm still blocks, and collision flags, reflex mode and the 10 N force guard still block below 5 Nm. Restart the policy workbench to load the saved setting; no controller restart or motion was performed by the agent.
