#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_simple_dual_robot.sh --mode sim [options]
  ./run_simple_dual_robot.sh --mode real [options]

One-command dual-robot vertical slice:
  sim   starts mock MoveIt, plans a fresh pair, then executes it in Isaac
  real  starts hardware MoveIt, builds fresh targets, then preflights or executes live

Common options:
  --assembly NAME              Assembly under the artifact root.
                               Default: plumbers_block
  --incoming-part-id ID        Selected-order part to bring to pre-insertion.
                               Default: first holder-active step (part 0 for plumbers_block)
  --artifact-root PATH         Default: artifacts/dual_grasp_planning
  --artifact-dir PATH
  --pair-id ID
  --holder-grasp-id ID
  --max-pair-attempts N         Default: 256
  --assembly-x M                Default: 0.55
  --assembly-y M                Default: 0.0
  --assembly-z M                Default: --floor-z (assembled prefix supported)
  --assembly-yaw-deg DEG        Default: 0
  --pickup-x M                  Default: 0.55
  --pickup-y M                  Default: 0.28
  --pickup-roll-deg DEG         Default: 0
  --pickup-pitch-deg DEG        Default: 0
  --pickup-yaw-deg DEG          Default: 0
  --inserter-arm NAME           lbr_one, lbr_two, or auto. Default: lbr_two
  --step-id ID                  Explicit compatibility override.
  --floor-z M                   Default: -0.030
  --plan-output PATH            Simulation MoveIt plan artifact.
  --task-output PATH            Real target-only task artifact.
  --attempt-output PATH         PITL/Isaac or real attempt artifact.
  --ros-domain-id ID            Default: DUAL_ROBOT_ROS_DOMAIN_ID,
                                then ROS_DOMAIN_ID, then 0
  --rviz
  --reuse-moveit               Reuse an already-running /lbr_dual_arm stack
  --keep-moveit                Leave a stack started by this script running
  --no-planning-debug-gui      Do not open the live pair/phase browser view.
                               It is enabled for visible sim and real runs.
  --debug-gui-port N           Stable real-mode debugger port. Default: 38825
                               (or DUAL_REAL_PLANNING_DEBUG_GUI_PORT).

Simulation options:
  --headless
  --holder-only
  --skip-joint-space-ranking   Keep Stage-3 order without seeded MoveIt transition ranking.
  --joint-rank-candidates N    Candidates to pre-plan in joint space. Default: 8
  --record-video PATH
  --isaac-python PATH
  --static-friction VALUE       Isaac contact friction. Default: 5.0
  --dynamic-friction VALUE      Isaac contact friction. Default: 4.0
  --gripper-effort-limit VALUE  Isaac finger effort limit. Default: 200
  --critical-damping-ratio ZETA Configuration-adaptive joint damping. Default: 1.0
  --gripper-close-duration-s S  Gentle quintic finger close duration. Default: 3.0
  --finger-contact-min-force-n N Bilateral contact latch threshold. Default: 0.25
  --gripper-contact-preload-m M Minimum inward preload after contact. Default: 0.0004

Real options:
  --execute                     Without this, only non-moving target IK is checked
  --allow-objectless-planning   Required for motion in the current simple scene
  --stop-after PHASE            Default: holder_pregrasp
  --skip-grippers               Only valid through holder_pregrasp
  --yes                         Skip typed confirmation
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
ASSEMBLY=""
INCOMING_PART_ID=""
ARTIFACT_ROOT="artifacts/dual_grasp_planning"
ARTIFACT_DIR=""
PAIR_ID=""
HOLDER_GRASP_ID=""
MAX_PAIR_ATTEMPTS="256"
ASSEMBLY_X="0.55"
ASSEMBLY_Y="0.0"
ASSEMBLY_Z=""
ASSEMBLY_YAW="0.0"
PICKUP_X="0.55"
PICKUP_Y="0.28"
PICKUP_ROLL="0.0"
PICKUP_PITCH="0.0"
PICKUP_YAW="0.0"
INSERTER_ARM="lbr_two"
STEP_ID=""
FLOOR_Z="-0.030"
PLAN_OUTPUT=""
TASK_OUTPUT=""
ATTEMPT_OUTPUT=""
ROS_DOMAIN_VALUE="${DUAL_ROBOT_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
RVIZ=0
REUSE_MOVEIT=0
KEEP_MOVEIT=0
HEADLESS=0
HOLDER_ONLY=0
PLANNING_DEBUG_GUI=1
PLANNING_DEBUG_GUI_PORT="${DUAL_REAL_PLANNING_DEBUG_GUI_PORT:-38825}"
JOINT_SPACE_RANKING=1
JOINT_RANK_CANDIDATES="8"
RECORD_VIDEO=""
ISAAC_PYTHON="${ISAAC_PYTHON:-/media/pdz/Elements1/IsaacLab/isaaclab.sh}"
STATIC_FRICTION="5.0"
DYNAMIC_FRICTION="4.0"
GRIPPER_EFFORT_LIMIT="200.0"
CRITICAL_DAMPING_RATIO="1.0"
GRIPPER_CLOSE_DURATION="3.0"
FINGER_CONTACT_MIN_FORCE="0.25"
GRIPPER_CONTACT_PRELOAD="0.0004"
EXECUTE_REAL=0
ALLOW_OBJECTLESS=0
STOP_AFTER="holder_pregrasp"
SKIP_GRIPPERS=0
ASSUME_YES=0
MOVEIT_PID=""

source_if_exists() {
  local setup_file="$1"
  if [[ -f "${setup_file}" ]]; then
    local nounset_was_enabled=0
    case "$-" in
      *u*) nounset_was_enabled=1 ;;
    esac
    set +u
    # shellcheck source=/dev/null
    source "${setup_file}"
    if [[ "${nounset_was_enabled}" -eq 1 ]]; then
      set -u
    fi
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${MOVEIT_PID}" && "${KEEP_MOVEIT}" -eq 0 ]]; then
    echo "[DUAL-RUN] Stopping MoveIt stack started by this command."
    kill "${MOVEIT_PID}" 2>/dev/null || true
    wait "${MOVEIT_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --assembly) ASSEMBLY="${2:-}"; shift 2 ;;
    --incoming-part-id) INCOMING_PART_ID="${2:-}"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="${2:-}"; shift 2 ;;
    --artifact-dir) ARTIFACT_DIR="${2:-}"; shift 2 ;;
    --pair-id) PAIR_ID="${2:-}"; shift 2 ;;
    --holder-grasp-id) HOLDER_GRASP_ID="${2:-}"; shift 2 ;;
    --max-pair-attempts) MAX_PAIR_ATTEMPTS="${2:-}"; shift 2 ;;
    --assembly-x) ASSEMBLY_X="${2:-}"; shift 2 ;;
    --assembly-y) ASSEMBLY_Y="${2:-}"; shift 2 ;;
    --assembly-z) ASSEMBLY_Z="${2:-}"; shift 2 ;;
    --assembly-yaw-deg) ASSEMBLY_YAW="${2:-}"; shift 2 ;;
    --pickup-x) PICKUP_X="${2:-}"; shift 2 ;;
    --pickup-y) PICKUP_Y="${2:-}"; shift 2 ;;
    --pickup-roll-deg) PICKUP_ROLL="${2:-}"; shift 2 ;;
    --pickup-pitch-deg) PICKUP_PITCH="${2:-}"; shift 2 ;;
    --pickup-yaw-deg) PICKUP_YAW="${2:-}"; shift 2 ;;
    --inserter-arm) INSERTER_ARM="${2:-}"; shift 2 ;;
    --step-id) STEP_ID="${2:-}"; shift 2 ;;
    --floor-z) FLOOR_Z="${2:-}"; shift 2 ;;
    --plan-output) PLAN_OUTPUT="${2:-}"; shift 2 ;;
    --task-output) TASK_OUTPUT="${2:-}"; shift 2 ;;
    --attempt-output) ATTEMPT_OUTPUT="${2:-}"; shift 2 ;;
    --ros-domain-id) ROS_DOMAIN_VALUE="${2:-}"; shift 2 ;;
    --rviz) RVIZ=1; shift ;;
    --reuse-moveit) REUSE_MOVEIT=1; shift ;;
    --keep-moveit) KEEP_MOVEIT=1; shift ;;
    --headless) HEADLESS=1; shift ;;
    --holder-only) HOLDER_ONLY=1; shift ;;
    --no-planning-debug-gui) PLANNING_DEBUG_GUI=0; shift ;;
    --debug-gui-port) PLANNING_DEBUG_GUI_PORT="${2:-}"; shift 2 ;;
    --skip-joint-space-ranking) JOINT_SPACE_RANKING=0; shift ;;
    --joint-rank-candidates) JOINT_RANK_CANDIDATES="${2:-}"; shift 2 ;;
    --record-video) RECORD_VIDEO="${2:-}"; shift 2 ;;
    --isaac-python) ISAAC_PYTHON="${2:-}"; shift 2 ;;
    --static-friction) STATIC_FRICTION="${2:-}"; shift 2 ;;
    --dynamic-friction) DYNAMIC_FRICTION="${2:-}"; shift 2 ;;
    --gripper-effort-limit) GRIPPER_EFFORT_LIMIT="${2:-}"; shift 2 ;;
    --critical-damping-ratio) CRITICAL_DAMPING_RATIO="${2:-}"; shift 2 ;;
    --gripper-close-duration-s) GRIPPER_CLOSE_DURATION="${2:-}"; shift 2 ;;
    --finger-contact-min-force-n) FINGER_CONTACT_MIN_FORCE="${2:-}"; shift 2 ;;
    --gripper-contact-preload-m) GRIPPER_CONTACT_PRELOAD="${2:-}"; shift 2 ;;
    --execute) EXECUTE_REAL=1; shift ;;
    --allow-objectless-planning) ALLOW_OBJECTLESS=1; shift ;;
    --stop-after) STOP_AFTER="${2:-}"; shift 2 ;;
    --skip-grippers) SKIP_GRIPPERS=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[DUAL-RUN] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "sim" && "${MODE}" != "real" ]]; then
  echo "[DUAL-RUN] --mode must be sim or real." >&2
  exit 1
fi
if [[ ! "${ROS_DOMAIN_VALUE}" =~ ^[0-9]+$ ]]; then
  echo "[DUAL-RUN] --ros-domain-id must be a non-negative integer." >&2
  exit 1
fi
if [[ ! "${JOINT_RANK_CANDIDATES}" =~ ^[0-9]+$ ]]; then
  echo "[DUAL-RUN] --joint-rank-candidates must be a non-negative integer." >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source_if_exists "/home/pdz/lbr-stack/install/setup.bash"
source_if_exists "${SCRIPT_DIR}/ros2_ws/install/setup.bash"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros-log}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_VALUE}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_DISCOVERY_SERVER
unset ROS_STATIC_PEERS
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE

trap cleanup EXIT INT TERM

MOVEIT_SERVICES="$(
  timeout 5s ros2 service list --no-daemon --spin-time 1.0 2>/dev/null || true
)"
MOVEIT_READY=0
if grep -qx "/lbr_dual_arm/compute_ik" <<<"${MOVEIT_SERVICES}" \
  && grep -qx "/lbr_dual_arm/plan_kinematic_path" <<<"${MOVEIT_SERVICES}"; then
  MOVEIT_READY=1
fi

if [[ "${MOVEIT_READY}" -eq 1 ]]; then
  if [[ "${REUSE_MOVEIT}" -ne 1 ]]; then
    echo "[DUAL-RUN] A dual MoveIt stack already exists on ROS domain ${ROS_DOMAIN_ID}." >&2
    echo "[DUAL-RUN] Stop it first, or pass --reuse-moveit after confirming it matches mode=${MODE}." >&2
    exit 1
  fi
  echo "[DUAL-RUN] Reusing the existing dual MoveIt stack."
else
  if [[ "${REUSE_MOVEIT}" -eq 1 ]]; then
    echo "[DUAL-RUN] --reuse-moveit was requested, but the live MoveIt services are not ready on ROS domain ${ROS_DOMAIN_ID}." >&2
    echo "[DUAL-RUN] Missing /lbr_dual_arm/compute_ik or /lbr_dual_arm/plan_kinematic_path; restart the selected MoveIt stack." >&2
    exit 1
  fi
  START_ARGS=(--mode "$([[ "${MODE}" == "sim" ]] && printf mock || printf hardware)")
  START_ARGS+=(--ros-domain-id "${ROS_DOMAIN_ID}")
  if [[ "${RVIZ}" -eq 1 ]]; then
    START_ARGS+=(--rviz)
  fi
  ./start_dual_lbr_moveit.sh "${START_ARGS[@]}" &
  MOVEIT_PID="$!"
  echo "[DUAL-RUN] Waiting for dual MoveIt services."
  ready=0
  for _ in $(seq 1 60); do
    if ros2 service type /lbr_dual_arm/compute_ik >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "[DUAL-RUN] Dual MoveIt did not become ready within 60 seconds." >&2
    exit 1
  fi
fi

COMMON_TASK_ARGS=(
  --artifact-root "${ARTIFACT_ROOT}"
  --assembly-x "${ASSEMBLY_X}"
  --assembly-y "${ASSEMBLY_Y}"
  --assembly-yaw-deg "${ASSEMBLY_YAW}"
  --pickup-x "${PICKUP_X}"
  --pickup-y "${PICKUP_Y}"
  --pickup-roll-deg "${PICKUP_ROLL}"
  --pickup-pitch-deg "${PICKUP_PITCH}"
  --pickup-yaw-deg "${PICKUP_YAW}"
  --floor-z "${FLOOR_Z}"
)
if [[ -n "${ASSEMBLY_Z}" ]]; then
  COMMON_TASK_ARGS+=(--assembly-z "${ASSEMBLY_Z}")
fi
if [[ -n "${ASSEMBLY}" ]]; then
  COMMON_TASK_ARGS+=(--assembly "${ASSEMBLY}")
fi
if [[ -n "${INCOMING_PART_ID}" ]]; then
  COMMON_TASK_ARGS+=(--incoming-part-id "${INCOMING_PART_ID}")
fi
if [[ -n "${ARTIFACT_DIR}" ]]; then
  COMMON_TASK_ARGS+=(--artifact-dir "${ARTIFACT_DIR}")
fi
if [[ -n "${STEP_ID}" ]]; then
  COMMON_TASK_ARGS+=(--step-id "${STEP_ID}")
fi
if [[ -n "${PAIR_ID}" ]]; then
  COMMON_TASK_ARGS+=(--pair-id "${PAIR_ID}")
fi
if [[ -n "${HOLDER_GRASP_ID}" ]]; then
  COMMON_TASK_ARGS+=(--holder-grasp-id "${HOLDER_GRASP_ID}")
fi

SELECTED_ARTIFACT_DIR="${ARTIFACT_DIR}"
if [[ -z "${SELECTED_ARTIFACT_DIR}" ]]; then
  SELECTED_ARTIFACT_DIR="${ARTIFACT_ROOT}/${ASSEMBLY:-plumbers_block}"
fi
OUTPUT_SUFFIX=""
if [[ -n "${STEP_ID}" ]]; then
  OUTPUT_SUFFIX="_${STEP_ID}"
elif [[ -n "${INCOMING_PART_ID}" ]]; then
  OUTPUT_SUFFIX="_part_${INCOMING_PART_ID}"
fi

if [[ "${MODE}" == "sim" ]]; then
  PLAN_PATH="${PLAN_OUTPUT:-${SELECTED_ARTIFACT_DIR}/simple_dual_robot_sim_plan${OUTPUT_SUFFIX}.json}"
  PLAN_ARGS=()
  if [[ "${HOLDER_ONLY}" -eq 1 ]]; then
    PLAN_ARGS+=(--holder-only)
  fi
  if [[ "${HEADLESS}" -eq 0 && "${PLANNING_DEBUG_GUI}" -eq 1 ]]; then
    PLAN_ARGS+=(--debug-gui)
  fi
  if [[ "${JOINT_SPACE_RANKING}" -eq 0 ]]; then
    PLAN_ARGS+=(--skip-joint-space-ranking)
  fi
  PLAN_ARGS+=(--joint-rank-candidates "${JOINT_RANK_CANDIDATES}")
  PLAN_ARGS+=(--inserter-arm "${INSERTER_ARM}")
  python3 scripts/plan_simple_dual_robot_sim.py \
    "${COMMON_TASK_ARGS[@]}" \
    --max-pair-attempts "${MAX_PAIR_ATTEMPTS}" \
    "${PLAN_ARGS[@]}" \
    --output "${PLAN_PATH}"
  ISAAC_ARGS=(
    -p scripts/run_simple_dual_robot_sim_in_isaac.py
    --plan-json "${PLAN_PATH}"
  )
  ISAAC_ARGS+=(
    --attempt-artifact
    "${ATTEMPT_OUTPUT:-${SELECTED_ARTIFACT_DIR}/simple_dual_robot_sim_attempt${OUTPUT_SUFFIX}.json}"
    --static-friction "${STATIC_FRICTION}"
    --dynamic-friction "${DYNAMIC_FRICTION}"
    --gripper-effort-limit "${GRIPPER_EFFORT_LIMIT}"
    --critical-damping-ratio "${CRITICAL_DAMPING_RATIO}"
    --gripper-close-duration-s "${GRIPPER_CLOSE_DURATION}"
    --finger-contact-min-force-n "${FINGER_CONTACT_MIN_FORCE}"
    --gripper-contact-preload-m "${GRIPPER_CONTACT_PRELOAD}"
  )
  if [[ "${HEADLESS}" -eq 1 ]]; then
    ISAAC_ARGS+=(--headless)
  fi
  if [[ "${HOLDER_ONLY}" -eq 1 ]]; then
    ISAAC_ARGS+=(--holder-only)
  fi
  if [[ -n "${RECORD_VIDEO}" ]]; then
    ISAAC_ARGS+=(--record-video "${RECORD_VIDEO}")
  fi
  TERM=xterm "${ISAAC_PYTHON}" "${ISAAC_ARGS[@]}"
  exit 0
fi

TASK_PATH="${TASK_OUTPUT:-${SELECTED_ARTIFACT_DIR}/simple_dual_robot_real_task${OUTPUT_SUFFIX}.json}"
TASK_BUILD_ARGS=(
  "${COMMON_TASK_ARGS[@]}"
  --max-pair-candidates "${MAX_PAIR_ATTEMPTS}"
  --output "${TASK_PATH}"
)
if [[ "${PLANNING_DEBUG_GUI}" -eq 1 ]]; then
  TASK_BUILD_ARGS+=(--debug-gui --debug-gui-port "${PLANNING_DEBUG_GUI_PORT}")
fi
python3 scripts/build_simple_dual_robot_task.py "${TASK_BUILD_ARGS[@]}"

REAL_ARGS=(
  --plan-json "${TASK_PATH}"
  --stop-after "${STOP_AFTER}"
  --attempt-artifact
  "${ATTEMPT_OUTPUT:-${SELECTED_ARTIFACT_DIR}/simple_dual_robot_real_attempt${OUTPUT_SUFFIX}.json}"
)
if [[ "${EXECUTE_REAL}" -eq 1 ]]; then
  REAL_ARGS+=(--execute)
fi
if [[ "${ALLOW_OBJECTLESS}" -eq 1 ]]; then
  REAL_ARGS+=(--allow-objectless-planning)
fi
if [[ "${SKIP_GRIPPERS}" -eq 1 ]]; then
  REAL_ARGS+=(--skip-grippers)
fi
if [[ "${ASSUME_YES}" -eq 1 ]]; then
  REAL_ARGS+=(--yes)
fi
if [[ "${PLANNING_DEBUG_GUI}" -eq 1 ]]; then
  REAL_ARGS+=(
    --debug-gui
    --debug-gui-port "${PLANNING_DEBUG_GUI_PORT}"
    --no-debug-gui-open-browser
  )
fi
python3 scripts/run_simple_dual_robot_real.py "${REAL_ARGS[@]}"
