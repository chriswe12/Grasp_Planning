#!/usr/bin/env bash
# Separate hardware overlay: never replace libraries mapped by a running robot.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" != --clean-environment ]]; then
  exec env -i HOME="$HOME" USER="${USER:-pdz}" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    LANG=C.UTF-8 PYTHONNOUSERSITE=1 bash "$0" --clean-environment
fi
base="$root/.cache/franka_compatible"
motion="$root/.cache/franka_velocity_driver"
python3 - "$motion/install/franka_hardware/lib/libfranka_hardware.so" <<'PY'
import sys
from pathlib import Path
library = sys.argv[1]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        maps = (proc / 'maps').read_text()
    except OSError:
        continue
    if library in maps:
        sys.exit('Velocity driver is in use. Stop the connect terminal before rebuilding.')
PY
test -f "$base/install/local_setup.bash" || { echo 'Build the compatible stack first.' >&2; exit 1; }
mkdir -p "$motion/src"
cp -a "$base/src/franka_ros2/franka_hardware" "$motion/src/"
# Patch only the isolated overlay, using the same filter tested offline.
cp "$root/scripts/franka_policy_velocity_filter.hpp" "$motion/src/franka_hardware/include/franka_hardware/"
python3 - "$motion/src/franka_hardware" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
p = root / 'include/franka_hardware/robot.hpp'
s = p.read_text()
old = 'bool velocity_command_rate_limit_active_{false};'
assert s.count(old) == 1, 'Unexpected pinned driver source'
p.write_text(s.replace(old, 'bool velocity_command_rate_limit_active_{true};'))
p = root / 'src/robot.cpp'
s = p.read_text()
start = s.index('  // If you are experiencing issues', s.index('void Robot::writeOnceJointVelocities'))
end = s.index('  active_control_->writeOnce(velocity_command);', start)
s = s[:start] + '''  velocity_command.dq = franka_policy::filterVelocity(
      velocities, current_state_.q_d, current_state_.dq_d, current_state_.ddq_d);

''' + s[end:]
s = '#include <franka_hardware/franka_policy_velocity_filter.hpp>\n' + s
p.write_text(s)
PY
set +u
source /opt/ros/humble/setup.bash
source "$base/install/local_setup.bash"
export CMAKE_PREFIX_PATH="$base/libfranka:/opt/openrobots:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$base/libfranka/lib:/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"
export MAKEFLAGS=-j3
colcon --log-base "$motion/log" build --base-paths "$motion/src" \
  --build-base "$motion/build" --install-base "$motion/install" \
  --packages-select franka_hardware --allow-overriding franka_hardware \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DFranka_DIR="$base/libfranka/lib/cmake/Franka"
python3 - "$motion" "$root/scripts/franka_policy_velocity_filter.hpp" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
library = root / 'install/franka_hardware/lib/libfranka_hardware.so'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'policy_driver_manifest.json').write_text(json.dumps(dict(
    version=2, library_sha256=digest(library), filter_sha256=digest(Path(sys.argv[2]))
), indent=2) + '\n')
PY
