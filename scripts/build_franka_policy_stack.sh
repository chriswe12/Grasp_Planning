#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Do not inherit an old colcon overlay, custom trajectory controller or Python site.
if [[ "${1:-}" != --clean-environment ]]; then
  exec env -i HOME="$HOME" USER="${USER:-pdz}" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    LANG=C.UTF-8 PYTHONNOUSERSITE=1 bash "$0" --clean-environment
fi
stack="$root/.cache/franka_compatible"
mkdir -p "$stack/src"
fetch_release() {
  local name="$1" tag="$2" sha="$3"
  if [[ ! -d "$stack/src/$name/.git" ]]; then
    git clone --depth 1 --branch "$tag" --recurse-submodules --shallow-submodules \
      "https://github.com/frankarobotics/$name.git" "$stack/src/$name"
  fi
  [[ "$(git -C "$stack/src/$name" rev-parse HEAD)" == "$sha" ]] || {
    echo "Unexpected checkout for $name; refusing to replace it." >&2; exit 1;
  }
}
fetch_release libfranka 0.18.0 c5c66bb25a987f122bb424549f5e3b283a41d2f2
fetch_release franka_ros2 v2.0.2 43d2535b135c9d49be42921fdf6b7a46d19fe38c
fetch_release franka_description 1.0.1 566ba44abc48be2fca16050c406562e22f85793b
# Humble's semantic-component headers belong to controller_interface; v2.0.2
# omits this direct build dependency. Keep the minimal compatibility patch explicit.
if git -C "$stack/src/franka_ros2" apply --check "$root/scripts/franka_ros2_humble.patch" 2>/dev/null; then
  git -C "$stack/src/franka_ros2" apply "$root/scripts/franka_ros2_humble.patch"
else
  git -C "$stack/src/franka_ros2" apply --reverse --check "$root/scripts/franka_ros2_humble.patch"
fi
cmake -S "$stack/src/libfranka" -B "$stack/build-libfranka" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$stack/libfranka" \
  -DCMAKE_PREFIX_PATH=/opt/openrobots -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build "$stack/build-libfranka" -j3
cmake --install "$stack/build-libfranka"
c++ -std=c++17 "$root/scripts/franka_read_state.cpp" -I"$stack/libfranka/include" \
  -L"$stack/libfranka/lib" -Wl,-rpath,"$stack/libfranka/lib" -Wl,-rpath,/opt/openrobots/lib \
  -lfranka -o "$stack/read_state"
# ROS setup scripts are not nounset-safe.
set +u
source /opt/ros/humble/setup.bash
export CMAKE_PREFIX_PATH="$stack/libfranka:/opt/openrobots:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$stack/libfranka/lib:/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"
export MAKEFLAGS=-j3
colcon --log-base "$stack/log" build \
  --base-paths "$stack/src/franka_ros2" "$stack/src/franka_description" \
  --build-base "$stack/build" --install-base "$stack/install" \
  --packages-up-to franka_hardware franka_gripper franka_robot_state_broadcaster franka_fr3_moveit_config franka_description \
  --executor sequential --cmake-clean-cache \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DFranka_DIR="$stack/libfranka/lib/cmake/Franka"
