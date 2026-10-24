#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${ROS_VERSION:-}" == "1" ]]; then
  echo "Open a fresh shell without sourcing ROS 1 Noetic." >&2
  exit 1
fi
if [[ ! -f /opt/ros/foxy/setup.bash ]]; then
  echo "Install ROS 2 Foxy first; see README.md." >&2
  exit 1
fi
set +u
source /opt/ros/foxy/setup.bash
set -u
colcon build --symlink-install --base-paths src --executor sequential
set +u
source install/setup.bash
set -u
colcon test --base-paths src --event-handlers console_direct+
colcon test-result --verbose
