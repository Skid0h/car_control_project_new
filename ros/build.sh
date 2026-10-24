#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo 'Expected ROS1 Noetic on Ubuntu 20.04. See ros/README.md.' >&2
  exit 1
fi
CAR_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CAR_WS="${CAR_WS:-$CAR_REPO/ros_ws}"
source /opt/ros/noetic/setup.bash
mkdir -p "$CAR_WS/src"
if [[ -e "$CAR_WS/src/car_control_ros" || -L "$CAR_WS/src/car_control_ros" ]]; then
  if [[ "$(readlink -f "$CAR_WS/src/car_control_ros")" != "$CAR_REPO/ros/car_control_ros" ]]; then
    echo 'Workspace already contains another car_control_ros; choose a different CAR_WS.' >&2
    exit 1
  fi
else
  ln -s "$CAR_REPO/ros/car_control_ros" "$CAR_WS/src/car_control_ros"
fi
cd "$CAR_WS"
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
echo "Built. Run: source \"$CAR_WS/devel/setup.bash\""
