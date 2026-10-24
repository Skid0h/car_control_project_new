# Validation status

Base: ros branch, 2e2e053c4c00561e78ef67a5b24c46204c5f4dae.

- Local Windows / Python 3.12.14: 27 unit/contract tests passed; one real ROS integration test skipped because rclpy is unavailable.
- Python sources parsed using the Python 3.8 grammar; package XML parsed successfully.
- Unit coverage includes 1000 incoming commands/s, slew/angle limits, heartbeat, stale input, invalid values, manual/auto switching, lost camera/PC, orange finish latch, and two simulated laps.
- Detector contract tests use mocked imports and real NumPy; they do NOT execute TensorRT/CUDA.
- ROS 2 Foxy CI is configured for Ubuntu 20.04 / Python 3.8 on AMD64 and ARM64, including colcon build/test, real UDP/rclpy integration and installed launch/entry points.
- Reference Arduino Uno compile is configured in CI; the actual board model is not verified.
- CI outcome is recorded on the GitHub pull request; configuration alone is not a passing build.
- No Jetson, ZED, trained cone engine or physical actuator is attached to this development host. GPU execution, hardware operation, steering calibration and real-time performance remain unverified until the on-device checklist in README.md is completed.

## Executed CI results (2026-09-27)

Run: https://github.com/salimadze2005-beep/car_control_project_new_Danis/actions/runs/36339417168
Tested runtime commit: f08bcfd805c577b2fa63afdcb377ca4283142215.

- ARM64: both ament packages built, Python **3.8.10**, **28 passed / 0 skipped**. Real rclpy/UDP integration passed.
- AMD64: both packages built, Python **3.8.10**, **28 passed / 0 skipped**.
- Installed launch arguments, preflight entry point and generated ConeArray interface checked on both architectures.
- Arduino Uno / AVR: sketch compiled successfully, 4614 bytes program storage and 278 bytes global memory.
- The runner hosts are Ubuntu 24.04; ROS build/test runs inside Ubuntu **20.04** containers. Jetson deployment itself is native, without Docker.
- Commit 99499b46117b79ecbee3593b7c10aa0c7700ddc6 adds STOP centring, timed Servo.detach(), manual STOP latching and firmware time simulation. No physical hardware or GPU inference result is implied.

## STOP and watchdog correction (2026-09-27)

Run: https://github.com/salimadze2005-beep/car_control_project_new_Danis/actions/runs/36340527534
Runtime commit: 99499b46117b79ecbee3593b7c10aa0c7700ddc6.

- ARM64 and AMD64 / Foxy / Python 3.8.10: **30 tests passed, 0 skipped** on each architecture.
- Arduino Uno sketch compiled; deterministic host simulation passed STOP, watchdog, centre, timed detach, restart and malformed packet checks.
- This establishes software behaviour. Mechanical centre, steering under load, servo power and actual Jetson timing remain to be checked on the vehicle.
