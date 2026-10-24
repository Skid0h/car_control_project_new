# Applied command telemetry, NOT an actuator input. throttle [-1,1], steering positive RIGHT.
# brake=1 means neutral ESC, not an active mechanical brake.
std_msgs/Header header
float32 throttle
float32 steering
float32 brake
