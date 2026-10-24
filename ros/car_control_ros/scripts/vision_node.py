#!/usr/bin/env python3
"""Feed ROS images to the project's TensorRT detector, on one worker thread."""
import math
import threading
import time
from pathlib import Path
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from car_control_ros.core import fresh
from car_control_ros.legacy import configuration, load_module
from car_control_ros.msg import Cone, ConeArray


def main():
    rospy.init_node('car_vision')
    root, config = configuration()
    config.yolo_model_path = str(Path(rospy.get_param('~model_path', config.yolo_model_path)).expanduser())
    if not Path(config.yolo_model_path).is_file():
        raise FileNotFoundError('Provide ~model_path to your cone TensorRT engine: ' + config.yolo_model_path)
    # Construct and call on the same thread: legacy detector owns a CUDA context.
    detector = load_module(root / 'Code' / 'Cone_detector.py', 'legacy_detector').ConeDetector(config)
    if detector.engine is None:
        raise RuntimeError('TensorRT engine failed to initialize')
    bridge = CvBridge()
    pub = rospy.Publisher('/car/cones', ConeArray, queue_size=1)
    lock = threading.Lock()
    latest = [None]
    calibration = [None]
    timeout = float(rospy.get_param('~sensor_timeout', 0.4))
    fov = float(rospy.get_param('~horizontal_fov_deg', 90.0))
    area_constant = float(rospy.get_param('~area_depth_constant', config.area_depth_constant))
    # Camera pose in vehicle coordinates; default is centred, aligned simulation camera.
    camera_forward = float(rospy.get_param('~camera_forward', 0.0))
    camera_left = float(rospy.get_param('~camera_left', 0.0))
    if not 0 < fov < 180 or timeout <= 0 or area_constant <= 0:
        raise ValueError('Invalid vision configuration')

    def receive(msg):
        with lock:
            latest[0] = (msg, time.monotonic())

    def info(msg):
        with lock:
            calibration[0] = msg

    rospy.Subscriber('~image', Image, receive, queue_size=1, buff_size=2**24)
    rospy.Subscriber('~camera_info', CameraInfo, info, queue_size=1)
    colors = {name: Cone.BLUE for name in config.blue_cones}
    colors.update({name: Cone.YELLOW for name in config.yellow_cones})
    colors.update({name: Cone.ORANGE for name in config.orange_cones})
    while not rospy.is_shutdown():
        with lock:
            sample, latest[0] = latest[0], None
            cam = calibration[0]
        if sample is None:
            time.sleep(0.005)
            continue
        msg, received = sample
        if not fresh(msg.header.stamp.to_sec(), rospy.Time.now().to_sec(), timeout):
            continue
        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = frame.shape[:2]
            if cam is not None and cam.width == width and cam.height == height and cam.K[0] > 0:
                fx, cx = cam.K[0], cam.K[2]
            else:
                fx, cx = width / (2 * math.tan(math.radians(fov) / 2)), width / 2
                rospy.logwarn_throttle(10, 'Using configured FOV; CameraInfo not available/matching')
            detections = detector.detect(frame)
            result = ConeArray()
            # Preserve original acquisition time, including inference latency.
            result.header.stamp, result.header.frame_id = msg.header.stamp, 'base_link'
            kept = []
            for det in sorted(detections, key=lambda d: -float(d['conf'])):
                if det['name'] not in colors:
                    continue
                u, v = det['center']
                if any(name == det['name'] and (u - ku)**2 + (v - kv)**2 < 625 for ku, kv, name in kept):
                    continue
                kept.append((u, v, det['name']))
                x1, y1, x2, y2 = det['bbox']
                depth = area_constant / math.sqrt(max(1, (x2 - x1) * (y2 - y1)))
                cone = Cone()
                cone.position.x = depth + camera_forward
                cone.position.y = -(u - cx) * depth / fx + camera_left
                cone.color = colors[det['name']]
                result.cones.append(cone)
            if fresh(received, time.monotonic(), timeout) and fresh(msg.header.stamp.to_sec(), rospy.Time.now().to_sec(), timeout):
                pub.publish(result)
        except Exception as error:
            rospy.logerr_throttle(2, 'Vision failed: %s', error)
            # No refresh of old detections; downstream watchdog applies the brake.


if __name__ == '__main__':
    main()
