"""ZED acquisition and TensorRT share one thread; no stale frame queue."""
import json
import math
import time
from datetime import datetime
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Bool, String
from car_interfaces.msg import Cone, ConeArray
from .common import configuration, value


class Vision(Node):
    def __init__(self):
        super().__init__('car_vision')
        self.camera = None
        self.detector = None
        self.writer = None
        self.recording = False
        self.web = None
        try:
            self.configure()
        except BaseException:
            self.close()
            self.destroy_node()
            raise

    def configure(self):
        if self.get_parameter('use_sim_time').value:
            raise ValueError('Real camera must not use simulated time')
        import cv2
        import pyzed.sl as sl
        from .detector import ConeDetector
        self.cv2, self.sl = cv2, sl
        self.cfg = configuration(self)
        model = value(self, 'model_path', '')
        if not model:
            model = self.cfg.yolo_model_path
        self.cfg.yolo_model_path = str(Path(model).expanduser().resolve())
        if not Path(self.cfg.yolo_model_path).is_file():
            raise FileNotFoundError('Cone-trained TensorRT engine required: ' + self.cfg.yolo_model_path)
        self.cfg.preprocess = value(self, 'preprocess', 'stretch')
        self.detector = ConeDetector(self.cfg)
        self.pub = self.create_publisher(ConeArray, '/car/cones', 1)
        self.status = self.create_publisher(String, '/car/vision_status', 1)
        self.create_subscription(Bool, '/car/record', self.record, 1)
        self.output_dir = Path(value(self, 'recording_dir', str(Path.home() / 'zed_recordings'))).expanduser()
        if value(self, 'web_enabled', False):
            from . import web
            self.web = web
            web.start()
        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.cfg.zed_resolution)
        init.camera_fps = int(self.cfg.zed_fps)
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NONE
        result = self.camera.open(init)
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('ZED open failed: ' + str(result))
        calibration = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.fx, self.cx = calibration.fx, calibration.cx
        if not math.isfinite(self.fx) or self.fx <= 0:
            raise ValueError('Invalid ZED calibration')
        self.runtime = sl.RuntimeParameters()
        self.runtime.enable_depth = False
        self.image = sl.Mat()
        self.colors = {name: Cone.BLUE for name in self.cfg.blue_cones}
        self.colors.update({name: Cone.YELLOW for name in self.cfg.yellow_cones})
        self.colors.update({name: Cone.ORANGE for name in self.cfg.orange_cones})
        self.timeout = value(self, 'sensor_timeout', 0.4)
        self.camera_right = value(self, 'camera_right', 0.0)
        if not 0 < self.timeout <= 1 or not math.isfinite(self.camera_right):
            raise ValueError('Invalid camera timeout/offset')
        self.last_warning = 0.0
        self.record_heartbeat = None
        self.create_timer(1 / min(self.cfg.target_fps, self.cfg.zed_fps), self.tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

    def record(self, msg):
        self.recording = bool(msg.data)
        self.record_heartbeat = time.monotonic()
        if not self.recording and self.writer is not None:
            self.writer.release()
            self.writer = None

    def report(self, connected, error=''):
        self.status.publish(String(data=json.dumps(dict(cam_connected=connected,
                            rec=self.writer is not None, error=error))))

    def tick(self):
        acquired = self.get_clock().now()
        start = time.monotonic()
        try:
            if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('ZED grab failed')
            self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT)
            frame = self.cv2.cvtColor(self.image.get_data(), self.cv2.COLOR_BGRA2BGR)
            detections = self.detector.detect(frame)
            msg = ConeArray()
            # Acquisition timestamp, never completion timestamp.
            msg.header.stamp = acquired.to_msg()
            msg.header.frame_id = 'base_link'
            for det in detections:
                if det['name'] not in self.colors:
                    continue
                x1, y1, x2, y2 = det['bbox']
                area = (x2 - x1) * (y2 - y1)
                if area <= 0:
                    continue
                depth = self.cfg.area_depth_constant / math.sqrt(area)
                if not self.cfg.min_depth < depth <= self.cfg.max_depth:
                    continue
                cone = Cone()
                cone.position.x = float(depth - self.cfg.camera_offset_z)
                cone.position.y = float(-((det['center'][0] - self.cx) * depth / self.fx + self.camera_right))
                cone.color = self.colors[det['name']]
                msg.cones.append(cone)
                self.cv2.rectangle(frame, (x1, y1), (x2, y2),
                                   tuple(self.cfg.cone_colors[det['name']]), 2)
            if time.monotonic() - start <= self.timeout:
                self.pub.publish(msg)
                self.report(True)
            else:
                self.report(True, 'Inference too slow; cones withheld')
            if self.record_heartbeat is None or time.monotonic() - self.record_heartbeat > 1:
                self.recording = False
            if self.recording:
                if self.writer is None:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    filename = self.output_dir / (datetime.now().strftime('zed_%Y%m%d_%H%M%S_%f') + '.avi')
                    height, width = frame.shape[:2]
                    self.writer = self.cv2.VideoWriter(str(filename), self.cv2.VideoWriter_fourcc(*'MJPG'),
                                                       min(self.cfg.target_fps, self.cfg.zed_fps), (width, height))
                    if not self.writer.isOpened():
                        self.writer.release()
                        self.writer = None
                        raise RuntimeError('Cannot open recording file')
                self.writer.write(frame)
            elif self.writer is not None:
                self.writer.release()
                self.writer = None
            if self.web is not None:
                self.web.set_frame(frame)
        except Exception as error:
            self.report(False, str(error))
            if time.monotonic() - self.last_warning > 2:
                self.get_logger().error(str(error))
                self.last_warning = time.monotonic()
            # Never republish old cones. Control process watchdog stops independently.

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        if self.camera is not None:
            self.camera.close()
            self.camera = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Vision()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
