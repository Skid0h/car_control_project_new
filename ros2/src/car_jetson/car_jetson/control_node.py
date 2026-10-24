"""Only this process owns Serial. Vision/GPU cannot block its watchdog."""
import json
import socket
import time
from dataclasses import fields
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String
from car_interfaces.msg import ConeArray, Drive
from .actuation import Actuator, Limits
from .core import Parameters
from .session import Session
from .common import configuration, value


class Control(Node):
    def __init__(self):
        super().__init__('car_control')
        self.serial = None
        self.sock = None
        self.session = None
        try:
            self.configure()
        except BaseException:
            self.close()
            self.destroy_node()
            raise

    def configure(self):
        if self.get_parameter('use_sim_time').value:
            raise ValueError('Real car must not use simulated time')
        cfg = configuration(self)
        defaults = Limits(neutral=cfg.neutral_speed, forward=cfg.forward_speed,
                          reverse=cfg.back_speed, center=cfg.center_steering,
                          steering_range=cfg.steering_range)
        params = {f.name: value(self, 'actuator.' + f.name, getattr(defaults, f.name))
                  for f in fields(Limits)}
        self.actuator = Actuator(Limits(**params))
        ctrl = Parameters(**{f.name: value(self, 'controller.' + f.name, getattr(cfg, f.name, f.default))
                             for f in fields(Parameters)})
        self.session = Session(self.actuator, ctrl,
                               value(self, 'sensor_timeout', 0.4),
                               value(self, 'link_timeout', 0.5))
        self.session.capture_speed_limits()
        self.dry_run = value(self, 'dry_run', True)
        self.last_peer = None
        self.camera_state = {}
        self.camera_time = 0.0
        self.last_telemetry = 0.0
        self.last_cone_stamp = -1.0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(ConeArray, '/car/cones', self.cones, qos)
        self.create_subscription(String, '/car/vision_status', self.vision_status, qos)
        self.status_pub = self.create_publisher(String, '/car/status', qos)
        self.drive_pub = self.create_publisher(Drive, '/car/command', qos)
        self.record_pub = self.create_publisher(Bool, '/car/record', qos)
        if not self.dry_run:
            import serial
            from serial.tools.list_ports import comports
            port = value(self, 'serial_port', '')
            if not port:
                candidates = [p.device for p in comports() if p.vid in (0x2341, 0x1A86)
                              or any(s in (p.description or '') for s in ('Arduino', 'CH340', 'USB Serial'))]
                if len(candidates) != 1:
                    raise RuntimeError('Specify serial_port: expected exactly one Arduino, found ' + repr(candidates))
                port = candidates[0]
            self.serial = serial.Serial(port, cfg.baud_rate, timeout=0, write_timeout=0.05,
                                        exclusive=True)
            time.sleep(cfg.arduino_init_delay)
            self.write(self.actuator.packet(time.monotonic()))
            time.sleep(cfg.arduino_post_stop_delay)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((value(self, 'udp_ip', cfg.udp_ip), value(self, 'udp_port', cfg.udp_port)))
        self.timer = self.create_timer(0.01, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('DRY RUN: no serial writes' if self.dry_run else 'Serial enabled; starting MANUAL/stopped')

    def cones(self, msg):
        now_ros = self.get_clock().now().nanoseconds / 1e9
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if msg.header.frame_id != 'base_link' or stamp <= self.last_cone_stamp or not 0 <= now_ros - stamp <= self.session.sensor_timeout:
            return
        self.last_cone_stamp = stamp
        colors = {0: 'blue', 1: 'yellow', 2: 'orange'}
        self.session.cones([(-c.position.y, c.position.x, colors.get(c.color, 'unknown'))
                            for c in msg.cones], time.monotonic(), now_ros - stamp)

    def vision_status(self, msg):
        try:
            self.camera_state = json.loads(msg.data)
            self.camera_time = time.monotonic()
        except (ValueError, TypeError):
            pass

    def write(self, packet):
        if packet is None:
            return
        if self.serial is not None:
            if self.serial.write(packet) != len(packet):
                raise IOError('Partial Serial write; stop and restart required')

    def tick(self):
        # Bounded work prevents a UDP flood from starving the watchdog.
        for _ in range(32):
            try:
                data, peer = self.sock.recvfrom(256)
            except BlockingIOError:
                break
            try:
                text = data.decode('ascii').strip()
            except UnicodeDecodeError:
                continue
            now = time.monotonic()
            if self.last_peer is not None and peer != self.last_peer and self.session.link is not None and now - self.session.link < self.session.link_timeout:
                continue
            if self.session.receive(text, now):
                self.last_peer = peer
            if text in ('S', 'Q', 'F') or self.session.message.startswith('Rejected'):
                self.write(self.session.tick(now))
                break
        now = time.monotonic()
        packet = self.session.tick(now)
        self.write(packet)
        if packet is not None:
            motor, angle = self.actuator.last_packet
            drive = Drive()
            drive.header.stamp = self.get_clock().now().to_msg()
            drive.header.frame_id = 'base_link'
            endpoint = self.actuator.p.forward if motor >= self.actuator.p.neutral else self.actuator.p.reverse
            span = abs(endpoint - self.actuator.p.neutral)
            drive.throttle = float((motor - self.actuator.p.neutral) / span if span else 0)
            drive.steering = float((self.actuator.p.center - angle) / self.actuator.p.steering_range)
            drive.brake = float(motor == self.actuator.p.neutral)
            self.drive_pub.publish(drive)
        if now - self.last_telemetry >= 0.1:
            self.last_telemetry = now
            camera_fresh = now - self.camera_time <= 0.5
            state = dict(mode='AUTO' if self.session.auto else 'MANUAL',
                         rec=bool(camera_fresh and self.camera_state.get('rec')),
                         cam_connected=bool(camera_fresh and self.camera_state.get('cam_connected')),
                         arduino_connected=self.serial is not None, dry_run=self.dry_run,
                         fwd=self.actuator.p.forward, bck=self.actuator.p.reverse,
                         msg=self.session.message or self.camera_state.get('error', ''),
                         finished=self.session.controller.finished)
            self.record_pub.publish(Bool(data=self.session.record))
            payload = json.dumps(state)
            self.status_pub.publish(String(data=payload))
            if self.last_peer:
                try:
                    self.sock.sendto(payload.encode('utf-8'), self.last_peer)
                except OSError:
                    pass
        if self.session.quit:
            raise SystemExit(0)

    def close(self):
        if self.session is not None:
            self.actuator.stop()
            try:
                self.write(self.actuator.packet(time.monotonic()))
            except Exception:
                pass  # Board watchdog is the fallback if USB already failed.
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Control()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
