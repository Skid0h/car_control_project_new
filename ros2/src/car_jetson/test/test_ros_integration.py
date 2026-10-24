"""Runs with real rclpy/generated messages after colcon build; no hardware."""
import importlib.util
import json
from pathlib import Path
import socket
import time
import unittest

HAS_ROS = importlib.util.find_spec('rclpy') is not None


@unittest.skipUnless(HAS_ROS, 'ROS 2 not installed on this test host')
class RosIntegrationTests(unittest.TestCase):
    def test_udp_cones_watchdog_and_shutdown(self):
        import rclpy
        from rclpy.node import Node
        from car_interfaces.msg import Cone, ConeArray
        from car_jetson.control_node import Control
        config = str(Path(__file__).resolve().parents[1] / 'config/project.jsonc')
        rclpy.init(args=['--ros-args', '-p', 'udp_port:=0', '-p', 'dry_run:=true',
                        '-p', 'project_config:=' + config])
        control = None
        probe = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        try:
            control = Control()
            probe = Node('integration_probe', use_global_arguments=False)
            pub = probe.create_publisher(ConeArray, '/car/cones', 1)
            target = ('127.0.0.1', control.sock.getsockname()[1])
            def spin(seconds):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    rclpy.spin_once(control, timeout_sec=0.005)
                    rclpy.spin_once(probe, timeout_sec=0.005)
            def send(text):
                sock.sendto(text.encode(), target)
                spin(0.06)
            spin(0.4)
            send('1,0')
            self.assertEqual(control.actuator.last_packet[0], 1570)
            send('nan,0')
            self.assertEqual(control.actuator.last_packet[0], 1500)
            send('A')
            msg = ConeArray()
            msg.header.frame_id = 'base_link'
            msg.header.stamp = probe.get_clock().now().to_msg()
            for y, color in ((0.75, Cone.BLUE), (-0.75, Cone.YELLOW)):
                c = Cone()
                c.position.x, c.position.y, c.color = 1.0, y, color
                msg.cones.append(c)
            pub.publish(msg)
            spin(0.1)
            self.assertGreater(control.actuator.last_packet[0], 1500)
            # Keep PC alive but freeze vision.
            for _ in range(6):
                send('0,0')
                spin(0.04)
            self.assertEqual(control.actuator.last_packet[0], 1500)
            self.assertFalse(control.session.auto)
            send('R')
            self.assertTrue(control.session.record)
            send('C')
            self.assertFalse(control.session.record)
            send('-1,0')
            self.assertLess(control.actuator.last_packet[0], 1500)
            spin(0.4)
            self.assertEqual(control.actuator.last_packet[0], 1500)
            telemetry = []
            while True:
                try:
                    telemetry.append(json.loads(sock.recv(4096)))
                except BlockingIOError:
                    break
            self.assertTrue(telemetry)
            self.assertIn('cam_connected', telemetry[-1])
            self.assertTrue(telemetry[-1]['dry_run'])
            sock.sendto(b'Q', target)
            with self.assertRaises(SystemExit):
                spin(0.2)
        finally:
            sock.close()
            if control is not None:
                control.close()
                control.destroy_node()
            if probe is not None:
                probe.destroy_node()
            rclpy.shutdown()
