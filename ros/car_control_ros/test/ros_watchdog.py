#!/usr/bin/env python3
import time
import unittest
import rospy
import rostest
from car_control_ros.msg import Cone, ConeArray, Drive


class Watchdog(unittest.TestCase):
    def test_dropout_stale_and_empty(self):
        self.command = None
        rospy.Subscriber('/car/command', Drive, lambda m: setattr(self, 'command', m), queue_size=1)
        pub = rospy.Publisher('/car/cones', ConeArray, queue_size=1)
        deadline = time.monotonic() + 10
        while pub.get_num_connections() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertGreater(pub.get_num_connections(), 0)
        msg = ConeArray()
        msg.header.frame_id = 'base_link'
        for y, color in ((0.75, Cone.BLUE), (-0.75, Cone.YELLOW)):
            c = Cone()
            c.position.x, c.position.y, c.color = 1.0, y, color
            msg.cones.append(c)

        def publish_for(duration, stale=False):
            until = time.monotonic() + duration
            while time.monotonic() < until:
                msg.header.stamp = rospy.Time.now() - rospy.Duration(2 if stale else 0)
                pub.publish(msg)
                time.sleep(0.03)

        publish_for(0.3)
        self.assertIsNotNone(self.command)
        self.assertGreater(self.command.throttle, 0)
        time.sleep(0.65)
        self.assertEqual(self.command.brake, 1)
        self.assertEqual(self.command.throttle, 0)
        publish_for(0.3, stale=True)
        self.assertEqual(self.command.brake, 1, 'Receipt of old data must not refresh motion')
        publish_for(0.3)
        self.assertGreater(self.command.throttle, 0)
        msg.cones = []
        publish_for(0.3)
        self.assertEqual(self.command.brake, 1)


if __name__ == '__main__':
    rospy.init_node('watchdog_test')
    rostest.rosrun('car_control_ros', 'ros_watchdog', Watchdog)
