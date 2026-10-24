#!/usr/bin/env python3
import time
import unittest
import rospy
import rostest
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
from car_control_ros.msg import Drive


class Smoke(unittest.TestCase):
    def test_movement_and_disable(self):
        self.command = None
        self.distance = 0.0
        rospy.Subscriber('/car/command', Drive, lambda msg: setattr(self, 'command', msg), queue_size=1)
        rospy.Subscriber('/car/sim/odom', Odometry,
                         lambda msg: setattr(self, 'distance', msg.pose.pose.position.x), queue_size=1)
        rospy.wait_for_service('/car/enable', timeout=10)
        enable = rospy.ServiceProxy('/car/enable', SetBool)
        deadline = time.monotonic() + 10
        while self.distance < 0.5 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertGreater(self.distance, 0.5, 'ROS closed loop did not move')
        enable(False)
        time.sleep(0.2)
        self.assertEqual(self.command.brake, 1)
        self.assertEqual(self.command.throttle, 0)
        enable(True)
        time.sleep(0.2)
        self.assertGreater(self.command.throttle, 0)


if __name__ == '__main__':
    rospy.init_node('ros_smoke_test')
    rostest.rosrun('car_control_ros', 'ros_smoke', Smoke)
