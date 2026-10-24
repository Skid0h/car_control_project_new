#!/usr/bin/env python3
"""Optional ROS1 FSDS protocol adapter. Install upstream fs_msgs first."""
import math
import time
import rospy
from nav_msgs.msg import Odometry
from fs_msgs.msg import ControlCommand, GoSignal, Track, Cone as FSCone
from car_control_ros.core import Command, fresh, world_to_cones
from car_control_ros.msg import Cone, ConeArray, Drive
from car_control_ros.transport import CommandInbox


class Node:
    def __init__(self):
        self.go = None
        self.track = []
        self.inbox = CommandInbox()
        self.pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/car/command', Drive, self.inbox.receive, queue_size=1)
        rospy.Subscriber('/fsds/signal/go', GoSignal, self.receive_go, queue_size=1)
        self.require_go = rospy.get_param('~require_go', True)
        if rospy.get_param('~ground_truth', True):
            self.cones_pub = rospy.Publisher('/car/cones', ConeArray, queue_size=1)
            rospy.Subscriber('/fsds/testing_only/track', Track, self.receive_track, queue_size=1)
            rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.receive_odom, queue_size=1)

    def receive_go(self, _):
        self.go = time.monotonic()

    def receive_track(self, msg):
        colors = {FSCone.BLUE: 'blue', FSCone.YELLOW: 'yellow',
                  FSCone.ORANGE_BIG: 'orange', FSCone.ORANGE_SMALL: 'orange'}
        self.track = [(c.location.x, c.location.y, colors.get(c.color, 'unknown')) for c in msg.track]

    def receive_odom(self, msg):
        if not fresh(msg.header.stamp.to_sec(), rospy.Time.now().to_sec(), 0.4):
            return
        pose = msg.pose.pose
        q = pose.orientation
        numbers = (pose.position.x, pose.position.y, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in numbers):
            return
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        visible = world_to_cones(self.track, pose.position.x, pose.position.y, yaw,
                                rospy.get_param('~max_depth', 15.0))
        result = ConeArray()
        result.header.stamp, result.header.frame_id = msg.header.stamp, 'base_link'
        colors = {'blue': Cone.BLUE, 'yellow': Cone.YELLOW, 'orange': Cone.ORANGE}
        for right, forward, color in visible:
            if color not in colors:
                continue
            cone = Cone()
            cone.position.x, cone.position.y, cone.color = forward, -right, colors[color]
            result.cones.append(cone)
        self.cones_pub.publish(result)

    def publish(self, command):
        msg = ControlCommand()
        msg.throttle, msg.steering, msg.brake = command.throttle, command.steering, command.brake
        self.pub.publish(msg)

    def run(self):
        try:
            while not rospy.is_shutdown():
                command = self.inbox.get()
                if self.require_go and not fresh(self.go, time.monotonic(), 4.0):
                    command = Command()
                self.publish(command)
                time.sleep(0.02)
        finally:
            self.publish(Command())


if __name__ == '__main__':
    rospy.init_node('fsds_adapter')
    Node().run()
