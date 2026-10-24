#!/usr/bin/env python3
import math
import time
import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from car_control_ros.core import Bicycle, circle_track, world_to_cones
from car_control_ros.msg import Cone, ConeArray, Drive
from car_control_ros.transport import CommandInbox


def main():
    rospy.init_node('car_simulator')
    radius = float(rospy.get_param('~radius', 8.0))
    width = float(rospy.get_param('~track_width', 1.5))
    if radius <= width or width <= 0:
        raise ValueError('Require radius > track_width > 0')
    track = circle_track(radius, width)
    car = Bicycle(rospy.get_param('~wheelbase', 0.32), rospy.get_param('~max_steering', 0.44),
                  rospy.get_param('~max_speed', 2.0))
    inbox = CommandInbox()
    rospy.Subscriber('/car/command', Drive, inbox.receive, queue_size=1)
    cones_pub = rospy.Publisher('/car/cones', ConeArray, queue_size=1)
    odom_pub = rospy.Publisher('/car/sim/odom', Odometry, queue_size=1)
    markers_pub = rospy.Publisher('/car/sim/markers', MarkerArray, queue_size=1, latch=True)
    last = time.monotonic()
    while not rospy.is_shutdown():
        now = time.monotonic()
        car.step(inbox.get(), min(now - last, 0.1))
        last = now
        stamp = rospy.Time.now()
        msg = ConeArray()
        msg.header.stamp, msg.header.frame_id = stamp, 'base_link'
        for right, forward, color in world_to_cones(track, car.x, car.y, car.yaw, 4.0):
            cone = Cone()
            cone.position.x, cone.position.y = forward, -right
            cone.color = Cone.BLUE if color == 'blue' else Cone.YELLOW
            msg.cones.append(cone)
        cones_pub.publish(msg)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'map', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = car.x, car.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(car.yaw / 2), math.cos(car.yaw / 2)
        odom.twist.twist.linear.x = car.speed
        odom_pub.publish(odom)
        markers = MarkerArray()
        for i, (x, y, color) in enumerate(track + [(car.x, car.y, 'car')]):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = 'map', stamp
            marker.ns, marker.id = 'simulation', i
            marker.type, marker.action = Marker.CYLINDER, Marker.ADD
            marker.pose.position.x, marker.pose.position.y = x, y
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.15 if color != 'car' else 0.32
            marker.scale.z = 0.25
            marker.color.a = 1.0
            marker.color.b = float(color == 'blue')
            marker.color.r = float(color == 'yellow')
            marker.color.g = float(color in ('yellow', 'car'))
            markers.markers.append(marker)
        markers_pub.publish(markers)
        time.sleep(0.02)


if __name__ == '__main__':
    main()
