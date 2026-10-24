import time
import rospy
from car_control_ros.core import Command, fresh, valid_command
from car_control_ros.msg import Drive


def drive_message(command):
    msg = Drive()
    msg.header.stamp = rospy.Time.now()
    msg.throttle, msg.steering, msg.brake = command.throttle, command.steering, command.brake
    return msg


class CommandInbox:
    """Watch both ROS source timestamps and local monotonic receipt times."""
    def __init__(self, timeout=0.4):
        self.timeout = timeout
        self.sample = None

    def receive(self, msg):
        command = Command(msg.throttle, msg.steering, msg.brake)
        self.sample = (command, msg.header.stamp.to_sec(), time.monotonic())

    def get(self):
        sample = self.sample
        if sample is None:
            return Command()
        command, stamp, received = sample
        if (valid_command(command) and fresh(stamp, rospy.Time.now().to_sec(), self.timeout)
                and fresh(received, time.monotonic(), self.timeout)):
            return command
        return Command()
