#!/usr/bin/env python3
import json
import threading
import time
from dataclasses import fields
import rospy
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse
from car_control_ros.core import Controller, Parameters, Command, fresh
from car_control_ros.legacy import configuration
from car_control_ros.msg import ConeArray, Drive
from car_control_ros.transport import drive_message


class Node:
    def __init__(self):
        _, config = configuration()
        values = {f.name: getattr(config, f.name) for f in fields(Parameters) if hasattr(config, f.name)}
        values.update(rospy.get_param('~controller', {}))
        self.controller = Controller(Parameters(**values))
        self.timeout = float(rospy.get_param('~sensor_timeout', 0.4))
        if self.timeout <= 0:
            raise ValueError('sensor_timeout must be positive')
        self.enabled = rospy.get_param('~auto_start', False)
        self.sample = None
        self.lock = threading.Lock()
        self.pub = rospy.Publisher('/car/command', Drive, queue_size=1)
        self.status = rospy.Publisher('/car/status', String, queue_size=1)
        self.sub = rospy.Subscriber('/car/cones', ConeArray, self.receive, queue_size=1)
        self.service = rospy.Service('/car/enable', SetBool, self.enable)

    def receive(self, msg):
        colors = {0: 'blue', 1: 'yellow', 2: 'orange'}
        if msg.header.frame_id != 'base_link':
            rospy.logwarn_throttle(5, 'Rejecting cones: expected base_link')
            return
        with self.lock:
            self.sample = ([( -c.position.y, c.position.x, colors.get(c.color, 'unknown')) for c in msg.cones],
                           msg.header.stamp.to_sec(), time.monotonic())

    def enable(self, request):
        with self.lock:
            self.enabled = request.data
            self.sample = None  # Require new measurements after every mode transition.
            self.controller.reset()
        return SetBoolResponse(True, 'enabled' if request.data else 'disabled')

    def run(self):
        last_sample = None
        last_step = time.monotonic()
        command = Command()
        try:
            while not rospy.is_shutdown():
                now = time.monotonic()
                with self.lock:
                    sample = self.sample
                    valid = (self.enabled and sample is not None
                             and fresh(sample[1], rospy.Time.now().to_sec(), self.timeout)
                             and fresh(sample[2], now, self.timeout))
                    if not valid:
                        command = Command()
                        # Keep an orange finish latched until explicit enable.
                        if not self.controller.finished:
                            self.controller.reset()
                        last_sample = None
                        last_step = now
                    elif sample is not last_sample:
                        command = self.controller.step(sample[0], now - last_step)
                        last_step, last_sample = now, sample
                    state = {'enabled': self.enabled, 'sensor_fresh': bool(valid),
                             'finished': self.controller.finished,
                             'braking': command.brake > 0}
                self.pub.publish(drive_message(command))
                self.status.publish(String(data=json.dumps(state)))
                time.sleep(0.02)  # Wall clock: watchdog still works if /clock freezes.
        finally:
            self.pub.publish(drive_message(Command()))


if __name__ == '__main__':
    rospy.init_node('car_controller')
    Node().run()
