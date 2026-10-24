"""Mode arbitration and PC protocol, independently testable."""
import math
from .core import Controller, Parameters


class Session:
    def __init__(self, actuator, parameters=None, sensor_timeout=0.4, link_timeout=0.5):
        if not 0 < sensor_timeout <= 1 or not 0 < link_timeout <= 2:
            raise ValueError('Invalid watchdog timeouts')
        self.actuator = actuator
        self.controller = Controller(parameters or Parameters())
        self.sensor_timeout, self.link_timeout = sensor_timeout, link_timeout
        self.auto = False
        self.manual_disarmed = True
        self.link = None
        self.sensor = None
        self.auto_since = None
        self.record = False
        self.quit = False
        self.message = ''
        self.capture_speed_limits()

    def disable(self, reason=''):
        self.auto = False
        self.manual_disarmed = True
        self.auto_since = None
        self.sensor = None
        self.controller.reset()
        self.actuator.stop()
        self.message = reason

    def receive(self, text, now):
        try:
            if text in ('A', 'S', 'Q', 'R', 'C', 'F'):
                self.link = now
                if text == 'A':
                    if not self.auto:
                        self.disable()
                        self.auto = True
                        self.auto_since = now
                elif text in ('S', 'Q', 'F'):
                    self.disable('Restart ros2 launch to reconnect hardware' if text == 'F' else '')
                    self.quit = text == 'Q'
                else:
                    self.record = text == 'R'
                return True
            if text.startswith('speed:'):
                forward, reverse = [int(v) for v in text[6:].split(',')]
                p = self.actuator.p
                # Remote controls can reduce speed, never exceed configured endpoints.
                if not p.neutral <= forward <= self.max_forward or not self.min_reverse <= reverse <= p.neutral:
                    raise ValueError('Speed exceeds configured limits')
                p.forward, p.reverse = forward, reverse
                self.link = now
                return True
            speed, steering = [float(v) for v in text.split(',')]
            if not all(math.isfinite(v) and -1 <= v <= 1 for v in (speed, steering)):
                raise ValueError('Invalid normalized command')
            self.link = now
            if self.auto and (speed != 0 or steering != 0):
                self.disable('Manual override')
            if not self.auto:
                # PC sends neutral 20 times/s. It must not silently cancel STOP.
                if speed != 0 or steering != 0:
                    self.manual_disarmed = False
                if not self.manual_disarmed:
                    self.actuator.set_target(speed, steering, now)
            return True
        except (ValueError, AttributeError):
            self.disable('Rejected malformed/out-of-range command')
            return False

    def capture_speed_limits(self):
        self.max_forward = self.actuator.p.forward
        self.min_reverse = self.actuator.p.reverse

    def cones(self, cones, now, age):
        if not self.auto or not math.isfinite(age) or not 0 <= age <= self.sensor_timeout:
            return
        if self.link is None or now - self.link > self.link_timeout:
            self.disable('PC heartbeat lost')
            return
        dt = 1 / 15 if self.sensor is None else max(0.001, now - self.sensor)
        self.sensor = now
        cmd = self.controller.step(cones, dt)
        if cmd.brake > 0:
            self.actuator.stop()
        else:
            self.actuator.set_target(cmd.throttle, cmd.steering, now)

    def tick(self, now):
        if self.link is None or not 0 <= now - self.link <= self.link_timeout:
            self.disable('PC heartbeat lost')
        if self.auto:
            reference = self.sensor if self.sensor is not None else self.auto_since
            if reference is None or not 0 <= now - reference <= self.sensor_timeout:
                self.disable('Camera/detection timeout; press A to rearm')
        return self.actuator.packet(now)
