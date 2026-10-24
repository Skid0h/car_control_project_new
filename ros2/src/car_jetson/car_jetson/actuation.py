"""Deterministic command shaping. No sleeping, hardware or ROS imports."""
from dataclasses import dataclass
import math


@dataclass
class Limits:
    neutral: int = 1500
    forward: int = 1570
    reverse: int = 1420
    center: int = 100
    steering_range: int = 50
    steering_min: int = 70
    steering_max: int = 130
    rate_hz: float = 20.0
    deadband_deg: float = 2.0
    slew_deg_s: float = 60.0
    heartbeat_s: float = 0.2
    command_timeout_s: float = 0.3

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('Actuation limits must be finite')
        if not 1000 <= self.reverse <= self.neutral <= self.forward <= 2000:
            raise ValueError('Invalid motor calibration')
        if not 0 <= self.steering_min < self.center < self.steering_max <= 180:
            raise ValueError('Invalid calibrated steering stops')
        if self.steering_range <= 0 or not 1 <= self.rate_hz <= 50:
            raise ValueError('Invalid steering range/rate')
        if not 0 < self.deadband_deg <= 10 or not 0 < self.slew_deg_s <= 360:
            raise ValueError('Invalid deadband/slew')
        if not 1 / self.rate_hz <= self.heartbeat_s < 0.5:
            raise ValueError('Heartbeat must fit rate and Arduino 500ms watchdog')
        if not 0 < self.command_timeout_s < 0.5:
            raise ValueError('Command timeout must be below Arduino watchdog')


class Actuator:
    def __init__(self, limits):
        self.p = limits
        self.target_motor = limits.neutral
        self.target_angle = float(limits.center)
        self.angle = float(limits.center)
        self.received = None
        self.last_tick = None
        self.last_sent = None
        self.last_packet = None
        self.last_stop_mode = None
        self.stopped = True

    def set_target(self, speed, steering, now):
        if not all(math.isfinite(v) for v in (speed, steering, now)):
            self.stop()
            return False
        if not -1 <= speed <= 1 or not -1 <= steering <= 1:
            self.stop()
            return False
        self.received = now
        endpoint = self.p.forward if speed >= 0 else self.p.reverse
        self.target_motor = round(self.p.neutral + abs(speed) * (endpoint - self.p.neutral))
        desired = max(self.p.steering_min, min(self.p.steering_max,
                      self.p.center - steering * self.p.steering_range))
        # Compare against accepted target, not last raw input: tiny drift accumulates.
        if abs(desired - self.target_angle) >= self.p.deadband_deg:
            self.target_angle = float(desired)
        self.stopped = False
        return True

    def stop(self):
        # Remove motor torque now; return steering to calibrated centre at the normal slew rate.
        self.target_motor = self.p.neutral
        self.target_angle = float(self.p.center)
        self.received = None
        self.stopped = True

    def packet(self, now):
        if self.received is None or not 0 <= now - self.received <= self.p.command_timeout_s:
            self.stop()
        dt = 0.0 if self.last_tick is None else max(0.0, min(now - self.last_tick, 1 / self.p.rate_hz))
        self.last_tick = now
        delta = self.target_angle - self.angle
        step = self.p.slew_deg_s * dt
        self.angle += max(-step, min(step, delta))
        value = (self.target_motor, int(round(self.angle)))
        emergency = (self.stopped and self.last_packet is not None
                     and (self.last_packet[0] != self.p.neutral or not self.last_stop_mode))
        elapsed = float('inf') if self.last_sent is None else now - self.last_sent
        if not emergency and elapsed + 1e-9 < 1 / self.p.rate_hz:
            return None
        if value == self.last_packet and self.stopped == self.last_stop_mode and elapsed + 1e-9 < self.p.heartbeat_s:
            return None
        self.last_packet, self.last_sent, self.last_stop_mode = value, now, self.stopped
        # Third field is a backwards-compatible stop marker: the old sketch ignores it.
        return (('<%d,%d,S>' if self.stopped else '<%d,%d>') % value).encode('ascii')
