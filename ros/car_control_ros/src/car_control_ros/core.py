"""Port of server.py boundary interpolation + EMA + heading PID.

Core coordinates match the original camera code: right, forward (metres).
No ROS, CUDA, camera or serial imports; usable in deterministic tests.
"""
from dataclasses import dataclass
import math


@dataclass
class Parameters:
    track_width: float = 1.5
    lookahead_distance: float = 0.5
    min_depth: float = 0.1
    max_depth: float = 4.0
    lateral_limit: float = 2.5
    kp_gain: float = 1.0
    ki_gain: float = 0.1
    kd_gain: float = 0.25
    max_integral: float = 1.5
    ema_alpha: float = 0.85
    max_steering_output: float = 1.0
    min_dt: float = 0.001
    throttle: float = 0.25
    stop_on_orange: bool = True
    stop_cone_z_threshold: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError('Non-finite parameter: ' + name)
        if not 0 < self.min_depth < self.lookahead_distance < self.max_depth:
            raise ValueError('Require 0 < min_depth < lookahead_distance < max_depth')
        if min(self.track_width, self.lateral_limit, self.min_dt) <= 0:
            raise ValueError('Geometry and min_dt must be positive')
        if not 0 < self.ema_alpha <= 1 or not 0 < self.max_steering_output <= 1:
            raise ValueError('Invalid smoothing or steering limit')
        if not 0 <= self.throttle <= 1 or self.max_integral < 0:
            raise ValueError('Invalid throttle or integral limit')


@dataclass(frozen=True)
class Command:
    throttle: float = 0.0
    steering: float = 0.0
    brake: float = 1.0


def clamp(value, low, high):
    return max(low, min(high, value))


def valid_command(command):
    return (all(math.isfinite(v) for v in (command.throttle, command.steering, command.brake))
            and 0 <= command.throttle <= 1 and -1 <= command.steering <= 1
            and 0 <= command.brake <= 1 and not (command.throttle > 0 and command.brake > 0))


def fresh(stamp, now, timeout):
    return stamp is not None and math.isfinite(stamp) and 0 <= now - stamp <= timeout


def boundary(points, z):
    points = sorted(points, key=lambda p: p[1])[:6]
    unique = []
    for point in points:
        if not unique or point[1] > unique[-1][1] + 0.001:
            unique.append(point)
    if not unique:
        return None
    for (x0, z0), (x1, z1) in zip(unique, unique[1:]):
        if z0 <= z <= z1:
            return x0 + (x1 - x0) * (z - z0) / (z1 - z0), unique[0][1], unique[-1][1]
    return (unique[0][0] if z < unique[0][1] else unique[-1][0], unique[0][1], unique[-1][1])


class Controller:
    def __init__(self, parameters):
        self.p = parameters
        self.reset()

    def reset(self):
        self.tx, self.tz = 0.0, self.p.lookahead_distance
        self.integral, self.last_error = 0.0, None
        self.finished = False

    def step(self, cones, dt):
        if self.finished:
            return Command()
        if not math.isfinite(dt) or dt <= 0:
            self.reset()
            return Command()
        p = self.p
        cones = [(x, z, color) for x, z, color in cones
                 if math.isfinite(x) and math.isfinite(z) and 0 < z <= p.max_depth
                 and abs(x) < p.lateral_limit]
        if p.stop_on_orange and any(c == 'orange' and z <= p.stop_cone_z_threshold for x, z, c in cones):
            self.finished = True
            return Command()
        blues = [(x, z) for x, z, c in cones if c == 'blue' and z > p.min_depth]
        yellows = [(x, z) for x, z, c in cones if c == 'yellow' and z > p.min_depth]
        if not blues and not yellows:
            self.reset()
            return Command()
        # Same 0.3m + 0.2m grid as the existing VisionLoop.
        grid = [0.3 + i * 0.2 for i in range(max(1, int(math.ceil((p.max_depth - 0.3) / 0.2))))]
        z = next((v for v in grid if v >= p.lookahead_distance), grid[-1])
        left, right = boundary(blues, z), boundary(yellows, z)
        vl = left is not None and left[1] - 0.4 <= z <= left[2] + 0.4
        vr = right is not None and right[1] - 0.4 <= z <= right[2] + 0.4
        if vl and vr:
            x = (left[0] + right[0]) / 2
        elif vl:
            x = left[0] + p.track_width / 2
        elif vr:
            x = right[0] - p.track_width / 2
        elif left is not None and right is not None:
            x = (left[0] + right[0]) / 2
        elif left is not None:
            x = left[0] + p.track_width / 2
        else:
            x = right[0] - p.track_width / 2
        self.tx += p.ema_alpha * (x - self.tx)
        self.tz += p.ema_alpha * (z - self.tz)
        error = math.atan2(self.tx, self.tz)
        dt = clamp(dt, p.min_dt, 0.2)
        self.integral = clamp(self.integral + error * dt, -p.max_integral, p.max_integral)
        derivative = 0.0 if self.last_error is None else (error - self.last_error) / dt
        self.last_error = error
        steering = clamp(p.kp_gain * error + p.ki_gain * self.integral + p.kd_gain * derivative,
                         -p.max_steering_output, p.max_steering_output)
        return Command(p.throttle, steering, 0.0)


def world_to_cones(track, x, y, yaw, max_depth, fov=math.pi / 2):
    result = []
    for wx, wy, color in track:
        dx, dy = wx - x, wy - y
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        right = math.sin(yaw) * dx - math.cos(yaw) * dy
        if 0 < forward <= max_depth and abs(math.atan2(right, forward)) < fov / 2:
            result.append((right, forward, color))
    return result


def circle_track(radius=8.0, width=1.5, count=100):
    return [(r * math.sin(a), radius - r * math.cos(a), color)
            for r, color in ((radius - width / 2, 'blue'), (radius + width / 2, 'yellow'))
            for a in (2 * math.pi * i / count for i in range(count))]


class Bicycle:
    def __init__(self, wheelbase=0.32, max_steering=0.44, max_speed=2.0):
        if not all(math.isfinite(v) and v > 0 for v in (wheelbase, max_steering, max_speed)) or max_steering >= math.pi / 2:
            raise ValueError('Invalid bicycle geometry')
        self.wheelbase, self.max_steering, self.max_speed = wheelbase, max_steering, max_speed
        self.x = self.y = self.yaw = self.speed = 0.0

    def step(self, command, dt):
        if not valid_command(command):
            command = Command()
        target = command.throttle * self.max_speed if command.brake == 0 else 0.0
        self.speed += clamp(target - self.speed, -4 * dt, 2 * dt)
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.yaw -= self.speed / self.wheelbase * math.tan(command.steering * self.max_steering) * dt
