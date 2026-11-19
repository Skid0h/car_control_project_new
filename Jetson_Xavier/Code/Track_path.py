"""Одна локальная траектория для руля и стрима. Координаты X/Z в метрах."""

import math

import numpy as np


def _curve(gates):
    """Кубическая интерполяция с ограниченными касательными между серединами пар."""
    x, z = np.asarray([(0., 0.), *[(p['x'], p['z']) for p in gates]]).T
    dz = np.diff(z)
    slopes = np.diff(x) / dz
    tangents = np.zeros(len(x))
    tangents[-1] = slopes[-1]
    for i in range(1, len(x) - 1):
        a, b = slopes[i - 1:i + 1]
        if a * b > 0:
            tangents[i] = 2 * a * b / (a + b)
    samples = np.linspace(0., z[-1], math.ceil(z[-1] / 0.05) + 1)
    i = np.clip(np.searchsorted(z, samples, side='right') - 1, 0, len(z) - 2)
    t = (samples - z[i]) / dz[i]
    lateral = ((2*t**3 - 3*t**2 + 1) * x[i] + (t**3 - 2*t**2 + t) * dz[i] * tangents[i] +
               (-2*t**3 + 3*t**2) * x[i+1] + (t**3 - t**2) * dz[i] * tangents[i+1])
    return np.column_stack((lateral, samples))


class TrackPath:
    def __init__(self, config, timeout):
        self.config = config
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.points = None
        self.floor = None
        self.observed_at = None
        self.frame_at = None
        self.finishing = False

    def _predict(self, travelled_m, steering):
        if self.points is None:
            return None
        # Модель дуги по последней команде руля. Без калибровки перенос только вперёд.
        radius = self.config.track_path['motion_min_radius_m']
        heading = steering * travelled_m / radius if radius else 0.
        half = heading / 2
        chord = travelled_m * (math.sin(half) / half if half else 1.)
        x = self.points[:, 0] - chord * math.sin(half)
        z = self.points[:, 1] - chord * math.cos(half)
        c, s = math.cos(heading), math.sin(heading)
        predicted = np.column_stack((c*x - s*z, s*x + c*z))
        # Используем только непрерывную часть, направленную вперёд.
        reversal = np.flatnonzero(np.diff(predicted[:, 1]) <= 0.)
        if len(reversal):
            predicted = predicted[:reversal[0] + 1]
        return predicted if len(predicted) >= 2 and predicted[-1, 1] >= 0.10 else None

    def _pairs(self, blue, yellow, predicted):
        width = self.config.track_width
        options = []
        for bi, b in enumerate(blue):
            bx, bz = b['pos_3d']
            for yi, y in enumerate(yellow):
                yx, yz = y['pos_3d']
                depth_gap = abs(bz - yz)
                span = math.hypot(bx - yx, depth_gap)
                x, z = (bx + yx) / 2, (bz + yz) / 2
                if (z < 0.10 or not 0.55 * width <= span <= 1.50 * width or
                        depth_gap >= self.config.pair_z_tolerance or abs(bx - yx) < 0.45 * span):
                    continue
                error = 0.
                if predicted is not None:
                    end = predicted[-1, 1]
                    expected = np.interp(z, predicted[:, 1], predicted[:, 0])
                    if z > end:
                        tail = predicted[-1] - predicted[-2]
                        expected += min(z - end, width) * np.clip(tail[0] / tail[1], -1.5, 1.5)
                    tolerance = self.config.track_path['match_tolerance_m'] + 0.12 * z
                    error = abs(x - expected)
                    if error > tolerance:
                        continue
                score = depth_gap + abs(span - width) + error
                options.append((score, bi, yi, {'x': x, 'z': z, 'cones': (b, y)}))

        used_blue, used_yellow = set(), set()
        pairs = []
        for _, bi, yi, pair in sorted(options, key=lambda option: option[0]):
            if bi not in used_blue and yi not in used_yellow:
                pairs.append(pair)
                used_blue.add(bi)
                used_yellow.add(yi)
        pairs.sort(key=lambda pair: pair['z'])
        selected = []
        for pair in pairs:
            if selected:
                dz = pair['z'] - selected[-1]['z']
                if dz < 0.15 or abs(pair['x'] - selected[-1]['x']) > 0.15 + 1.5 * dz:
                    continue
            selected.append(pair)
            if len(selected) == 3:
                break
        return selected

    def _update_floor(self, gates, alpha):
        samples = [(1 / cone['z_cam'], cone['bbox'][3])
                   for gate in gates for cone in gate['cones']]
        inverse_z, rows = np.asarray(samples).T
        if np.ptp(inverse_z) < 0.03:
            return
        design = np.column_stack((inverse_z, np.ones_like(inverse_z)))
        floor = np.linalg.lstsq(design, rows, rcond=None)[0]
        if floor[0] > 0:
            self.floor = floor if self.floor is None else self.floor + alpha * (floor - self.floor)

    def update(self, blue, yellow, finish, captured_at, speed_kmh, steering):
        if self.frame_at is not None and captured_at <= self.frame_at:
            return
        if self.observed_at is not None and captured_at - self.observed_at >= self.timeout:
            self.reset()
        dt = 0. if self.frame_at is None else captured_at - self.frame_at
        self.frame_at = captured_at
        predicted = self._predict(max(0., speed_kmh) / 3.6 * dt, steering)
        alpha = -math.expm1(-dt / self.config.target_smoothing_s)
        gates = self._pairs(blue, yellow, predicted if not self.finishing else None)
        if finish is not None:
            x, z = finish['pos_3d']
            gates = [{'x': x, 'z': z, 'cones': (finish,)}] if z >= 0.10 else []
        confirmed = len(gates) >= (1 if finish is not None else 2)
        if confirmed:
            points = _curve(gates)
            self._update_floor(gates, alpha)
            if predicted is not None:
                z = points[:, 1]
                previous_x = np.interp(z, predicted[:, 1], predicted[:, 0])
                weight = alpha + (1 - alpha) * np.clip((z - predicted[-1, 1]) / 0.5, 0., 1.)
                points[:, 0] = previous_x + weight * (points[:, 0] - previous_x)
            self.observed_at = captured_at
            self.finishing = finish is not None
        elif predicted is not None:
            z = np.linspace(0., predicted[-1, 1], math.ceil(predicted[-1, 1] / 0.05) + 1)
            points = np.column_stack((np.interp(z, predicted[:, 1], predicted[:, 0]), z))
        else:
            self.points = None
            return

        # Возвращаем начало на ось машинки, плавно сводя поправку к нулю впереди.
        z = points[:, 1]
        t = np.clip(z / 0.35, 0., 1.)
        offset = points[0, 0]
        points[:, 0] -= offset * (1 - t)**2 * (1 + 2*t)
        self.points = points

    def target(self, lookahead):
        if self.points is None or self.floor is None or self.points[-1, 1] < 0.10:
            return None
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        return tuple(float(np.interp(lookahead, stations, axis)) for axis in self.points.T)
