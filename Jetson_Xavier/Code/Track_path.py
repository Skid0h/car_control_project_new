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
    samples = np.union1d(z, np.linspace(0., z[-1], math.ceil(z[-1] / 0.05) + 1))
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
        self.floor_reference = None
        self.follow_curve = False
        self.observed_at = None
        self.frame_at = None
        self.anchor_index = None
        self.has_pair = False

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

    def _pairs(self, blue, yellow):
        width = self.config.track_width
        options = []
        for bi, b in enumerate(blue):
            bx, bz = b['pos_3d']
            for yi, y in enumerate(yellow):
                yx, yz = y['pos_3d']
                depth_gap = abs(bz - yz)
                lateral_gap = yx - bx
                span = math.hypot(lateral_gap, depth_gap)
                if (min(bz, yz) < 0.10 or not 0.40 * width <= span <= 1.50 * width or
                        depth_gap >= self.config.pair_z_tolerance or lateral_gap < 0.45 * span):
                    continue
                x, z = (bx + yx) / 2, (bz + yz) / 2
                # Сначала пара, у которой оба конуса ближе. Дальняя пара не забирает конус первой.
                priority = (max(bz, yz), z, abs(span - width))
                options.append((priority, bi, yi, {'x': x, 'z': z, 'cones': (b, y)}))

        used_blue, used_yellow = set(), set()
        selected = []
        for _, bi, yi, pair in sorted(options, key=lambda option: option[0]):
            if bi in used_blue or yi in used_yellow:
                continue
            if selected:
                previous = selected[-1]
                dz = pair['z'] - previous['z']
                if dz < 0.15 or abs(pair['x'] - previous['x']) > 0.15 + 1.5 * dz:
                    continue
                if any(current['pos_3d'][1] <= earlier['pos_3d'][1]
                       for current, earlier in zip(pair['cones'], previous['cones'])):
                    continue
            selected.append(pair)
            used_blue.add(bi)
            used_yellow.add(yi)
            if len(selected) == 3:
                break
        return selected

    def _single_cone_gate(self, blue, yellow):
        candidates = []
        for cones, offset in ((blue, self.config.virtual_point_offset),
                              (yellow, -self.config.virtual_point_offset)):
            for cone in cones:
                x, z = cone['pos_3d']
                if z >= 0.10:
                    candidates.append({'x': x + offset, 'z': z, 'cones': (cone,)})
        return min(candidates, key=lambda gate: gate['z'], default=None)

    def _update_floor(self, gates, alpha):
        samples = [(1 / cone['z_cam'], cone['bbox'][3])
                   for gate in gates for cone in gate['cones']]
        inverse_z, rows = np.asarray(samples).T
        nearest = gates[0]['cones']
        depths = np.array([cone['z_cam'] for cone in nearest])
        feet = [cone['bbox'][3] for cone in nearest]
        reference = np.array([depths.mean(), np.average(feet, weights=depths)])
        if self.floor_reference is None:
            self.floor_reference = reference
        else:
            self.floor_reference += alpha * (reference - self.floor_reference)
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
        anchor_depth = None
        has_pair = self.has_pair
        if finish is not None:
            x, z = finish['pos_3d']
            gates = [{'x': x, 'z': z, 'cones': (finish,)}] if z >= 0.10 else []
        else:
            gates = self._pairs(blue, yellow)
            if not gates:
                single = self._single_cone_gate(blue, yellow)
                if single is not None:
                    gates = [single]

        if gates:
            has_pair = finish is None and len(gates[0]['cones']) == 2
            follow_curve = len(gates) >= 2 or finish is not None
            self._update_floor(gates, alpha)
            if finish is None:
                first = gates[0]
                # Переносим предыдущую цель в текущую систему координат и
                # фильтруем её по обеим осям до построения общей линии.
                if predicted is not None and self.anchor_index is not None and self.anchor_index < len(predicted):
                    old_x, old_z = predicted[self.anchor_index]
                    if old_z >= 0.10:
                        z = old_z + alpha * (first['z'] - old_z)
                        if len(gates) > 1:
                            z = min(z, gates[1]['z'] - 0.05)
                        first = {**first, 'x': old_x + alpha * (first['x'] - old_x), 'z': z}
                        gates[0] = first
                anchor_depth = first['z']

            if follow_curve:
                points = _curve(gates)
            else:
                points = np.array([(0., 0.), (gates[0]['x'], gates[0]['z'])])
            if anchor_depth is not None and predicted is not None and follow_curve:
                z = points[:, 1]
                previous_x = np.interp(z, predicted[:, 1], predicted[:, 0])
                difference = previous_x - points[:, 0]
                anchor_index = np.searchsorted(z, anchor_depth)
                tolerance = self.config.track_path['match_tolerance_m'] + 0.12 * anchor_depth
                if abs(difference[anchor_index]) <= tolerance:
                    overlap = np.clip((predicted[-1, 1] - z) / 0.5, 0., 1.)
                    beyond_anchor = np.clip((z - anchor_depth) / 0.35, 0., 1.)
                    points[:, 0] += (1 - alpha) * overlap * beyond_anchor * difference
            self.observed_at = captured_at
            self.follow_curve = follow_curve
        elif predicted is not None:
            z = np.linspace(0., predicted[-1, 1], math.ceil(predicted[-1, 1] / 0.05) + 1)
            if self.anchor_index is not None and self.anchor_index < len(predicted):
                depth = predicted[self.anchor_index, 1]
                if depth >= 0.10:
                    anchor_depth = depth
                    z = np.union1d(z, [depth])
            points = np.column_stack((np.interp(z, predicted[:, 1], predicted[:, 0]), z))
        else:
            self.points = None
            self.anchor_index = None
            return

        # Возвращаем начало на ось машинки, плавно сводя поправку к нулю впереди.
        z = points[:, 1]
        t = np.clip(z / 0.35, 0., 1.)
        offset = points[0, 0]
        points[:, 0] -= offset * (1 - t)**2 * (1 + 2*t)
        self.points = points
        self.anchor_index = None if anchor_depth is None else np.searchsorted(z, anchor_depth)
        self.has_pair = has_pair

    def pair_center(self):
        if not self.has_pair or self.anchor_index is None:
            return None
        return tuple(map(float, self.points[self.anchor_index]))

    def target(self, lookahead):
        if self.points is None or self.points[-1, 1] < 0.10:
            return None
        if self.anchor_index is not None:
            return tuple(map(float, self.points[self.anchor_index]))
        if not self.follow_curve:
            return tuple(map(float, self.points[-1]))
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        return tuple(float(np.interp(lookahead, stations, axis)) for axis in self.points.T)

    def turn_percent(self):
        """100% на прямой, 0% при изменении направления на 90° и больше."""
        if self.points is None or len(self.points) < 2:
            return None
        segments = np.diff(self.points, axis=0)
        headings = np.arctan2(segments[:, 0], segments[:, 1])
        angle = np.ptp(np.r_[0., headings])
        return 100. * max(0., 1. - angle / (math.pi / 2))
