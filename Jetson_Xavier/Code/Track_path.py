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


def _smooth_curve(points, anchor_depth):
    """Сглаживает короткие изломы с сохранением старта, цели руля и конца."""
    x, z = points.T
    grid = np.linspace(0., z[-1], math.ceil(z[-1] / 0.04) + 1)
    step = grid[1] - grid[0]
    sigma = min(0.18, z[-1] / 6)
    radius = math.ceil(2.5 * sigma / step)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets * step / sigma)**2)
    lateral = np.interp(grid, z, x)
    padding = np.arange(1, radius + 1)
    padded = np.r_[lateral[0] - padding[::-1] * (lateral[1] - lateral[0]),
                   lateral,
                   lateral[-1] + padding * (lateral[-1] - lateral[-2])]
    smooth = np.convolve(padded, kernel / kernel.sum(), mode='valid')
    delta = np.interp(z, grid, smooth) - x

    fixed_z = np.array([0., anchor_depth, z[-1]])
    fixed_delta = np.interp(fixed_z, z, delta)
    left = np.searchsorted(fixed_z[1:-1], z, side='right')
    t = (z - fixed_z[left]) / (fixed_z[left + 1] - fixed_z[left])
    weight = t*t * (3 - 2*t)
    delta -= fixed_delta[left] + weight * (fixed_delta[left + 1] - fixed_delta[left])
    # Ограничиваем поперечную поправку, чтобы не срезать поворот.
    x += delta * (0.12 / max(0.12, np.max(np.abs(delta))))


class TrackPath:
    PREVIEW_LENGTH_M = 1.0
    PREVIEW_HOLD_S = 1.0
    PAIR_MEMORY_S = 0.25
    PAIR_DEPTH_ADVANTAGE_M = 0.20
    HANDOFF_RATE_MPS = 2.0

    def __init__(self, config, timeout):
        self.config = config
        self.timeout = timeout
        self.reset()

    def reset(self, *, keep_preview=False):
        self.points = None
        if not keep_preview:
            self.floor = None
            self.floor_reference = None
            self.preview = None
            self.preview_until = 0.
            self.curvature = 0.
            self.curvature_at = -math.inf
        self.ends_at_finish = False
        self.follow_curve = False
        self.observed_at = None
        self.frame_at = None
        self.anchor_index = None
        self.target_index = None
        self.handoff = None
        self.handoff_limited = False
        self.has_pair = False
        self.tracked_pairs = []

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

    @staticmethod
    def _match_cones(tracked, cones, color, captured_at):
        candidates = []
        for pair_index, (seen_at, pair, _) in enumerate(tracked):
            x1, y1, x2, y2 = pair[color]['bbox']
            width, height = x2 - x1, y2 - y1
            age = captured_at - seen_at
            for cone_index, cone in enumerate(cones):
                left, top, right, bottom = cone['bbox']
                height_ratio = max(1, bottom - top) / max(1, height)
                if not 0.55 <= height_ratio <= 1.80:
                    continue
                dx = (left + right - x1 - x2) / 2
                dy = bottom - y2
                # Сравниваем основания рамок: верх чувствительнее к их размеру.
                distance = math.hypot(dx / (45 + 360 * age + 0.35 * width),
                                      dy / (22 + 140 * age + 0.30 * height))
                if distance <= 1.:
                    cost = distance + 0.5 * abs(math.log(height_ratio))
                    candidates.append((cost, pair_index, cone_index))

        matches, used = {}, set()
        for _, pair_index, cone_index in sorted(candidates):
            if pair_index not in matches and cone_index not in used:
                matches[pair_index] = cone_index
                used.add(cone_index)
        return matches

    @staticmethod
    def _follows(previous, pair):
        dz = pair['z'] - previous['z']
        return (dz >= 0.15 and abs(pair['x'] - previous['x']) <= 0.15 + 1.5 * dz and
                all(current['pos_3d'][1] > earlier['pos_3d'][1]
                    for current, earlier in zip(pair['cones'], previous['cones'])))

    def _pairs(self, blue, yellow, captured_at):
        width = self.config.track_width
        options = {}
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
                options[bi, yi] = (priority, {'x': x, 'z': z, 'cones': (b, y)})

        tracked = [record for record in self.tracked_pairs
                   if captured_at - record[0] < self.PAIR_MEMORY_S]
        blue_matches = self._match_cones(tracked, blue, 0, captured_at)
        yellow_matches = self._match_cones(tracked, yellow, 1, captured_at)
        retained = {}
        incomplete = []
        used_blue, used_yellow = set(), set()
        for i, (seen_at, pair, pair_id) in enumerate(tracked):
            bi, yi = blue_matches.get(i), yellow_matches.get(i)
            if bi is not None and yi is not None:
                if (bi, yi) in options:
                    retained[bi, yi] = pair_id
                continue
            # Увиденный конус не получает нового партнёра сразу после пропуска кадра.
            if bi is not None:
                used_blue.add(bi)
            if yi is not None:
                used_yellow.add(yi)
            incomplete.append((seen_at, (blue[bi] if bi is not None else pair[0],
                                         yellow[yi] if yi is not None else pair[1]), pair_id))

        def pair_priority(key):
            farther_depth, center_depth, width_error = options[key][0]
            # Существующая пара получает преимущество только при близких расстояниях.
            if key in retained:
                farther_depth -= self.PAIR_DEPTH_ADVANTAGE_M
            return farther_depth, center_depth, width_error

        selected = []
        for bi, yi in sorted(options, key=pair_priority):
            if bi in used_blue or yi in used_yellow:
                continue
            _, pair = options[bi, yi]
            index = sum(prior['z'] < pair['z'] for prior in selected)
            if index and not self._follows(selected[index - 1], pair):
                continue
            if index < len(selected) and not self._follows(pair, selected[index]):
                continue
            pair['id'] = retained.get((bi, yi), object())
            selected.insert(index, pair)
            used_blue.add(bi)
            used_yellow.add(yi)
            if len(selected) == 3:
                break

        self.tracked_pairs = [(captured_at, gate['cones'], gate['id']) for gate in selected] + incomplete
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

    def _handoff_depth(self, first, next_pair):
        clearance = min(cone.get('edge_clearance', math.inf) for cone in first['cones'])
        fraction = min(1., max(0., (0.85 - clearance) / 0.75))
        pair_ids = (first['id'], next_pair['id'])
        if self.handoff is not None and self.handoff[:2] == pair_ids:
            fraction = max(fraction, self.handoff[2])
        self.handoff = (*pair_ids, fraction)
        weight = fraction**2 * (3. - 2. * fraction)
        return first['z'] + weight * (next_pair['z'] - first['z'])

    def update(self, blue, yellow, captured_at, speed_kmh, steering, *, finish=None):
        if self.frame_at is not None and captured_at <= self.frame_at:
            return
        if self.observed_at is not None and captured_at - self.observed_at >= self.timeout:
            self.reset(keep_preview=True)
        dt = 0. if self.frame_at is None else captured_at - self.frame_at
        self.frame_at = captured_at
        predicted = self._predict(max(0., speed_kmh) / 3.6 * dt, steering)
        alpha = -math.expm1(-dt / self.config.target_smoothing_s)
        anchor_depth = None
        target_depth = None
        handoff_active = self.handoff_limited or (self.handoff is not None and self.handoff[2] > 0.)
        has_pair = self.has_pair
        finishing = finish is not None and finish['pos_3d'][1] >= 0.10
        if finishing:
            x, z = finish['pos_3d']
            gates = [{'x': x, 'z': z, 'cones': (finish,)}]
        else:
            gates = self._pairs(blue, yellow, captured_at)
            if not gates:
                single = self._single_cone_gate(blue, yellow)
                if single is not None:
                    gates = [single]

        if gates:
            has_pair = len(gates[0]['cones']) == 2
            follow_curve = len(gates) >= 2
            self._update_floor(gates, alpha)
            if not finishing:
                first = gates[0]
                # Переносим предыдущую цель в текущую систему координат и
                # фильтруем её по обеим осям до построения общей линии.
                previous_index = self.anchor_index
                if self.handoff is not None and first.get('id') is self.handoff[1]:
                    previous_index = self.target_index
                if predicted is not None and previous_index is not None and previous_index < len(predicted):
                    old_x, old_z = predicted[previous_index]
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
            if len(gates) >= 2:
                _smooth_curve(points, anchor_depth)
            if len(gates) >= 2 and 'id' in gates[1]:
                target_depth = self._handoff_depth(gates[0], gates[1])
            else:
                self.handoff = None
                target_depth = gates[0]['z'] if finishing else anchor_depth

            handoff_active = handoff_active or (self.handoff is not None and self.handoff[2] > 0.)
            self.handoff_limited = False
            if (not finishing and handoff_active and target_depth is not None and predicted is not None and
                    self.target_index is not None and self.target_index < len(predicted)):
                previous_target = predicted[self.target_index, 1]
                step = self.HANDOFF_RATE_MPS * dt
                bounded = min(previous_target + step, max(previous_target - step, target_depth))
                self.handoff_limited = bounded != target_depth
                target_depth = float(np.clip(bounded, 0.10, points[-1, 1]))
            self.observed_at = captured_at
            self.follow_curve = follow_curve
            self.ends_at_finish = finishing
        elif predicted is not None:
            z = np.linspace(0., predicted[-1, 1], math.ceil(predicted[-1, 1] / 0.05) + 1)
            if self.anchor_index is not None and self.anchor_index < len(predicted):
                depth = predicted[self.anchor_index, 1]
                if depth >= 0.10:
                    anchor_depth = depth
                    z = np.union1d(z, [depth])
            if self.target_index is not None and self.target_index < len(predicted):
                depth = predicted[self.target_index, 1]
                if depth >= 0.10:
                    target_depth = depth
            points = np.column_stack((np.interp(z, predicted[:, 1], predicted[:, 0]), z))
        else:
            self.points = None
            self.anchor_index = None
            self.target_index = None
            self.handoff = None
            self.handoff_limited = False
            return

        if target_depth is not None:
            z = np.union1d(points[:, 1], [target_depth])
            points = np.column_stack((np.interp(z, points[:, 1], points[:, 0]), z))

        # Возвращаем начало на ось машинки, плавно сводя поправку к нулю впереди.
        z = points[:, 1]
        t = np.clip(z / 0.35, 0., 1.)
        offset = points[0, 0]
        points[:, 0] -= offset * (1 - t)**2 * (1 + 2*t)
        self.points = points
        self.anchor_index = None if anchor_depth is None else np.searchsorted(z, anchor_depth)
        self.target_index = None if target_depth is None else np.searchsorted(z, target_depth)
        self.has_pair = has_pair
        self._update_preview(captured_at, alpha, observed=bool(gates))

    def _update_preview(self, captured_at, alpha, *, observed):
        points = self.points
        self.preview_until = self.observed_at + self.PREVIEW_HOLD_S
        if self.ends_at_finish:
            self.preview = points
            return

        # Берём направление и изгиб по отрезкам, чтобы шум последней точки
        # не превращался в резкий поворот красной линии.
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        step = min(0.30, length[-1] / 2)
        stations = length[-1] - step * np.array([2., 1., 0.])
        tail = np.column_stack([np.interp(stations, length, axis) for axis in points.T])
        segments = np.diff(tail, axis=0)
        headings = np.unwrap(np.arctan2(segments[:, 0], segments[:, 1]))
        if observed and self.follow_curve and step >= 0.20:
            curvature = float(np.clip((headings[1] - headings[0]) / step, -1.8, 1.8))
            if captured_at - self.curvature_at >= self.PREVIEW_HOLD_S:
                self.curvature = curvature
            else:
                self.curvature += alpha * (curvature - self.curvature)
            self.curvature_at = captured_at

        age = captured_at - self.curvature_at
        curvature = self.curvature * max(0., 1. - age / self.PREVIEW_HOLD_S)
        direction = headings[-1] + curvature * step / 2
        distances = np.linspace(0., self.PREVIEW_LENGTH_M, 21)
        interval = np.diff(distances)
        angles = direction + curvature * (distances[:-1] + interval / 2)
        offsets = interval[:, None] * np.column_stack((np.sin(angles), np.cos(angles)))
        extension = points[-1] + np.cumsum(offsets, axis=0)
        self.preview = np.vstack((points, extension))

    def display_paths(self, now):
        if self.preview is None or now >= self.preview_until:
            return None, None
        if (self.points is not None and self.observed_at == self.frame_at and
                now < self.observed_at + self.timeout):
            return self.points, self.preview[len(self.points) - 1:]
        return None, self.preview

    def pair_center(self):
        if not self.has_pair or self.target_index is None:
            return None
        return tuple(map(float, self.points[self.target_index]))

    def target(self, lookahead):
        if self.points is None or self.points[-1, 1] < 0.10:
            return None
        if self.target_index is not None:
            return tuple(map(float, self.points[self.target_index]))
        if not self.follow_curve:
            return tuple(map(float, self.points[-1]))
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        return tuple(float(np.interp(lookahead, stations, axis)) for axis in self.points.T)

    def turn_percent(self, now):
        """100% на прямой, 0% при изменении направления на 90° и больше."""
        if self.preview is None or now >= self.preview_until:
            return None
        if self.has_pair and not self.follow_curve:
            return 0.
        segments = np.diff(self.preview, axis=0)
        headings = np.unwrap(np.r_[0., np.arctan2(segments[:, 0], segments[:, 1])])
        angle = np.ptp(headings)
        return 100. * max(0., 1. - angle / (math.pi / 2))
