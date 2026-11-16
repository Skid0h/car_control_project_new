"""Геометрия трассы в координатах (поперёк, вперёд), метры."""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass
class TrackGeometry:
    blue: np.ndarray
    yellow: np.ndarray
    center: Optional[np.ndarray] = None


def _chain(cones, max_gap):
    points = [np.asarray(cone['pos_3d'], dtype=float) for cone in cones
              if cone['pos_3d'][1] > 0 and all(map(math.isfinite, cone['pos_3d']))]
    if not points:
        return np.empty((0, 2))
    chain = [points.pop(min(range(len(points)), key=lambda i: np.linalg.norm(points[i])))]
    heading = np.array([0., 1.])
    while points:
        steps = np.asarray(points) - chain[-1]
        lengths = np.linalg.norm(steps, axis=1)
        # Не возвращаемся назад вдоль уже найденной границы.
        candidates = np.flatnonzero((lengths >= 0.05) & (lengths <= max_gap) &
                                    (steps @ heading >= -0.25 * lengths))
        if not len(candidates):
            break
        i = int(candidates[np.argmin(lengths[candidates])])
        heading = steps[i] / lengths[i]
        chain.append(points.pop(i))
    return np.asarray(chain)


def _stations(line):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]


def _at(line, stations, values):
    return np.stack([np.interp(values, stations, line[:, axis]) for axis in (0, 1)], axis=-1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def build_track_geometry(blue_cones, yellow_cones, track_width, settings):
    blue = _chain(blue_cones, settings['max_cone_gap_m'])
    yellow = _chain(yellow_cones, settings['max_cone_gap_m'])
    track = TrackGeometry(blue, yellow)
    if min(len(blue), len(yellow)) < 2:
        return track

    blue_s, yellow_s = _stations(blue), _stations(yellow)
    # Согласуем относительный прогресс по обеим границам: на дуге внешняя длиннее.
    progress = np.unique(np.round(np.r_[blue_s / blue_s[-1], yellow_s / yellow_s[-1],
                                       np.linspace(0, 1, 25)], 6))
    blue = _at(blue, blue_s, progress * blue_s[-1])
    yellow = _at(yellow, yellow_s, progress * yellow_s[-1])
    centers = (blue + yellow) / 2
    across = yellow - blue
    widths = np.linalg.norm(across, axis=1)
    tangent = np.gradient(centers, axis=0)
    tangent_length = np.linalg.norm(tangent, axis=1)
    sides = _cross(tangent, across)
    clearance = settings['boundary_clearance_m']
    valid = ((widths >= max(track_width * 0.5, 2 * clearance + 0.05)) &
             (widths <= track_width * 1.6) & (tangent_length > 1e-5) &
             (np.abs(sides) >= 0.6 * tangent_length * widths) & (sides * sides[0] > 0))
    invalid = np.flatnonzero(~valid)
    count = int(invalid[0]) if len(invalid) else len(valid)
    if count < 3:
        return track
    blue, yellow, centers = blue[:count], yellow[:count], centers[:count]
    stations = _stations(centers)
    if stations[-1] < 0.10:
        return track

    # Сглаживаем локально, затем проверяем точки и середины отрезков
    # в коридоре, сдвинутом внутрь на запас до каждой границы.
    inset = across[:count] * (clearance / widths[:count])[:, None]
    safe_blue, safe_yellow = blue + inset, yellow - inset
    cells = np.stack([safe_blue[:-1], safe_yellow[:-1], safe_yellow[1:], safe_blue[1:]], axis=1)
    edges = np.roll(cells, -1, axis=1) - cells
    sample_s = np.linspace(0., stations[-1], math.ceil(stations[-1] / 0.10) + 1)
    center = _at(centers, stations, sample_s)
    if len(center) >= 5:
        filtered = center.copy()
        filtered[2:-2] = (center[:-4] + 4 * center[1:-3] + 6 * center[2:-2] +
                          4 * center[3:-1] + center[4:]) / 16
        center = filtered
    samples = np.empty((2 * len(center) - 1, 2))
    samples[::2] = center
    samples[1::2] = (center[:-1] + center[1:]) / 2
    crosses = _cross(edges[None, :, :, :], samples[:, None, None, :] - cells[None, :, :, :])
    inside = np.all(crosses >= -1e-7, axis=2) | np.all(crosses <= 1e-7, axis=2)
    if not np.all(np.any(inside, axis=1)):
        return track

    return TrackGeometry(track.blue, track.yellow, center)
