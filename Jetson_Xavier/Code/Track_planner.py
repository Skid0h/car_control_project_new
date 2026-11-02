"""
Чистая логика: геометрия кадра, восстановление координат конусов, пары, траектория, руление.

Модуль НЕ зависит от cv2 / torch / pyzed, поэтому проверяется на любой машине:
    python -m Code.selfcheck_planner

Соглашение по осям (как в ZED): x - вправо от продольной оси машины, z - вперёд, метры.
"""

import math

# Ближе этой дистанции точка считается недостоверной (защита от деления на ~0 и от
# срабатываний прямо перед объективом).
MIN_Z_CAR = 0.05


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


# --------------------------------------------------------------------------
# 1. Геометрия предварительной обработки кадра
# --------------------------------------------------------------------------

def compute_preprocess_plan(img_w, img_h, max_w, max_h, crop_top_frac, crop_bottom_frac):
    """План уменьшения/обрезки кадра и коэффициенты ОБРАТНОГО пересчёта координат.

    Возвращает словарь:
      resized_w, resized_h - размер кадра после уменьшения (ДО обрезки)
      crop_top, crop_bottom- сколько пикселей отрезается сверху/снизу (после уменьшения)
      detect_w, detect_h   - размер кадра, который уходит в детектор
      scale_x, scale_y     - множители: координата детекции -> пиксель исходного кадра
      offset_x, offset_y   - сдвиги, которые прибавляются ПОСЛЕ умножения

    Ключевое: scale_y считается от высоты ПОСЛЕ УМЕНЬШЕНИЯ (resized_h), а не от высоты
    обрезанного кадра, и к вертикали добавляется offset_y = crop_top * scale_y.
    Если делить на высоту обрезанного кадра, рамки растягиваются по вертикали,
    площадь завышается, а оценка глубины z = K / sqrt(area) систематически занижается.
    """
    img_w = int(img_w)
    img_h = int(img_h)
    resized_w, resized_h = img_w, img_h

    if max_w and max_h and (img_w > max_w or img_h > max_h):
        scale = min(max_w / float(img_w), max_h / float(img_h))
        if scale < 1.0:
            resized_w = max(32, int(round(img_w * scale)))
            resized_h = max(32, int(round(img_h * scale)))

    crop_top_frac = _clamp(float(crop_top_frac or 0.0), 0.0, 0.9)
    crop_bottom_frac = _clamp(float(crop_bottom_frac or 0.0), 0.0, 0.9 - crop_top_frac)

    crop_top = _clamp(int(round(resized_h * crop_top_frac)), 0, max(0, resized_h - 1))
    crop_bottom = _clamp(int(round(resized_h * crop_bottom_frac)),
                         0, max(0, resized_h - crop_top - 1))

    detect_w = resized_w
    detect_h = max(1, resized_h - crop_top - crop_bottom)

    scale_x = img_w / float(resized_w)
    scale_y = img_h / float(resized_h)

    return {
        'resized_w': resized_w,
        'resized_h': resized_h,
        'crop_top': crop_top,
        'crop_bottom': crop_bottom,
        'detect_w': detect_w,
        'detect_h': detect_h,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'offset_x': 0.0,
        'offset_y': crop_top * scale_y,
    }


def map_point(x, y, plan):
    """Точка в системе детектора -> точка в исходном кадре."""
    return (x * plan['scale_x'] + plan['offset_x'],
            y * plan['scale_y'] + plan['offset_y'])


def rescale_detections(detections, plan):
    """Пересчитывает bbox и center детекций в координаты исходного кадра (in-place)."""
    sx, sy = plan['scale_x'], plan['scale_y']
    ox, oy = plan['offset_x'], plan['offset_y']
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        det['bbox'] = (int(round(x1 * sx + ox)), int(round(y1 * sy + oy)),
                       int(round(x2 * sx + ox)), int(round(y2 * sy + oy)))
        center = det.get('center')
        if center is not None:
            det['center'] = (int(round(center[0] * sx + ox)),
                             int(round(center[1] * sy + oy)))
    return detections


# --------------------------------------------------------------------------
# 2. Пиксели -> метры
# --------------------------------------------------------------------------

def detections_to_cones(detections, cfg, fx, cx):
    """Оценка расстояния по площади рамки и переход в координаты машины.

    z = area_depth_constant / sqrt(area) - площадь берётся в пикселях ИСХОДНОГО кадра.
    x_cam - положение относительно оптической оси камеры;
    x_car = x_cam + camera_offset_x - положение относительно продольной оси машины.
    """
    cones = []
    if not detections or fx <= 0:
        return cones

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        area = width * height
        if area <= 0:
            continue

        z = cfg.area_depth_constant / math.sqrt(area)
        if not (cfg.min_depth < z <= cfg.max_depth):
            continue

        u, _v = det['center']
        x_cam = (u - cx) * z / fx
        x_car = x_cam + cfg.camera_offset_x
        z_car = z - cfg.camera_offset_z
        if z_car <= MIN_Z_CAR:
            continue

        det['pos_3d'] = (x_car, z_car)
        cones.append(det)

    return cones


# --------------------------------------------------------------------------
# 3. Пары и точки траектории
# --------------------------------------------------------------------------

def pair_z_tolerance(z, cfg):
    """Допуск по глубине для пары растёт вместе с дистанцией.

    Ошибка оценки z по площади пропорциональна самой z, поэтому фиксированный допуск
    на 4 метрах либо отсекает настоящие пары, либо пропускает соседние.
    Итог ограничен сверху pair_z_tolerance и снизу pair_z_tolerance_min.
    """
    tol = cfg.pair_z_tolerance_multiplier * max(0.0, z)
    return _clamp(tol, cfg.pair_z_tolerance_min, cfg.pair_z_tolerance)


def build_waypoints(cones, cfg):
    """Пары синий/жёлтый + виртуальные точки от одиночных конусов + стоп-конус.

    Возвращает (waypoints, stop_z). waypoints отсортированы по возрастанию z.

    Назначение пар глобальное: перебираются все комбинации, допустимые по обоим
    допускам, сортируются по суммарной стоимости и разбираются от лучшей к худшей.
    Это не зависит от порядка конусов в списке и не даёт одному синему конусу
    "забрать" жёлтый, который намного лучше подходит другому.

    Виртуальные точки добавляются для ВСЕХ непарных конусов всегда, а не только
    когда пар не нашлось вовсе. Раньше при одной найденной паре остальные конусы
    просто выбрасывались, и цель скачком перескакивала на другую точку.
    """
    blue = [c for c in cones if c.get('name') in cfg.blue_cones]
    yellow = [c for c in cones if c.get('name') in cfg.yellow_cones]
    orange = [c for c in cones if c.get('name') in cfg.orange_cones]

    blue.sort(key=lambda c: c['pos_3d'][1])
    yellow.sort(key=lambda c: c['pos_3d'][1])

    x_tol = max(0.1, cfg.track_width * cfg.pair_x_tolerance_multiplier)
    virtual_limit = max(cfg.track_width * 2.0, 0.5)

    candidates = []
    for bi, b_cone in enumerate(blue):
        b_x, b_z = b_cone['pos_3d']
        z_tol = pair_z_tolerance(b_z, cfg)
        for yi, y_cone in enumerate(yellow):
            y_x, y_z = y_cone['pos_3d']
            dz = abs(b_z - y_z)
            dx = abs(b_x - y_x)
            if dz <= z_tol and dx <= x_tol:
                candidates.append((dz / z_tol + dx / x_tol, bi, yi))

    candidates.sort(key=lambda item: item[0])

    used_blue = set()
    used_yellow = set()
    waypoints = []

    for _cost, bi, yi in candidates:
        if bi in used_blue or yi in used_yellow:
            continue
        used_blue.add(bi)
        used_yellow.add(yi)
        b_x, b_z = blue[bi]['pos_3d']
        y_x, y_z = yellow[yi]['pos_3d']
        waypoints.append({
            'x': (b_x + y_x) / 2.0,
            'z': (b_z + y_z) / 2.0,
            'type': 'pair',
            'steer_k': cfg.pair_steer_k,
            'b_center': blue[bi]['center'],
            'y_center': yellow[yi]['center'],
        })

    # Синий - правая граница, поэтому виртуальная точка смещается в минус по x (внутрь трассы).
    for bi, b_cone in enumerate(blue):
        if bi in used_blue:
            continue
        b_x, b_z = b_cone['pos_3d']
        waypoints.append({
            'x': _clamp(b_x - cfg.virtual_point_offset, -virtual_limit, virtual_limit),
            'z': b_z,
            'type': 'virtual_blue',
            'steer_k': cfg.virtual_steer_k,
            'b_center': b_cone['center'],
        })

    # Жёлтый - левая граница, смещение в плюс по x.
    for yi, y_cone in enumerate(yellow):
        if yi in used_yellow:
            continue
        y_x, y_z = y_cone['pos_3d']
        waypoints.append({
            'x': _clamp(y_x + cfg.virtual_point_offset, -virtual_limit, virtual_limit),
            'z': y_z,
            'type': 'virtual_yellow',
            'steer_k': cfg.virtual_steer_k,
            'y_center': y_cone['center'],
        })

    stop_z = None
    if orange:
        closest = min(orange, key=lambda c: c['pos_3d'][1])
        o_x, o_z = closest['pos_3d']
        stop_z = o_z
        waypoints.append({
            'x': o_x,
            'z': o_z,
            'type': 'stop',
            'steer_k': cfg.pair_steer_k,
            'o_center': closest['center'],
        })

    waypoints.sort(key=lambda wp: wp['z'])
    return waypoints, stop_z


def select_target(waypoints, lookahead_distance):
    """Выбор целевой точки с упреждением.

    Берётся ПЕРВАЯ точка с z >= lookahead_distance, а не ближайшая вообще.
    При lookahead_distance = 0 поведение прежнее - цель это ближайшая точка.
    Если все точки ближе дистанции упреждения, берётся самая дальняя из них.
    """
    if not waypoints:
        return None
    if lookahead_distance <= 0:
        return waypoints[0]
    for wp in waypoints:
        if wp['z'] >= lookahead_distance:
            return wp
    return waypoints[-1]


# --------------------------------------------------------------------------
# 4. Сглаживание руления
# --------------------------------------------------------------------------

class SteeringSmoother:
    """EMA по управляющему сигналу + затухание при потере цели.

    Сглаживается именно произведение error * steer_k * kp_gain, а не голая ошибка:
    тогда переключение типа точки (пара 2.0 -> одиночный конус 2.5) не даёт скачка руля.
    Старт с нуля: после включения авто-режима руль набирает угол за 2-3 такта,
    а не встаёт сразу в полное отклонение.
    """

    def __init__(self):
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def step(self, error, steer_k, cfg, have_target):
        if have_target:
            raw = error * steer_k * cfg.kp_gain
            alpha = cfg.ema_alpha
            self.value = raw * alpha + self.value * (1.0 - alpha)
        else:
            self.value *= cfg.error_decay_rate
        return _clamp(self.value, -cfg.max_steering_output, cfg.max_steering_output)