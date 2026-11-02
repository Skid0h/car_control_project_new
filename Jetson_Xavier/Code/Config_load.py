"""
Загрузка конфигурации из JSONC (JSON с комментариями // и /* */).

Схема описана таблицей _SPEC: (секция, ключ, атрибут, значение по умолчанию).
Пропущенный ключ не роняет программу - берётся умолчание и пишется WARNING,
чтобы старый конфиг продолжал работать после обновления кода.
Неизвестный ключ тоже даёт WARNING: это ловит опечатки, из-за которых параметр
молча применялся бы со значением по умолчанию.

После загрузки выполняется _validate(): несогласованные пределы (например
center_steering вне steer_min..steer_max или neutral_speed вне esc_min..esc_max)
не пускают программу в работу вместо того, чтобы проявиться на трассе.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Ошибка конфигурации: файл не найден, не разбирается или противоречив."""


def strip_jsonc_comments(text):
    """Убирает // и /* */ комментарии, не трогая содержимое строк.

    Наивная замена re.sub(r'//.*', '') ломает значения, в которых встречается '//'
    (например путь или URL), поэтому разбор посимвольный.
    """
    out = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == '\\' and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < n and text[i + 1] == '/':
            while i < n and text[i] != '\n':
                i += 1
            continue
        if ch == '/' and i + 1 < n and text[i + 1] == '*':
            i += 2
            while i + 1 < n and not (text[i] == '*' and text[i + 1] == '/'):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


# (секция, ключ, атрибут, умолчание)
_SPEC = [
    ('network', 'udp_ip', 'udp_ip', '0.0.0.0'),
    ('network', 'udp_port', 'udp_port', 5005),

    ('web', 'host', 'web_host', '0.0.0.0'),
    ('web', 'port', 'web_port', 5000),
    ('web', 'publish_fps', 'web_publish_fps', 8.0),
    ('web', 'jpeg_quality', 'jpeg_quality', 70),

    ('car', 'baud_rate', 'baud_rate', 9600),
    ('car', 'arduino_port', 'arduino_port', ''),
    ('car', 'neutral_speed', 'neutral_speed', 1500),
    ('car', 'forward_speed', 'forward_speed', 1570),
    ('car', 'back_speed', 'back_speed', 1420),
    ('car', 'esc_min', 'esc_min', 1400),
    ('car', 'esc_max', 'esc_max', 1600),
    ('car', 'center_steering', 'center_steering', 90),
    ('car', 'steering_range', 'steering_range', 50),
    ('car', 'steer_min', 'steer_min', 0),
    ('car', 'steer_max', 'steer_max', 180),
    ('car', 'command_interval', 'command_interval', 0.02),
    ('car', 'watchdog_timeout', 'watchdog_timeout', 0.4),
    ('car', 'arduino_retry_interval', 'arduino_retry_interval', 2.0),

    ('control', 'control_rate_hz', 'control_rate_hz', 30.0),
    ('control', 'detection_max_age', 'detection_max_age', 0.5),
    ('control', 'target_loss_max_ticks', 'target_loss_max_ticks', 12),

    ('pid', 'kp_gain', 'kp_gain', 1.0),
    ('pid', 'ema_alpha', 'ema_alpha', 0.4),
    ('pid', 'error_decay_rate', 'error_decay_rate', 0.85),
    ('pid', 'max_steering_output', 'max_steering_output', 1.0),
    # Зарезервировано: в v1.0 не используются, загружаются только чтобы не терять конфиг.
    ('pid', 'ki_gain', 'ki_gain', 0.0),
    ('pid', 'kd_gain', 'kd_gain', 0.0),
    ('pid', 'max_integral', 'max_integral', 0.0),
    ('pid', 'min_dt', 'min_dt', 0.001),

    ('autopilot', 'max_depth', 'max_depth', 4.0),
    ('autopilot', 'min_depth', 'min_depth', 0.1),
    ('autopilot', 'track_width', 'track_width', 1.5),
    ('autopilot', 'lookahead_distance', 'lookahead_distance', 0.5),
    ('autopilot', 'pair_z_tolerance', 'pair_z_tolerance', 1.0),
    ('autopilot', 'pair_z_tolerance_min', 'pair_z_tolerance_min', 0.25),
    ('autopilot', 'pair_z_tolerance_multiplier', 'pair_z_tolerance_multiplier', 0.25),
    ('autopilot', 'pair_x_tolerance_multiplier', 'pair_x_tolerance_multiplier', 1.5),
    ('autopilot', 'pair_steer_k', 'pair_steer_k', 2.0),
    ('autopilot', 'virtual_point_offset', 'virtual_point_offset', 0.7),
    ('autopilot', 'virtual_steer_k', 'virtual_steer_k', 2.5),
    ('autopilot', 'stop_cone_z_threshold', 'stop_cone_z_threshold', 0.5),
    ('autopilot', 'area_depth_constant', 'area_depth_constant', 129.1),

    ('vision', 'yolo_model_path', 'yolo_model_path', ''),
    ('vision', 'confidence_threshold', 'confidence_threshold', 0.5),
    ('vision', 'iou_threshold', 'iou_threshold', 0.5),
    ('vision', 'target_fps', 'target_fps', 15.0),
    ('vision', 'output_folder', 'output_folder', 'zed_recordings'),
    ('vision', 'camera_offset_x', 'camera_offset_x', 0.0),
    ('vision', 'camera_offset_z', 'camera_offset_z', 0.0),
    ('vision', 'zed_resolution', 'zed_resolution', 'HD720'),
    ('vision', 'zed_fps', 'zed_fps', 15),
    ('vision', 'coordinate_units', 'coordinate_units', 'METER'),
    ('vision', 'detect_max_width', 'detect_max_width', 480),
    ('vision', 'detect_max_height', 'detect_max_height', 270),
    ('vision', 'detect_crop_top', 'detect_crop_top', 0.20),
    ('vision', 'detect_crop_bottom', 'detect_crop_bottom', 0.20),
    ('vision', 'cone_base_v', 'cone_base_v', 0.65),
    ('vision', 'point_of_view_offset_y', 'point_of_view_offset_y', 0.8),
    ('vision', 'target_cross_size', 'target_cross_size', 25),
    ('vision', 'target_cross_thickness', 'target_cross_thickness', 3),

    ('detection', 'cone_colors', 'cone_colors', {}),
    ('detection', 'class_names', 'class_names', {}),
    ('detection', 'blue_cones', 'blue_cones', ['blue']),
    ('detection', 'yellow_cones', 'yellow_cones', ['yellow']),
    ('detection', 'orange_cones', 'orange_cones', ['orange']),
    ('detection', 'circle_marker_radius', 'circle_marker_radius', 4),
    ('detection', 'circle_marker_color', 'circle_marker_color', [255, 255, 255]),
    ('detection', 'text_scale', 'text_scale', 0.5),
    ('detection', 'text_thickness', 'text_thickness', 2),
    ('detection', 'z_text_scale', 'z_text_scale', 0.4),
    ('detection', 'z_text_thickness', 'z_text_thickness', 1),
    ('detection', 'z_text_color', 'z_text_color', [255, 255, 255]),

    ('display', 'draw_detections', 'draw_detections', True),
    ('display', 'draw_trajectory', 'draw_trajectory', True),
    ('display', 'draw_target', 'draw_target', True),
    ('display', 'draw_fps', 'draw_fps', True),
    ('display', 'draw_target_z', 'draw_target_z', True),
    ('display', 'draw_rec', 'draw_rec', True),
    ('display', 'fps_text_scale', 'fps_text_scale', 0.7),
    ('display', 'fps_text_thickness', 'fps_text_thickness', 2),
    ('display', 'fps_text_color', 'fps_text_color', [0, 255, 0]),
    ('display', 'target_z_text_scale', 'target_z_text_scale', 0.6),
    ('display', 'target_z_text_thickness', 'target_z_text_thickness', 2),
    ('display', 'target_z_text_color', 'target_z_text_color', [0, 255, 255]),
    ('display', 'rec_text_scale', 'rec_text_scale', 0.7),
    ('display', 'rec_text_thickness', 'rec_text_thickness', 2),
    ('display', 'rec_text_color', 'rec_text_color', [0, 0, 255]),
    ('display', 'trajectory_thickness', 'trajectory_thickness', 2),
    ('display', 'trajectory_color', 'trajectory_color', [0, 255, 0]),
    ('display', 'waypoint_radius', 'waypoint_radius', 6),
    ('display', 'waypoint_color_pair', 'waypoint_color_pair', [0, 255, 0]),
    ('display', 'waypoint_color_virtual', 'waypoint_color_virtual', [255, 200, 0]),
    ('display', 'waypoint_color_stop', 'waypoint_color_stop', [0, 0, 255]),
    ('display', 'pair_line_thickness', 'pair_line_thickness', 1),
    ('display', 'pair_line_color', 'pair_line_color', [0, 255, 255]),

    ('video', 'temp_codec', 'temp_codec', 'MJPG'),
    ('video', 'output_codec', 'output_codec', 'libx264'),
    ('video', 'output_preset', 'output_preset', 'fast'),
    ('video', 'output_crf', 'output_crf', '23'),
    ('video', 'output_pix_fmt', 'output_pix_fmt', 'yuv420p'),
    ('video', 'output_extension', 'output_extension', 'mp4'),
    ('video', 'temp_extension', 'temp_extension', 'avi'),
    ('video', 'output_prefix', 'output_prefix', 'zed_recording'),
    ('video', 'fps_update_interval', 'fps_update_interval', 0.5),
    ('video', 'rec_queue_size', 'rec_queue_size', 30),

    ('timing', 'vision_thread_join_timeout', 'vision_thread_join_timeout', 3.0),
    ('timing', 'thread_stop_timeout', 'thread_stop_timeout', 2.0),
    ('timing', 'message_clear_timeout', 'message_clear_timeout', 3.0),
    ('timing', 'socket_timeout', 'socket_timeout', 0.2),
    ('timing', 'arduino_init_delay', 'arduino_init_delay', 2.0),
    ('timing', 'arduino_post_stop_delay', 'arduino_post_stop_delay', 0.5),
    ('timing', 'arduino_close_delay', 'arduino_close_delay', 0.1),
]

_COLOR_ATTRS = {
    'circle_marker_color', 'z_text_color', 'fps_text_color', 'target_z_text_color',
    'rec_text_color', 'trajectory_color', 'waypoint_color_pair',
    'waypoint_color_virtual', 'waypoint_color_stop', 'pair_line_color',
}


def _to_color(value):
    try:
        return tuple(int(c) for c in value)
    except (TypeError, ValueError):
        return (255, 255, 255)


def _parse_file(path):
    if not os.path.exists(path):
        raise ConfigError("Файл конфигурации '%s' не найден." % path)
    # utf-8-sig: файл с BOM (его легко получить в Windows-редакторе) иначе не разбирается.
    with open(path, 'r', encoding='utf-8-sig') as handle:
        content = handle.read()
    try:
        return json.loads(strip_jsonc_comments(content))
    except json.JSONDecodeError as exc:
        raise ConfigError("Не удалось разобрать '%s': %s" % (path, exc))


class Config:
    def __init__(self, config_path="config.jsonc"):
        self.config_path = os.path.abspath(config_path)
        self._apply(_parse_file(self.config_path), self.config_path)

    # ------------------------------------------------------------------
    def _apply(self, raw, path):
        if not isinstance(raw, dict):
            raise ConfigError("'%s': ожидался JSON-объект верхнего уровня." % path)

        values = {}
        missing = []
        for section, key, attr, default in _SPEC:
            block = raw.get(section)
            if not isinstance(block, dict):
                missing.append('%s (нет секции "%s")' % (attr, section))
                values[attr] = default
                continue
            if key not in block:
                missing.append('%s' % attr)
                values[attr] = default
            else:
                values[attr] = block[key]

        if missing:
            logger.warning("В '%s' отсутствуют ключи, взяты умолчания: %s",
                           path, ', '.join(missing))

        known = {}
        for section, key, _attr, _default in _SPEC:
            known.setdefault(section, set()).add(key)
        unknown = []
        for section, block in raw.items():
            if not isinstance(block, dict):
                unknown.append(section)
                continue
            for key in block:
                if key not in known.get(section, set()):
                    unknown.append('%s.%s' % (section, key))
        if unknown:
            logger.warning("Неизвестные ключи в '%s' (возможна опечатка, они игнорируются): %s",
                           path, ', '.join(unknown))

        for attr in _COLOR_ATTRS:
            if attr in values:
                values[attr] = _to_color(values[attr])

        if isinstance(values.get('cone_colors'), dict):
            values['cone_colors'] = {
                name: _to_color(color) for name, color in values['cone_colors'].items()
            }

        if isinstance(values.get('class_names'), dict):
            mapped = {}
            for class_id, name in values['class_names'].items():
                try:
                    mapped[int(class_id)] = str(name)
                except (TypeError, ValueError):
                    logger.warning("class_names: ключ '%s' не является числом, пропущен.", class_id)
            values['class_names'] = mapped

        for attr in ('blue_cones', 'yellow_cones', 'orange_cones'):
            value = values.get(attr)
            values[attr] = set(value) if isinstance(value, (list, tuple, set)) else set()

        values['arduino_port'] = str(values.get('arduino_port') or '').strip()
        values['yolo_model_path'] = os.path.expanduser(str(values.get('yolo_model_path') or ''))

        self.__dict__.update(values)
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        problems = []

        def need(cond, message):
            if not cond:
                problems.append(message)

        need(self.esc_min < self.esc_max,
             'esc_min (%s) должен быть меньше esc_max (%s)' % (self.esc_min, self.esc_max))
        need(self.esc_min <= self.neutral_speed <= self.esc_max,
             'neutral_speed (%s) вне [esc_min; esc_max] = [%s; %s]'
             % (self.neutral_speed, self.esc_min, self.esc_max))
        need(self.esc_min <= self.forward_speed <= self.esc_max,
             'forward_speed (%s) вне [esc_min; esc_max] = [%s; %s]. Прошивка отбросит такой пакет ЦЕЛИКОМ.'
             % (self.forward_speed, self.esc_min, self.esc_max))
        need(self.esc_min <= self.back_speed <= self.esc_max,
             'back_speed (%s) вне [esc_min; esc_max] = [%s; %s]. Прошивка отбросит такой пакет ЦЕЛИКОМ.'
             % (self.back_speed, self.esc_min, self.esc_max))

        need(self.steer_min < self.steer_max,
             'steer_min (%s) должен быть меньше steer_max (%s)' % (self.steer_min, self.steer_max))
        need(self.steer_min <= self.center_steering <= self.steer_max,
             'center_steering (%s) вне [steer_min; steer_max] = [%s; %s]'
             % (self.center_steering, self.steer_min, self.steer_max))
        need(self.center_steering - self.steering_range >= self.steer_min
             and self.center_steering + self.steering_range <= self.steer_max,
             'center_steering ± steering_range = [%s; %s] выходит за [steer_min; steer_max] = [%s; %s]'
             % (self.center_steering - self.steering_range,
                self.center_steering + self.steering_range,
                self.steer_min, self.steer_max))
        need(self.steering_range >= 0, 'steering_range не может быть отрицательным')

        need(0 < self.watchdog_timeout, 'watchdog_timeout должен быть больше 0')
        need(self.command_interval >= 0, 'command_interval не может быть отрицательным')
        need(0 < self.control_rate_hz <= 500, 'control_rate_hz вне разумного диапазона (0; 500]')
        need(1.0 / self.control_rate_hz < self.watchdog_timeout,
             'период control_rate_hz (%.3f с) должен быть меньше watchdog_timeout (%.3f с)'
             % (1.0 / self.control_rate_hz, self.watchdog_timeout))
        need(self.target_loss_max_ticks >= 1, 'target_loss_max_ticks должен быть >= 1')
        need(self.detection_max_age > 0, 'detection_max_age должен быть больше 0')

        need(0 < self.ema_alpha <= 1.0, 'ema_alpha должен быть в (0; 1.0]')
        need(0 <= self.error_decay_rate <= 1.0, 'error_decay_rate должен быть в [0; 1.0]')
        need(self.error_decay_rate < 1.0,
             'error_decay_rate = 1.0 означает, что руль не возвращается в центр при потере цели')
        need(self.max_steering_output > 0, 'max_steering_output должен быть больше 0')
        need(self.kp_gain > 0, 'kp_gain должен быть больше 0')

        need(self.min_depth < self.max_depth,
             'min_depth (%s) должен быть меньше max_depth (%s)' % (self.min_depth, self.max_depth))
        need(self.area_depth_constant > 0, 'area_depth_constant должен быть больше 0')
        need(self.track_width > 0, 'track_width должен быть больше 0')
        need(self.pair_z_tolerance_min > 0, 'pair_z_tolerance_min должен быть больше 0')
        need(self.pair_z_tolerance_min <= self.pair_z_tolerance,
             'pair_z_tolerance_min (%s) не должен превышать pair_z_tolerance (%s)'
             % (self.pair_z_tolerance_min, self.pair_z_tolerance))
        need(self.pair_z_tolerance_multiplier >= 0, 'pair_z_tolerance_multiplier не может быть отрицательным')
        need(self.pair_x_tolerance_multiplier > 0, 'pair_x_tolerance_multiplier должен быть больше 0')
        need(0 <= self.lookahead_distance < self.max_depth,
             'lookahead_distance (%s) должен быть в [0; max_depth = %s), иначе цель почти всегда самая дальняя точка'
             % (self.lookahead_distance, self.max_depth))
        need(self.stop_cone_z_threshold > 0, 'stop_cone_z_threshold должен быть больше 0')
        need(self.virtual_point_offset >= 0, 'virtual_point_offset не может быть отрицательным')

        need(self.zed_fps > 0, 'zed_fps должен быть больше 0')
        need(self.target_fps > 0, 'target_fps должен быть больше 0')
        need(self.web_publish_fps > 0, 'web_publish_fps должен быть больше 0')
        need(0 <= self.detect_crop_top < 1.0, 'detect_crop_top должен быть в [0; 1)')
        need(0 <= self.detect_crop_bottom < 1.0, 'detect_crop_bottom должен быть в [0; 1)')
        need(self.detect_crop_top + self.detect_crop_bottom <= 0.9,
             'detect_crop_top + detect_crop_bottom = %.2f оставляет слишком мало кадра (нужно <= 0.9)'
             % (self.detect_crop_top + self.detect_crop_bottom))
        need(self.detect_max_width >= 32 and self.detect_max_height >= 32,
             'detect_max_width/height должны быть не меньше 32')
        need(0 < self.point_of_view_offset_y <= 1.0, 'point_of_view_offset_y должен быть в (0; 1.0]')
        need(0 < self.confidence_threshold <= 1.0, 'confidence_threshold должен быть в (0; 1.0]')
        need(0 < self.iou_threshold <= 1.0, 'iou_threshold должен быть в (0; 1.0]')
        need(bool(self.yolo_model_path), 'yolo_model_path пуст')
        need(bool(self.class_names), 'detection.class_names пуст: детектор не сможет назвать ни один класс')
        need(bool(self.blue_cones) and bool(self.yellow_cones),
             'blue_cones и yellow_cones не могут быть пустыми: без них не строится ни одна пара')

        if problems:
            raise ConfigError('Проверка конфигурации не пройдена:\n  - ' + '\n  - '.join(problems))

    # ------------------------------------------------------------------
    def reload(self, config_path=None):
        """Перечитать конфиг без перезапуска. Возвращает True при успехе.

        При ошибке текущие значения сохраняются: на трассе лучше остаться
        с рабочим конфигом, чем упасть из-за опечатки в файле.
        """
        path = os.path.abspath(config_path or self.config_path)
        try:
            fresh = Config(path)
        except ConfigError as exc:
            logger.error("Перезагрузка конфигурации отменена: %s", exc)
            return False
        snapshot = {key: value for key, value in fresh.__dict__.items()}
        self.__dict__.update(snapshot)
        self.config_path = path
        logger.info("Конфигурация перечитана: %s", path)
        return True