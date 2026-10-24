"""Загрузка и проверка конфигурации проекта из JSONC-файла."""

import json
from pathlib import Path


class ConfigError(ValueError):
    """Конфигурация отсутствует либо содержит недопустимые значения."""


def _strip_jsonc_comments(content):
    """Удаляет комментарии JSONC, не затрагивая ``//`` внутри строк JSON."""
    result = []
    in_string = False
    escaped = False
    index = 0

    while index < len(content):
        char = content[index]
        next_char = content[index + 1] if index + 1 < len(content) else ""

        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == chr(92):
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
        elif char == "/" and next_char == "/":
            index += 2
            while index < len(content) and content[index] not in "\r\n":
                index += 1
        elif char == "/" and next_char == "*":
            index += 2
            while index < len(content):
                if content[index] == "*" and index + 1 < len(content) and content[index + 1] == "/":
                    index += 2
                    break
                if content[index] in "\r\n":
                    result.append(content[index])
                index += 1
            else:
                raise ConfigError("Незакрытый блочный комментарий в JSONC-конфигурации.")
        else:
            result.append(char)
            index += 1

    return "".join(result)


class Config:
    """Only settings used by the ROS 2 Jetson runtime."""
    def __init__(self, config_path):
        import math
        self.path = Path(config_path).expanduser().resolve()
        raw = json.loads(_strip_jsonc_comments(self.path.read_text(encoding='utf-8')))
        required = {'network': ['udp_ip', 'udp_port'], 'car': ['baud_rate', 'neutral_speed', 'forward_speed', 'back_speed', 'center_steering', 'steering_range'], 'pid': ['kp_gain', 'ki_gain', 'kd_gain', 'max_integral', 'ema_alpha', 'max_steering_output', 'min_dt'], 'autopilot': ['max_depth', 'min_depth', 'track_width', 'lookahead_distance', 'stop_cone_z_threshold', 'area_depth_constant'], 'vision': ['yolo_model_path', 'confidence_threshold', 'iou_threshold', 'target_fps', 'camera_offset_z', 'zed_resolution', 'zed_fps'], 'detection': ['cone_colors', 'class_names', 'blue_cones', 'yellow_cones', 'orange_cones'], 'timing': ['arduino_init_delay', 'arduino_post_stop_delay']}
        for group, names in required.items():
            for name in names:
                try:
                    value = raw[group][name]
                except KeyError as error:
                    raise ConfigError('Missing configuration: ' + group + '.' + name) from error
                if isinstance(value, (float, int)) and not math.isfinite(value):
                    raise ConfigError('Non-finite configuration: ' + name)
                setattr(self, name, value)
        if not 1 <= self.udp_port <= 65535 or self.baud_rate <= 0:
            raise ConfigError('Invalid network port/baud rate')
        if self.zed_fps <= 0 or self.target_fps <= 0 or self.area_depth_constant <= 0:
            raise ConfigError('FPS and area constant must be positive')
        if not 0 <= self.confidence_threshold <= 1 or not 0 <= self.iou_threshold <= 1:
            raise ConfigError('Detection thresholds must be within 0..1')
        if self.arduino_init_delay < 0 or self.arduino_post_stop_delay < 0:
            raise ConfigError('Serial delays must be nonnegative')
