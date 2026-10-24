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
    """Плоское представление настроек с ранней проверкой критичных значений."""

    DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.jsonc"

    def __init__(self, config_path=None):
        self.path = self._resolve_path(config_path)
        cfg = self._read_config(self.path)

        try:
            # Network
            net = cfg['network']
            self.udp_ip = net['udp_ip']
            self.udp_port = net['udp_port']

            # Car
            car = cfg['car']
            self.baud_rate = car['baud_rate']
            self.neutral_speed = car['neutral_speed']
            self.forward_speed = car['forward_speed']
            self.back_speed = car['back_speed']
            self.center_steering = car['center_steering']
            self.command_interval = car['command_interval']
            self.watchdog_timeout = car['watchdog_timeout']
            self.steering_range = car['steering_range']

            # PID
            pid = cfg['pid']
            self.kp_gain = pid['kp_gain']
            self.ki_gain = pid['ki_gain']
            self.kd_gain = pid['kd_gain']
            self.max_integral = pid['max_integral']
            self.ema_alpha = pid['ema_alpha']
            self.error_decay_rate = pid['error_decay_rate']
            self.max_steering_output = pid['max_steering_output']
            self.min_dt = pid['min_dt']

            # Autopilot
            ap = cfg['autopilot']
            self.max_depth = ap['max_depth']
            self.min_depth = ap['min_depth']
            self.track_width = ap['track_width']
            self.lookahead_distance = ap['lookahead_distance']
            self.pair_z_tolerance = ap['pair_z_tolerance']
            self.pair_x_tolerance_multiplier = ap['pair_x_tolerance_multiplier']
            self.virtual_point_offset = ap['virtual_point_offset']
            self.virtual_steer_k = ap['virtual_steer_k']
            self.stop_cone_z_threshold = ap['stop_cone_z_threshold']
            self.area_depth_constant = ap['area_depth_constant']

            # Vision
            vis = cfg['vision']
            self.yolo_model_path = vis['yolo_model_path']
            self.confidence_threshold = vis['confidence_threshold']
            self.iou_threshold = vis['iou_threshold']
            self.target_fps = vis['target_fps']
            self.output_folder = vis['output_folder']
            self.camera_offset_x = vis['camera_offset_x']
            self.camera_offset_z = vis['camera_offset_z']
            self.zed_resolution = vis['zed_resolution']
            self.zed_fps = vis['zed_fps']
            self.depth_mode = vis['depth_mode']
            self.coordinate_units = vis['coordinate_units']
            self.cone_base_v = vis['cone_base_v']
            self.point_of_view_offset_y = vis['point_of_view_offset_y']
            self.target_cross_size = vis['target_cross_size']
            self.target_cross_thickness = vis['target_cross_thickness']

            # Detection
            det = cfg['detection']
            self.cone_colors = det['cone_colors']
            self.class_names = det['class_names']
            self.blue_cones = det['blue_cones']
            self.yellow_cones = det['yellow_cones']
            self.orange_cones = det['orange_cones']
            self.circle_marker_radius = det['circle_marker_radius']
            self.circle_marker_color = det['circle_marker_color']
            self.text_scale = det['text_scale']
            self.text_thickness = det['text_thickness']
            self.z_text_scale = det['z_text_scale']
            self.z_text_thickness = det['z_text_thickness']
            self.z_text_color = det['z_text_color']

            # Display
            disp = cfg['display']
            self.draw_detections = disp['draw_detections']
            self.draw_trajectory = disp['draw_trajectory']
            self.draw_target = disp['draw_target']
            self.draw_fps = disp['draw_fps']
            self.draw_target_z = disp['draw_target_z']
            self.draw_rec = disp['draw_rec']
            self.fps_text_scale = disp['fps_text_scale']
            self.fps_text_thickness = disp['fps_text_thickness']
            self.fps_text_color = disp['fps_text_color']
            self.target_z_text_scale = disp['target_z_text_scale']
            self.target_z_text_thickness = disp['target_z_text_thickness']
            self.target_z_text_color = disp['target_z_text_color']
            self.rec_text_scale = disp['rec_text_scale']
            self.rec_text_thickness = disp['rec_text_thickness']
            self.rec_text_color = disp['rec_text_color']
            self.trajectory_thickness = disp['trajectory_thickness']
            self.trajectory_color = disp['trajectory_color']
            self.waypoint_radius = disp['waypoint_radius']
            self.waypoint_color_pair = disp['waypoint_color_pair']
            self.waypoint_color_virtual = disp['waypoint_color_virtual']
            self.waypoint_color_stop = disp['waypoint_color_stop']
            self.pair_line_thickness = disp['pair_line_thickness']
            self.pair_line_color = disp['pair_line_color']

            # Video
            vid = cfg['video']
            self.temp_codec = vid['temp_codec']
            self.output_codec = vid['output_codec']
            self.output_preset = vid['output_preset']
            self.output_crf = vid['output_crf']
            self.output_pix_fmt = vid['output_pix_fmt']
            self.output_extension = vid['output_extension']
            self.temp_extension = vid['temp_extension']
            self.output_prefix = vid['output_prefix']
            self.fps_update_interval = vid['fps_update_interval']

            # Timing
            tim = cfg['timing']
            self.vision_thread_join_timeout = tim['vision_thread_join_timeout']
            self.message_clear_timeout = tim['message_clear_timeout']
            self.socket_timeout = tim['socket_timeout']
            self.arduino_init_delay = tim['arduino_init_delay']
            self.arduino_post_stop_delay = tim['arduino_post_stop_delay']
            self.arduino_close_delay = tim['arduino_close_delay']

        except KeyError as error:
            raise ConfigError(
                f"В конфигурации {self.path} отсутствует обязательный параметр: {error.args[0]!r}."
            ) from error

        self.publish_fps = cfg["vision"].get("publish_fps", 0.0)
        self.rec_fps = cfg["video"].get("rec_fps", self.zed_fps)
        self.rec_queue_size = cfg["video"].get("rec_queue_size", 120)
        self._validate()

    @classmethod
    def _resolve_path(cls, config_path):
        if config_path is None:
            return cls.DEFAULT_PATH
        return Path(config_path).expanduser().resolve()

    @staticmethod
    def _read_config(config_path):
        if not config_path.is_file():
            raise ConfigError(f"Файл конфигурации не найден: {config_path}")

        try:
            content = config_path.read_text(encoding="utf-8")
            return json.loads(_strip_jsonc_comments(content))
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"Некорректный JSONC в {config_path}: строка {error.lineno}, "
                f"столбец {error.colno}: {error.msg}"
            ) from error

    def _validate(self):
        if not 1 <= int(self.udp_port) <= 65535:
            raise ConfigError("network.udp_port должен быть в диапазоне 1..65535.")
        if int(self.zed_fps) <= 0:
            raise ConfigError("vision.zed_fps должен быть больше нуля.")
        if float(self.target_fps) <= 0:
            raise ConfigError("vision.target_fps должен быть больше нуля.")
        if int(self.rec_fps) <= 0:
            raise ConfigError("video.rec_fps должен быть больше нуля.")
        if int(self.rec_queue_size) <= 0:
            raise ConfigError("video.rec_queue_size должен быть больше нуля.")
        for name in ("confidence_threshold", "iou_threshold", "point_of_view_offset_y"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} должен быть в диапазоне 0..1.")

    def reload(self, config_path=None):
        """Перезагружает текущий файл, если другой путь не передан."""
        self.__init__(self.path if config_path is None else config_path)
