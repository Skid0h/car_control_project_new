"""
Загрузка конфигурации из JSONC-файла с поддержкой комментариев.
Все параметры вынесены в атрибуты класса для быстрого доступа.
"""

import os
import sys
import json
import math
import re


class Config:
    # Совместимость с examples/calibrate_depth.py, который читает эти атрибуты.
    area_depth_constant = 100.0
    cone_colors = {"yellow": (0, 255, 255), "orange": (0, 165, 255), "blue": (255, 0, 0)}

    def cone_distance_from_area(self, area, cone_name):
        """Расстояние от камеры, м; калибровка по площади bbox в исходном кадре."""
        calibration = self.area_depth_calibration.get(cone_name)
        if calibration is None:
            return self.area_depth_constant / math.sqrt(area)
        z = calibration['scale'] / math.sqrt(area) + calibration['offset']
        anchors = calibration.get('distance_correction', [])
        if not anchors or z <= anchors[0][0]:
            return z

        # Между контрольными замерами плавно меняем поправку.
        for (near_z, near_true), (far_z, far_true) in zip(anchors, anchors[1:]):
            if z <= far_z:
                weight = (z - near_z) / (far_z - near_z)
                return near_true + weight * (far_true - near_true)
        last_z, last_true = anchors[-1]
        return z + (last_true - last_z)

    def __init__(self, config_path="config.jsonc"):
        if not os.path.exists(config_path):
            print(f"Файл конфигурации '{config_path}' не найден.")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Сохраняем строки JSON, включая URL и экранированные кавычки.
        content = re.sub(r'("(?:\\.|[^"\\])*")|//[^\r\n]*',
                         lambda match: match.group(1) or '', content)
        cfg = json.loads(content)
        
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
        self.esc_min = car['esc_min']
        self.esc_max = car['esc_max']
        self.auto_speed = cfg['auto_speed']
        speed = self.auto_speed
        speed.setdefault('ramp_pwm', speed['start_pwm'])
        speed.setdefault('ramp_interval_s', 0.3)
        speed.setdefault('settle_time_s', 0.0)
        for key, value in speed.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'auto_speed.{key} должно быть конечным числом')
        if not (self.neutral_speed < speed['min_pwm'] <= speed['start_pwm'] <=
                speed['ramp_pwm'] <= speed['max_pwm'] <= self.esc_max):
            raise ValueError('Некорректные границы ШИМ auto_speed')
        for key in ('min_pwm', 'start_pwm', 'ramp_pwm', 'max_pwm', 'step_pwm'):
            if type(speed[key]) is not int:
                raise ValueError(f'auto_speed.{key} должно быть целым числом')
        for key in ('target_kmh', 'step_pwm', 'increase_interval_s', 'decrease_interval_s',
                    'startup_timeout_s', 'sensor_timeout_s', 'ramp_interval_s'):
            if speed[key] <= 0:
                raise ValueError(f'auto_speed.{key} должно быть положительным')
        if not 0 <= speed['deadband_kmh'] < speed['target_kmh']:
            raise ValueError('Некорректный допуск скорости')
        if speed['settle_time_s'] < 0:
            raise ValueError('auto_speed.settle_time_s не может быть отрицательным')
        
        # PID
        pid = cfg['pid']
        self.pid_debug = pid.get('debug', False)
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
        self.area_depth_calibration = ap.get('area_depth_calibration', {})
        self.graceful_degradation_speed = ap['graceful_degradation_speed']
        
        # Vision
        vis = cfg['vision']
        self.yolo_model_path = vis['yolo_model_path']
        self.confidence_threshold = vis['confidence_threshold']
        self.iou_threshold = vis['iou_threshold']
        # fps задаёт отрисовку и стрим, camera_fps — захват и основной цикл.
        self.fps = int(vis.get('fps', 15))
        self.zed_fps = int(vis.get('camera_fps', 30))
        if not 1 <= self.fps <= self.zed_fps <= 30:
            raise ValueError('Требуется 1 <= vision.fps <= vision.camera_fps <= 30')
        self.target_fps = float(self.zed_fps)
        self.output_folder = vis['output_folder']
        self.camera_offset_x = vis['camera_offset_x']
        self.camera_offset_z = vis['camera_offset_z']
        self.camera_to_front_m = vis.get('camera_to_front_m', self.camera_offset_z)
        self.zed_resolution = vis['zed_resolution']
        self.depth_mode = vis['depth_mode']
        self.coordinate_units = vis['coordinate_units']
        self.point_of_view_offset_y = vis['point_of_view_offset_y']
        self.target_cross_size = vis['target_cross_size']
        self.target_cross_thickness = vis['target_cross_thickness']
        
        # Detection
        det = cfg['detection']
        self.class_names = det['class_names']
        self.blue_cones = det['blue_cones']
        self.yellow_cones = det['yellow_cones']
        self.orange_cones = det['orange_cones']
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
        self.draw_speed = disp['draw_speed']
        self.draw_rec = disp['draw_rec']
        self.fps_text_scale = disp['fps_text_scale']
        self.fps_text_thickness = disp['fps_text_thickness']
        self.fps_text_color = disp['fps_text_color']
        self.speed_text_scale = disp['speed_text_scale']
        self.speed_text_thickness = disp['speed_text_thickness']
        self.speed_text_color = disp['speed_text_color']
        self.target_z_text_scale = disp['target_z_text_scale']
        self.target_z_text_thickness = disp['target_z_text_thickness']
        self.target_z_text_color = disp['target_z_text_color']
        self.rec_text_scale = disp['rec_text_scale']
        self.rec_text_thickness = disp['rec_text_thickness']
        self.rec_text_color = disp['rec_text_color']
        self.trajectory_thickness = disp['trajectory_thickness']
        self.trajectory_color = disp['trajectory_color']
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
    
    def reload(self, config_path="config.jsonc"):
        """Перезагрузка конфига без перезапуска программы"""
        self.__init__(config_path)
