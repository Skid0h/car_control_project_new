"""
Загрузка конфигурации из JSONC-файла с поддержкой однострочных комментариев (//).
Все параметры вынесены в атрибуты класса для быстрого доступа и автодополнения в IDE.
"""

import os
import sys
import json
import re


class Config:
    def __init__(config_self, config_path="config.jsonc"):
        if not os.path.exists(config_path):
            print(f"Файл конфигурации '{config_path}' не найден.")
            print(f"Текущая рабочая директория: {os.getcwd()}")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = re.sub(r'//.*', '', content)
        
        
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON в {config_path}:")
            print(f"Строка {e.lineno}, колонка {e.colno}: {e.msg}")
            sys.exit(1)
        
        # Network
        net = cfg.get('network', {})
        config_self.udp_ip = net.get('udp_ip', '0.0.0.0')
        config_self.udp_port = net.get('udp_port', 5005)
        
        # Car
        car = cfg.get('car', {})
        config_self.baud_rate = car.get('baud_rate', 9600)
        config_self.neutral_speed = car.get('neutral_speed', 1500)
        config_self.forward_speed = car.get('forward_speed', 1570)
        config_self.max_speed_pwm = car.get('max_speed_pwm', 1600) # Добавлено для dynamic lookahead
        config_self.back_speed = car.get('back_speed', 1430)
        config_self.center_steering = car.get('center_steering', 135)
        config_self.command_interval = car.get('command_interval', 0.1)
        config_self.watchdog_timeout = car.get('watchdog_timeout', 0.4)
        config_self.steering_range = car.get('steering_range', 90)
        
        # Autopilot & PID (объединены для удобства, как в идеальном конфиге)
        ap = cfg.get('autopilot', {})
        config_self.pid_kp = ap.get('pid_kp', 1.5)
        config_self.pid_ki = ap.get('pid_ki', 0.1)
        config_self.pid_kd = ap.get('pid_kd', 0.3)
        config_self.max_integral = ap.get('max_integral', 1.5)
        config_self.error_decay_rate = ap.get('error_decay_rate', 0.85)
        
        config_self.lookahead_min = ap.get('lookahead_min', 0.3)
        config_self.lookahead_max = ap.get('lookahead_max', 1.2)
        config_self.max_depth = ap.get('max_depth', 3.0)
        config_self.min_depth = ap.get('min_depth', 0.1)
        config_self.track_width = ap.get('track_width', 1.4)
        config_self.memory_timeout = ap.get('memory_timeout', 0.8) # Добавлено
        config_self.stop_cone_z_threshold = ap.get('stop_cone_z_threshold', 0.4)
        
        # Старые ключи для обратной совместимости (если вдруг где-то используются)
        config_self.lookahead_distance = ap.get('lookahead_distance', 0.5)
        config_self.pair_z_tolerance = ap.get('pair_z_tolerance', 1.0)
        config_self.pair_x_tolerance_multiplier = ap.get('pair_x_tolerance_multiplier', 1.5)
        config_self.virtual_point_offset = ap.get('virtual_point_offset', 0.7)
        config_self.area_depth_constant = ap.get('area_depth_constant', 150.0)
        
        # Vision
        vis = cfg.get('vision', {})
        config_self.yolo_model_path = vis.get('yolo_model_path', '')
        config_self.yolo_img_size = vis.get('yolo_img_size', 640) # Добавлено
        config_self.roi_crop_top = vis.get('roi_crop_top', 0.4)   # Добавлено
        config_self.roi_crop_bottom = vis.get('roi_crop_bottom', 1.0) # Добавлено
        config_self.confidence_threshold = vis.get('confidence_threshold', 0.5)
        config_self.iou_threshold = vis.get('iou_threshold', 0.5)
        config_self.target_fps = vis.get('target_fps', 10.0)
        config_self.output_folder = vis.get('output_folder', 'zed_recordings')
        config_self.camera_offset_x = vis.get('camera_offset_x', -0.06)
        config_self.zed_resolution = vis.get('zed_resolution', 'HD720')
        config_self.zed_fps = vis.get('zed_fps', 15)
        config_self.depth_mode = vis.get('depth_mode', 'PERFORMANCE')
        config_self.coordinate_units = vis.get('coordinate_units', 'METER')
        config_self.cone_base_v = vis.get('cone_base_v', 0.65)
        config_self.point_of_view_offset_y = vis.get('point_of_view_offset_y', 0.8)
        config_self.target_cross_size = vis.get('target_cross_size', 25)
        config_self.target_cross_thickness = vis.get('target_cross_thickness', 3)
        
        # Detection
        det = cfg.get('detection', {})
        config_self.cone_colors = det.get('cone_colors', {})
        config_self.class_names = det.get('class_names', {})
        config_self.blue_cones = det.get('blue_cones', ['blue'])
        config_self.yellow_cones = det.get('yellow_cones', ['yellow'])
        config_self.orange_cones = det.get('orange_cones', ['orange'])
        config_self.circle_marker_radius = det.get('circle_marker_radius', 4)
        config_self.circle_marker_color = det.get('circle_marker_color', [255, 255, 255])
        config_self.text_scale = det.get('text_scale', 0.5)
        config_self.text_thickness = det.get('text_thickness', 2)
        config_self.z_text_scale = det.get('z_text_scale', 0.4)
        config_self.z_text_thickness = det.get('z_text_thickness', 1)
        config_self.z_text_color = det.get('z_text_color', [255, 255, 255])
        
        # Display
        disp = cfg.get('display', {})
        config_self.draw_detections = disp.get('draw_detections', True)
        config_self.draw_trajectory = disp.get('draw_trajectory', True)
        config_self.draw_target = disp.get('draw_target', True)
        config_self.draw_fps = disp.get('draw_fps', True)
        config_self.draw_target_z = disp.get('draw_target_z', True)
        config_self.draw_rec = disp.get('draw_rec', True)
        config_self.fps_text_scale = disp.get('fps_text_scale', 0.7)
        config_self.fps_text_thickness = disp.get('fps_text_thickness', 2)
        config_self.fps_text_color = disp.get('fps_text_color', [0, 255, 0])
        config_self.target_z_text_scale = disp.get('target_z_text_scale', 0.6)
        config_self.target_z_text_thickness = disp.get('target_z_text_thickness', 2)
        config_self.target_z_text_color = disp.get('target_z_text_color', [0, 255, 255])
        config_self.rec_text_scale = disp.get('rec_text_scale', 0.7)
        config_self.rec_text_thickness = disp.get('rec_text_thickness', 2)
        config_self.rec_text_color = disp.get('rec_text_color', [0, 0, 255])
        config_self.trajectory_thickness = disp.get('trajectory_thickness', 2)
        config_self.trajectory_color = disp.get('trajectory_color', [0, 255, 0])
        config_self.waypoint_radius = disp.get('waypoint_radius', 6)
        config_self.waypoint_color_pair = disp.get('waypoint_color_pair', [0, 255, 0])
        config_self.waypoint_color_virtual = disp.get('waypoint_color_virtual', [255, 200, 0])
        config_self.waypoint_color_stop = disp.get('waypoint_color_stop', [0, 0, 255])
        config_self.pair_line_thickness = disp.get('pair_line_thickness', 1)
        config_self.pair_line_color = disp.get('pair_line_color', [0, 255, 255])
        
        # Video
        vid = cfg.get('video', {})
        config_self.temp_codec = vid.get('temp_codec', 'MJPG')
        config_self.output_codec = vid.get('output_codec', 'libx264')
        config_self.output_preset = vid.get('output_preset', 'fast')
        config_self.output_crf = vid.get('output_crf', '23')
        config_self.output_pix_fmt = vid.get('output_pix_fmt', 'yuv420p')
        config_self.output_extension = vid.get('output_extension', 'mp4')
        config_self.temp_extension = vid.get('temp_extension', 'avi')
        config_self.output_prefix = vid.get('output_prefix', 'zed_recording')
        config_self.fps_update_interval = vid.get('fps_update_interval', 0.5)
        
        # Timing
        tim = cfg.get('timing', {})
        config_self.vision_thread_join_timeout = tim.get('vision_thread_join_timeout', 3.0)
        config_self.message_clear_timeout = tim.get('message_clear_timeout', 3.0)
        config_self.socket_timeout = tim.get('socket_timeout', 0.2)
        config_self.arduino_init_delay = tim.get('arduino_init_delay', 2.0)
        config_self.arduino_post_stop_delay = tim.get('arduino_post_stop_delay', 0.5)
        config_self.arduino_close_delay = tim.get('arduino_close_delay', 0.1)
    
    def reload(self, config_path="config.jsonc"):
        """Перезагрузка конфига без перезапуска программы"""
        self.__init__(config_path)
