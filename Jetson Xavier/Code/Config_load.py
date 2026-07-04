"""
Загрузка конфигурации из JSONC-файла с поддержкой комментариев.
Все параметры вынесены в атрибуты класса для быстрого доступа.
"""

import os
import sys
import json
import re


class Config:
    def __init__(self, config_path="config.jsonc"):
        if not os.path.exists(config_path):
            print(f"Файл конфигурации '{config_path}' не найден.")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = re.sub(r'//.*', '', content)
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
        self.draw_cone_quad = disp['draw_cone_quad']
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
    
    def reload(self, config_path="config.jsonc"):
        """Перезагрузка конфига без перезапуска программы"""
        self.__init__(config_path)