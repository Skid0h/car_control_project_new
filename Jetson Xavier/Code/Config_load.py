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
        
        # Autopilot
        ap = cfg['autopilot']
        self.max_depth = ap['max_depth']
        self.min_depth = ap['min_depth']
        self.track_width = ap['track_width']
        self.pair_z_tolerance = ap['pair_z_tolerance']
        self.pair_x_tolerance_multiplier = ap['pair_x_tolerance_multiplier']
        self.area_depth_constant = ap['area_depth_constant']
        
        # Vision
        vis = cfg['vision']
        self.yolo_model_path = vis['yolo_model_path']
        self.confidence_threshold = vis['confidence_threshold']
        self.iou_threshold = vis['iou_threshold']
        self.output_folder = vis['output_folder']
        self.camera_offset_x = vis['camera_offset_x']
        self.camera_offset_z = vis['camera_offset_z']
        self.zed_resolution = vis['zed_resolution']
        self.zed_fps = vis['zed_fps']
        self.coordinate_units = vis['coordinate_units']
        self.cone_base_v = vis['cone_base_v']
        self.target_cross_size = vis['target_cross_size']
        self.target_cross_thickness = vis['target_cross_thickness']
        
        # Detection
        det = cfg['detection']
        self.blue_cones = det['blue_cones']
        self.yellow_cones = det['yellow_cones']
        self.orange_cones = det['orange_cones']
        
        # Display
        disp = cfg['display']
        self.draw_detections = disp['draw_detections']
        self.draw_trajectory = disp['draw_trajectory']
        self.draw_target = disp['draw_target']
        self.draw_fps = disp['draw_fps']
        self.draw_rec = disp['draw_rec']
        self.draw_cone_quad = disp['draw_cone_quad']
        self.fps_text_scale = disp['fps_text_scale']
        self.fps_text_thickness = disp['fps_text_thickness']
        self.fps_text_color = disp['fps_text_color']
        self.rec_text_scale = disp['rec_text_scale']
        self.rec_text_thickness = disp['rec_text_thickness']
        self.rec_text_color = disp['rec_text_color']
        
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
    
    def reload(self, config_path="config.jsonc"):
        """Перезагрузка конфига без перезапуска программы"""
        self.__init__(config_path)
