"""
Запускается на Jetson.
АЛГОРИТМ: Обычный автопилот (Оценка глубины по площади + Пары + Блокировка виртуальных точек).
ФАЙЛ КОНФИГУРАЦИИ. ПОМЕХОЗАЩИЩЕННЫЙ UART.
"""

import socket
import time
import logging
import sys
import json
import math
import subprocess
from datetime import datetime
import cv2
import pyzed.sl as sl
import threading
import numpy as np
import os

from Code.Config_load import load_config
from Code.Car_control import CarController
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

config = load_config()

start()

class VisionLoop:
    def __init__(self, config, detector, car, robot_state):
        self.vision = config['vision']
        self.video = config['video']
        self.display = config['display']
        self.ap = config['autopilot']
        self.detection = config['detection']
        self.timing = config['timing']
        self.detector = detector
        self.car = car
        self.robot_state = robot_state

        self.zed = sl.Camera()
        self.running = True
        self.is_recording = False
        self.output_folder = self.vision['output_folder']
       
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
       
        self.fx = 0
        self.cx_cam = 0
        
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _convert_video(self, input_path, output_path, fps):
        try:
            cmd = ['ffmpeg', '-i', input_path, '-r', str(fps), '-c:v', self.video['output_codec'], 
                   '-preset', self.video['output_preset'], '-crf', str(self.video['output_crf']), 
                   '-pix_fmt', self.video['output_pix_fmt'], '-y', output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Видео сконвертировано: {output_path}")
            os.remove(input_path)
            self.robot_state['msg'] = "ВИДЕО СОХРАНЕНО!"
            self.robot_state['msg_time'] = time.time()
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")

    def _vision_loop(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self.vision['zed_resolution'], sl.RESOLUTION.HD720)
        init_params.camera_fps = self.vision['zed_fps']
        init_params.coordinate_units = getattr(sl.UNIT, self.vision['coordinate_units'], sl.UNIT.METER)
       
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            logger.error("Не удалось открыть ZED-камеру.")
            self.robot_state['cam_connected'] = False
            self.running = False
            return

        self.robot_state['cam_connected'] = True
        cam_info = self.zed.get_camera_information()
        self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx

        runtime_params = sl.RuntimeParameters()
        image_zed = sl.Mat()
        
        fps_counter = 0
        current_fps = 0
        fps_last_time = time.time()
        
        video_writer = None
        temp_video_path = None
        final_video_path = None

        logger.info(f"Обычный автопилот: Оценка глубины по площади + Блокировка виртуальных точек.")

        while self.running:
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                
                img_data = image_zed.get_data()
                if img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    image_np = img_data
                    
                set_frame(image_np)
                
                detections = self.detector.detect(image_np)
                
                blue_cones = []
                yellow_cones = []
                orange_cones = []
                
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    width = max(x2 - x1, 1)
                    height = max(y2 - y1, 1)
                    area = width * height
                    
                    z = self.ap['area_depth_constant'] / math.sqrt(area)
                    
                    if self.ap['min_depth'] < z <= self.ap['max_depth']:
                        u, v = det['center']
                        x_cam = (u - self.cx_cam) * z / self.fx
                        det['pos_3d'] = (x_cam, z)
                        
                        if self.display['draw_target_z']:
                            cv2.putText(image_np, f"Z:{z:.1f}m", (x1, y1-25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.detection['z_text_scale'], 
                                       self.detection['z_text_color'], 
                                       self.detection['z_text_thickness'])
                        
                        cone_name = det.get('name', '')
                        if cone_name in self.detection['blue_cones']:
                            blue_cones.append(det)
                        elif cone_name in self.detection['yellow_cones']:
                            yellow_cones.append(det)
                        elif cone_name in self.detection['orange_cones']:
                            orange_cones.append(det)

                waypoints_3d = [] 
                
                blue_cones.sort(key=lambda c: c['pos_3d'][1])
                yellow_cones.sort(key=lambda c: c['pos_3d'][1])
                
                used_yellows = set()
                pairs_found_count = 0
                
                for b_cone in blue_cones:
                    best_y = None
                    best_diff = float('inf')
                    b_x, b_z = b_cone['pos_3d']
                    
                    for i, y_cone in enumerate(yellow_cones):
                        if i in used_yellows: continue
                        y_x, y_z = y_cone['pos_3d']
                        
                        z_diff = abs(b_z - y_z)
                        x_dist = abs(b_x - y_x) 
                        
                        if z_diff < self.ap['pair_z_tolerance'] and x_dist < (self.ap['track_width'] * self.ap['pair_x_tolerance_multiplier']):
                            if z_diff < best_diff:
                                best_diff = z_diff
                                best_y = (i, y_cone)
                    
                    if best_y:
                        y_idx, y_cone = best_y
                        used_yellows.add(y_idx)
                        y_x, y_z = y_cone['pos_3d']
                        
                        mid_x = (b_x + y_x) / 2.0
                        mid_z = (b_z + y_z) / 2.0
                        waypoints_3d.append({'x': mid_x, 'z': mid_z, 'type': 'pair', 'b_cone': b_cone, 'y_cone': y_cone})
                        if self.display['draw_detections']:
                            cv2.line(image_np, b_cone['center'], y_cone['center'], 
                                    self.display['pair_line_color'], 
                                    self.display['pair_line_thickness'])
                        pairs_found_count += 1

                if pairs_found_count == 0:
                    for b_cone in blue_cones:
                        b_x, b_z = b_cone['pos_3d']
                        waypoints_3d.append({'x': b_x + self.ap['virtual_point_offset'], 'z': b_z, 'type': 'virtual_blue'})
                        
                    for i, y_cone in enumerate(yellow_cones):
                        if i not in used_yellows:
                            y_x, y_z = y_cone['pos_3d']
                            waypoints_3d.append({'x': y_x - self.ap['virtual_point_offset'], 'z': y_z, 'type': 'virtual_yellow'})

                waypoints_3d.sort(key=lambda wp: wp['z'])

                target_detected = False
                if orange_cones:
                    closest_orange = min(orange_cones, key=lambda c: c['pos_3d'][1])
                    o_x, o_z = closest_orange['pos_3d']
                    waypoints_3d.append({'x': o_x, 'z': o_z, 'type': 'stop'})
                    if o_z < self.ap['stop_cone_z_threshold']: 
                        target_detected = True

                if self.display['draw_trajectory']:
                    pts_2d = [[image_np.shape[1]//2, image_np.shape[0]]]
                    for wp in waypoints_3d:
                        u = int((wp['x'] * self.fx / wp['z']) + self.cx_cam)
                        v = int(image_np.shape[0] * self.vision['cone_base_v'])
                        pts_2d.append([u, v])
                    if len(pts_2d) > 1:
                        pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image_np, [pts_arr], isClosed=False, 
                                     color=self.display['trajectory_color'], 
                                     thickness=self.display['trajectory_thickness'])

                target_x, target_z = None, None
                if len(waypoints_3d) > 0:
                    target_x = waypoints_3d[0]['x']
                    target_z = waypoints_3d[0]['z']
                    
                    if self.display['draw_target']:
                        target_u = int((target_x * self.fx / target_z) + self.cx_cam)
                        target_v = int(image_np.shape[0] * self.vision['cone_base_v'])
                        cv2.drawMarker(image_np, (target_u, target_v), (0, 0, 255), 
                                      cv2.MARKER_CROSS, 
                                      self.vision['target_cross_size'], 
                                      self.vision['target_cross_thickness'])

                if self.robot_state.get('auto_mode', False):
                    if target_detected:
                        self.robot_state['auto_mode'] = False
                        self.robot_state['msg'] = "ФИНИШ! ОРАНЖЕВЫЙ КОНУС."
                        self.robot_state['msg_time'] = time.time()
                        self.car.stop()
                    elif target_x is not None:
                        error = math.atan2(target_x, target_z)
                        steering = max(-1.0, min(1.0, error * 2.0))
                        self.car.update(1.0, steering)

                fps_counter += 1
                if time.time() - fps_last_time >= self.video['fps_update_interval']:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_last_time = time.time()
                
                if self.display['draw_fps']:
                    cv2.putText(image_np, f"FPS: {current_fps} Mode: {'AUTO' if self.robot_state.get('auto_mode') else 'MANUAL'}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               self.display['fps_text_scale'], self.display['fps_text_color'], self.display['fps_text_thickness'])
                if target_x is not None and self.display['draw_target_z']:
                    cv2.putText(image_np, f"Target Z: {target_z:.2f}m", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                               self.display['target_z_text_scale'], self.display['target_z_text_color'], self.display['target_z_text_thickness'])
                
                if self.is_recording:
                    if self.display['draw_rec']:
                        cv2.putText(image_np, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                                   self.display['rec_text_scale'], self.display['rec_text_color'], self.display['rec_text_thickness'])
                    if video_writer is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_video_path = os.path.join(self.output_folder, f"temp_{timestamp}.{self.video['temp_extension']}")
                        final_video_path = os.path.join(self.output_folder, f"{self.video['output_prefix']}_{timestamp}.{self.video['output_extension']}")
                        height, width = image_np.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*self.video['temp_codec'])
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.vision['zed_fps'], (width, height))
                    video_writer.write(image_np)
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, self.vision['zed_fps'])).start()

        if video_writer is not None:
            video_writer.release()
        self.zed.close()
        self.robot_state['cam_connected'] = False

    def start_recording(self):
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

    def close(self):
        self.running = False
        self.vision_thread.join(timeout=self.timing['vision_thread_join_timeout'])


def main():
    udp_ip = config['network']['udp_ip']
    udp_port = config['network']['udp_port']
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config['timing']['socket_timeout'])
    try:
        sock.bind((udp_ip, udp_port))
    except:
        sys.exit(1)
    
    detector = ConeDetector(config)
    car = CarController(config)
    robot_state = {'auto_mode': False, 'cam_connected': False, 'msg': '', 'msg_time': 0}
    loop = VisionLoop(config, detector, car, robot_state)
    running = True
    
    try:
        while running:
            try:
                data, addr = sock.recvfrom(1024)
                command = data.decode('utf-8').strip()
                if command == "Q":
                    running = False
                    break
                elif command == "A":
                    if not robot_state['auto_mode']:
                        robot_state['auto_mode'] = True
                        robot_state['msg'] = ''
                elif command == "S":
                    if robot_state['auto_mode']:
                        robot_state['auto_mode'] = False
                        car.stop()
                elif command == "R":
                    loop.start_recording()
                elif command == "C":
                    loop.stop_recording()
                elif command.startswith("speed:"):
                    try:
                        fwd, bck = map(int, command[6:].split(','))
                        car.set_speeds(fwd, bck)
                    except:
                        pass
                else:
                    if not robot_state['auto_mode']:
                        try:
                            speed, steering = map(float, command.split(','))
                            car.update(speed, steering)
                        except:
                            pass

                msg_clear_timeout = config['timing']['message_clear_timeout']
                if time.time() - robot_state['msg_time'] > msg_clear_timeout:
                    robot_state['msg'] = ''
                telemetry = {
                    "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                    "rec": loop.is_recording,
                    "cam_connected": robot_state['cam_connected'],
                    "fwd": car.forward_speed,
                    "bck": car.back_speed,
                    "msg": robot_state['msg']
                }
                sock.sendto(json.dumps(telemetry).encode('utf-8'), addr)
            except socket.timeout:
                if not robot_state['auto_mode']:
                    car.check_stop()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        car.close()
        sock.close()


if __name__ == "__main__":
    main()
