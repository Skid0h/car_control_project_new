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

from Code.Config_load import Config
from Code.Car_control import CarController
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

config = Config("config.jsonc")

start()  # Запуск Web

class VisionLoop:
    def __init__(self, config, detector, car, robot_state):
        self.config = config
        self.detector = detector
        self.car = car
        self.robot_state = robot_state

        self.zed = sl.Camera()
        self.running = True
        self.is_recording = False
        self.frame_counter = 0
        self.process_every = max(1, int(round(self.config.zed_fps / max(1.0, self.config.target_fps))))
        self.publish_every = max(1, self.process_every)
        self.last_detections = []
        self.last_waypoints_3d = []
        self.last_target_x = None
        self.last_target_z = None
        self.last_target_detected = False
       
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
       
        self.fx = 0
        self.cx_cam = 0
        
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _convert_video(self, input_path, output_path, fps):
        try:
            cmd = ['ffmpeg', '-i', input_path, '-r', str(fps), '-c:v', self.config.output_codec, 
                   '-preset', self.config.output_preset, '-crf', str(self.config.output_crf), 
                   '-pix_fmt', self.config.output_pix_fmt, '-y', output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Видео сконвертировано: {output_path}")
            os.remove(input_path)
            self.robot_state['msg'] = "ВИДЕО СОХРАНЕНО!"
            self.robot_state['msg_time'] = time.time()
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")

    def _vision_loop(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self.config.zed_resolution, sl.RESOLUTION.HD720)
        init_params.camera_fps = self.config.zed_fps
        init_params.coordinate_units = getattr(sl.UNIT, self.config.coordinate_units, sl.UNIT.METER)
       
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
        
        grab_error_count = 0
        max_grab_errors = 3

        logger.info(f"Обычный автопилот: Оценка глубины по площади + Блокировка виртуальных точек.")

        while self.running:
            try:
                grab_result = self.zed.grab(runtime_params)
                
                if grab_result != sl.ERROR_CODE.SUCCESS:
                    grab_error_count += 1
                    
                    if grab_error_count >= max_grab_errors:
                        logger.error("Ошибка захвата кадра. Переподключение камеры...")
                        self.robot_state['cam_connected'] = False
                        self._reconnect_camera(init_params)
                        grab_error_count = 0
                        continue
                    
                    time.sleep(0.1)
                    continue
                
                grab_error_count = 0
            except Exception as e:
                logger.error(f"Исключение при grab(): {e}")
                self.robot_state['cam_connected'] = False
                time.sleep(0.5)
                continue

            if grab_result == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                
                img_data = image_zed.get_data()
                if img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    image_np = img_data

                detect_frame = image_np
                if image_np.shape[1] > 640 or image_np.shape[0] > 480:
                    target_width = 480
                    target_height = 270
                    scale = min(1.0, target_width / image_np.shape[1], target_height / image_np.shape[0])
                    if scale < 1.0:
                        new_w = max(320, int(image_np.shape[1] * scale))
                        new_h = max(180, int(image_np.shape[0] * scale))
                        detect_frame = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

                self.frame_counter += 1
                should_process = (self.frame_counter % self.process_every) == 0

                detections = self.last_detections
                waypoints_3d = self.last_waypoints_3d
                target_x = self.last_target_x
                target_z = self.last_target_z
                target_detected = self.last_target_detected

                if should_process:
                    detections = self.detector.detect(detect_frame)
                    if detect_frame is not image_np:
                        scale_x = image_np.shape[1] / detect_frame.shape[1]
                        scale_y = image_np.shape[0] / detect_frame.shape[0]
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            det['bbox'] = (
                                int(x1 * scale_x),
                                int(y1 * scale_y),
                                int(x2 * scale_x),
                                int(y2 * scale_y),
                            )
                            center = det.get('center')
                            if center is not None:
                                det['center'] = (int(center[0] * scale_x), int(center[1] * scale_y))

                    blue_cones = []
                    yellow_cones = []
                    orange_cones = []
                    
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        width = max(x2 - x1, 1)
                        height = max(y2 - y1, 1)
                        area = width * height
                        
                        z = self.config.area_depth_constant / math.sqrt(area)
                        
                        if self.config.min_depth < z <= self.config.max_depth:
                            u, v = det['center']
                            x_cam = (u - self.cx_cam) * z / self.fx
                            det['pos_3d'] = (x_cam, z)
                            
                            if self.config.draw_target_z:
                                cv2.putText(image_np, f"Z:{z:.1f}m", (x1, y1-25), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.config.z_text_scale, 
                                           self.config.z_text_color, 
                                           self.config.z_text_thickness)
                            
                            cone_name = det.get('name', '')
                            if cone_name in self.config.blue_cones:
                                blue_cones.append(det)
                            elif cone_name in self.config.yellow_cones:
                                yellow_cones.append(det)
                            elif cone_name in self.config.orange_cones:
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
                            
                            if z_diff < self.config.pair_z_tolerance and x_dist < (self.config.track_width * self.config.pair_x_tolerance_multiplier):
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
                            pairs_found_count += 1

                    if pairs_found_count == 0:
                        for b_cone in blue_cones:
                            b_x, b_z = b_cone['pos_3d']
                            waypoints_3d.append({'x': b_x + self.config.virtual_point_offset, 'z': b_z, 'type': 'virtual_blue'})
                            
                        for i, y_cone in enumerate(yellow_cones):
                            if i not in used_yellows:
                                y_x, y_z = y_cone['pos_3d']
                                waypoints_3d.append({'x': y_x - self.config.virtual_point_offset, 'z': y_z, 'type': 'virtual_yellow'})

                    waypoints_3d.sort(key=lambda wp: wp['z'])

                    target_detected = False
                    if orange_cones:
                        closest_orange = min(orange_cones, key=lambda c: c['pos_3d'][1])
                        o_x, o_z = closest_orange['pos_3d']
                        waypoints_3d.append({'x': o_x, 'z': o_z, 'type': 'stop'})
                        if o_z < self.config.stop_cone_z_threshold: 
                            target_detected = True

                    if self.config.draw_trajectory and should_process:
                        pts_2d = [[image_np.shape[1]//2, image_np.shape[0]]]
                        for wp in waypoints_3d:
                            u = int((wp['x'] * self.fx / wp['z']) + self.cx_cam)
                            v = int(image_np.shape[0] * self.config.cone_base_v)
                            pts_2d.append([u, v])
                        if len(pts_2d) > 1:
                            pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(image_np, [pts_arr], isClosed=False, 
                                         color=self.config.trajectory_color, 
                                         thickness=self.config.trajectory_thickness)

                    target_x, target_z = None, None
                    if len(waypoints_3d) > 0:
                        target_x = waypoints_3d[0]['x']
                        target_z = waypoints_3d[0]['z']
                        
                        if self.config.draw_target:
                            target_u = int((target_x * self.fx / target_z) + self.cx_cam)
                            target_v = int(image_np.shape[0] * self.config.cone_base_v)
                            cv2.drawMarker(image_np, (target_u, target_v), (0, 0, 255), 
                                          cv2.MARKER_CROSS, 
                                          self.config.target_cross_size, 
                                          self.config.target_cross_thickness)

                    # ОТРИСОВКА КОНУСОВ (draw_detections) 
                    if self.config.draw_detections and should_process:
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            cone_name = det.get('name', '')
                            
                            if cone_name in self.config.blue_cones:
                                color = (255, 0, 0)
                            elif cone_name in self.config.yellow_cones:
                                color = (0, 255, 255)
                            elif cone_name in self.config.orange_cones:
                                color = (0, 165, 255)
                            else:
                                color = (255, 255, 255)
                            
                            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image_np, cone_name, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # ЛИНИИ МЕЖДУ ПАРАМИ 
                    if self.config.pair_line_color and pairs_found_count > 0:
                        for wp in waypoints_3d:
                            if wp.get('type') == 'pair':
                                cv2.line(image_np, wp['b_cone']['center'], wp['y_cone']['center'], 
                                        self.config.pair_line_color, 
                                        self.config.pair_line_thickness)

                    # УПРАВЛЕНИЕ 
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

                    self.last_detections = detections
                    self.last_waypoints_3d = waypoints_3d
                    self.last_target_x = target_x
                    self.last_target_z = target_z
                    self.last_target_detected = target_detected

                else:
                    detections = self.last_detections
                    waypoints_3d = self.last_waypoints_3d
                    target_x = self.last_target_x
                    target_z = self.last_target_z
                    target_detected = self.last_target_detected

                # FPS 
                fps_counter += 1
                if time.time() - fps_last_time >= self.config.fps_update_interval:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_last_time = time.time()
                
                if self.config.draw_fps:
                    cv2.putText(image_np, f"FPS: {current_fps} Mode: {'AUTO' if self.robot_state.get('auto_mode') else 'MANUAL'}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               self.config.fps_text_scale, self.config.fps_text_color, self.config.fps_text_thickness)
                if target_x is not None and self.config.draw_target_z:
                    cv2.putText(image_np, f"Target Z: {target_z:.2f}m", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                               self.config.target_z_text_scale, self.config.target_z_text_color, self.config.target_z_text_thickness)
                
                # ЗАПИСЬ 
                if self.is_recording:
                    if self.config.draw_rec:
                        cv2.putText(image_np, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                                   self.config.rec_text_scale, self.config.rec_text_color, self.config.rec_text_thickness)
                    if video_writer is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_video_path = os.path.join(self.config.output_folder, f"temp_{timestamp}.{self.config.temp_extension}")
                        final_video_path = os.path.join(self.config.output_folder, f"{self.config.output_prefix}_{timestamp}.{self.config.output_extension}")
                        height, width = image_np.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*self.config.temp_codec)
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.config.zed_fps, (width, height))
                    video_writer.write(image_np)
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, self.config.zed_fps)).start()

                # ОТПРАВКА НА WEB
                if self.frame_counter % self.publish_every == 0:
                    set_frame(image_np)

        if video_writer is not None:
            video_writer.release()
        self.zed.close()
        self.robot_state['cam_connected'] = False
    
    def _reconnect_camera(self, init_params):
        """Попытка переподключения к камере"""
        try:
            self.zed.close()
            time.sleep(0.5)
        except Exception as e:
            pass
        
        try:
            self.zed = sl.Camera()
            if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
                self.robot_state['cam_connected'] = True
                cam_info = self.zed.get_camera_information()
                self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
                self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx
            else:
                self.robot_state['cam_connected'] = False
        except Exception as e:
            logger.error(f"Ошибка переподключения камеры: {e}")
            self.robot_state['cam_connected'] = False

    def close(self):
        self.running = False
        self.robot_state['cam_connected'] = False
        try:
            if getattr(self, 'zed', None) is not None:
                self.zed.close()
        except Exception:
            pass
        if getattr(self, 'vision_thread', None) is not None and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=self.config.vision_thread_join_timeout)
        self.zed = None

    def restart(self):
        """Перезагрузка камеры"""
        self.close()
        time.sleep(0.5)
        self.running = True
        self.is_recording = False
        self.fx = 0
        self.cx_cam = 0
        self.zed = sl.Camera()
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)
    try:
        sock.bind((config.udp_ip, config.udp_port))
    except:
        sys.exit(1)
    
    detector = ConeDetector(config)
    car = CarController(config)
    robot_state = {'auto_mode': False, 'cam_connected': False, 'arduino_connected': False, 'msg': '', 'msg_time': 0}
    robot_state['arduino_connected'] = car.arduino is not None
    loop = VisionLoop(config, detector, car, robot_state)
    running = True
    last_addr = None
    
    try:
        while running:
            try:
                data, addr = sock.recvfrom(1024)
                last_addr = addr
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
                    loop.is_recording = True
                elif command == "C":
                    loop.is_recording = False
                elif command == "F":
                    # ПЕРЕЗАГРУЗКА: отключение и повторное включение системы
                    robot_state['auto_mode'] = False
                    robot_state['msg'] = 'ПЕРЕЗАГРУЗКА...'
                    robot_state['msg_time'] = time.time()
                    logger.info("Инициирована перезагрузка системы...")
                    
                    # Отключение Arduino
                    car.close()
                    time.sleep(0.5)
                    
                    # Отключение камеры
                    loop.close()
                    time.sleep(0.5)
                    
                    # Повторное включение Arduino
                    car.restart()
                    time.sleep(0.5)
                    
                    # Повторное включение камеры
                    robot_state['cam_connected'] = False
                    loop.restart()
                    time.sleep(1.0)
                    
                    # Обновляем статусы подключения
                    robot_state['cam_connected'] = loop.robot_state.get('cam_connected', False)
                    robot_state['arduino_connected'] = car.arduino is not None
                    robot_state['msg'] = 'СИСТЕМА ПЕРЕЗАГРУЖЕНА!'
                    robot_state['msg_time'] = time.time()
                    logger.info("Система успешно перезагружена!")
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

                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
                    robot_state['msg'] = ''
                
                # Обновляем статус подключения Arduino
                robot_state['arduino_connected'] = car.arduino is not None
                
                telemetry = {
                    "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                    "rec": loop.is_recording,
                    "cam_connected": robot_state['cam_connected'],
                    "arduino_connected": robot_state['arduino_connected'],
                    "fwd": car.config.forward_speed,
                    "bck": car.config.back_speed,
                    "msg": robot_state['msg']
                }
                if last_addr:
                    sock.sendto(json.dumps(telemetry).encode('utf-8'), last_addr)
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
