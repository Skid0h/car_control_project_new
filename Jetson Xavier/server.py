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
import queue

try:
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except:
    CUDA_AVAILABLE = False

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

	# --- Добавлено для нового алгоритма нахождения пути ---
	self.memory_cones = []
        self.smooth_tx = 0.0
        self.smooth_tz = self.config.lookahead_distance
       
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
       
        self.fx = 0
        self.cx_cam = 0

        self.rec_queue = queue.Queue(maxsize=30)
        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()

        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _get_boundary_data(self, cones, z_targets):
        """ДОБАВЛЕНО Интерполяция X-координат с фильтрацией выбросов"""
        valid_cones = [c for c in cones if abs(c[0]) < 2.5]
        
        if not valid_cones:
            return None, 999.0, -1.0
            
        z_vals = [c[1] for c in valid_cones]
        x_vals = [c[0] for c in valid_cones]
        min_z, max_z = min(z_vals), max(z_vals)
        
        if len(valid_cones) == 1:
            bound_x = np.full_like(z_targets, x_vals[0])
        else:
            bound_x = np.interp(z_targets, z_vals, x_vals, left=x_vals[0], right=x_vals[-1])
            
        return bound_x, min_z, max_z

    def _rec_loop(self):
        video_writer = None
        temp_video_path = None
        final_video_path = None
        rec_frame_count = 0
        rec_start_time = None

        while self.running:
            try:
                item = self.rec_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    rec_duration = time.time() - rec_start_time
                    real_fps = rec_frame_count / rec_duration if rec_duration > 0 else self.config.zed_fps
                    threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, real_fps)).start()
                continue

            frame, timestamp = item
            if video_writer is None:
                temp_video_path = os.path.join(self.config.output_folder, f"temp_{timestamp}.{self.config.temp_extension}")
                final_video_path = os.path.join(self.config.output_folder, f"{self.config.output_prefix}_{timestamp}.{self.config.output_extension}")
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.config.temp_codec)
                video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.config.zed_fps, (w, h))
                rec_frame_count = 0
                rec_start_time = time.time()
            video_writer.write(frame)
            rec_frame_count += 1

        if video_writer is not None:
            video_writer.release()

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
        rec_timestamp = None
        was_recording = False

        grab_error_count = 0
        max_grab_errors = 5

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
                if CUDA_AVAILABLE and img_data.shape[2] == 4:
                    gpu_src = cv2.cuda_GpuMat()
                    gpu_src.upload(img_data)
                    image_np = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGRA2BGR).download()
                elif img_data.shape[2] == 4:
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
                        if CUDA_AVAILABLE:
                            gpu_img = cv2.cuda_GpuMat()
                            gpu_img.upload(image_np)
                            detect_frame = cv2.cuda.resize(gpu_img, (new_w, new_h)).download()
                        else:
                            detect_frame = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

                crop_top = int(detect_frame.shape[0] * 0.20)
                crop_bottom = int(detect_frame.shape[0] * 0.20)
                if crop_top > 0 or crop_bottom > 0:
                    detect_frame = detect_frame[crop_top:detect_frame.shape[0] - crop_bottom, :]

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

current_time = time.time()
                    
                    # 1. Очистка старых конусов из памяти
                    mem_timeout = getattr(self.config, 'memory_timeout', 0.8)
                    self.memory_cones = [c for c in self.memory_cones if current_time - c['updated_at'] < mem_timeout]
                    
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        width = max(x2 - x1, 1)
                        height = max(y2 - y1, 1)
                        area = width * height
                        
                        z = self.config.area_depth_constant / math.sqrt(area)
                        
                        if self.config.min_depth < z <= self.config.max_depth:
                            u, v = det['center']
                            x_cam = (u - self.cx_cam) * z / self.fx
                            cone_pos = (x_cam, z - self.config.camera_offset_z)
                            det['pos_3d'] = cone_pos
                            cone_name = det.get('name', '')
                            
                            # 2. Обновление памяти конусов
                            found = False
                            for mc in self.memory_cones:
                                if mc['name'] == cone_name:
                                    mx, mz = mc['pos_3d']
                                    dist = math.sqrt((x_cam - mx)**2 + (z - self.config.camera_offset_z - mz)**2)
                                    if dist < 0.4:
                                        mc['pos_3d'] = cone_pos
                                        mc['updated_at'] = current_time
                                        found = True
                                        break
                            
                            if not found:
                                self.memory_cones.append({'name': cone_name, 'pos_3d': cone_pos, 'updated_at': current_time})
                            
                            if self.config.draw_target_z:
                                cv2.putText(image_np, f"Z:{z:.1f}m", (x1, y1-25), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.config.z_text_scale, 
                                           self.config.z_text_color, 
                                           self.config.z_text_thickness)

                    # 3. Извлекаем координаты из памяти для алгоритма
                    blues = sorted([c['pos_3d'] for c in self.memory_cones if c['name'] in self.config.blue_cones], key=lambda p: p[1])[:6]
                    yellows = sorted([c['pos_3d'] for c in self.memory_cones if c['name'] in self.config.yellow_cones], key=lambda p: p[1])[:6]
                    orange_cones = [c for c in self.memory_cones if c['name'] in self.config.orange_cones]

                    # 4. Интерполяция траектории (алгоритм server2.py)
                    centerline = []
                    half_track = self.config.track_width / 2.0
                    z_grid = np.arange(0.3, self.config.max_depth, 0.2)
                    
                    left_bound_x, l_min_z, l_max_z = self._get_boundary_data(blues, z_grid)
                    right_bound_x, r_min_z, r_max_z = self._get_boundary_data(yellows, z_grid)
                    
                    for i, z in enumerate(z_grid):
                        lx = left_bound_x[i] if left_bound_x is not None else None
                        rx = right_bound_x[i] if right_bound_x is not None else None
                        
                        valid_l = lx is not None and (l_min_z - 0.4 <= z <= l_max_z + 0.4)
                        valid_r = rx is not None and (r_min_z - 0.4 <= z <= r_max_z + 0.4)
                        
                        if valid_l and valid_r:
                            cx = (lx + rx) / 2.0
                        elif valid_l:
                            cx = lx + half_track
                        elif valid_r:
                            cx = rx - half_track
                        else:
                            if lx is not None and rx is not None:
                                cx = (lx + rx) / 2.0
                            elif lx is not None:
                                cx = lx + half_track
                            elif rx is not None:
                                cx = rx - half_track
                            else:
                                cx = 0.0 
                                
                        centerline.append((cx, z))

                    # Подготавливаем массив для совместимости со старой отрисовкой
                    waypoints_3d = [{'x': cx, 'z': cz, 'type': 'centerline'} for cx, cz in centerline]

                    # 5. ВЫБОР ЦЕЛИ (Статичный lookahead)
                    lookahead_dist = self.config.lookahead_distance
                    target_wp = None
                    for cx, cz in centerline:
                        if cz >= lookahead_dist:
                            target_wp = (cx, cz)
                            break
                            
                    if target_wp is None and len(centerline) > 0:
                        target_wp = centerline[-1]

                    # 6. EMA Сглаживание целевой точки
                    if target_wp is not None:
                        tx, tz = target_wp
                        alpha = getattr(self.config, 'ema_alpha', 0.4)
                        self.smooth_tx = self.smooth_tx + alpha * (tx - self.smooth_tx)
                        self.smooth_tz = self.smooth_tz + alpha * (tz - self.smooth_tz)
                    else:
                        decay = getattr(self.config, 'error_decay_rate', 0.85)
                        self.smooth_tx *= decay

                    target_x = self.smooth_tx
                    target_z = self.smooth_tz

                    # 7. Проверка стоп-конуса
                    target_detected = False
                    if orange_cones:
                        stop_threshold = getattr(self.config, 'stop_cone_z_threshold', 0.5)
                        if any(oc['pos_3d'][1] <= stop_threshold for oc in orange_cones):
                            target_detected = True

                    # ОТРИСОВКА ТРАЕКТОРИИ
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

                    # ОТРИСОВКА ЦЕЛИ
                    if target_x is not None and target_z > 0:
                        if self.config.draw_target:
                            target_u = int((target_x * self.fx / target_z) + self.cx_cam)
                            target_v = int(image_np.shape[0] * self.config.cone_base_v)
                            cv2.drawMarker(image_np, (target_u, target_v), (0, 0, 255), 
                                          cv2.MARKER_CROSS, 
                                          self.config.target_cross_size, 
                                          self.config.target_cross_thickness)

                    # ОТРИСОВКА КОНУСОВ
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

                    # УПРАВЛЕНИЕ 
                    if self.robot_state.get('auto_mode', False):
                        if target_detected:
                            self.robot_state['auto_mode'] = False
                            self.robot_state['msg'] = "ФИНИШ! ОРАНЖЕВЫЙ КОНУС."
                            self.robot_state['msg_time'] = time.time()
                            def _brake(car=self.car):
                                car.stop()
                            threading.Thread(target=_brake, daemon=True).start()
                        elif target_x is not None and target_z > 0:
                            error = math.atan2(target_x, target_z)
                            steer_k = self.config.kp_gain  # <--- Теперь рулит по kp_gain
                            steering = max(-1.0, min(1.0, error * steer_k))
                            self.car.update(1.0, steering)

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
                    if not was_recording:
                        rec_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        was_recording = True
                    if not self.rec_queue.full():
                        self.rec_queue.put((image_np.copy(), rec_timestamp))
                else:
                    if was_recording:
                        self.rec_queue.put(None)
                        was_recording = False

                # ОТПРАВКА НА WEB
                if self.frame_counter % self.publish_every == 0:
                    set_frame(image_np)

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
        if getattr(self, 'rec_thread', None) is not None and self.rec_thread.is_alive():
            self.rec_queue.put(None)
            self.rec_thread.join(timeout=3.0)
        self.zed = None

    def restart(self):
        self.close()
        time.sleep(0.5)
        self.running = True
        self.is_recording = False
        self.fx = 0
        self.cx_cam = 0
        self.zed = sl.Camera()
        self.rec_queue = queue.Queue(maxsize=30)
        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()
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
                    try:
                        speed, steering = map(float, command.split(','))
                        if robot_state['auto_mode'] and (speed != 0.0 or steering != 0.0):
                            robot_state['auto_mode'] = False
                            robot_state['msg_time'] = time.time()
                        if not robot_state['auto_mode']:
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
