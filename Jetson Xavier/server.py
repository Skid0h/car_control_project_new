"""
Тестовый server.py с закомментированными блоками для пошаговой диагностики FPS.
Оставлены только: Захват ZED, Веб-стрим (set_frame) и счетчик FPS.
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
        self.smooth_tx = 0.0
        self.smooth_tz = self.config.lookahead_distance
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.last_pid_time = time.time()
        self.fx = 0
        self.cx_cam = 0
        
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
        
        self.detect_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()
        
        self.zed = sl.Camera()
        
        self.rec_queue = queue.Queue(maxsize=30)
        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()
        
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        
    def _detect_loop(self):
        while self.running:
            try:
                item = self.detect_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            frame, orig_shape = item
            detections = self.detector.detect(frame, orig_shape=orig_shape)
            
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(detections)

    def _get_boundary_data(self, cones, z_targets):
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

        grab_error_count = 0
        max_grab_errors = 5

        while self.running:
            try:
                grab_result = self.zed.grab(runtime_params)
                if grab_result != sl.ERROR_CODE.SUCCESS:
                    grab_error_count += 1
                    if grab_error_count >= max_grab_errors:
                        logger.error("Ошибка захвата кадра.")
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

                self.frame_counter += 1
                should_process = (self.frame_counter % self.process_every) == 0

                # ==========================================================
                # === ШАГ 2: ШИНА ПАМЯТИ CUDA (ОТКЛЮЧЕНА) ===
                # ==========================================================
                """
                if CUDA_AVAILABLE and img_data.shape[2] == 4:
                    gpu_src = cv2.cuda_GpuMat()
                    gpu_src.upload(img_data)
                    gpu_bgr = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGRA2BGR)
                    image_np = gpu_bgr.download()
                    if should_process:
                        gpu_resized = cv2.cuda.resize(gpu_bgr, (640, 640))
                        detect_frame = gpu_resized.download()
                else:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR) if img_data.shape[2] == 4 else img_data
                    if should_process:
                        detect_frame = cv2.resize(image_np, (640, 640), interpolation=cv2.INTER_LINEAR)
                """
                # Временно используем стандартное преобразование CPU:
                image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR) if img_data.shape[2] == 4 else img_data
                detect_frame = None

                # ==========================================================
                # === ШАГ 1: ИНФЕРЕНС YOLO (ОТКЛЮЧЕН) ===
                # ==========================================================
                """
                try:
                    fresh_detections = self.result_queue.get_nowait()
                    self.last_detections = fresh_detections
                except queue.Empty:
                    pass
                active_detections = self.last_detections

                if should_process and detect_frame is not None:
                    if not self.detect_queue.full():
                        self.detect_queue.put((detect_frame, image_np.shape[:2]))
                """
                active_detections = []

                # ==========================================================
                # === ШАГ 3: МАТЕМАТИКА ТРАЕКТОРИИ И ГЛУБИН (ОТКЛЮЧЕНА) ===
                # ==========================================================
                """
                current_time = time.time()
                current_cones = []
                for det in active_detections:
                    x1, y1, x2, y2 = det['bbox']
                    width = max(x2 - x1, 1)
                    height = max(y2 - y1, 1)
                    area = width * height
                    z = self.config.area_depth_constant / math.sqrt(area)
                    if self.config.min_depth < z <= self.config.max_depth:
                        u, v = det['center']
                        x_cam = (u - self.cx_cam) * z / self.fx
                        cone_pos = (x_cam, z - self.config.camera_offset_z)
                        current_cones.append({'name': det.get('name', ''), 'pos_3d': cone_pos})

                blues = sorted([c['pos_3d'] for c in current_cones if c['name'] in self.config.blue_cones], key=lambda p: p[1])[:6]
                yellows = sorted([c['pos_3d'] for c in current_cones if c['name'] in self.config.yellow_cones], key=lambda p: p[1])[:6]
                left_bound_x, l_min_z, l_max_z = self._get_boundary_data(blues, z_grid)
                right_bound_x, r_min_z, r_max_z = self._get_boundary_data(yellows, z_grid)
                """
                target_x, target_z = None, None

                # ==========================================================
                # === ШАГ 4: ОТРИСОВКА ОВЕРЛЕЕВ (ОТКЛЮЧЕНА) ===
                # ==========================================================
                """
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

                if self.config.draw_detections and should_process:
                    for det in active_detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                """

                # ==========================================================
                # === ДИАГНОСТИЧЕСКИЙ ВЫВОД: СЧЕТЧИК FPS И ВЕБ-СТРИМ ===
                # ==========================================================
                fps_counter += 1
                if time.time() - fps_last_time >= self.config.fps_update_interval:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_last_time = time.time()
                
                # Текст прямо на видеокадре для контроля в браузере
                cv2.putText(image_np, f"TEST MODE - FPS: {current_fps}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 255, 0), 2)

                # Публикация картинки на Web
                if self.frame_counter % self.publish_every == 0:
                    set_frame(image_np)

        self.zed.close()
        self.robot_state['cam_connected'] = False

    def _reconnect_camera(self, init_params):
        try:
            self.zed.close()
            time.sleep(0.5)
        except Exception:
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
        
        self.detect_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()
        
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
                        loop.pid_integral = 0.0
                        loop.pid_last_error = 0.0
                elif command == "S":
                    if robot_state['auto_mode']:
                        robot_state['auto_mode'] = False
                        car.stop()
                elif command == "R":
                    loop.is_recording = True
                elif command == "C":
                    loop.is_recording = False
                elif command == "F":
                    robot_state['auto_mode'] = False
                    robot_state['msg'] = 'ПЕРЕЗАГРУЗКА...'
                    robot_state['msg_time'] = time.time()
                    car.close()
                    time.sleep(0.5)
                    loop.close()
                    time.sleep(0.5)
                    car.restart()
                    time.sleep(0.5)
                    robot_state['cam_connected'] = False
                    loop.restart()
                    time.sleep(1.0)
                    robot_state['cam_connected'] = loop.robot_state.get('cam_connected', False)
                    robot_state['arduino_connected'] = car.arduino is not None
                    robot_state['msg'] = 'СИСТЕМА ПЕРЕЗАГРУЖЕНА!'
                    robot_state['msg_time'] = time.time()
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
