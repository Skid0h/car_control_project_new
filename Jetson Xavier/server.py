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
 
        logger.info("Автопилот: 2 пары конусов → плоскость → точка-цель.")
 
        while self.running:
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
 
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
 
                img_data = image_zed.get_data()
                if img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    image_np = img_data
 
                detections = self.detector.detect(image_np)
 
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
 
                # === МАКСИМАЛЬНО ПРОСТОЙ АВТОПИЛОТ: ЗМЕЙКА ===
                # Едем в середину между ближайшим синим и ближайшим жёлтым конусом
                
                # Сортируем по дальности Z (от ближних к дальним)
                blue_sorted = sorted([c for c in blue_cones if 'pos_3d' in c], key=lambda c: c['pos_3d'][1])
                yellow_sorted = sorted([c for c in yellow_cones if 'pos_3d' in c], key=lambda c: c['pos_3d'][1])
                orange_sorted = sorted([c for c in orange_cones if 'pos_3d' in c], key=lambda c: c['pos_3d'][1])
                
                target_x, target_z = None, None
                target_detected = False
                
                # Стоп-конус (оранжевый)
                if orange_sorted:
                    o_x, o_z = orange_sorted[0]['pos_3d']
                    if o_z < self.config.stop_cone_z_threshold:
                        target_detected = True
                
                # Цель = середина между ближайшим синим и ближайшим жёлтым
                if blue_sorted and yellow_sorted:
                    bx, bz = blue_sorted[0]['pos_3d']
                    yx, yz = yellow_sorted[0]['pos_3d']
                    target_x = (bx + yx) / 2.0
                    target_z = (bz + yz) / 2.0
                
                # Отрисовка цели
                if target_x is not None and self.config.draw_target:
                    target_u = int((target_x * self.fx / max(target_z, 0.01)) + self.cx_cam)
                    target_v = int(image_np.shape[0] * self.config.cone_base_v)
                    cv2.drawMarker(image_np, (target_u, target_v), (0, 0, 255),
                                  cv2.MARKER_CROSS,
                                  self.config.target_cross_size,
                                  self.config.target_cross_thickness)
                    cv2.putText(image_np, f"Target Z: {target_z:.2f}m", (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                               self.config.target_z_text_scale, self.config.target_z_text_color, self.config.target_z_text_thickness)
                    # Линия между конусами пары
                    b_u = int((bx * self.fx / max(bz, 0.01)) + self.cx_cam)
                    b_v = int(image_np.shape[0] * self.config.cone_base_v)
                    y_u = int((yx * self.fx / max(yz, 0.01)) + self.cx_cam)
                    y_v = int(image_np.shape[0] * self.config.cone_base_v)
                    cv2.line(image_np, (b_u, b_v), (y_u, y_v), self.config.trajectory_color, self.config.trajectory_thickness)
 
                # ОТРИСОВКА КОНУСОВ (draw_detections)
                if self.config.draw_detections:
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
                        self.car.stop()
                    elif target_x is not None:
                        # Угол до цели
                        error = math.atan2(target_x, max(target_z, 0.01))
                        steering = max(-1.0, min(1.0, error * 1.5))
                        self.car.update(1.0, steering)  # скорость не трогаем
                    else:
                        # Нет конусов — едем прямо
                        self.car.update(1.0, 0.0)
 
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
                set_frame(image_np)
 
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
        self.vision_thread.join(timeout=self.config.vision_thread_join_timeout)
 
 
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)
    try:
        sock.bind((config.udp_ip, config.udp_port))
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
 
                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
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
