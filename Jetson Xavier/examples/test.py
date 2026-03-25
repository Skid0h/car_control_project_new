"""Запускается на Jetson.
  Слушает все интерфейсы (0.0.0.0) на порту UDP_PORT.
  Включена 3D-навигация (Pure Pursuit) и плавное рулевое управление.
"""

import socket
import serial
import serial.tools.list_ports
import time
import logging
import threading
import sys
import os
import json
import math
from datetime import datetime
import cv2
import pyzed.sl as sl
import subprocess
import numpy as np
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

UDP_IP = "0.0.0.0" 
UDP_PORT = 5005
Baud = 9600

base_speed = 90
forward_speed = 98
back_speed = 82

base_rotation = 90
right_rotation = 0
left_rotation = 180

# Параметры детекции
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.4
ORANGE_CONE_CLASS_ID = 1

CONE_COLORS = {
   0: (0, 255, 255),   # Желтый
   1: (0, 165, 255),   # Оранжевый
   2: (255, 0, 0),     # Синий
   3: (0, 0, 255)      # Красный
}

CLASS_NAMES = {
   0: "Yellow", 1: "Orange", 2: "Blue", 3: "Red"
}

def find_arduino_port():
   ports = serial.tools.list_ports.comports()
   for port in ports:
       if ('Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description):
           return port.device
   for port in ports:
       if port.vid and port.pid:
           if (port.vid == 0x2341) or (port.vid == 0x1A86):
               return port.device
   return None

class ConeDetector:
   def __init__(self, model_path):
       self.model = None
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       logger.info(f"Загрузка модели YOLO с {model_path} на {self.device}")
       try:
           self.model = YOLO(model_path)
           if hasattr(self.model, 'names'):
               logger.info(f"Классы модели: {self.model.names}")
       except Exception as e:
           logger.error(f"Ошибка загрузки модели: {e}")
           self.model = None
   
   def detect(self, frame):
       if self.model is None:
           return frame, []
       try:
           results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, device=self.device)
           detections = []
           for result in results:
               if result.boxes is not None:
                   for box in result.boxes:
                       x1, y1, x2, y2 = map(int, box.xyxy[0])
                       conf = float(box.conf[0])
                       cls = int(box.cls[0])
                       
                       detections.append({
                           'bbox': (x1, y1, x2, y2),
                           'conf': conf,
                           'class': cls,
                           'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                       })
                       
                       color = CONE_COLORS.get(cls, (0, 255, 0))
                       cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                       class_name = CLASS_NAMES.get(cls, f"Class_{cls}")
                       cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 4, (255, 255, 255), -1)
           return frame, detections
       except Exception as e:
           logger.error(f"Ошибка детекции: {e}")
           return frame, []

class VisionSystem:
   def __init__(self, detector, car, robot_state):
       self.detector = detector
       self.car = car
       self.robot_state = robot_state
       self.zed = sl.Camera()
       self.is_recording = False
       self.video_writer = None
       self.running = True
       self.output_folder = "zed_recordings"
       
       self.target_fps = 15.0 
       self.frame_time = 1.0 / self.target_fps
       
       if not os.path.exists(self.output_folder):
           os.makedirs(self.output_folder)
           
       self.thread = threading.Thread(target=self._vision_loop)
       self.thread.start()
       
   def _convert_video(self, input_path, output_path):
       try:
           cmd = ['ffmpeg', '-i', input_path, '-r', str(self.target_fps), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-y', output_path]
           subprocess.run(cmd, check=True, capture_output=True)
           logger.info(f"Видео сконвертировано: {output_path}")
           os.remove(input_path)
       except Exception as e:
           logger.error(f"Ошибка конвертации: {e}")
           
   def _vision_loop(self):
       init_params = sl.InitParameters()
       init_params.camera_resolution = sl.RESOLUTION.HD720
       init_params.camera_fps = int(self.target_fps) 
       init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
       init_params.coordinate_units = sl.UNIT.METER 
       
       if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
           logger.error("Не удалось открыть ZED-камеру.")
           self.running = False
           return

       cam_info = self.zed.get_camera_information()
       fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
       cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx

       runtime_params = sl.RuntimeParameters()
       image_zed = sl.Mat()
       depth_zed = sl.Mat()
       
       frame_count = 0
       fps_counter = 0
       current_fps = 0
       fps_last_time = time.time()
       
       temp_video_path = None
       final_video_path = None

       logger.info("Система зрения запущена.")

       while self.running:
           loop_start_time = time.time()
           
           if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
               self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
               image_np = image_zed.get_data()
               if image_np.shape[2] == 4:
                   image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
                   
               self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
               depth_data = depth_zed.get_data()
               
               image_np, detections = self.detector.detect(image_np)
               
               blue_cones_3d = []
               yellow_cones_3d = []
               target_detected = False
               
               for det in detections:
                   u, v = det['center']
                   if 0 <= v < depth_data.shape[0] and 0 <= u < depth_data.shape[1]:
                       z = depth_data[v, u] 
                       if np.isfinite(z) and z > 0:
                           x = (u - cx_cam) * z / fx
                           det['pos_3d'] = (x, z)
                           cv2.putText(image_np, f"X:{x:.2f} Z:{z:.2f}", (det['bbox'][0], det['bbox'][1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                           
                           if det['class'] == 2:   # Blue (слева)
                               blue_cones_3d.append(det)
                           elif det['class'] == 0: # Yellow (справа)
                               yellow_cones_3d.append(det)
                           elif det['class'] == 1 and z <= 0.4: # Orange stop
                               target_detected = True

               # --- РАСЧЕТ УПРАВЛЕНИЯ (PURE PURSUIT) ---
               gate_x = None
               gate_z = None
               auto_steering = 0.0 # По умолчанию едем прямо
               
               # Сценарий 1: Видим оба конуса (Идеально)
               if blue_cones_3d and yellow_cones_3d:
                   closest_blue = min(blue_cones_3d, key=lambda c: c['pos_3d'][1])
                   closest_yellow = min(yellow_cones_3d, key=lambda c: c['pos_3d'][1])
                   
                   gate_x = (closest_blue['pos_3d'][0] + closest_yellow['pos_3d'][0]) / 2.0
                   gate_z = (closest_blue['pos_3d'][1] + closest_yellow['pos_3d'][1]) / 2.0
                   
                   gate_u = int((gate_x * fx / gate_z) + cx_cam)
                   gate_v = int((closest_blue['center'][1] + closest_yellow['center'][1]) / 2)
                   cv2.circle(image_np, (gate_u, gate_v), 8, (0, 255, 0), -1)
               
               # Сценарий 2: Видим только синий конус
               elif blue_cones_3d:
                   closest_blue = min(blue_cones_3d, key=lambda c: c['pos_3d'][1])
                   gate_x = closest_blue['pos_3d'][0] + 1.0 # Центр трассы = 1 метр правее синего
                   gate_z = closest_blue['pos_3d'][1]
                   
               # Сценарий 3: Видим только желтый конус
               elif yellow_cones_3d:
                   closest_yellow = min(yellow_cones_3d, key=lambda c: c['pos_3d'][1])
                   gate_x = closest_yellow['pos_3d'][0] - 1.0 # Центр трассы = 1 метр левее желтого
                   gate_z = closest_yellow['pos_3d'][1]

               # Если мы вычислили целевую точку, переводим ее в угол руля
               if gate_x is not None and gate_z is not None:
                   # Угол к цели в радианах (от -pi до pi)
                   alpha = math.atan2(gate_x, gate_z)
                   
                   # П-регулятор: Коэффициент чувствительности руля
                   Kp = 1.5 
                   
                   auto_steering = alpha * Kp
                   # Жестко ограничиваем команды пределами [-1.0, 1.0]
                   auto_steering = max(-1.0, min(1.0, auto_steering))
                   
                   if frame_count % 15 == 0:
                       logger.info(f"📍 Цель: X={gate_x:.2f}m, Угол={math.degrees(alpha):.1f}°, Руль={auto_steering:.2f}")


               # --- ПРИМЕНЕНИЕ КОМАНД В АВТОМАТИЧЕСКОМ РЕЖИМЕ ---
               if self.robot_state.get('auto_mode', False):
                   if target_detected:
                       logger.info("Оранжевый конус! АВТОПИЛОТ ОСТАНОВЛЕН.")
                       self.robot_state['auto_mode'] = False
                       self.robot_state['msg'] = "ОРАНЖЕВЫЙ КОНУС! АВТОПИЛОТ ОСТАНОВЛЕН."
                       self.robot_state['msg_time'] = time.time()
                       self.car.stop()
                   else:
                       # Едем вперед (1) с вычисленным плавным углом поворота руля
                       self.car.update(1, auto_steering)
               # ------------------------------------------------

               fps_counter += 1
               if time.time() - fps_last_time >= 1.0:
                   current_fps = fps_counter
                   fps_counter = 0
                   fps_last_time = time.time()
               
               cv2.putText(image_np, f"FPS: {current_fps} Mode: {'AUTO' if self.robot_state.get('auto_mode') else 'MANUAL'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
               
               if self.is_recording:
                   cv2.putText(image_np, f"REC: {frame_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   if self.video_writer is None:
                       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                       temp_video_path = os.path.join(self.output_folder, f"temp_{timestamp}.avi")
                       final_video_path = os.path.join(self.output_folder, f"zed_recording_{timestamp}.mp4")
                       height, width = image_np.shape[:2]
                       fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                       self.video_writer = cv2.VideoWriter(temp_video_path, fourcc, float(self.target_fps), (width, height))
                       frame_count = 0
                       
                   self.video_writer.write(image_np)
                   frame_count += 1
               else:
                   if self.video_writer is not None:
                       self.video_writer.release()
                       self.video_writer = None
                       threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path)).start()
           
           processing_time = time.time() - loop_start_time
           if processing_time < self.frame_time:
               time.sleep(self.frame_time - processing_time)

       if self.video_writer:
           self.video_writer.release()
       self.zed.close()

   def start_recording(self):
       self.is_recording = True
   def stop_recording(self):
       self.is_recording = False
   def close(self):
       self.running = False
       self.thread.join(timeout=5.0)

class CarController:
   def __init__(self):
       self.lock = threading.Lock()
       self.forward_speed = forward_speed
       self.back_speed = back_speed
       self.last_command_time = 0
       self.last_sent_cmd = ""
       self.last_sent_time = 0
       
       port = find_arduino_port()
       self.arduino = None
       if port is None:
           logger.error("Arduino не найден")
           return
           
       try:
           self.arduino = serial.Serial(port, Baud, timeout=1)
           time.sleep(2)
           self.stop()
           time.sleep(0.5)
           logger.info(f"Arduino подключен: {port}")
       except Exception as e:
           logger.error(f"Ошибка подключения: {e}")
           self.arduino = None
   
   def set_speeds(self, forward, back):
       self.forward_speed = forward
       self.back_speed = back
   
   def update(self, speed, steering):
       """
       speed: -1 (назад), 0 (стоп), 1 (вперед)
       steering: float от -1.0 (максимально влево) до 1.0 (максимально вправо)
       """
       if not self.arduino: return
       
       motor_value = base_speed
       if speed > 0: motor_value = self.forward_speed
       elif speed < 0: motor_value = self.back_speed
       
       # Ограничиваем входящее значение от -1.0 до 1.0 для безопасности
       steering_clamped = max(-1.0, min(1.0, float(steering)))
       
       # Переводим дробное значение в градусы для сервопривода
       steer_value = int(base_rotation - (steering_clamped * 90))
       steer_value = max(0, min(180, steer_value))
       
       command = f"{motor_value},{steer_value}\n"
       
       current_time = time.time()
       # В авто-режиме шлем не чаще 10 раз в секунду
       if command != self.last_sent_cmd or (current_time - self.last_sent_time) > 0.1:
           with self.lock:
               try:
                   self.arduino.write(command.encode('utf-8'))
                   self.last_sent_cmd = command
                   self.last_sent_time = current_time
                   self.last_command_time = current_time
               except serial.SerialException as e:
                   logger.error(f"Ошибка отправки: {e}")
                   self.arduino = None
   
   def stop(self):
       if not self.arduino: return
       with self.lock:
           try:
               self.arduino.write(f"{base_speed},{base_rotation}\n".encode('utf-8'))
               self.last_sent_cmd = f"{base_speed},{base_rotation}\n"
               self.last_command_time = time.time()
           except: pass
   
   def check_stop(self):
       if self.arduino and time.time() - self.last_command_time > 1.0:
           self.stop()
           
   def close(self):
       self.stop()
       time.sleep(0.1)
       if self.arduino: self.arduino.close()

def main():
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.settimeout(0.5)
   try:
       sock.bind((UDP_IP, UDP_PORT))
       logger.info(f"Сервер слушает порт {UDP_PORT}")
   except OSError as e:
       logger.error(f"Ошибка порта {UDP_PORT}: {e}")
       sys.exit(1)
   
   model_path = "/mnt/ArdorSSD/car_control_project_new/Datasets/cone_detector_v3.engine"
   detector = ConeDetector(model_path)
   car = CarController()
   
   robot_state = {
       'auto_mode': False,
       'msg': '',
       'msg_time': 0
   }
   
   vision = VisionSystem(detector, car, robot_state)
   running = True
   
   logger.info("Сервер готов. Включен алгоритм Pure Pursuit.")
   
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
                       logger.info("Включен АВТОМАТИЧЕСКИЙ режим")
                       robot_state['auto_mode'] = True
                       robot_state['msg'] = ''
               elif command == "S":
                   if robot_state['auto_mode']:
                       logger.info("Включен РУЧНОЙ режим")
                       robot_state['auto_mode'] = False
                       car.stop()
               elif command == "R":
                   vision.start_recording()
               elif command == "C":
                   vision.stop_recording()
               elif command.startswith("speed:"):
                   try:
                       fwd, bck = map(int, command[6:].split(','))
                       car.set_speeds(fwd, bck)
                   except: pass
               else:
                   if not robot_state['auto_mode']:
                       try:
                           # Клиент присылает -1, 0, 1 для поворота
                           speed, steering = map(float, command.split(','))
                           car.update(speed, steering)
                       except: pass

               if time.time() - robot_state['msg_time'] > 3.0:
                   robot_state['msg'] = ''

               telemetry = {
                   "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                   "rec": vision.is_recording,
                   "fwd": car.forward_speed,
                   "bck": car.back_speed,
                   "msg": robot_state['msg']
               }
               sock.sendto(json.dumps(telemetry).encode('utf-8'), addr)

           except socket.timeout:
               if not robot_state['auto_mode']:
                   car.check_stop()
                   
   except KeyboardInterrupt:
       logger.info("Остановка сервера...")
   finally:
       vision.close()
       car.close()
       sock.close()
       logger.info("Сервер остановлен")

if __name__ == "__main__":
   main()
