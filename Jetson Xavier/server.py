"""Запускается на Jetson.
ЛИНЕЙНАЯ АРХИТЕКТУРА. ПИД-АВТОПИЛОТ. 
ФАЙЛ КОНФИГУРАЦИИ. ПОМЕХОЗАЩИЩЕННЫЙ UART.
ОПТИМИЗАЦИЯ: Строго последовательный цикл, убрана лишняя фильтрация (минимальная задержка), скорость постоянная.
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

# Загрузка конфигурации
def load_config(path="config.json"):
  if not os.path.exists(path):
      logger.error(f"Файл конфигурации '{path}' не найден! Пожалуйста, создайте его.")
      sys.exit(1)
  with open(path, 'r', encoding='utf-8') as f:
      return json.load(f)

config = load_config()

CONE_COLORS = {0: (0, 255, 255), 1: (0, 165, 255), 2: (255, 0, 0), 3: (0, 0, 255)}
CLASS_NAMES = {0: "Yellow", 1: "Orange", 2: "Blue", 3: "Red"}

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
  def __init__(self, cfg):
      self.cfg = cfg
      self.model = None
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      logger.info(f"Загрузка модели YOLO с {self.cfg['yolo_model_path']} на {self.device}")
      try:
          self.model = YOLO(self.cfg['yolo_model_path'])
          if hasattr(self.model, 'names'):
              logger.info(f"Классы модели: {self.model.names}")
      except Exception as e:
          logger.error(f"Ошибка загрузки модели: {e}")
          self.model = None
  
  def detect(self, frame):
      if self.model is None:
          return frame, []
      try:
          results = self.model(frame, conf=self.cfg['confidence_threshold'], iou=self.cfg['iou_threshold'], verbose=False, device=self.device)
          detections = []
          for result in results:
              if result.boxes is not None:
                  for box in result.boxes:
                      x1, y1, x2, y2 = map(int, box.xyxy[0])
                      conf = float(box.conf[0])
                      cls = int(box.cls[0])
                      
                      # Целимся в нижнюю широкую часть конуса (80% высоты рамки)
                      center_x = (x1 + x2) // 2
                      center_y = int(y1 + (y2 - y1) * 0.8) 
                      
                      detections.append({
                          'bbox': (x1, y1, x2, y2),
                          'conf': conf,
                          'class': cls,
                          'center': (center_x, center_y)
                      })
                      
                      color = CONE_COLORS.get(cls, (0, 255, 0))
                      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                      class_name = CLASS_NAMES.get(cls, f"Class_{cls}")
                      cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                      cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)
          return frame, detections
      except Exception as e:
          logger.error(f"Ошибка детекции: {e}")
          return frame, []

class VisionSystem:
  def __init__(self, detector, car, robot_state, cfg):
      self.detector = detector
      self.car = car
      self.robot_state = robot_state
      self.cfg = cfg
      self.zed = sl.Camera()
      
      self.running = True
      self.is_recording = False
      self.output_folder = self.cfg['vision']['output_folder']
      
      if not os.path.exists(self.output_folder):
          os.makedirs(self.output_folder)
          
      self.reset_pid_event = threading.Event()
      
      self.fx = 0
      self.cx_cam = 0
      
      # ЕДИНСТВЕННЫЙ поток для всей работы с камерой, нейросетью и логикой
      self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
      self.vision_thread.start()

  def reset_pid(self):
      self.reset_pid_event.set()
      
  def _convert_video(self, input_path, output_path, fps):
      try:
          # Запускаем ffmpeg как отдельный процесс, он не блокирует Python GIL
          cmd = ['ffmpeg', '-i', input_path, '-r', str(fps), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-y', output_path]
          subprocess.run(cmd, check=True, capture_output=True)
          logger.info(f"Видео сконвертировано: {output_path}")
          os.remove(input_path)
          
          self.robot_state['msg'] = "ВИДЕО УСПЕШНО СОХРАНЕНО!"
          self.robot_state['msg_time'] = time.time()
          
      except Exception as e:
          logger.error(f"Ошибка конвертации: {e}")
          self.robot_state['msg'] = "ОШИБКА СОХРАНЕНИЯ ВИДЕО!"
          self.robot_state['msg_time'] = time.time()

  def _vision_loop(self):
      # 1. Инициализация камеры
      init_params = sl.InitParameters()
      init_params.camera_resolution = sl.RESOLUTION.HD720
      init_params.camera_fps = 15 # Жесткая хардверная привязка цикла к 15 FPS
      init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
      init_params.coordinate_units = sl.UNIT.METER 
      
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
      depth_zed = sl.Mat()

      # 2. Инициализация переменных ПИД-регулятора
      pid_cfg = self.cfg['pid']
      
      prev_error = 0.0
      integral = 0.0
      smoothed_error = 0.0
      last_time = time.time()
      
      # 3. Переменные записи и FPS
      fps_counter = 0
      current_fps = 0
      fps_last_time = time.time()
      
      video_writer = None
      temp_video_path = None
      final_video_path = None

      logger.info("Единый Линейный Поток (ZED -> YOLO -> PID -> REC) запущен.")

      # 4. Основной последовательный цикл
      while self.running:
          # Хардверное ожидание нового кадра от ZED (без лишней нагрузки на CPU)
          if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
              
              # Сброс ПИД при необходимости
              if self.reset_pid_event.is_set():
                  integral = 0.0
                  prev_error = 0.0
                  smoothed_error = 0.0
                  self.reset_pid_event.clear()
                  logger.info("ПИД-регулятор сброшен.")

              # --- ШАГ 1: Получение данных (Без копирования .copy()) ---
              self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
              self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
              
              img_data = image_zed.get_data()
              if img_data.shape[2] == 4:
                  # cvtColor создает новый массив BGR, с которым безопасно работать
                  image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
              else:
                  image_np = img_data
                  
              depth_data = depth_zed.get_data()
              
              # --- ШАГ 2: Детекция ---
              image_np, detections = self.detector.detect(image_np)
              
              blue_cones_3d = []
              yellow_cones_3d = []
              target_detected = False
              
              # --- ШАГ 3: Проекция в 3D ---
              for det in detections:
                  u, v = det['center']
                  if 0 <= v < depth_data.shape[0] and 0 <= u < depth_data.shape[1]:
                      z = depth_data[v, u] 
                      if np.isfinite(z) and 0.1 < z < 10.0:
                          x_cam = (u - self.cx_cam) * z / self.fx
                          det['pos_3d'] = (x_cam, z)
                          
                          cv2.putText(image_np, f"X:{x_cam:.2f} Z:{z:.2f}", (det['bbox'][0], det['bbox'][1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                          
                          if det['class'] == 2:   
                              blue_cones_3d.append(det)
                          elif det['class'] == 0: 
                              yellow_cones_3d.append(det)
                          elif det['class'] == 1 and z <= 0.4: 
                              target_detected = True

              raw_gate_x, raw_gate_z = None, None
              
              if blue_cones_3d and yellow_cones_3d:
                  closest_blue = min(blue_cones_3d, key=lambda c: c['pos_3d'][1])
                  closest_yellow = min(yellow_cones_3d, key=lambda c: c['pos_3d'][1])
                  raw_gate_x = (closest_blue['pos_3d'][0] + closest_yellow['pos_3d'][0]) / 2.0
                  raw_gate_z = (closest_blue['pos_3d'][1] + closest_yellow['pos_3d'][1]) / 2.0
              
              elif blue_cones_3d:
                  closest_blue = min(blue_cones_3d, key=lambda c: c['pos_3d'][1])
                  raw_gate_x = closest_blue['pos_3d'][0] + 0.8 
                  raw_gate_z = closest_blue['pos_3d'][1]
                  
              elif yellow_cones_3d:
                  closest_yellow = min(yellow_cones_3d, key=lambda c: c['pos_3d'][1])
                  raw_gate_x = closest_yellow['pos_3d'][0] - 0.8
                  raw_gate_z = closest_yellow['pos_3d'][1]

              # --- ШАГ 4: Расчет ПИД ---
              current_time = time.time()
              dt = current_time - last_time
              if dt <= 0.001: dt = 0.001
              last_time = current_time

              if raw_gate_x is not None and raw_gate_z is not None:
                  # Отрисовка целевой точки без лишней фильтрации
                  raw_gate_u = int((raw_gate_x * self.fx / raw_gate_z) + self.cx_cam)
                  raw_gate_v = image_np.shape[0] // 2 
                  cv2.circle(image_np, (raw_gate_u, raw_gate_v), 12, (0, 255, 0), -1)

                  # Расчет ошибки и ее сглаживание
                  raw_error = math.atan2(raw_gate_x, raw_gate_z)
                  smoothed_error = (pid_cfg['ema_alpha'] * raw_error) + ((1.0 - pid_cfg['ema_alpha']) * smoothed_error)
              else:
                  # Плавное затухание ошибки, если ворота потеряны
                  smoothed_error *= 0.85 

              error = smoothed_error

              integral += error * dt
              integral = max(-pid_cfg['max_integral'], min(pid_cfg['max_integral'], integral))
              
              derivative = (error - prev_error) / dt
              prev_error = error
              
              auto_steering = (pid_cfg['kp_gain'] * error) + (pid_cfg['ki_gain'] * integral) + (pid_cfg['kd_gain'] * derivative)
              auto_steering = max(-1.0, min(1.0, auto_steering))
              
              # --- ШАГ 5: Обновление управления ---
              if self.robot_state.get('auto_mode', False):
                  if target_detected:
                      self.robot_state['auto_mode'] = False
                      self.robot_state['msg'] = "ОРАНЖЕВЫЙ КОНУС! СТОП."
                      self.robot_state['msg_time'] = time.time()
                      self.car.stop()
                  else:
                      # Передаем константу 1.0 (максимальный разрешенный газ)
                      self.car.update(1.0, auto_steering)

              # Подсчет реального FPS цикла
              fps_counter += 1
              if time.time() - fps_last_time >= 1.0:
                  current_fps = fps_counter
                  fps_counter = 0
                  fps_last_time = time.time()
              
              cv2.putText(image_np, f"FPS: {current_fps} Mode: {'AUTO' if self.robot_state.get('auto_mode') else 'MANUAL'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
              
              # --- ШАГ 6: Видеозапись прямо в цикле ---
              if self.is_recording:
                  cv2.putText(image_np, f"REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  
                  if video_writer is None:
                      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                      temp_video_path = os.path.join(self.output_folder, f"temp_{timestamp}.avi")
                      final_video_path = os.path.join(self.output_folder, f"zed_recording_{timestamp}.mp4")
                      height, width = image_np.shape[:2]
                      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                      # Записываем с FPS, равным целевому (15 FPS)
                      video_writer = cv2.VideoWriter(temp_video_path, fourcc, 15.0, (width, height))
                  
                  video_writer.write(image_np)
              else:
                  if video_writer is not None:
                      video_writer.release()
                      video_writer = None
                      # Отправляем задачу конвертации в фоновый поток
                      threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, 15)).start()

      # Завершение работы
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
      self.vision_thread.join(timeout=3.0)

class CarController:
  def __init__(self, cfg):
      self.cfg = cfg
      self.lock = threading.Lock()
      self.forward_speed = self.cfg['max_auto_speed']
      self.back_speed = self.cfg['max_reverse_speed']
      self.neutral = self.cfg['neutral_speed']
      self.center_steering = self.cfg['center_steering']
      
      self.last_command_time = 0
      self.last_sent_cmd = ""
      self.last_sent_time = 0
      
      port = find_arduino_port()
      self.arduino = None
      if port is None:
          logger.error("Arduino не найден")
          return
          
      try:
          self.arduino = serial.Serial(port, self.cfg['baud_rate'], timeout=1)
          time.sleep(2)
          self.stop()
          time.sleep(0.5)
      except Exception as e:
          logger.error(f"Ошибка подключения: {e}")
          self.arduino = None
  
  def set_speeds(self, forward, back):
      self.forward_speed = forward
      self.back_speed = back
  
  def update(self, speed, steering):
      if not self.arduino: return
      
      speed_clamped = max(-1.0, min(1.0, float(speed)))
      
      motor_value = self.neutral
      if speed_clamped > 0:
          motor_value = int(self.neutral + (self.forward_speed - self.neutral) * speed_clamped)
      elif speed_clamped < 0:
          motor_value = int(self.neutral + (self.back_speed - self.neutral) * abs(speed_clamped))
      
      steering_clamped = max(-1.0, min(1.0, float(steering)))
      steer_value = int(self.center_steering - (steering_clamped * 90))
      steer_value = max(0, min(180, steer_value))
      
      command = f"<{motor_value},{steer_value}>"
      
      current_time = time.time()
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
              cmd = f"<{self.neutral},{self.center_steering}>"
              self.arduino.write(cmd.encode('utf-8'))
              self.last_sent_cmd = cmd
              self.last_command_time = time.time()
          except: pass
  
  def check_stop(self):
      if self.arduino and time.time() - self.last_command_time > 0.4:
          self.stop()
          
  def close(self):
      self.stop()
      time.sleep(0.1)
      if self.arduino: self.arduino.close()

def main():
  udp_ip = config['network']['udp_ip']
  udp_port = config['network']['udp_port']

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.settimeout(0.2) 
  try:
      sock.bind((udp_ip, udp_port))
      logger.info(f"Сервер слушает порт {udp_port}")
  except OSError as e:
      logger.error(f"Ошибка порта {udp_port}: {e}")
      sys.exit(1)
  
  detector = ConeDetector(config['vision'])
  car = CarController(config['car'])
  
  robot_state = {
      'auto_mode': False,
      'cam_connected': False,
      'msg': '',
      'msg_time': 0
  }
  
  vision = VisionSystem(detector, car, robot_state, config)
  running = True
  
  logger.info("Сервер готов. Конфигурация загружена. UART-протокол активен.")
  
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
                      vision.reset_pid()
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
                          speed, steering = map(float, command.split(','))
                          car.update(speed, steering)
                      except: pass

              if time.time() - robot_state['msg_time'] > 3.0:
                  robot_state['msg'] = ''

              telemetry = {
                  "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                  "rec": vision.is_recording,
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
      logger.info("Остановка сервера...")
  finally:
      vision.close()
      car.close()
      sock.close()
      logger.info("Сервер остановлен")

if __name__ == "__main__":
  main()
