"""
Джетсон: YOLO-детекция конусов + отрисовка + отправка кадра на ноутбук.
"""

import socket
import struct
import time
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ========== НАСТРОЙКИ ==========
# Сеть
STREAM_IP = "192.168.137.1"   # IP ноутбука
STREAM_PORT = 5555

# Камера
ZED_RESOLUTION = sl.RESOLUTION.HD720
ZED_FPS = 15

# Модель
YOLO_MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Классы конусов (подставьте свои названия из model.names)
BLUE_CONES = ['blue', 'blue_cone']
YELLOW_CONES = ['yellow', 'yellow_cone']
ORANGE_CONES = ['orange', 'orange_cone']

# Цвета для боксов
BLUE_COLOR = (255, 0, 0)      # BGR
YELLOW_COLOR = (0, 255, 255)
ORANGE_COLOR = (0, 140, 255)
TEXT_COLOR = (255, 255, 255)

# JPEG
JPEG_QUALITY = 60
CHUNK_SIZE = 60000
# ===============================

# Загрузка модели
logger.info(f"Загрузка YOLO: {YOLO_MODEL_PATH} на {DEVICE}")
model = YOLO(YOLO_MODEL_PATH)
logger.info(f"Классы: {model.names}")

# Словарь: ID класса → название
class_names = model.names  # {0: 'blue', 1: 'yellow', 2: 'orange', ...}

# ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = ZED_RESOLUTION
init_params.camera_fps = ZED_FPS
init_params.depth_mode = sl.DEPTH_MODE.NONE

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    logger.error("Не удалось открыть ZED")
    exit()

logger.info(f"ZED открыта: {ZED_RESOLUTION}, {ZED_FPS} FPS")

# Сокет для стрима
stream_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
stream_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
stream_addr = (STREAM_IP, STREAM_PORT)

image_zed = sl.Mat()
frame_id = 0
fps_counter = 0
fps_last_time = time.time()
current_fps = 0

logger.info(f"Стрим на {STREAM_IP}:{STREAM_PORT}")
logger.info("Работает. Ctrl+C для выхода.")

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Получаем кадр
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            
            # RGBA → BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # ===== YOLO ДЕТЕКЦИЯ =====
            results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, 
                          verbose=False, device=DEVICE)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Название класса
                        name = class_names.get(cls_id, f'cls_{cls_id}')
                        
                        # Цвет по типу конуса
                        if name in BLUE_CONES:
                            color = BLUE_COLOR
                            label = f"BLUE {conf:.2f}"
                        elif name in YELLOW_CONES:
                            color = YELLOW_COLOR
                            label = f"YELLOW {conf:.2f}"
                        elif name in ORANGE_CONES:
                            color = ORANGE_COLOR
                            label = f"ORANGE {conf:.2f}"
                        else:
                            color = (200, 200, 200)
                            label = f"{name} {conf:.2f}"
                        
                        # Рисуем бокс
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Подпись
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_last_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_last_time = time.time()
            
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ===== ОТПРАВКА НА НОУТБУК =====
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            data = jpeg.tobytes()
            
            total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            for i in range(total_chunks):
                chunk = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                header = struct.pack('!IHHI', frame_id, total_chunks, i, len(chunk))
                try:
                    stream_sock.sendto(header + chunk, stream_addr)
                except:
                    pass
            
            frame_id += 1

except KeyboardInterrupt:
    pass
finally:
    zed.close()
    stream_sock.close()
    logger.info("Остановлено")
