"""
Сервер видеопотока ZED 2.
Запускать на Jetson.
Отправляет JPEG-кадры через UDP на ноутбук.
"""

import pyzed.sl as sl
import socket
import struct
import cv2
import numpy as np
import time

# Настройки
STREAM_IP = "192.168.137.1"  # IP ноутбука
STREAM_PORT = 5555
JPEG_QUALITY = 70
RESOLUTION = sl.RESOLUTION.HD720
FPS = 15

# Инициализация ZED 2
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = RESOLUTION
init_params.camera_fps = FPS
init_params.depth_mode = sl.DEPTH_MODE.NONE

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Ошибка камеры: {status}")
    exit()

print(f"Стрим запущен: {STREAM_IP}:{STREAM_PORT}")

# Сокет для отправки
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

# Буфер для кадра
image = sl.Mat()
width = 0
height = 0

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Получаем левый кадр
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Конвертируем в numpy
            frame = image.get_data()
            
            # ZED возвращает RGBA, конвертируем в BGR для OpenCV
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            if width == 0:
                height, width = frame.shape[:2]
                print(f"Разрешение: {width}x{height}")
            
            # Сжатие в JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            data = jpeg.tobytes()
            
            # Отправка с заголовком (размер кадра)
            header = struct.pack('!I', len(data))
            sock.sendto(header + data, (STREAM_IP, STREAM_PORT))
            
            time.sleep(1.0 / (FPS + 5))  # Чуть быстрее FPS для буфера

except KeyboardInterrupt:
    pass
finally:
    zed.close()
    sock.close()
    print("Стрим остановлен")
