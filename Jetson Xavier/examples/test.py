"""
Сервер видеопотока ZED 2 с разбивкой кадров на чанки.
"""

import pyzed.sl as sl
import socket
import struct
import cv2
import time

STREAM_IP = "192.168.137.1"
STREAM_PORT = 5555
CHUNK_SIZE = 60000  # Безопасный размер для UDP
JPEG_QUALITY = 60   # Чуть ниже качество — меньше размер

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 15
init_params.depth_mode = sl.DEPTH_MODE.NONE

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Ошибка: {status}")
    exit()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
image = sl.Mat()
frame_id = 0

print(f"Стрим на {STREAM_IP}:{STREAM_PORT}")

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()
        
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Сжатие
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        data = jpeg.tobytes()
        
        # Разбивка на чанки
        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for i in range(total_chunks):
            chunk = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            
            # Заголовок: frame_id (4) + total_chunks (2) + chunk_index (2) + размер чанка (4)
            header = struct.pack('!IHHI', frame_id, total_chunks, i, len(chunk))
            packet = header + chunk
            
            sock.sendto(packet, (STREAM_IP, STREAM_PORT))
        
        frame_id += 1
        time.sleep(0.05)
