"""Детекция конусов с ZED-камерой и записью по нажатию клавиш (pygame окно)."""

from ultralytics import YOLO
import pyzed.sl as sl
import cv2
import numpy as np
import time
import os
from datetime import datetime
import subprocess
import pygame
import sys

# Инициализация pygame
pygame.init()
screen = pygame.display.set_mode((300, 130))
pygame.display.set_caption("Управление записью")

model = YOLO("/mnt/ArdorSSD/car_control_project_new/Datasets/cone_detector_v3.engine")

# Параметры детекции
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.4

# Цвета для конусов
CONE_COLORS = {
    0: (0, 255, 255),   # Желтый
    1: (0, 165, 255),   # Оранжевый
    2: (255, 0, 0)      # Синий
}

CLASS_NAMES = {
    0: "Yellow",
    1: "Orange", 
    2: "Blue"
}

# Класс для кнопок
class Button:
    def __init__(self, x, y, w, h, text, color, hover_color):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.clicked = False
        
    def draw(self, screen, font):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, self.hover_color, self.rect)
        else:
            pygame.draw.rect(screen, self.color, self.rect)
        
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
                return True
        return False

def main():
    output_folder = "zed_recordings"
    os.makedirs(output_folder, exist_ok=True)

    # Инициализация ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Ошибка инициализации ZED")
        return

    video_writer = None
    recording = False
    frame_count = 0
    temp_path = None
    final_path = None
    actual_fps = 0

    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    
    target_fps = 15
    frame_time = 1.0 / target_fps
    last_time = time.time()
    
    # Для расчета FPS
    fps_counter = 0
    fps_last_time = time.time()
    current_fps = 0
    
    # Создание кнопок
    button_font = pygame.font.Font(None, 36)
    btn_r = Button(20, 30, 80, 40, "R", (0, 100, 0), (0, 200, 0))
    btn_c = Button(110, 30, 80, 40, "C", (100, 0, 0), (200, 0, 0))
    btn_q = Button(200, 30, 80, 40, "Q", (100, 100, 100), (150, 150, 150))
    
    # Статус текст
    font = pygame.font.Font(None, 28)

    try:
        while True:
            current_time = time.time()
            
            # Обработка событий pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if btn_r.handle_event(event):
                    if not recording:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_path = os.path.join(output_folder, f"temp_{timestamp}.avi")
                        final_path = os.path.join(output_folder, f"recording_{timestamp}.mp4")
                        
                        height, width = 720, 1280
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        
                        actual_fps = current_fps if current_fps > 0 else 15
                        video_writer = cv2.VideoWriter(temp_path, fourcc, actual_fps, (width, height))
                        
                        if video_writer.isOpened():
                            recording = True
                            frame_count = 0
                            print(f"\n>>> ЗАПИСЬ: {final_path} с FPS {actual_fps}")
                
                if btn_c.handle_event(event):
                    if recording:
                        recording = False
                        if video_writer:
                            video_writer.release()
                            print(f"\n>>> СТОП. Кадров: {frame_count}")
                            
                            try:
                                subprocess.run([
                                    'ffmpeg', '-i', temp_path,
                                    '-c:v', 'libx264', '-preset', 'fast',
                                    '-pix_fmt', 'yuv420p', '-y', final_path
                                ], check=True, capture_output=True)
                                os.remove(temp_path)
                                print(f"Сохранено: {final_path}")
                            except Exception as e:
                                print(f"Ошибка конвертации: {e}")
                                print(f"Временный файл: {temp_path}")
                
                if btn_q.handle_event(event):
                    print("\nВыход...")
                    pygame.quit()
                    sys.exit()
            
            # Обновление pygame окна
            screen.fill((30, 30, 30))
            btn_r.draw(screen, button_font)
            btn_c.draw(screen, button_font)
            btn_q.draw(screen, button_font)
            
            status_text = "RECORDING" if recording else "STOPPED"
            status_color = (255, 0, 0) if recording else (255, 255, 255)
            text = font.render(f"Status: {status_text}", True, status_color)
            screen.blit(text, (20, 90))
            
            pygame.display.flip()
            
            if current_time - last_time >= frame_time:
                if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                    
                    frame = image_zed.get_data()[:, :, :3].copy()
                    depth_data = depth_zed.get_data()

                    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            
                            color = CONE_COLORS.get(cls, (0, 255, 0))
                            class_name = CLASS_NAMES.get(cls, model.names[cls])
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            depth_text = "N/A"
                            
                            if 0 <= center_y < depth_data.shape[0] and 0 <= center_x < depth_data.shape[1]:
                                depth_val = depth_data[center_y, center_x]
                                if np.isfinite(depth_val) and depth_val > 0:
                                    depth_text = f"{depth_val:.2f}m"
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"Depth: {depth_text}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Расчет FPS
                    fps_counter += 1
                    if current_time - fps_last_time >= 1.0:
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_last_time = current_time

                    # Отображение FPS на видео
                    cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if recording:
                        cv2.putText(frame, f"REC {frame_count}", (10, 55), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        video_writer.write(frame)
                        frame_count += 1

                    cv2.imshow("ZED Cones Detection", frame)
                    cv2.waitKey(1)
                    
                    last_time = current_time

    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        if recording and video_writer:
            video_writer.release()
        zed.close()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()
