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
# Но лучше сделать видимое маленькое окно для фокуса
pygame.display.set_mode((300, 100))
pygame.display.set_caption("Управление записью - нажимайте R, C, Q")

model = YOLO("/mnt/ArdorSSD/car_control_project_new/Datasets/cone_detector.pt")

class_colors = {
    "Yellow": (0, 255, 255),
    "Blue": (255, 0, 0),
    "Orange": (0, 165, 255)
}

def main():
    # Создаем папку для записей
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

    print("\n" + "="*50)
    print("УПРАВЛЕНИЕ (pygame окно):")
    print("  R - начать запись")
    print("  C - остановить запись")
    print("  Q - выход")
    print("="*50 + "\n")
    print("  Нажимайте клавиши в окне 'Управление записью'")

    # Переменные записи
    video_writer = None
    recording = False
    frame_count = 0
    temp_path = None
    final_path = None
    
    # Состояния клавиш
    key_r_pressed = False
    key_c_pressed = False
    key_q_pressed = False

    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    
    target_fps = 10
    frame_time = 1.0 / target_fps
    last_time = time.time()

    # Для отображения статуса в pygame окне
    font = pygame.font.Font(None, 36)

    try:
        while True:
            current_time = time.time()
            
            # Обработка событий pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    key_q_pressed = True
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        key_r_pressed = True
                        print(">>> R нажата (pygame)")
                    elif event.key == pygame.K_c:
                        key_c_pressed = True
                        print(">>> C нажата (pygame)")
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        key_q_pressed = True
                        print(">>> Q нажата (pygame)")
            
            # Обновление pygame окна
            screen = pygame.display.get_surface()
            screen.fill((30, 30, 30))
            
            status_text = "REC" if recording else "STOP"
            status_color = (255, 0, 0) if recording else (255, 255, 255)
            
            text = font.render(f"Status: {status_text}", True, status_color)
            screen.blit(text, (20, 20))
            
            text = font.render("R:Start C:Stop Q:Quit", True, (200, 200, 200))
            screen.blit(text, (20, 60))
            
            pygame.display.flip()
            
            # Проверка команд
            if key_q_pressed:
                print("\nВыход...")
                break
            
            if key_r_pressed and not recording:
                key_r_pressed = False
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = os.path.join(output_folder, f"temp_{timestamp}.avi")
                final_path = os.path.join(output_folder, f"recording_{timestamp}.mp4")
                
                height, width = 720, 1280  # HD720
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(temp_path, fourcc, target_fps, (width, height))
                
                if video_writer.isOpened():
                    recording = True
                    frame_count = 0
                    print(f"\n>>> ЗАПИСЬ: {final_path}")
            
            if key_c_pressed and recording:
                key_c_pressed = False
                recording = False
                if video_writer:
                    video_writer.release()
                    print(f"\n>>> СТОП. Кадров: {frame_count}")
                    
                    # Конвертация
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

            if current_time - last_time >= frame_time:
                if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                    
                    frame = image_zed.get_data()[:, :, :3].copy()
                    depth_data = depth_zed.get_data()

                    # YOLO детекция
                    results = model(frame)

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            label = model.names[cls]
                            color = class_colors.get(label, (0, 255, 0))
                            
                            # Глубина
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            depth_text = "N/A"
                            
                            if 0 <= center_y < depth_data.shape[0] and 0 <= center_x < depth_data.shape[1]:
                                depth_val = depth_data[center_y, center_x]
                                if np.isfinite(depth_val) and depth_val > 0:
                                    depth_text = f"{depth_val:.2f}m"
                            
                            # Отрисовка
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"Depth: {depth_text}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Индикатор записи
                    if recording:
                        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(frame, f"REC {frame_count}", (50, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        video_writer.write(frame)
                        frame_count += 1

                    cv2.putText(frame, "R=Start C=Stop Q=Exit (pygame)", (10, frame.shape[0]-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
