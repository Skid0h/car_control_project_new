"""Простой пример работы с ZED-камерой: запись видео."""

import pyzed.sl as sl
import time
import os
from datetime import datetime
import cv2
import subprocess

def convert_to_compatible_format(input_path, output_path):
    """Конвертирует видео в формат, совместимый со встроенным просмотрщиком Ubuntu"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except:
        return False

def main():
    # Создаем папку для сохранения видео
    output_folder = "zed_recordings"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_video_path = os.path.join(output_folder, f"temp_{timestamp}.avi")
    final_video_path = os.path.join(output_folder, f"zed_recording_{timestamp}.mp4")
    
    print(f"Видео будет сохранено в: {final_video_path}")

    # Инициализация ZED-камеры
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Ошибка: не удалось инициализировать ZED-камеру")
        return

    print("ZED-камера успешно инициализирована. Нажмите Ctrl+C для выхода.")

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    video_writer = None
    frame_count = 0
    target_fps = 10
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    # Получаем первый кадр для инициализации
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        image_np = image_zed.get_data()
        
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        
        height, width = image_np.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (width, height))
        
        if not video_writer.isOpened():
            print("Ошибка: не удалось открыть VideoWriter")
            zed.close()
            return
        
        print(f"Начало записи. Разрешение: {width}x{height}, FPS: {target_fps}")

    try:
        while True:
            current_time = time.time()
            
            if current_time - last_frame_time >= frame_time:
                if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Получение изображения
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    image_np = image_zed.get_data()
                    
                    if image_np.shape[2] == 4:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

                    # Запись кадра
                    video_writer.write(image_np)
                    frame_count += 1
                    
                    last_frame_time = current_time
                
    except KeyboardInterrupt:
        print("\nОстановка работы скрипта...")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Временное видео сохранено: {temp_video_path}")
            print(f"Всего кадров: {frame_count}")
            
            # Конвертируем в совместимый формат
            print("Конвертация видео в совместимый формат...")
            if convert_to_compatible_format(temp_video_path, final_video_path):
                print(f"Готовое видео: {final_video_path}")
                os.remove(temp_video_path)
                print("Временный файл удален")
            else:
                print("Ошибка конвертации. Временный файл сохранен.")
        
        zed.close()
        print("ZED-камера отключена")

if __name__ == "__main__":
    main()
