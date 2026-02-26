"""Работа с ZED-камерой: отображение видео с камеры и отметка ближайшей точки"""

import pyzed.sl as sl
import cv2
import numpy as np
import time

def find_closest_point(depth_data, max_distance=5.0):
    """
    Находит координаты ближайшей точки на карте глубины
    Возвращает (x, y, distance) или (None, None, None) если нет данных
    """
    # Убираем inf и nan
    depth_clean = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Создаем маску для валидных точек (больше 0 и меньше max_distance)
    valid_mask = (depth_clean > 0) & (depth_clean < max_distance)
    
    if not np.any(valid_mask):
        return None, None, None
    
    # Находим индекс минимального значения среди валидных точек
    masked_depth = np.where(valid_mask, depth_clean, np.inf)
    min_idx = np.argmin(masked_depth)
    y, x = np.unravel_index(min_idx, depth_clean.shape)
    min_distance = depth_clean[y, x]
    
    return x, y, min_distance

def main():
    # Инициализация ZED-камеры
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Ошибка: не удалось инициализировать ZED-камеру")
        return

    print("ZED-камера успешно инициализирована")
    print("Нажмите 'q' для выхода")

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    
    # Для измерения FPS
    fps_counter = 0
    fps_time = time.time()
    
    # Максимальное расстояние для поиска ближайшей точки (метры)
    max_search_distance = 5.0

    try:
        while True:
            # Захват кадра
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                
                # Получаем изображение с камеры
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                image_np = image_zed.get_data()
                
                # Получаем карту глубины
                zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                depth_data = depth_zed.get_data()
                
                # Находим ближайшую точку
                x, y, min_distance = find_closest_point(depth_data, max_search_distance)
                
                # Рисуем на изображении
                if x is not None and y is not None:
                    # Рисуем круг в ближайшей точке
                    cv2.circle(image_np, (x, y), 10, (0, 0, 255), -1)  # Красный круг
                    cv2.circle(image_np, (x, y), 15, (255, 255, 255), 2)  # Белая обводка
                    
                    # Добавляем текст с расстоянием
                    text = f"Ближайшая точка: {min_distance:.2f} м"
                    cv2.putText(image_np, text, (x + 20, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(image_np, text, (x + 20, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Добавляем информацию о FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps_text = f"FPS: {fps_counter}"
                    fps_counter = 0
                    fps_time = current_time
                
                cv2.putText(image_np, f"FPS: {fps_counter}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_np, f"Max distance: {max_search_distance}m", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Отображаем изображение
                cv2.imshow("ZED Camera - Closest Point", image_np)
                
                # Выход по нажатию клавиши 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nОстановка работы скрипта...")
    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("ZED-камера отключена")

if __name__ == "__main__":
    main()
