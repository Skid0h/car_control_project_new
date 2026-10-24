import cv2
import torch
from ultralytics import YOLO
import time

MODEL_NAME = 'yolo11n.pt'  
VIDEO_PATH = 'Test_Yolo11n.mp4'    
OUTPUT_PATH = 'output_video.mp4' 
CONFIDENCE = 0.3         
CLASS_PERSON = 0          

print("Загрузка модели YOLOv11n...")
model = YOLO(MODEL_NAME)
print(f"Модель загружена на: {next(model.model.parameters()).device}")

print(f"\n Открытие видео: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

# Получаем параметры видео
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"   Параметры видео:")
print(f"   Размер: {width}x{height}")
print(f"   FPS: {fps}")
print(f"   Кадров: {total_frames}")
print(f"   Длительность: {total_frames/fps:.1f} сек")

# Создаем VideoWriter для сохранения
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ================= ОБРАБОТКА ВИДЕО =================
frame_count = 0
people_count_history = []
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Прогресс каждые 50 кадров
    if frame_count % 50 == 0:
        elapsed = time.time() - start_time
        fps_current = frame_count / elapsed
        print(f"   Обработано: {frame_count}/{total_frames} кадров ({frame_count/total_frames*100:.1f}%) | FPS: {fps_current:.1f}")
    
    # Детекция людей на кадре
    results = model(frame, 
                    conf=CONFIDENCE,
                    classes=[CLASS_PERSON],  
                    verbose=False,
                    device='cuda')  
    
    # Количество людей на текущем кадре
    people_count = 0
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Проверяем что это человек (дополнительная проверка)
            cls = int(box.cls[0])
            if cls == CLASS_PERSON:
                people_count += 1
                
                # Координаты bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Рисуем bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Подпись с уверенностью
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    people_count_history.append(people_count)
    
    # Статистика в углу кадра
    stats_text = f"People: {people_count} | Frame: {frame_count}/{total_frames}"
    cv2.putText(frame, stats_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Сохраняем кадр
    out.write(frame)
    
    # Показываем в реальном времени (можно отключить)
    cv2.imshow('YOLOv11 - People Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n Остановка...")
        break

# ================= ЗАВЕРШЕНИЕ =================
cap.release()
out.release()
cv2.destroyAllWindows()

# Статистика
processing_time = time.time() - start_time
print(f"\n ОБРАБОТКА ЗАВЕРШЕНА!")
print(f"   СТАТИСТИКА:")
print(f"   Общее время: {processing_time:.1f} сек")
print(f"   Средний FPS: {frame_count/processing_time:.1f}")
print(f"   Максимум людей в кадре: {max(people_count_history)}")
print(f"   Среднее людей в кадре: {sum(people_count_history)/len(people_count_history):.1f}")
print(f"   Результат сохранен: {OUTPUT_PATH}")

