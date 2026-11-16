from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

current_jpeg = None
frame_lock = threading.Lock()
frame_cond = threading.Condition(frame_lock)  # Condition — издатель и подписчики
frame_seq = 0  # Счётчик кадров для отслеживания новых кадров

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Camera</title>
    <style>
        body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        img { max-width: 100%; max-height: 100vh; }
    </style>
</head>
<body>
    <img src="/video">
</body>
</html>
"""

def set_frame(frame):
    """Основной код вызывает эту функцию чтобы обновить кадр"""
    global current_jpeg, frame_seq
    if frame is None:
        return False
    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return False
    jpeg_bytes = jpeg.tobytes()
    with frame_cond:
        current_jpeg = jpeg_bytes
        frame_seq += 1  # Увеличиваем счётчик для отслеживания новых кадров
        frame_cond.notify_all()  # Будим всех ждущих зрителей
    return True

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    def generate():
        last_seq = 0
        while True:
            # Ждём новый кадр с таймаутом для поддержки keep-alive
            with frame_cond:
                if frame_seq == last_seq:
                    frame_cond.wait(timeout=1.0)
                jpeg_bytes = current_jpeg
                seq = frame_seq
            
            # Если это тот же кадр - пропускаем
            if jpeg_bytes is None or seq == last_seq:
                continue
            
            last_seq = seq
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            
            # Небольшая задержка чтобы не нагружать сеть
            time.sleep(0.01)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True).start()
    print("Web server started on port 5000")
