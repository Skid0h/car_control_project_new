from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading

app = Flask(__name__)

# Сюда будет записываться кадр из основного кода
current_frame = None
frame_lock = threading.Lock()
# Флаг, который будет сообщать циклу, что приехал свежий кадр
new_frame_event = threading.Event() 

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
    global current_frame
    with frame_lock:
        if frame is not None:
            current_frame = frame.copy()
            # Даем сигнал генератору, что кадр обновился
            new_frame_event.set()

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    def generate():
        while True:
            # Спим и ждем, пока set_frame не вызовет new_frame_event.set()
            # Это освобождает 99% ресурсов процессора!
            new_frame_event.wait()
            # Сбрасываем флаг обратно, чтобы уснуть на следующем круге
            new_frame_event.clear()
            
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Чуть снизил качество JPEG с 80 до 70. Разницы глазом не увидишь, 
            # а процессору станет еще легче.
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start():
    """Запуск веб-сервера в отдельном потоке"""
    # Добавил подавление стандартных логов Flask, чтобы они не засоряли консоль
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True).start()
    print("Web server started on port 5000")
