from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading

app = Flask(__name__)

current_jpeg = None
frame_lock = threading.Lock()
frame_event = threading.Event()

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
    global current_jpeg
    if frame is None:
        return
    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return
    jpeg_bytes = jpeg.tobytes()
    with frame_lock:
        current_jpeg = jpeg_bytes
    frame_event.set()

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    def generate():
        while True:
            frame_event.wait(timeout=1.0)
            frame_event.clear()
            with frame_lock:
                jpeg_bytes = current_jpeg
            if jpeg_bytes is None:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True).start()
    print("Web server started on port 5000")
