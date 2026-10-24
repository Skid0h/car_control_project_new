from flask import Flask, Response, render_template_string
import logging
import threading

import cv2
import numpy as np

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Последний кадр и общий JPEG-кэш для всех подключённых браузеров.
current_frame = None
current_frame_id = 0
current_jpeg = None
encoded_frame_id = -1
active_streams = 0
frame_condition = threading.Condition()
encoder_thread = None
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 80]

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
    """Публикует новый кадр; JPEG-кодирование выполнит отдельный поток."""
    global current_frame, current_frame_id

    if frame is None:
        return

    with frame_condition:
        current_frame = frame.copy()
        current_frame_id += 1
        frame_condition.notify_all()


def _encode_loop():
    """Кодирует только свежие кадры и только пока к потоку подключён клиент."""
    global current_jpeg, encoded_frame_id

    last_frame_id = -1
    while True:
        with frame_condition:
            frame_condition.wait_for(
                lambda: (
                    active_streams > 0
                    and current_frame is not None
                    and current_frame_id != last_frame_id
                )
            )
            frame = current_frame.copy()
            frame_id = current_frame_id

        try:
            success, jpeg = cv2.imencode(".jpg", frame, JPEG_PARAMS)
        except Exception:
            logger.exception("Не удалось закодировать кадр веб-стрима")
            last_frame_id = frame_id
            continue

        last_frame_id = frame_id
        if not success:
            logger.warning("OpenCV не смог закодировать кадр веб-стрима")
            continue

        with frame_condition:
            current_jpeg = jpeg.tobytes()
            encoded_frame_id = frame_id
            frame_condition.notify_all()


def _start_encoder():
    """Запускает единственный кодировщик, даже если start() вызван повторно."""
    global encoder_thread

    with frame_condition:
        if encoder_thread is not None and encoder_thread.is_alive():
            return
        encoder_thread = threading.Thread(
            target=_encode_loop,
            name="web-jpeg-encoder",
            daemon=True,
        )
        encoder_thread.start()


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video")
def video():
    def generate():
        global active_streams

        last_jpeg_id = -1
        with frame_condition:
            active_streams += 1
            frame_condition.notify_all()

        try:
            while True:
                with frame_condition:
                    frame_condition.wait_for(
                        lambda: (
                            current_jpeg is not None
                            and encoded_frame_id != last_jpeg_id
                        )
                    )
                    jpeg = current_jpeg
                    last_jpeg_id = encoded_frame_id

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
        finally:
            with frame_condition:
                active_streams = max(0, active_streams - 1)
                frame_condition.notify_all()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start():
    """Запускает кодировщик и веб-сервер в отдельных фоновых потоках."""
    _start_encoder()
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False,
        ),
        name="web-server",
        daemon=True,
    ).start()
    logger.info("Web server started on port 5000")
