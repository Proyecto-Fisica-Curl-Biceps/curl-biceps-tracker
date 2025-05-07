import cv2
from config import VIDEO_PATH, FRAME_WIDTH, FRAME_HEIGHT, MAX_FRAMES



def iter_frames():
    """Devuelve (t, frame) usando el timestamp real del frame."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # t en segundos según el timestamp del vídeo
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0

        yield t, frame

    cap.release()

