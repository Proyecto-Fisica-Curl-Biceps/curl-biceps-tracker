# config.py

# Ruta al video de entrada
VIDEO_PATH = "videos/curl6.mp4"

# Parámetros de calibración:
#   long_ref_m: longitud real (metros) de un objeto en la escena
#   pix_ref: longitud en píxeles de ese objeto (medido offline)
LONG_REF_M = 0.30        # ej: barra de 30 cm
PIX_REF = 150            # medida en píxeles

# Escala píxeles→metros
PIXEL_TO_METER = LONG_REF_M / PIX_REF

# MediaPipe / OpenCV settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_FRAMES = None        # None = todo el video

# Landmark indexes (MediaPipe Pose)
LANDMARKS = {
    "RIGHT_SHOULDER": 12,
    "RIGHT_ELBOW":    14,
    "RIGHT_WRIST":    16,
}

# Filtrado sencillo (p. ej. media móvil)
SMOOTHING_WINDOW = 5

# fracción a lo largo del antebrazo donde está la mancuerna:
# 0.0 → codo, 1.0 → muñeca. Ajusta según cómo agarres la pesa.
COM_FRACTION = 1.0 

# Size de las flechas de overlay
ARROW_SIZE: int = 500
