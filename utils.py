
from collections import deque
from config import SMOOTHING_WINDOW

def smooth_sequence(values):
    """Media móvil simple para secuencias de tuplas (x, y)."""
    window = deque(maxlen=SMOOTHING_WINDOW)
    for v in values:
        window.append(v)
        dims = len(v)
        avg = tuple(
            sum(pt[i] for pt in window) / len(window)
            for i in range(dims)
        )
        yield avg
