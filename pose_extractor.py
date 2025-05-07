import mediapipe as mp
from config import LANDMARKS

class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame):
        """Devuelve dict con coordenadas {(x_px, y_px), ...} o None."""
        results = self.pose.process(frame[..., ::-1])
        if not results.pose_landmarks:
            return None

        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark
        return {
            name: (lm[idx].x * w, lm[idx].y * h)
            for name, idx in LANDMARKS.items()
        }
