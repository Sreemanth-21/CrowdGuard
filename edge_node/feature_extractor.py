import numpy as np
import cv2
from itertools import combinations

class CrowdFeatureExtractor:
    def __init__(self):
        self.prev_gray = None

    def _compute_density(self, count, frame_shape):
        h, w = frame_shape[:2]
        area = h * w
        return count / area

    def _avg_inter_person_distance(self, centers):
        if len(centers) < 2:
            return 0.0

        distances = []
        for (x1, y1), (x2, y2) in combinations(centers, 2):
            distances.append(np.linalg.norm([x1 - x2, y1 - y2]))

        return float(np.mean(distances))

    def _motion_magnitude(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.prev_gray = gray

        return float(np.mean(mag))

    def extract(self, frame, boxes):
        count = len(boxes)

        centers = [
            ((x1 + x2) / 2, (y1 + y2) / 2)
            for x1, y1, x2, y2 in boxes
        ]

        density = self._compute_density(count, frame.shape)
        avg_distance = self._avg_inter_person_distance(centers)
        motion = self._motion_magnitude(frame)

        crowd_pressure = density * motion

        return {
            "count": count,
            "density": density,
            "avg_distance": avg_distance,
            "motion": motion,
            "pressure": crowd_pressure
        }
