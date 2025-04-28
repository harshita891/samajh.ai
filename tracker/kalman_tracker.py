import cv2
import numpy as np
from typing import Dict, List, Tuple
from utils.bbox_utils import get_box_center

class KalmanTracker:
    def __init__(self):
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}

    def update_kalman_filter(self, obj_id: int, box: List[float]) -> np.ndarray:
        if obj_id not in self.kalman_filters:
            kalman = cv2.KalmanFilter(6, 2)
            kalman.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]], np.float32)
            kalman.transitionMatrix = np.array([
                [1,0,1,0,0.5,0],
                [0,1,0,1,0,0.5],
                [0,0,1,0,1,0],
                [0,0,0,1,0,1],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]
            ], np.float32)
            kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
            self.kalman_filters[obj_id] = kalman

        center = get_box_center(box)
        self.kalman_filters[obj_id].correct(np.array([[center[0]], [center[1]]], dtype=np.float32))
        return self.kalman_filters[obj_id].predict()[:2]

    def remove_filter(self, obj_id: int):
        self.kalman_filters.pop(obj_id, None)