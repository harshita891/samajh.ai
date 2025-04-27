import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment
from utils.bbox_utils import compute_iou, get_box_center, compute_distance
from configs.settings import TRACKING_IOU_THRESHOLD, TEMPORAL_WINDOW, MISSING_TOLERANCE, NEW_TOLERANCE, REID_THRESHOLD, MAX_OBJECTS, CLEANUP_INTERVAL
from tracker.kalman_tracker import KalmanTracker
import cv2

class ObjectHistory:
    def __init__(self):
        self.detection_history = deque(maxlen=TEMPORAL_WINDOW)
        self.object_history: Dict[int, Dict] = {}
        self.kalman_tracker = KalmanTracker()
        self.next_id = 0
        self.frame_count = 0
        self.tracking_iou_threshold = TRACKING_IOU_THRESHOLD
        self.missing_tolerance = MISSING_TOLERANCE
        self.new_tolerance = NEW_TOLERANCE
        self.reid_threshold = REID_THRESHOLD
        self.max_objects = MAX_OBJECTS
        self.cleanup_interval = CLEANUP_INTERVAL

    def assign_tracking_ids(self, detections: List[Tuple], image: np.ndarray) -> List[Tuple]:
        if not self.detection_history:
            return self._initialize_new_ids(detections)

        prev_detections = self.detection_history[-1] if self.detection_history else []
        cost_matrix = np.full((len(detections), len(prev_detections)), np.inf)

        for i, curr_det in enumerate(detections):
            curr_box = curr_det[:4]
            curr_features = curr_det[6]
            for j, prev_det in enumerate(prev_detections):
                prev_box = prev_det[:4]
                prev_features = prev_det[6]
                if curr_det[5] != prev_det[5]:
                    continue
                predicted_center = self.kalman_tracker.update_kalman_filter(prev_det[7], prev_box)
                curr_center = get_box_center(curr_box)
                distance = compute_distance(predicted_center, curr_center)
                feature_sim = cv2.compareHist(curr_features, prev_features, cv2.HISTCMP_CORREL)
                size_sim = 1 - abs(
                    (curr_box[2]-curr_box[0])*(curr_box[3]-curr_box[1]) -
                    (prev_box[2]-prev_box[0])*(prev_box[3]-prev_box[1])
                ) / max(
                    (curr_box[2]-curr_box[0])*(curr_box[3]-curr_box[1]),
                    (prev_box[2]-prev_box[0])*(prev_box[3]-prev_box[1])
                )
                iou = compute_iou(curr_box, prev_box)
                cost_matrix[i, j] = (
                    0.3 * (distance / 100) +
                    0.3 * (1 - feature_sim) +
                    0.2 * (1 - size_sim) +
                    0.2 * (1 - iou)
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < 0.7]

        tracked_detections = []
        used_current = set()
        used_previous = set()

        for i, j in matches:
            prev_id = prev_detections[j][7]
            tracked_detections.append((*detections[i][:6], detections[i][6], prev_id))
            self.kalman_tracker.update_kalman_filter(prev_id, detections[i][:4])
            used_current.add(i)
            used_previous.add(j)

        for i in range(len(detections)):
            if i not in used_current:
                matched_id = self._reidentify_missing(detections[i], image)
                if matched_id is not None:
                    tracked_detections.append((*detections[i][:6], detections[i][6], matched_id))
                    self.kalman_tracker.update_kalman_filter(matched_id, detections[i][:4])
                elif len(self.object_history) < self.max_objects:
                    tracked_detections.append((*detections[i][:6], detections[i][6], self.next_id))
                    self.next_id += 1

        self.detection_history.append(tracked_detections)
        return tracked_detections

    def update_object_history(self, detections: List[Tuple]) -> None:
        current_ids = {det[7] for det in detections}
        for det in detections:
            obj_id = det[7]
            x1, y1, x2, y2 = det[:4]
            score = det[4]
            cls = det[5]
            feature = det[6]
            if obj_id not in self.object_history:
                self.object_history[obj_id] = {
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'frames_visible': 1,
                    'frames_missing': 0,
                    'confidence': 1.0,
                    'status': 'new',
                    'current_box': (x1, y1, x2, y2),
                    'class': cls,
                    'feature': feature
                }
            else:
                hist = self.object_history[obj_id]
                hist['last_seen'] = self.frame_count
                hist['frames_visible'] += 1
                hist['frames_missing'] = 0
                hist['confidence'] = min(1.0, hist['confidence'] + 0.1)
                hist['current_box'] = (x1, y1, x2, y2)
                hist['class'] = cls
                hist['feature'] = feature
                if hist['status'] == 'new' and hist['frames_visible'] > self.new_tolerance:
                    hist['status'] = 'tracking'

        for obj_id in list(self.object_history.keys()):
            if obj_id not in current_ids:
                hist = self.object_history[obj_id]
                hist['frames_missing'] += 1
                hist['confidence'] = max(0.0, hist['confidence'] - 0.05)
                if obj_id in self.kalman_tracker.kalman_filters:
                    prediction = self.kalman_tracker.kalman_filters[obj_id].predict()
                    last_box = hist['current_box']
                    w = last_box[2] - last_box[0]
                    h = last_box[3] - last_box[1]
                    new_x1 = prediction[0][0] - w/2
                    new_y1 = prediction[1][0] - h/2
                    new_x2 = prediction[0][0] + w/2
                    new_y2 = prediction[1][0] + h/2
                    hist['current_box'] = (new_x1, new_y1, new_x2, new_y2)
                if hist['frames_missing'] > self.missing_tolerance:
                    hist['status'] = 'missing'
                if hist['confidence'] <= 0 or len(self.object_history) > self.max_objects:
                    del self.object_history[obj_id]
                    self.kalman_tracker.remove_filter(obj_id)

        self.frame_count += 1

    def cleanup_memory(self) -> None:
        expired_ids = [
            obj_id for obj_id, hist in self.object_history.items()
            if self.frame_count - hist['last_seen'] > self.missing_tolerance * 3
        ]
        for obj_id in expired_ids:
            self.object_history.pop(obj_id, None)
            self.kalman_tracker.remove_filter(obj_id)

        while len(self.object_history) > self.max_objects:
            oldest_id = min(self.object_history, key=lambda k: self.object_history[k]['last_seen'])
            self.object_history.pop(oldest_id, None)
            self.kalman_tracker.remove_filter(oldest_id)

    def _reidentify_missing(self, detection: Tuple, image: np.ndarray) -> Optional[int]:
        best_match = None
        best_score = 0
        curr_features = detection[6]
        curr_center = get_box_center(detection[:4])

        for obj_id, hist in self.object_history.items():
            if hist['status'] != 'missing' or self.frame_count - hist['last_seen'] > self.missing_tolerance * 2:
                continue
            last_box = hist['current_box']
            predicted_center = get_box_center(last_box)
            distance = compute_distance(predicted_center, curr_center)
            feature_sim = cv2.compareHist(curr_features, hist['feature'], cv2.HISTCMP_CORREL)
            score = 0.4 * (1 - distance/100) + 0.6 * feature_sim
            if score > self.reid_threshold and score > best_score:
                best_score = score
                best_match = obj_id

        return best_match if best_score > self.reid_threshold else None

    def _initialize_new_ids(self, detections: List[Tuple]) -> List[Tuple]:
        return [(*det[:6], det[6], self.next_id + i) for i, det in enumerate(detections)]