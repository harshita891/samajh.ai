import numpy as np

class KalmanFilter:
    def __init__(self, dt=1, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.R = np.eye(2) * measurement_noise
        self.Q = np.eye(4) * process_noise

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))


class ObjectTracker:
    def __init__(self, max_lost_frames=30):
        self.trackers = {}
        self.lost_frames = {}
        self.max_lost_frames = max_lost_frames
        self.next_id = 0

    def add_object(self, initial_position):
        tracker = KalmanFilter()
        tracker.x[0] = initial_position[0]
        tracker.x[1] = initial_position[1]

        obj_id = self.next_id
        self.trackers[obj_id] = tracker
        self.lost_frames[obj_id] = 0
        self.next_id += 1
        return obj_id

    def update(self, detections):
        updated_objects = {}

        if len(self.trackers) == 0:
            for bbox, score, class_id in detections:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                obj_id = self.add_object((cx, cy))
                updated_objects[obj_id] = bbox
            return updated_objects

        obj_ids = list(self.trackers.keys())
        predicted_positions = []
        for obj_id in obj_ids:
            pred = self.trackers[obj_id].predict()
            predicted_positions.append((obj_id, pred[0, 0], pred[1, 0]))

        assigned = set()
        for bbox, score, class_id in detections:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            min_distance = float('inf')
            best_id = None

            for obj_id, pred_x, pred_y in predicted_positions:
                if obj_id in assigned:
                    continue
                dist = np.sqrt((pred_x - cx)**2 + (pred_y - cy)**2)
                if dist < min_distance:
                    min_distance = dist
                    best_id = obj_id

            if best_id is not None and min_distance < 50:  # Threshold to match existing object
                self.trackers[best_id].update(np.array([[cx], [cy]]))
                updated_objects[best_id] = bbox
                self.lost_frames[best_id] = 0
                assigned.add(best_id)
            else:
                new_id = self.add_object((cx, cy))
                updated_objects[new_id] = bbox

        # Update lost frames
        for obj_id in list(self.trackers.keys()):
            if obj_id not in updated_objects:
                self.lost_frames[obj_id] += 1
                if self.lost_frames[obj_id] > self.max_lost_frames:
                    del self.trackers[obj_id]
                    del self.lost_frames[obj_id]

        return updated_objects
