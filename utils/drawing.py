import cv2
import numpy as np
from typing import Dict, Tuple
import time
def draw_detections(frame: np.ndarray, object_history: Dict, alert_active: bool, alert_start_time: float, missing_count: int, fps: float) -> Tuple[np.ndarray, bool, int]:
    h, w = frame.shape[:2]
    for obj_id, hist in object_history.items():
        if hist['confidence'] < 0.3:
            continue
        x1, y1, x2, y2 = hist['current_box']
        x1 = max(0, min(int(x1), w-1))
        y1 = max(0, min(int(y1), h-1))
        x2 = max(0, min(int(x2), w-1))
        y2 = max(0, min(int(y2), h-1))
        if x2 <= x1 or y2 <= y1:
            continue
        if hist['status'] == 'missing':
            missing_count += 1
            color = (0, 0, 255)
            thickness = 2
            label = f"MISSING {hist['class']} {obj_id}"
            if (time.time() - alert_start_time) % 0.5 < 0.25:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        else:
            color = (0, 255, 0) if hist['status'] == 'new' else (255, 0, 0)
            thickness = 1
            label = f"{hist['class']} {obj_id} ({hist['confidence']:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-30), (x1+text_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, alert_active, missing_count

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame