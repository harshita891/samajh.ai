import cv2
import time
from configs.settings import ALERT_DURATION
import numpy as np
from typing import Tuple    
def monitor_presence(frame: np.ndarray, missing_count: int, alert_active: bool, alert_start_time: float) -> Tuple[np.ndarray, bool, float]:
    h, w = frame.shape[:2]
    if missing_count > 0:
        alert_text = f"ALERT: {missing_count} OBJECT(S) MISSING!"
        (text_width, text_height), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, alert_text, (w//2 - text_width//2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CHECK AREA!", (w//2 - text_width//2 + 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        alert_active = True
        alert_start_time = time.time()
    elif alert_active and (time.time() - alert_start_time) < ALERT_DURATION:
        cv2.putText(frame, "ALERT CLEARED", (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        alert_active = False
    return frame, alert_active, alert_start_time