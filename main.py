import cv2
import time
import numpy as np
from collections import deque
import gc
from detector.yolo_detector import YOLOv8Detector
from tracker.object_history import ObjectHistory
from utils.logger import setup_logger
from utils.drawing import draw_detections, draw_fps
from utils.presence_monitor import monitor_presence
from configs.settings import CLEANUP_INTERVAL

def main(video_source: str = '0', model_path: str = "models/yolov8s.onnx"):
    logger = setup_logger()
    try:
        detector = YOLOv8Detector(model_path=model_path)
        object_history = ObjectHistory()

        cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
        if not cap.isOpened():
            logger.error("Failed to open video source")
            return

        fps_history = deque(maxlen=30)
        alert_active = False
        alert_start_time = 0
        missing_count = 0

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break

                detections = detector.detect(frame)
                tracked_detections = object_history.assign_tracking_ids(detections, frame)
                object_history.update_object_history(tracked_detections)

                if object_history.frame_count % CLEANUP_INTERVAL == 0:
                    object_history.cleanup_memory()
                    gc.collect()

                frame, alert_active, missing_count = draw_detections(frame, object_history.object_history, alert_active, alert_start_time, missing_count, 0)
                frame, alert_active, alert_start_time = monitor_presence(frame, missing_count, alert_active, alert_start_time)
                fps = 1.0 / (time.time() - start_time)
                fps_history.append(fps)
                frame = draw_fps(frame, np.mean(fps_history))

                cv2.imshow("Advanced Object Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            gc.collect()

    except Exception as e:
        logger.error(f"Main loop failed: {e}")
        raise

if __name__ == "__main__":
    main()