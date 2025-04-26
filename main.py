import cv2
import yaml
import time
import os
import torch
from utils.logger import Logger
from utils.drawing import Drawing
from utils.presence_monitor import PresenceMonitor
from detector.yolo_detector import YOLODetector
from tracker.kalman_tracker import ObjectTracker

# Load Config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Components
logger = Logger(log_file=config['log_file'], enable_logging=config['enable_logging'])
drawing = Drawing()
presence_monitor = PresenceMonitor(
    zone=(config['zone']['x1'], config['zone']['y1'], config['zone']['x2'], config['zone']['y2']),
    max_lost_frames=config['max_lost_frames']
)
detector = YOLODetector(model_path="models/yolov8s1.onnx",
                        confidence_threshold=config['confidence_threshold'],
                        device='cuda' if torch.cuda.is_available() else 'cpu')
tracker = ObjectTracker(max_lost_frames=config['max_lost_frames'])

# Setup Video
cap = cv2.VideoCapture(config['video_source'])
if not cap.isOpened():
    raise IOError(f"Cannot open video source {config['video_source']}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or config['target_fps']

if config['output_video']:
    output_path = config['output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
else:
    out = None

prev_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    formatted_detections = []
    for det in detections:
        bbox = det[:4]
        score = det[4]
        class_id = det[5]
        if score >= config['min_new_object_score']:
            formatted_detections.append((bbox, score, class_id))

    active_objects = tracker.update(formatted_detections)
    events = presence_monitor.update(active_objects)

    for event_type, obj_id in events:
        text = f"{event_type.upper()} - ID {obj_id}"
        logger.log_event(text)
        print(text)  # 🔥 Console pe bhi print hoga

    boxes = []
    scores = []
    class_ids = []
    status_flags = []

    for obj_id, bbox in active_objects.items():
        boxes.append(bbox)
        scores.append(1.0)  # Tracker doesn't have score, assume 1.0
        class_ids.append(0)  # Dummy class
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        in_zone = (presence_monitor.zone[0] <= cx <= presence_monitor.zone[2]) and (presence_monitor.zone[1] <= cy <= presence_monitor.zone[3])
        status_flags.append(in_zone)

    frame = drawing.draw_zone(frame, presence_monitor.zone)
    frame = drawing.draw_detections(frame, boxes, scores, class_ids, ["Object"], status_flags)

    # FPS Calculation
    frame_count += 1
    if frame_count >= 10:
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        logger.log_fps(fps)
        frame = drawing.draw_fps(frame, fps)
        prev_time = curr_time
        frame_count = 0

    if out:
        out.write(frame)

    cv2.imshow('Real-Time Monitoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
