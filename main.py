import cv2
import yaml
import time
import os
import torch
from utils.logger import Logger
from utils.presence_monitor import PresenceMonitor
from detector.yolo_detector import YOLODetector
from tracker.kalman_tracker import ObjectTracker
from utils.drawing import Drawing

# --- Load Config ---
def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Initialize Components ---
def initialize_components(config):
    logger = Logger(log_file=config['log_file'], enable_logging=config['enable_logging'])

    presence_monitor = PresenceMonitor(
        zone=(config['zone']['x1'], config['zone']['y1'], config['zone']['x2'], config['zone']['y2']),
        max_lost_frames=config['max_lost_frames']
    )

    detector = YOLODetector(
        model_path="models/yolov8s1.onnx",
        confidence_threshold=config['confidence_threshold'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    tracker = ObjectTracker(max_lost_frames=config['max_lost_frames'])

    drawer = Drawing()

    return logger, presence_monitor, detector, tracker, drawer

# --- Setup Video ---
def setup_video(config):
    cap = cv2.VideoCapture(config['video_source'])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if config['output_video']:
        output_path = config['output_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), config['target_fps'], (frame_width, frame_height))
    else:
        out = None

    return cap, out, frame_width, frame_height

# --- Handle Events ---
def handle_events(events, logger, overlay_texts):
    timestamp = time.time()
    for event_type, obj_id in events:
        if event_type == 'new':
            logger.log_event(f"NEW OBJECT: ID {obj_id} has entered the monitored zone.")
            overlay_texts.append((f"New Object ID: {obj_id}", timestamp))
        elif event_type == 'missing':
            logger.log_event(f"MISSING OBJECT: ID {obj_id} is no longer detected.")
            overlay_texts.append((f"Missing Object ID: {obj_id}", timestamp))

# --- Draw Overlays ---
def draw_overlay_texts(frame, overlay_texts, overlay_duration=3):
    current_time = time.time()
    y_offset = 50
    active_texts = []

    # Keep overlays that are within the duration limit
    for text, timestamp in overlay_texts:
        if current_time - timestamp <= overlay_duration:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40
            active_texts.append((text, timestamp))

    # Return only active overlays
    return active_texts

# --- Calculate FPS ---
def calculate_fps(frame_count, prev_time, logger):
    if frame_count >= 10:
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        logger.log_fps(fps)
        prev_time = curr_time
        frame_count = 0
    else:
        frame_count += 1
    return frame_count, prev_time

# --- Process Detections ---
def process_detections(detector, tracker, frame, config):
    detections = detector.detect(frame)
    formatted_detections = []
    for det in detections:
        bbox = det[:4]
        score = det[4]
        class_id = det[5]
        if score >= config['confidence_threshold']:
            formatted_detections.append((bbox, score, class_id))

    active_objects = tracker.update(formatted_detections)
    return active_objects, formatted_detections

# --- Clip Bounding Box to Frame ---
def clip_bbox_to_frame(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    # Clip bounding box to frame dimensions
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))
    return [x1, y1, x2, y2]

# --- Main Loop ---
def main():
    config = load_config()
    logger, presence_monitor, detector, tracker, drawer = initialize_components(config)
    cap, out, frame_width, frame_height = setup_video(config)

    prev_time = time.time()
    frame_count = 0
    overlay_texts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detection and Tracking
        active_objects, detections = process_detections(detector, tracker, frame, config)

        # Presence Monitoring
        events = presence_monitor.update(active_objects)
        handle_events(events, logger, overlay_texts)

        # Draw zone
        drawer.draw_zone(frame, (config['zone']['x1'], config['zone']['y1'], config['zone']['x2'], config['zone']['y2']))

        # Handle Detections (draw detection boxes only for newly detected objects)
        for det in detections:
            bbox, score, class_id = det
            # Clip detection bbox to ensure it stays inside the frame
            bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)
            x1, y1, x2, y2 = bbox

            # Draw detection box (Green)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_id} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Handle Tracking (draw tracking boxes for objects that are being tracked)
        for obj_id, bbox in active_objects.items():
            # Clip tracking bbox to ensure it stays inside the frame
            bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)

            x1, y1, x2, y2 = bbox

            # Draw tracking box (Blue)
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw overlays (removes text after duration)
        overlay_texts = draw_overlay_texts(frame, overlay_texts)

        # FPS Calculation
        frame_count, prev_time = calculate_fps(frame_count, prev_time, logger)

        # Show frame
        cv2.imshow('Real-Time Monitoring', frame)
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
