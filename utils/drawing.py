import cv2

class Drawing:
    def __init__(self):
        pass

    def draw_zone(self, frame, zone):
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 0), 2)
        return frame

    def draw_detections(self, frame, boxes, scores, class_ids, class_names, status_flags):
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if status_flags[i] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_names[class_id]} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
