import numpy as np
from typing import List, Tuple

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float) -> List[int]:
    indices = []
    unique_classes = np.unique(class_ids)
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        if len(cls_boxes) == 0:
            continue
        x1, y1, x2, y2 = cls_boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = cls_scores.argsort()[::-1]
        while len(order) > 0:
            i = order[0]
            indices.append(np.where(cls_mask)[0][i])
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            keep = np.where(iou <= iou_threshold)[0]
            order = order[keep + 1]
    return indices

def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

def get_box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compute_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)