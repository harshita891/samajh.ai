import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
from utils.image_utils import preprocess, get_appearance_features
from utils.bbox_utils import non_max_suppression
from configs.settings import CONF_THRESHOLD, IOU_THRESHOLD, CLASSES
import logging

logger = logging.getLogger(__name__)

class YOLOv8Detector:
    def __init__(self, model_path: str):
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            logger.info(f"Loaded model {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.conf_threshold = CONF_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.classes = CLASSES

    def detect(self, image: np.ndarray) -> List[Tuple]:
        try:
            input_image, scale, orig_shape = preprocess(image)
            outputs = self.session.run(None, {self.input_name: input_image})[0]
            outputs = np.transpose(outputs, (0, 2, 1))

            boxes, scores, class_ids = [], [], []
            for pred in outputs[0]:
                box = pred[:4]
                cls_scores = pred[4:]
                score = np.max(cls_scores)
                if score < self.conf_threshold:
                    continue
                class_id = np.argmax(cls_scores)
                cx, cy, w, h = box
                x1 = (cx - w/2) / scale
                y1 = (cy - h/2) / scale
                x2 = (cx + w/2) / scale
                y2 = (cy + h/2) / scale
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)

            if boxes:
                boxes = np.array(boxes)
                scores = np.array(scores)
                class_ids = np.array(class_ids)
                indices = non_max_suppression(boxes, scores, class_ids, self.iou_threshold)
                detections = [(
                    boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                    scores[i], self.classes[class_ids[i]],
                    get_appearance_features(image, boxes[i])
                ) for i in indices]
                return detections
            return []

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []