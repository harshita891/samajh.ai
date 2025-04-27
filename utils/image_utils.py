import cv2
import numpy as np
from typing import Tuple
from configs.settings import CONF_THRESHOLD
from utils.logger import setup_logger
import logging
from typing import List
def preprocess(image: np.ndarray, input_shape: Tuple[int, int]=(640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(input_shape[0] / h, input_shape[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    limg = clahe.apply(l)
    processed = cv2.merge((limg, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    input_image = padded.astype(np.float32) / 255.0
    input_image = input_image.transpose(2, 0, 1)
    return np.expand_dims(input_image, axis=0), scale, (h, w)

def get_appearance_features(image: np.ndarray, box: List[float]) -> np.ndarray:
    try:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((64,), dtype=np.float32)
        crop = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, None).flatten()
    except Exception as e:
        logging.warning(f"Failed to compute appearance features: {e}")
        return np.zeros((64,), dtype=np.float32)