import onnxruntime as ort
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path='models/yolov8s1.onnx', confidence_threshold=0.5, iou_threshold=0.5, device='cpu'):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, frame):
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Invalid frame received")
            return []

        original_height, original_width = frame.shape[:2]
        
        # Preprocess
        img, scale, pad_w, pad_h = self.letterbox_resize(frame, (640, 640))
        img = img / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Inference
        inputs = {self.input_name: img}
        outputs = self.session.run([self.output_name], inputs)
        detections = outputs[0]

        # Postprocess
        results = []
        for detection in detections[0]:
            if len(detection) >= 6:
                x_center, y_center, w, h, conf = detection[:5]
                cls_probs = detection[5:]
                cls_id = np.argmax(cls_probs)

                if conf >= self.confidence_threshold:
                    # Undo letterboxing
                    x_center = (x_center * 640 - pad_w) / scale
                    y_center = (y_center * 640 - pad_h) / scale
                    w = w * 640 / scale
                    h = h * 640 / scale

                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2

                    # Clamp to image size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(original_width, x2), min(original_height, y2)

                    results.append((x1, y1, x2, y2, float(conf), cls_id))

        # Apply NMS
        results = self.non_max_suppression(results, self.iou_threshold)

        print(f"Detections after NMS: {results}")
        return results

    @staticmethod
    def letterbox_resize(image, new_shape=(640, 640), color=(114, 114, 114)):
        height, width = image.shape[:2]
        new_w, new_h = new_shape

        scale = min(new_w / width, new_h / height)
        resized_w = int(width * scale)
        resized_h = int(height * scale)

        resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
        pad_w = (new_w - resized_w) // 2
        pad_h = (new_h - resized_h) // 2
        canvas[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = resized_image

        return canvas, scale, pad_w, pad_h

    @staticmethod
    def non_max_suppression(detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []

        detections = sorted(detections, key=lambda x: x[4], reverse=True)  # Sort by confidence
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                det for det in detections
                if YOLODetector.compute_iou(best[:4], det[:4]) < iou_threshold
            ]

        return keep

    @staticmethod
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / float(box1_area + box2_area - inter_area)
