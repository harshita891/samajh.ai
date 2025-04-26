import onnx
import onnxruntime as ort
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov5.onnx', confidence_threshold=0.5, device='cpu'):
        """
        Initializes the YOLO detector using an ONNX model.
        
        Args:
            model_path (str): Path to the ONNX model file.
            confidence_threshold (float): Confidence threshold for object detection.
            device (str): 'cpu' or 'cuda' for selecting the computation device.
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load the ONNX model using ONNX Runtime
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'])
        
        # Get input/output names for later usage
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, frame):
        """
        Detects objects in the given frame.
        
        Args:
            frame (numpy.ndarray): The input image frame to perform object detection on.
            
        Returns:
            list: A list of detections in the form (x1, y1, x2, y2, confidence, class_id).
        """
        # Preprocess the frame for YOLO (assuming YOLOv5 preprocessing)
        frame_resized = cv2.resize(frame, (640, 640))  # Resize to 640x640 (common input size for YOLO)
        frame_resized = frame_resized / 255.0  # Normalize to [0, 1]
        frame_resized = frame_resized.transpose(2, 0, 1)  # Change to CHW format
        frame_resized = np.expand_dims(frame_resized, axis=0).astype(np.float32)  # Add batch dimension

        # Run inference using ONNX Runtime
        inputs = {self.input_name: frame_resized}
        outputs = self.session.run([self.output_name], inputs)

        # Post-process the model outputs
        detections = outputs[0]  # Assuming the first output is the detections
        
        # Print the output for debugging (shape and content)
        print("Detection Output Shape:", detections.shape)
        print("Detection Output (first example):", detections[0])

        results = []
        for detection in detections[0]:
            # Print each detection for inspection
            print("Detection Data:", detection)

            # Assuming the output contains [x1, y1, x2, y2, confidence, cls_probabilities]
            if len(detection) >= 6:  # Make sure there are enough values in the detection
                x1, y1, x2, y2, conf = detection[:5]
                cls_probabilities = detection[5:]
                cls = np.argmax(cls_probabilities)  # Find the class with the highest probability
                
                if conf >= self.confidence_threshold:
                    results.append((int(x1), int(y1), int(x2), int(y2), float(conf), cls))

        return results
