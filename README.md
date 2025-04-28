# YOLOv8 Real-Time Object Detection and Tracking

This project implements a real-time object detection and tracking system using the YOLOv8 model (ONNX format) with OpenCV, NumPy, and ONNX Runtime. It includes object detection, tracking with Kalman filters, appearance-based re-identification, and alerts for missing objects. The application is modularized for maintainability and packaged in a Docker container for easy deployment.

## Features

* **Object Detection:** Uses YOLOv8 to detect objects in real-time from a webcam or video file.
* **Object Tracking:** Tracks objects across frames using Kalman filters and appearance features.
* **Re-identification:** Re-identifies missing objects based on appearance and position.
* **Alerts:** Displays alerts for missing objects with a flashing bounding box.
* **Performance Monitoring:** Shows FPS and object confidence scores.
* **Modular Design:** Code is organized into modules for detection, tracking, utilities, and configuration.
## Prerequisites

* **Docker:** Installed on your system ([Docker Installation Guide](https://docs.docker.com/engine/install/)).
* **Git:** For cloning the repository ([Git Installation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)).
* **Webcam (optional):** For real-time detection, ensure a webcam is connected.
* **X11 (for GUI):** Required for OpenCV's `cv2.imshow` on Linux/macOS. On Windows, use WSL2 with X11 forwarding or modify the code for non-GUI output.
* **YOLOv8 ONNX Model:** Download a YOLOv8 model in ONNX format (e.g., `yolov8s.onnx`).




## Setup via GitHub

1.  **Clone the Repository:**
    ```bash
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```
    *(Remember to replace `<REPOSITORY_URL>` and `<REPOSITORY_NAME>` with the actual values)*

2.  **Download YOLOv8 ONNX Model:**
    Obtain a YOLOv8 ONNX model (e.g., `yolov8s.onnx`) from a source like [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) or convert a YOLOv8 model to ONNX format. Place the model in a `models/` directory at the project root:
    ```bash
    mkdir models
    mv yolov8s.onnx models/
    ```
    Alternatively, update the `model_path` in `app/main.py` to point to your model location.

3.  **Verify Directory Structure:**
    Ensure all files are in place as per the project structure above. The repository should include all Python modules, `Dockerfile`, and `requirements.txt`.

## Running with Docker

1.  **Build the Docker Image:**
    From the project root, run:
    ```bash
    docker build -t yolov8-detector .
    ```
    This builds an image based on Python 3.9.13 with all dependencies installed.

2.  **Run the Container:**

    * **With Webcam (for real-time detection):**
        ```bash
        docker run --rm -it --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/models:/app/models yolov8-detector
        ```
        * `--device=/dev/video0`: Grants webcam access.
        * `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`: Enables OpenCV GUI via X11 forwarding.
        * `-v $(pwd)/models:/app/models`: Mounts the model directory.

    * **With Video File:**
        Modify `video_source` in `app/main.py` to the path of your video file (e.g., `/app/video.mp4`), then run:
        ```bash
        docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/models:/app/models -v $(pwd)/video.mp4:/app/video.mp4 yolov8-detector
        ```

3.  **Interact with the Application:**
    The application displays a window with detected objects, bounding boxes, and alerts for missing objects. Press `q` to quit the application. FPS and object details (class, ID, confidence) are shown on the frame.

## Troubleshooting

* **Webcam Access:** Ensure the webcam is connected and accessible (`ls /dev/video*`). Add your user to the video group: `sudo usermod -a -G video $USER`.
* **X11 Forwarding:** On Linux, ensure an X server is running (e.g., Xorg). On macOS, install XQuartz and run `xhost +`. On Windows (WSL2), use an X server like VcXsrv. For non-GUI setups, modify `app/main.py` to save output frames instead of using `cv2.imshow`.
* **Model Not Found:** Verify the `models/` directory contains `yolov8s.onnx` or update the `model_path` in `app/main.py`.
* **Docker Issues:** Ensure Docker is running: `docker info`. Check image build logs for errors if the build fails.

## Dependencies

### Python 3.9.13 Libraries (see `requirements.txt`):

numpy==1.23.5
opencv-python==4.7.0.72
onnxruntime==1.14.1
scipy==1.10.1


### System packages (installed in Docker):

libgl1-mesa-glx
libglib2.0-0


## Notes

* The application uses a modular design for easy maintenance and extension.
* The Docker image is based on `python:3.9.13-slim` for compatibility with the specified Python version.
* For production use, consider optimizing the model or running on a GPU-enabled setup (requires modifying the `Dockerfile` and ONNX Runtime providers).
* The code assumes the YOLOv8 ONNX model is compatible with the input shape (640x640). Adjust `input_shape` in `utils/image_utils.py` if needed.

## Contributing

Feel free to open issues or submit pull requests on the GitHub repository for bug fixes, improvements, or new features.

## License

This project is licensed under the MIT License.
