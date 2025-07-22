# Real-time Object Tracking with YOLOv8 and StrongSORT

This repository provides a robust Python implementation for real-time object tracking using YOLOv8 for detection and StrongSORT for tracking. The system supports webcam and allows for user-selected single-object tracking with persistent IDs and visual trails.

## Implementation Overview

The tracker integrates YOLOv8 for object detection and StrongSORT for object tracking. On startup, the user selects an object to track by drawing a bounding box. The system then identifies the best matching detection and tracks it across frames, maintaining the object's ID even through brief occlusions. The code is optimized for CUDA enabled GPUs.

Key features:
- Real-time detection and tracking with YOLOv8 and StrongSORT
- User-driven object selection for focused tracking
- Persistent object IDs and visual trails
- GPU acceleration (CUDA required)
- Modular, clean codebase for easy extension

## Requirements

- Python 3.8+
- CUDA-compatible GPU (required for real-time performance)
- Webcam (for live tracking)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Download Required Model Weights

Before running the tracker, download the following model weights and place them in the project root directory:

- **YOLOv8s (detection):** [yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
- **OSNet ReID weights (for StrongSORT):** [osnet_x0_25_msmt17.pt](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v0.2/osnet_x0_25_msmt17.pt)

If any link is broken, refer to the official [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics/releases) and [BoxMOT/StrongSORT releases](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases).

## Usage

### Webcam Tracking
Run the tracker on your default webcam:
```bash
py Object-Tracker.py
```

To use a specific camera:
```bash
py Object-Tracker.py --camera 1
```

### Model Selection and Confidence
You can specify a different YOLOv8 model or adjust the detection confidence threshold:
```bash
py Object-Tracker.py --model yolov8n.pt --confidence 0.5
```

### Controls
- Draw a box to select the object to track
- Press `Enter` to confirm selection
- Press `r` to reset selection
- Press `n` to get the next frame
- Press `q` to quit

## File Structure

```
Object-Tracker-Task/
├── Object-Tracker.py           # Main tracking script
├── requirements.txt            # Python dependencies
├── yolov8s.pt                  # YOLOv8 model (default)
├── yolov8n.pt, yolov8x.pt, ... # Other YOLOv8 models (optional)            
├── osnet_x0_25_msmt17.pt       # ReID weights for StrongSORT
└── Yolov5_StrongSORT_OSNet/    # StrongSORT and dependencies
```

## Deployment
1. Clone the repository and navigate to the project directory.
2. Ensure all required model files are present (see above).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the tracker:
   ```bash
   py Object-Tracker.py
   ```

## Notes
- CUDA is required for real-time performance. The script will raise an error if CUDA is not available.
- You can use different YOLOv8 model weights by specifying the `--model` argument.
- The `Yolov5_StrongSORT_OSNet` directory contains the StrongSORT implementation and its dependencies.

