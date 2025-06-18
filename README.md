# City View - YOLO Crowd Detection System

A real-time crowd detection and counting system using YOLO (You Only Look Once) object detection model, specifically designed for monitoring urban areas and public spaces.

## 🚀 Features

- **Real-time Detection**: Detect and count people in images and videos
- **High Accuracy**: Uses YOLO-Crowd model optimized for crowd detection
- **Multiple Input Sources**: Support for images, videos, webcam, and RTSP streams
- **Configurable Parameters**: Easy configuration through YAML files
- **Cross-platform**: Works on Windows, Linux, and macOS
- **GPU Acceleration**: CUDA support for faster inference

## 📋 Requirements

- Python 3.8+
- PyTorch 1.7+
- OpenCV 4.5+
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd city-view
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**
   ```bash
   # The model weights should be placed in the yolo_crowd/weights/ directory
   # yolo-crowd.pt - Main crowd detection model
   ```

## 🎯 Quick Start

### Basic Usage

1. **Detect people in an image**
   ```bash
   python yolo_crowd/inference_crowd.py
   ```

2. **Using the CrowdInference class**
   ```python
   from yolo_crowd.inference_crowd import CrowdInference
   
   # Initialize with default config
   crowd_inference = CrowdInference()
   crowd_inference.load_model()
   
   # Run inference on image
   result = crowd_inference.inference('path/to/image.jpg')
   cv2.imshow('Result', result)
   cv2.waitKey(0)
   ```

### Using Configuration

The system supports configuration through YAML files:

```python
from yolo_crowd.config.get_config import get_inference_config

# Load configuration
config = get_inference_config()
print(f"Image size: {config['imgsz']}")
print(f"Confidence threshold: {config['conf_thres']}")
```

## ⚙️ Configuration

Edit `yolo_crowd/config/config.yaml` to customize detection parameters:

```yaml
inference:
  imgsz: 640          # Input image size
  conf_thres: 0.35    # Confidence threshold
  iou_thres: 0.45     # IoU threshold for NMS
  classes: 0          # Filter by class (0 = person)
  agnostic_nms: True  # Class-agnostic NMS
  device: '0'         # CUDA device (use 'cpu' for CPU)
```

## 📁 Project Structure

```
city-view/
├── yolo_crowd/              # Main Python package
│   ├── inference_crowd.py   # Main inference script and CrowdInference class
│   ├── models/              # Model definitions
│   │   ├── yolo.py          # YOLO model implementation
│   │   ├── common.py        # Common model components
│   │   ├── experimental.py  # Experimental models
│   │   ├── yolo_crowd.yaml  # YOLO-Crowd model config
│   │   └── hub/             # Model hub configurations
│   ├── utils/               # Utility functions
│   │   ├── datasets.py      # Dataset loading utilities
│   │   ├── general.py       # General utilities
│   │   ├── plots.py         # Plotting utilities
│   │   ├── torch_utils.py   # PyTorch utilities
│   │   └── ...              # Other utility modules
│   ├── config/              # Configuration files
│   │   ├── config.yaml      # Main configuration
│   │   └── get_config.py    # Config loader
│   ├── weights/             # Model weights
│   │   └── yolo-crowd.pt    # Pre-trained crowd detection model
│   └── data/                # Data and sample images
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Configuration Parameters

### Inference Settings (from config.yaml)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imgsz` | int | `640` | Input image size (pixels) |
| `conf_thres` | float | `0.35` | Confidence threshold |
| `iou_thres` | float | `0.45` | IoU threshold for NMS |
| `classes` | int | `0` | Filter by class (0 = person) |
| `agnostic_nms` | bool | `True` | Class-agnostic NMS |
| `device` | str | `'0'` | CUDA device (use 'cpu' for CPU) |
