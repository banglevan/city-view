# City View - YOLO & T2I Crowd Detection System

A comprehensive crowd detection and counting system using both YOLO (You Only Look Once) object detection and T2I (Text-to-Image) zero-shot counting models, specifically designed for monitoring urban areas and public spaces.

## 🚀 Features

### YOLO Crowd Detection
- **Real-time Detection**: Detect and count people in images and videos
- **High Accuracy**: Uses YOLO-Crowd model optimized for crowd detection
- **Multiple Input Sources**: Support for images, videos, webcam, and RTSP streams
- **Configurable Parameters**: Easy configuration through YAML files
- **Cross-platform**: Works on Windows, Linux, and macOS
- **GPU Acceleration**: CUDA support for faster inference

### T2I Zero-Shot Counting
- **Zero-Shot Learning**: Count objects without training on specific classes
- **Text-Guided Counting**: Use natural language prompts to count objects
- **High Resolution Support**: Handles images with height of 384px while maintaining aspect ratio
- **Attention-Based**: Uses CLIP tokenizer for cross-modal attention
- **Density Map Generation**: Produces density maps for precise counting
- **Flexible Object Classes**: Count any object type through text prompts

## 🧮 MPCount (Patch-based Crowd Counting)

MPCount là module đếm đám đông dựa trên chia ảnh thành các patch lớn, phù hợp cho ảnh độ phân giải cao hoặc đám đông dày đặc.

### Cách sử dụng

#### 1. Chạy bằng dòng lệnh

```bash
python mp_count/inference.py --img_path "path/to/image.jpg" --model_path "weights/sta.pth" --save_path "results.txt" --vis_dir "visualize" --unit_size 16 --patch_size 3584 --log_para 1000 --device "cuda"
```

#### 2. Sử dụng file cấu hình YAML

Tạo file `mp_count/configs/config.yaml` với nội dung:

```yaml
img_path: "D:\\city-view\\yolo_crowd\\data\\Crowd_in_street.jpg"  # Đường dẫn ảnh hoặc thư mục ảnh
model_path: "weights/sta.pth"  # Đường dẫn file trọng số model
save_path: null                # File lưu kết quả dự đoán (mặc định: không lưu)
vis_dir: "visualize"           # Thư mục lưu ảnh trực quan hóa kết quả
unit_size: 16                  # Kích thước đơn vị resize (thường để 16)
patch_size: 3584               # Kích thước patch (giảm nếu bị OOM)
log_para: 1000                 # Tham số log transform (thường để 1000)
device: "cuda"                 # Thiết bị chạy model ("cuda" hoặc "cpu")
```

Sau đó chạy:
```bash
python mp_count/inference.py --config mp_count/configs/config.yaml
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.7+
- OpenCV 4.5+
- Transformers 4.0+
- PIL (Pillow)
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
   # YOLO model weights should be placed in the yolo_crowd/weights/ directory
   # yolo-crowd.pt - Main crowd detection model
   
   # T2I model weights should be placed in the t2i_vlm_crowd/weights/ directory
   # best_model_paper.pth - T2I counting model
   ```

## 🎯 Quick Start

### YOLO Crowd Detection

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

### T2I Zero-Shot Counting

1. **Count objects using text prompts**
   ```python
   from t2i_vlm_crowd.inference_t2i import T2ICountInference
   
   # Initialize T2I model
   t2i_inference = T2ICountInference()
   
   # Count people in an image
   t2i_inference.inference("path/to/image.jpg", "person")
   
   # Count cars in an image
   t2i_inference.inference("path/to/image.jpg", "car")
   
   # Count any object
   t2i_inference.inference("path/to/image.jpg", "strawberries")
   ```

2. **Using the T2ICountInference class**
   ```python
   from t2i_vlm_crowd.inference_t2i import T2ICountInference
   
   # Initialize with custom model path
   t2i_inference = T2ICountInference(model_path='path/to/model.pth')
   
   # The model automatically:
   # - Resizes images to height=384px while maintaining aspect ratio
   # - Processes text prompts using CLIP tokenizer
   # - Generates attention masks for cross-modal attention
   # - Returns density maps and count predictions
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

### YOLO Configuration

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

### T2I Configuration

T2I model parameters can be adjusted in the `T2ICountInference` class:

```python
# Key parameters:
crop_size = 384       # Patch size for processing large images
batch_size = 16       # Batch size for inference
target_height = 384   # Target image height (maintains aspect ratio)
```

## 📁 Project Structure

```
city-view/
├── yolo_crowd/              # YOLO crowd detection package
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
├── t2i_vlm_crowd/           # T2I zero-shot counting package
│   ├── inference_t2i.py     # Main T2I inference script
│   ├── models/              # T2I model components
│   │   ├── reg_model.py     # Regression model for counting
│   │   ├── decoder.py       # Decoder components
│   │   └── diff_unet.py     # Diffusion UNet
│   ├── utils/               # T2I utilities
│   │   ├── tools.py         # Patch extraction and reassembly
│   │   ├── helper.py        # Helper functions
│   │   └── trainer.py       # Training utilities
│   ├── configs/             # T2I model configurations
│   │   ├── v1-inference.yaml # Inference configuration
│   │   └── v1-5-pruned-emaonly.ckpt # Model checkpoint
│   ├── ldm/                 # Latent Diffusion Model components
│   │   ├── models/          # LDM model definitions
│   │   └── modules/         # LDM modules
│   └── weights/             # T2I model weights
│       └── best_model_paper.pth # Pre-trained T2I model
├── .gitignore              # Git ignore rules
├── .gitmodules             # Git submodules configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration Parameters

### YOLO Inference Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imgsz` | int | `640` | Input image size (pixels) |
| `conf_thres` | float | `0.35` | Confidence threshold |
| `iou_thres` | float | `0.45` | IoU threshold for NMS |
| `classes` | int | `0` | Filter by class (0 = person) |
| `agnostic_nms` | bool | `True` | Class-agnostic NMS |
| `device` | str | `'0'` | CUDA device (use 'cpu' for CPU) |

### T2I Model Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crop_size` | int | `384` | Patch size for processing |
| `batch_size` | int | `16` | Batch size for inference |
| `target_height` | int | `384` | Target image height |
| `device` | str | `'cuda'` | Device for inference |

## 🎯 Use Cases

### YOLO Crowd Detection
- **Real-time monitoring**: Live video streams from cameras
- **Event management**: Crowd counting at events and gatherings
- **Safety monitoring**: Occupancy limits and social distancing
- **Traffic analysis**: Pedestrian flow in urban areas

### T2I Zero-Shot Counting
- **Flexible object counting**: Count any object type through text prompts
- **High-precision counting**: Density map-based approach
- **No training required**: Zero-shot learning capabilities
- **Research applications**: Novel object counting scenarios

## 🔍 Model Comparison

| Feature | YOLO Crowd | T2I Zero-Shot |
|---------|------------|---------------|
| **Speed** | Fast real-time | Moderate |
| **Accuracy** | High for people | High for any object |
| **Training** | Requires training data | Zero-shot |
| **Flexibility** | Fixed to trained classes | Any object via text |
| **Input** | Images/videos | Images + text prompts |
| **Output** | Bounding boxes + counts | Density maps + counts |

## 🚀 Advanced Usage

### Combining Both Models

```python
from yolo_crowd.inference_crowd import CrowdInference
from t2i_vlm_crowd.inference_t2i import T2ICountInference

# Initialize both models
yolo_inference = CrowdInference()
t2i_inference = T2ICountInference()

# Use YOLO for real-time people detection
yolo_result = yolo_inference.inference('crowd.jpg')

# Use T2I for specific object counting
t2i_result = t2i_inference.inference('crowd.jpg', 'person')
```

### Custom Text Prompts for T2I

```python
# Count different object types
prompts = [
    "person", "car", "bicycle", "motorcycle",
    "bus", "truck", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
]

for prompt in prompts:
    result = t2i_inference.inference('image.jpg', prompt)
    print(f"Count of {prompt}: {result}")
```

## 📝 Notes

- **Image Resizing**: T2I model automatically resizes images to height=384px while maintaining aspect ratio
- **Memory Usage**: T2I model may require more GPU memory due to transformer architecture
- **Batch Processing**: T2I supports batch processing for multiple images
- **Attention Masks**: T2I uses CLIP tokenizer to generate attention masks for text prompts
- **Refactored Structure**: Project has been refactored into separate `yolo_crowd` and `t2i_vlm_crowd` packages for better organization
