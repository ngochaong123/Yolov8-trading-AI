# Training an AI Model to Recognize Objects in Videos Using YOLOv8 Oriented Bounding Boxes

This project demonstrates how to use YOLOv8 with Oriented Bounding Boxes (OBB) for precise object detection in video streams, specifically tailored for trading environments. The training process is powered by Roboflow and optimized for NVIDIA GPUs with CUDA support, using Python 3.10.0.

## Key Features

- **YOLOv8 with Oriented Bounding Boxes**: Enhanced object detection for video-based applications.
- **Roboflow Integration**: Simplifies dataset management and training workflows.
- **GPU Acceleration**: Leverages NVIDIA CUDA for faster training and inference.

## Video Demonstration

Watch the implementation in action by clicking the video below:

[![Watch the video](https://img.youtube.com/vi/djSFUutVCRk/maxresdefault.jpg)](https://youtu.be/djSFUutVCRk)

## Webcam Requirements

The project uses the Hikvision DS-U02 Full HD (1920×1080) webcam for real-time video input. Ensure the webcam is connected and recognized by your system.

## Setup Instructions

### Prerequisites

- Python 3.10.0
- NVIDIA GPU with CUDA support

### Installation Steps

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone clone https://github.com/ngochaong123/Yolov8-trading-AI.git
    cd Yolov8-trading-AI
    ```

2. Install the required Python packages:
    ```bash
    pip install ultralytics roboflow
    ```

3. Install the CUDA-compatible version of PyTorch:
    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Dataset Preparation

The dataset is hosted on Roboflow. Use the following script to download it:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="puWNSw1Du2OCFEHTegQs")
project = rf.workspace("trading-al").project("project-trading-yolov5-v1")
version = project.version(3)
dataset = version.download("yolov8-obb")
```

## Training the Model

After downloading the dataset, train the YOLOv8 model with the following command:

```bash
yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```  
