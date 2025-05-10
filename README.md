# Trading AI Image Recognition with YOLOv8 and Oriented Bounding Boxes

This project utilizes YOLOv8 for image recognition in trading environments, leveraging Oriented Bounding Boxes (OBB) for precise object detection in video streams. The training pipeline is powered by Roboflow and optimized for NVIDIA GPU acceleration using CUDA and Python 3.10.0.

## Features

- **YOLOv8 with Oriented Bounding Boxes**: Advanced object detection tailored for video-based applications in trading.
- **Roboflow Integration**: Simplifies dataset management, versioning, and training workflow.
- **GPU Acceleration**: Utilizes NVIDIA CUDA for high-speed training and real-time inference.

## Video Demonstration

Click the image below to watch a demonstration of the full implementation:

[![Watch the video](https://img.youtube.com/vi/djSFUutVCRk/maxresdefault.jpg)](https://youtu.be/djSFUutVCRk)

## Webcam Setup

The project uses the Hikvision DS-U02 Full HD (1920×1080) webcam for real-time video input. Ensure the device is properly connected and recognized by your system before starting.

## Setup Guide

### Prerequisites

- Python 3.10.0
- NVIDIA GPU with CUDA support

### Installation Steps

1. Clone the repository and navigate to the directory:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install required Python packages:
    ```bash
    pip install ultralytics roboflow
    ```

3. Install the CUDA-compatible version of PyTorch:
    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Dataset Preparation

The dataset is hosted and managed through Roboflow. Use the script below to download the dataset:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="puWNSw1Du2OCFEHTegQs")
project = rf.workspace("trading-al").project("project-trading-yolov5-v1")
version = project.version(3)
dataset = version.download("yolov8-obb")
```

### Training  
Once the dataset is downloaded, train the YOLOv8 model using the provided dataset:  
```bash  
yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640  
```  
