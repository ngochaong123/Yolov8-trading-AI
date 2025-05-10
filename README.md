
# Trading AL Image Recognition with YOLOv8 and Oriented Bounding Boxes  

This project leverages YOLOv8 for image recognition in trading applications, utilizing oriented bounding boxes (OBB) for precise object detection in video data. The training process is powered by Roboflow, optimized for NVIDIA GPU acceleration with CUDA and Python 3.10.0.  

## Features  
- **YOLOv8 with Oriented Bounding Boxes**: Advanced object detection for video-based image recognition.  
- **Roboflow Integration**: Streamlined dataset management and training pipeline.  
- **GPU Acceleration**: Harnesses NVIDIA CUDA for faster training and inference.  

## Video Demonstration  
Watch the video illustrating the actual implementation of the project:  

[![Watch the video](https://img.youtube.com/vi/djSFUutVCRk/maxresdefault.jpg)](https://youtu.be/djSFUutVCRk)  

## Webcam Setup  
This project uses the Hikvision DS-U02 Full HD (1920×1080) webcam for video input. Ensure the webcam is properly connected and recognized by your system.  

## Setup  

### Prerequisites  
1. **Python 3.10.0**: Ensure Python 3.10.0 is installed for CUDA compatibility.  
2. **NVIDIA GPU with CUDA**: Required for GPU acceleration.  

### Installation  
1. Clone the repository and navigate to the project directory:  
    ```bash  
    git clone <repository_url>  
    cd <repository_name>  
    ```  
2. Install the required dependencies:  
    ```bash  
    pip install ultralytics roboflow  
    ```  
3. Install the correct version of PyTorch for CUDA compatibility:  
    ```bash  
    pip uninstall torch torchvision torchaudio -y  
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
    ```  

### Dataset Preparation  
The dataset is managed using Roboflow. Use the following script to download the dataset:  
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

