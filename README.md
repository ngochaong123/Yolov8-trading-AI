# Trading AL Image Recognition with YOLOv8 and Oriented Bounding Boxes  

This project focuses on using YOLOv8 for image recognition in trading applications, leveraging oriented bounding boxes (OBB) for precise object detection in video data. The training process is powered by Roboflow, and the implementation is optimized for NVIDIA GPU acceleration using CUDA with Python 3.10.0.  

## Features  
- **YOLOv8 with Oriented Bounding Boxes**: Enhanced object detection for video-based image recognition.  
- **Roboflow Integration**: Simplified dataset management and training pipeline.  
- **GPU Acceleration**: Utilizes NVIDIA CUDA for faster training and inference.  

## Setup  

### Prerequisites  
1. **Python 3.10.0**: Ensure Python 3.10.0 is installed to support CUDA compatibility.  
2. **NVIDIA GPU with CUDA**: Required for GPU acceleration.  

### Installation  
1. Clone the repository and navigate to the project directory.  
2. Install the required dependencies:  
    ```bash  
    pip install ultralytics roboflow  
    ```  
3. Ensure the correct version of PyTorch is installed for CUDA compatibility:  
    ```bash
    # !pip uninstall torch torchvision torchaudio -y
    # !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
Once the dataset is downloaded, you can train the YOLOv8 model using the provided dataset.  

## Video Demonstration  
For a detailed demonstration of the project, watch the video [here](#).  

## Contact  
Feel free to reach out for further details or collaboration opportunities.  
