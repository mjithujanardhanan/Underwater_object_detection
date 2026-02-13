# Underwater Object Detection and Localization

Real-time underwater object detection and localization system using YOLOv11, optimized for GPUs. This project is designed to detect underwater bots and other relevant objects from an ROV’s point of view.

## 📌 Project Overview

This project focuses on detecting and localizing underwater objects in real-time using a custom-trained YOLOv11 model. The model is fine-tuned on a dataset created from ROV-captured footage in underwater environments.

## 🚀 Features

- ✅ Real-time detection with optimized YOLOv11
- ✅ Custom underwater dataset created and annotated using Roboflow
- ✅ Easy deployment with minimal dependencies

## 🐳 Dataset

- 📸 Captured using an underwater ROV
- 🖼️ Annotated using Roboflow
- 🧊 Contains objects such as underwater bots, debris, and natural formations
- Format: YOLO format (txt + image)

## 🧠 Model

- Architecture: YOLOv11 (nano/small for Jetson compatibility)
- Training Framework: PyTorch
- Optimized with: TensorRT (optional)
- Input Size: 640x640

## 🛠️ Setup

### Requirements

- Python 3.8+
- Any Nvidia Gpu
- Intel Realsense d435i depth camera
- torch, torchvision
- OpenCV
- YOLOv11 repo (https://github.com/ultralytics/yolov5)

### Installation

```bash
git clone https://github.com/yourusername/underwater-object-detection.git
cd underwater-object-detection
pip install -r requirements.txt
