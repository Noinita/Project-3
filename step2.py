# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:52:46 2023

@author: noini
"""
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import models
from torchvision.transforms import ToTensor
from ultralytics import YOLO

#  avoid library duplication issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 2: Train YOLO V8 Model

# YOLOv8 model using the yolov8n.pt pretrained model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
    data='C:/Users/noini/Documents/GitHub/Project 3/data/data.yaml',  # Path to YAML file 
    epochs=50,    # Number of training epochs
    batch=4,      # Batch size 
    imgsz=900,    # Size of input images 
    name='AER850Project3'  # Name for the trained model 
)
