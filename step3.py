# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:12:28 2023

@author: noini
"""
import os
from ultralytics import YOLO
import cv2

# Set OpenMP environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLOv8 model
model = YOLO('AER850Project33/weights/best.pt')  

# Evaluation images
evaluation_folder = 'data/evaluation'  

# Get a list of evaluation image files
evaluation_images = [os.path.join(evaluation_folder, img) for img in os.listdir(evaluation_folder)]

# Loop through each evaluation image
for img_path in evaluation_images:
    
    # Make predictions using the trained model
    results = model.predict(img_path)

    # Display annotated frame
    annotated_frame = results[0].plot()

    # Resize and display the annotated frame using OpenCV
    resized_frame = cv2.resize(annotated_frame, (800, 600))  
    # Display the resized annotated frame in a new window
    cv2.imshow("Resized Annotated Frame", resized_frame)
    
    # Wait for a key press before moving to the next image
    cv2.waitKey(0)

    # Close the current window before moving to the next image
    cv2.destroyAllWindows()

