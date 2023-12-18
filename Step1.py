import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import models
from torchvision.transforms import ToTensor
import yolov8

# 2.1 Image Processing

# Loading the original image from the project folder path
image_path = 'motherboard_image.JPEG'
original_image = cv2.imread(image_path)

# Defining target dimensions for when the code is run
display_width, display_height = 800, 600
target_width, target_height = int(1500), int(2000)  

# Resizing the original image for the final display
resized_image_display = cv2.resize(original_image, (display_width, display_height))

# Resizing the original image for processing
resized_image = cv2.resize(original_image, (target_width, target_height))

# Applyng Gaussian blur for noise reduction
blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)  
# Convering the image to gray to simplify the image
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# Applying thresholding to create a binary image
threshold_value = 100
_, bin_threshold_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Inverting the binary image to create a mask. This is because the initial mask
# was outlining the table.
maskinvert = cv2.bitwise_not(bin_threshold_image)

# Finding contours
contours, _ = cv2.findContours(maskinvert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Creating a copy of the image to draw contours
contoursimg = resized_image.copy()

# Finding the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Creating a mask filled with white using the largest contour
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Creating a copy of the image to draw the largest contour
contoursimg = resized_image.copy()
cv2.drawContours(contoursimg, [largest_contour], -1, (0, 255, 0), 2)

# Extracting the PCB using the created mask
pcbextracted = cv2.bitwise_and(resized_image, resized_image, mask=mask)

# Resizing images for display as they were initially too large
resized_gray_image = cv2.resize(gray, (display_width, display_height))
resized_binary_thresholded_image = cv2.resize(bin_threshold_image, (display_width, display_height))
resized_contour_mask = cv2.resize(mask, (display_width, display_height))
resized_pcb_extracted = cv2.resize(pcbextracted, (display_width, display_height))
resized_image_with_contours = cv2.resize(contoursimg, (display_width, display_height))

# Displaying the resized images
cv2.imshow('Original Image', resized_gray_image)
cv2.imshow('Edge Image', resized_binary_thresholded_image)
cv2.imshow('Contour Mask', resized_contour_mask)
cv2.imshow('Extracted Image', resized_pcb_extracted)
cv2.imshow('Largest Contour', resized_image_with_contours)


cv2.waitKey(0)
cv2.destroyAllWindows()


