'''
   STUDENT NAME: Poojit Maddineni (NUID:002856550)
   STUDENT NAME: Ruohe Zhou (NUID: 002747606)
   DATE: 04-April, 2024
   DESCRIPTION: Extension 1 : Analyze and visualize the 1st layer of Resnet18.
'''

# Import the necessary libraries
import torch
import torchvision.models as models
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Load ResNet18 model
model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
conv1_weights = model.conv1.weight.data
filters = conv1_weights.cpu().numpy()

# Visualize the filters
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[0]:
        ax.imshow(filters[i, 0], cmap='gray')  
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

# Path to the sample image
image_path = './greek_train/alpha/alpha_001.png'  

# Check if the image file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The specified image file does not exist: {image_path}")

# Load the image and convert it to grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Resize the image to match the input size of ResNet18
image = cv2.resize(image, (224, 224))  

# Apply each filter to the image and store the filtered images
filtered_images = []
with torch.no_grad():
    for i in range(filters.shape[0]):
        kernel = filters[i, 0]
        filtered_image = cv2.filter2D(image, -1, kernel)
        filtered_images.append(filtered_image)

# Visualize the filtered images
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    if i < len(filtered_images):
        ax.imshow(filtered_images[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
