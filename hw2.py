#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:06:07 2025

@author: johnpaulmbagwu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_and_binarize(image_path):
    # Let's Load the grayscale image
    img = cv2.imread('tools.pgm', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Image not found.")
        return None
    
    #Let's Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Let's Use Otsu's thresholding for automatic binary conversion
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_img

def display_image(image, title="Binary Image"):
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Let's Define the image path (Ensure tools.pgm is in the working directory)
image_path = "tools.pgm"

# Let's Convert image to binary
binary_image = preprocess_and_binarize(image_path)


# Let's Display the binary image
if binary_image is not None:
    display_image(binary_image)
