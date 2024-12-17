# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:51:59 2024

@author: almus
"""
#-------------#
# Chapter 07
#-------------#

#%%
# Sobel Edge Detector

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Define a sample 8-bit grayscale image (larger than 3x3 to demonstrate Sobel properly)
# image = np.array([[10, 100, 200, 120, 60],
#                   [150, 50, 80, 90, 30],
#                   [60, 90, 40, 30, 10],
#                   [50, 60, 90, 200, 180],
#                   [30, 40, 10, 120, 250]], dtype=np.uint8)

# image = np.array([[50, 50, 50],
#                   [100, 100, 100],
#                   [150, 150, 150],
#                   [50, 60, 90]], dtype=np.uint8)


      
# Load and convert the image to grayscale
image_path = 'C:/Users/almus/Desktop/cameraman.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Custom Zero Padding Function
def zero_pad(image, pad_height, pad_width):
    # Create a new array of zeros with the desired padded shape
    padded_image = np.zeros((image.shape[0] + 2 * pad_height, image.shape[1] + 2 * pad_width), dtype=image.dtype)
    
    # Place the original image in the center of the padded image
    padded_image[pad_height:pad_height + image.shape[0], pad_width:pad_width + image.shape[1]] = image
    
    return padded_image

# Step 2: Define Gaussian Kernel
def gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size += 1
    
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    normal = 1 / (2.0 * np.pi * sigma**2)
    
    for x in range(-center, center + 1):
        for y in range(-center, center + 1):
            kernel[x + center, y + center] = normal * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / kernel.sum()

# Convolution function using custom zero padding
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = zero_pad(image, pad_height, pad_width)
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(kernel * padded_image[i:i+kernel_height, j:j+kernel_width])
    
    return output

# Apply Zero Padding for Original Image
# Define kernel size
kernel_height, kernel_width = 3, 3
# Calculate padding based on kernel size
pad_height, pad_width = kernel_height // 2, kernel_width // 2
# Apply zero padding to the image
padded_original_image = zero_pad(image, pad_height, pad_width)

print("\nAdd Zero Padding To Original Image:\n", padded_original_image)

# Apply Gaussian Smoothing
gaussian_filter = gaussian_kernel(5, sigma=1.0)
blurred_image = convolve(image, gaussian_filter)

print("\nGaussian Smoothing:\n", gaussian_filter)
print("\nBlurred Image:\n", blurred_image)

# Display Gaussian Filter and Blurred Image
plt.gray()
plt.subplot(121)
plt.imshow(gaussian_filter)
plt.title('Gaussian Filter')
plt.subplot(122)
plt.imshow(blurred_image)
plt.title('Blurred Image')
plt.show()

# Apply Zero Padding for Blurred Image
# Define kernel size
kernel_height, kernel_width = 3, 3
# Calculate padding based on kernel size
pad_height, pad_width = kernel_height // 2, kernel_width // 2
# Apply zero padding to the image
padded_blurred_image = zero_pad(blurred_image, pad_height, pad_width)

print("\nAdd Zero Padding To Blurred Image:\n", padded_blurred_image)

# Step 3: Sobel Filters (Gradient Calculation)
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

# Convolve with Sobel Kernels
Ix = convolve(blurred_image, Kx)
Iy = convolve(blurred_image, Ky)

print("\nGradient in x-Direction (Gx):\n", Ix)
print("\nGradient in y-Direction (Gy):\n", Iy)

# Display Gradients Ix and Iy
plt.subplot(121)
plt.imshow(Ix)
plt.title('Gradient in X (Ix)')
plt.subplot(122)
plt.imshow(Iy)
plt.title('Gradient in Y (Iy)')
plt.show()

# Step 4: Gradient Magnitude and Direction
gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
gradient_direction = np.arctan2(Iy, Ix) * (180 / np.pi)

# Display Gradient Magnitude and Direction
plt.subplot(121)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.subplot(122)
plt.imshow(gradient_direction, cmap='gray')
plt.title('Gradient Direction')
plt.show()

# Step 5: Thresholding
def apply_threshold(magnitude, threshold=100):
    thresholded = np.zeros_like(magnitude)
    thresholded[magnitude > threshold] = 255
    return thresholded

# Apply Thresholding to Gradient Magnitude
threshold = 100  # Set a threshold value
edges = apply_threshold(gradient_magnitude, threshold)

# Display Thresholded Edges
plt.imshow(edges, cmap='gray')
plt.title(f'Thresholded Edges (Threshold = {threshold})')
plt.show()

# Print Gradient Magnitude, Direction, and Thresholded Edges
print("\nGradient Magnitude:\n", gradient_magnitude)
print("\nGradient Direction:\n", gradient_direction)
print("\nThresholded Edges:\n", edges)
