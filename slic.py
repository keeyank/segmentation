# Import necessary libraries
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage import segmentation
import matplotlib.pyplot as plt

def display_two(img1, img2):
   plt.figure(figsize=(10,5))
   plt.subplot(121), plt.imshow(img1), plt.axis('off')
   plt.subplot(122), plt.imshow(img2), plt.axis('off')
   plt.tight_layout()
   plt.show()

# Load and display original image
image_path = './thin-sections/w15/w15_composite.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Downscale by factor of 4
height, width = img_rgb.shape[:2]
img_small = cv2.resize(img_rgb, (width//4, height//4))

print(f"Original size: {img_rgb.shape}")
print(f"Resized to: {img_small.shape}")

plt.figure(figsize=(10,10))
plt.imshow(img_small)
plt.title('Resized Image')
plt.show()

img_small_preprocessed = cv2.bilateralFilter(img_small, 7, 75, 75)
img_small_preprocessed = cv2.bilateralFilter(img_small_preprocessed, 7, 75, 75)
display_two(img_small, img_small_preprocessed)

def visualize_segments(image, segments):
    """
    Visualize original image, segments, and boundaries
    """
    # Create boundary image
    boundaries = segmentation.find_boundaries(segments)
    img_with_boundaries = image.copy()
    img_with_boundaries[boundaries] = [255, 0, 0]  # Red boundaries
    
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original')
    
    plt.subplot(132)
    plt.imshow(segments, cmap='nipy_spectral')
    plt.title(f'SLIC Segments\n{len(np.unique(segments))} segments')
    
    plt.subplot(133)
    plt.imshow(img_with_boundaries)
    plt.title('Boundaries')
    
    plt.tight_layout()
    plt.show()

# Apply SLIC
segments = slic(img_small_preprocessed, n_segments=10, compactness=10, sigma=5, enforce_connectivity=True)
visualize_segments(img_small_preprocessed, segments)