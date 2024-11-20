import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from TatToolkit.util import resize_image_with_pad, common_input_validate, HWC3

class OkayBuddy:
    def __init__(self, n_clusters=12, smoothing_kernel_size=(3, 3)):
        self.n_clusters = n_clusters
        self.smoothing_kernel_size = smoothing_kernel_size

    def calculate_luminance(self, image):
        # Ensure image is 3D
        if len(image.shape) == 4:
            image = image[0]  # Take first frame if 4D
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    def process_image(self, image, mask):
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.reshape(mask.shape[0], mask.shape[1])
        
        # Create output array
        result = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
        
        # Apply mask
        if mask.any():
            result[mask] = image[mask]
        
        return result

    def __call__(self, input_image, output_type="pil", detect_resolution=512):
        # Ensure input is 3D
        if len(input_image.shape) == 4:
            input_image = input_image[0]
        
        # Convert to uint8 if not already
        if input_image.dtype != np.uint8:
            input_image = (input_image * 255).astype(np.uint8)
        
        # Calculate luminance
        luminance = self.calculate_luminance(input_image)
        
        # Create masks
        dark_threshold = np.percentile(luminance, 10)
        bright_threshold = np.percentile(luminance, 95)
        
        dark_mask = (luminance <= dark_threshold)
        bright_mask = (luminance >= bright_threshold)
        mid_mask = ~(dark_mask | bright_mask)
        
        # Process dark regions
        dark_regions = np.zeros_like(input_image)
        dark_regions[dark_mask] = 0  # Pure black
        
        # Process mid regions (simplified)
        mid_regions = input_image.copy()
        mid_regions[~mid_mask] = 0
        
        # Process bright regions
        bright_regions = np.zeros_like(input_image)
        bright_regions[bright_mask] = 255  # Pure white
        
        # Combine regions
        result = dark_regions + mid_regions + bright_regions
        
        # Apply minimal smoothing
        if self.smoothing_kernel_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.smoothing_kernel_size)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        if output_type == "pil":
            result = Image.fromarray(result)
            
        return result