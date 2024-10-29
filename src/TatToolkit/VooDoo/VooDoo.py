import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from TatToolkit.util import resize_image_with_pad, common_input_validate, HWC3

class VooDoo:
    def __init__(self, n_clusters=4, smoothing_kernel_size=(8, 8)):
        self.n_clusters = n_clusters
        self.smoothing_kernel_size = smoothing_kernel_size

    def calculate_luminance(self, image):
        """Convert an RGB image to luminance using standard Y = 0.299*R + 0.587*G + 0.114*B"""
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def cluster_brightness_zone(self, image, mask, n_clusters):
        """Cluster pixels within the specified mask using KMeans"""
        pixels = image[mask].reshape(-1, 3)  # Only consider pixels in the mask
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.predict(pixels)
        clustered_pixels = centers[labels].reshape(image[mask].shape)
        return clustered_pixels

    def merge_clusters(self, image, dark_clusters, mid_clusters, bright_clusters, dark_mask, mid_mask, bright_mask):
        """Merge clustered zones back into the full image"""
        result = np.zeros_like(image)
        result[dark_mask] = dark_clusters
        result[mid_mask] = mid_clusters
        result[bright_mask] = bright_clusters
        return result

    def __call__(self, input_image, output_type="pil", detect_resolution=512):
        input_image, output_type = common_input_validate(input_image, output_type)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, "INTER_CUBIC")

        # Calculate luminance
        luminance = self.calculate_luminance(input_image)

        # Define masks for different brightness zones
        dark_mask = (luminance <= 25)  # 0-5% brightness (0-12.75 in pixel value)
        bright_mask = (luminance >= 220)  # 95-100% brightness (242.25-255)
        mid_mask = (~dark_mask & ~bright_mask)  # Everything in between (5-95%)

        # Cluster each zone separately
        dark_clusters = self.cluster_brightness_zone(input_image, dark_mask, n_clusters=1)  # One cluster for dark areas
        mid_clusters = self.cluster_brightness_zone(input_image, mid_mask, n_clusters=self.n_clusters)  # User adjustable
        bright_clusters = self.cluster_brightness_zone(input_image, bright_mask, n_clusters=1)  # One cluster for bright areas

        # Merge the clusters
        clustered_image = self.merge_clusters(input_image, dark_clusters, mid_clusters, bright_clusters, dark_mask, mid_mask, bright_mask)

        # Smooth the final image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.smoothing_kernel_size)
        smoothed_image = cv2.morphologyEx(clustered_image, cv2.MORPH_CLOSE, kernel)

        smoothed_image = remove_pad(smoothed_image)
        if output_type == "pil":
            processed_image = Image.fromarray(smoothed_image)
        else:
            processed_image = smoothed_image

        return processed_image
