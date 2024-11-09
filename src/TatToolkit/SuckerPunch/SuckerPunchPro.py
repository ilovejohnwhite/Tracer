import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from TatToolkit.util import resize_image_with_pad, common_input_validate, HWC3

class SuckerPunchPro:
    def __init__(self, n_clusters=8, smoothing_kernel_size=(10, 10)):
        self.n_clusters = n_clusters
        self.smoothing_kernel_size = smoothing_kernel_size

    def calculate_luminance(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def cluster_brightness_zone(self, image, mask, n_clusters):
        pixels = image[mask].reshape(-1, 3)  
        if pixels.size == 0:
            return np.zeros(image[mask].shape)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.predict(pixels)
        clustered_pixels = centers[labels].reshape(image[mask].shape)
        return clustered_pixels

    def merge_clusters(self, image, dark_clusters, mid_clusters, bright_clusters, dark_mask, mid_mask, bright_mask):
        result = np.zeros_like(image)
        result[dark_mask] = dark_clusters
        result[mid_mask] = mid_clusters
        result[bright_mask] = bright_clusters
        return result

    def __call__(self, input_image, output_type="pil", detect_resolution=512):
        input_image, output_type = common_input_validate(input_image, output_type)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, "INTER_CUBIC")

        luminance = self.calculate_luminance(input_image)

        dark_mask = (luminance <= 12.75)  # 0-5% brightness (0-12.75 in pixel value)
        bright_mask = (luminance >= 242.25)  # 95-100% brightness (242.25-255)
        mid_mask = (~dark_mask & ~bright_mask)  # Everything in between (5-95%)

        dark_clusters = self.cluster_brightness_zone(input_image, dark_mask, n_clusters=1)  
        mid_clusters = self.cluster_brightness_zone(input_image, mid_mask, n_clusters=self.n_clusters)  
        bright_clusters = self.cluster_brightness_zone(input_image, bright_mask, n_clusters=1)  

        clustered_image = self.merge_clusters(input_image, dark_clusters, mid_clusters, bright_clusters, dark_mask, mid_mask, bright_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.smoothing_kernel_size)
        smoothed_image = cv2.morphologyEx(clustered_image, cv2.MORPH_CLOSE, kernel)

        smoothed_image = remove_pad(smoothed_image)
        if output_type == "pil":
            processed_image = Image.fromarray(smoothed_image)
        else:
            processed_image = smoothed_image

        return processed_image
