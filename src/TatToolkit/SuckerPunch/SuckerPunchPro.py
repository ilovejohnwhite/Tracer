import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from TatToolkit.util import resize_image_with_pad, common_input_validate, HWC3

class SuckerPunchPro:
    def __init__(self, n_clusters=7, smoothing_kernel_size=(9, 9)):
        self.n_clusters = n_clusters
        self.smoothing_kernel_size = smoothing_kernel_size

    def cluster_colors(self, image):
        image = HWC3(image)
        pixels = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(pixels)

        centers = kmeans.cluster_centers_
        labels = kmeans.predict(pixels)
        clustered_image = centers[labels].reshape(image.shape).astype(np.uint8)

        return clustered_image

    def smooth_clusters(self, clustered_image):
        # Create an elliptical/oval kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.smoothing_kernel_size)
        smoothed_image = cv2.morphologyEx(clustered_image, cv2.MORPH_CLOSE, kernel)
        return smoothed_image

    def __call__(self, input_image, output_type="pil", detect_resolution=512):
        input_image, output_type = common_input_validate(input_image, output_type)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, "INTER_CUBIC")

        clustered_image = self.cluster_colors(input_image)
        smoothed_image = self.smooth_clusters(clustered_image)

        smoothed_image = remove_pad(smoothed_image)
        if output_type == "pil":
            processed_image = Image.fromarray(smoothed_image)
        else:
            processed_image = smoothed_image

        return processed_image
