import cv2
import numpy as np
from skimage import morphology, segmentation
from PIL import Image

class KillMe:
    def __init__(self):
        pass

    def process_dark_regions(self, input_image, threshold=20.5, epsilon=2, thickness=2): 
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            luminance = np.dot(input_image[..., :3], [0.299, 0.587, 0.114])
        else:
            luminance = input_image.copy()

        dark_mask = luminance <= threshold

        kernel = np.ones((3,3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        edges_image = np.full(input_image.shape[:2], 255, dtype=np.uint8)

        contours, _ = cv2.findContours(dark_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        min_contour_area = 30
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        for contour in significant_contours:
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            cv2.drawContours(edges_image, [approx], -1, (0), thickness)
            
            cv2.drawContours(edges_image, [approx], -1, (0), max(1, thickness-1))
        #blur on the outlines
        edges_image = cv2.GaussianBlur(edges_image, (1,1), 0)

        return edges_image

    def __call__(self, input_image, output_type="np", detect_resolution=None):
        edges_image = self.process_dark_regions(input_image)

        if output_type == "pil":
            edges_image = Image.fromarray(edges_image)

        return edges_image