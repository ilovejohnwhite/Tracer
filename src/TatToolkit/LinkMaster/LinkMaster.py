import cv2
import numpy as np
from skimage import morphology, segmentation, feature
from skimage.filters import gaussian
from PIL import Image

def create_custom_dashed_line(img, start, end, dash_length_px=7, gap_length_px=7, color=0, thickness=1):
    total_length = np.linalg.norm(end - start)
    num_dashes = int(total_length // (dash_length_px + gap_length_px))

    direction = (end - start) / total_length

    for dash_num in range(num_dashes):
        dash_start = start + (dash_num * (dash_length_px + gap_length_px)) * direction
        dash_end = dash_start + dash_length_px * direction
        cv2.line(img, tuple(np.round(dash_start).astype(int)), tuple(np.round(dash_end).astype(int)), color, thickness)

class LinkMaster:
    def __init__(self):
        pass

    def draw_spaced_outlines(self, img, contour, color=0, dash_length=6, gap_length=10, thickness=1):
        for i in range(1, len(contour)):
            start_point = contour[i-1][0]
            end_point = contour[i][0]
            create_custom_dashed_line(img, start_point, end_point, dash_length, gap_length, color, thickness)

    def draw_solid_outline(self, img, edges, color=0, thickness=1):
        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if edges[y, x]:
                    cv2.circle(img, (x, y), thickness, color, -1)                

    def apply_canny_edge_detection(self, img, sigma=3):
        img_smoothed = gaussian(img, sigma=sigma)
        edges = feature.canny(img_smoothed, sigma=sigma)
        return edges

    def find_edges_and_smooth(self, segmented_image, num_thinning_iterations=3, epsilon=5, thickness=1):
        edges_image = np.full(segmented_image.shape, 255, dtype=np.uint8)
        unique_segments = np.unique(segmented_image)
        sorted_segments = sorted(unique_segments)

        for idx, segment in enumerate(sorted_segments):
            if idx in [1,3,5]: #skipping darkest region off of black and 2nd region off of white (6 regions with normal set to 4 clusters and 1 cluster for each outlier mask, darkest is 0)
                continue
            mask = (segmented_image == segment)
            segment_edges = segmentation.find_boundaries(mask, mode='outer', connectivity=1)
            if segment_edges.ndim == 3:
                segment_edges = segment_edges.any(axis=2)

            contours, _ = cv2.findContours(segment_edges.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)            
            for contour in contours:
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if idx != 0:  #which region has the solid outline
                    self.draw_spaced_outlines(edges_image, approx, thickness=thickness)
                else:  
                    cv2.drawContours(edges_image, [approx], -1, (0), thickness)

        return edges_image

    def __call__(self, input_image, output_type="np", detect_resolution=None):
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        processed_image = morphology.opening(input_image, morphology.disk(5))

        edges_image = self.find_edges_and_smooth(processed_image)

        if output_type == "pil":
            edges_image = Image.fromarray((edges_image * 255).astype(np.uint8))

        return edges_image
