import cv2
import numpy as np
from skimage.filters import prewitt_h, prewitt_v

class Edge_Features:
    def edge_detection_canny(self, image, lower=None, upper=None):

        #Calculate median pixel value
        med_value = np.median(image)

        if lower is None:
            # Lower bound is either 0 or 70% of the median value, whichever is higher.
            lower = int(max(0, 0.7 * med_value))

        if upper is None:
            # Upper bound is either 255 or 30% above the median value, whichever is lower.
            upper = int(min(255, 1.3 * med_value))

        # It helps to blur the image first, so we don't pick up minor edges.
        blurred_image = cv2.blur(image, ksize=(5,5))

        edges = cv2.Canny(blurred_image, lower, upper)

        return edges

    def horizontal_edge_prewitt(self, image):
        return prewitt_h(image)

    def vertical_edge_prewitt(self, image):
        return prewitt_v(image)