import numpy as np
import cv2

class pixel_features:

    def mean_pixel_value(self,image):

        # Separate Red, Green, and Blue channel
        b, g, r = cv2.split(image)

        # Find Average pixel value of individual channel
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)

        # Find overall Average
        mean_overall = np.mean(image)

        # Store average values in a numpy array
        mean = np.array([mean_b, mean_g, mean_r, mean_overall])

        return mean


    def median_pixel_value(self,image):

        # Separate Red, Green, and Blue channel
        b, g, r = cv2.split(image)

        # Find Median pixel value of individual channel
        median_b = np.median(b)
        median_g = np.median(g)
        median_r = np.median(r)

        # Find overall Median pixel value
        median_overall = np.median(image)

        # Store median values in a numpy array
        median = np.array([median_b, median_g, median_r, median_overall])

        return median


    def mode_pixel_value(self,image):

        # Separate Red, Green, and Blue channel
        color = ('b', 'g', 'r')
        mode_channels = []
        maximum = 0

        # Find mode values using histogram
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])

            # Find mode value of individual channel
            mode_channels.append(np.where(hist == max(hist))[0][0])

            # Condition to calculate overall mode
            if max(hist) > maximum:
                mode_overall = np.where(hist == max(hist))[0][0]
                maximum = max(hist)

        # Store mode values in a numpy array
        mode = np.array([mode_channels[0], mode_channels[1], mode_channels[2], mode_overall])

        return mode