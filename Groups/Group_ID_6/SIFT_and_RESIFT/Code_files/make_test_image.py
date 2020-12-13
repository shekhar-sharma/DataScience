from libs import *
# Create test image by adding Scale Invariance and Rotational Invariance
# Transformation
def make_test_image(training_image):
    test_image = cv2.pyrDown(training_image)
    test_image = cv2.pyrDown(test_image)
    num_rows, num_cols = test_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    return test_image,test_gray