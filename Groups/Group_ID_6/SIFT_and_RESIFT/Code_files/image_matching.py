from libs import *
def Image_matching(train_descriptor,train_keypoints,training_image,test_descriptor,test_keypoints,test_gray):
    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    
    # Perform the matching between the SIFT descriptors of the training image and the test image
    matches = bf.match(train_descriptor, test_descriptor)
    
    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key = lambda x : x.distance)
    result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)
    return result,matches