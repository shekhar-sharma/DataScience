from libs import *
def SIFT_algo(training_image,training_gray,test_image,test_gray):
    #test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    #training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
    
    # Creating SIFT Object
    sift = cv2.SIFT_create()
    
    # Detecting features
    train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
    test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)
    keypoints_without_size = np.copy(training_image)
    keypoints_with_size = np.copy(training_image)
    
    # Drawing keypoints and extent of their importance as  descriptor
    cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
    cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints_with_size,keypoints_without_size,train_descriptor,train_keypoints,test_descriptor,test_keypoints