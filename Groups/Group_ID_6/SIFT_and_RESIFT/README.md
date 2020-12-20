# Group ID: 6

This following folder contains a detailed description and implementation of SIFT and RESIFT.

## Project Title:
Feature Extraction from Image

## Algorithms:
Scale Invariant Feature Transform (SIFT) <br>
Reliability weighted Scale Invariant Feature Transform (reSIFT)

## Scale Invariant Feature Transform (SIFT):
SIFT technique is used for extracting image features. The features extracted using
the SIFT algorithm are invariant to image scaling, rotation, the transformation it
means the features should be detectable even in such transformations. It locates the
points on the image, such points are called interest points or key points, they lie on
some of the regions of an image.

## Reliability weighted Scale Invariant Feature Transform (reSIFT):
reSIFT (Reliability-Weighted-Scale Invariant Feature Transformation) is image
quality estimator based on the SIFT descriptor over reliability features. Resift is
used to quantify the perpetual quality of images under noise, compression,
communication types. In resift algorithm the image is first smoothened and
normalized, then on that image the sift algorithm functions, resulting in more
accurate keypoints for features.

## Prerequisite:
The SIFT object works with opencv so run below commands before moving to
code to take care of dependencies: <br><br>
pip install opencv-python <br>
pip install opencv-contrib-python <br><br>
You also need to add the correct image path if you are running on local machine.

## This folder contains:
### 1) Code files Folder
This Folder contains all the files related to our project.
#### 1.1) SIFT and reSIFT Implementation file
This jupyter notebook in which all the code is being merged to
perform the tasks. Firstly, it shows the sift algorithm and then resift
with the help of example images.
#### 1.2) Libs.py
This python file contains the necessary libraries that we used in our
project.
#### 1.3) Make_test_image.py
This python file contains code to create test image by adding Scale
Invariance and Rotational Invariance
#### 1.4) Resift.py
This python file contains the function for resift algorithm.
#### 1.5) Sift.py
This python file contains the function for sift algorithm.
#### 1.6) Ex1.png, ex2.png, ex3.png
They are the example images we have used to show the working of
our project.
#### 1.7) Image_matching.py
This python file is used to perform the matching between the SIFT
descriptors of the training image and the test image. It was not the
necessary part for our project, we simply added this for a better
understanding of the working of the project.

### 2) Code_Documentation.html file
This html file contains documentation of the code, also include the
functions that are being used as well as the description about each code file
used.
### 3) Example.html file
This html file contains an example, showing the process as well as the
working of the project. It includes both the sift and resift implementation
with an input image, processing and final output observed by applying both
the algorithms one by one.
### 4) Report SIFT_and_RESIFT.pdf
This pdf file contains the report on our project. It includes both the
algorithms (sift and resift) in a single file. Each chapter first describes the
sift algorithm followed by resift. In this way, all the topics used in our
project are covered in the same document.

## Work Description:
Our work uses a training image. Then, it creates test image making by
making the image scale and rotation invariant (which is pre-requirement for
applying the algorithms). First, we applied the sift algorithm on the image,
and have used opencv for this. We found the output as the keypoints
detected using the sift. We then used this for image matching. Next, we
worked on resift, in which we used the sift function, and that was defined
earlier and shown the output in the same way as the previous one.

## Work done:
#### 1) Mrunali Patil – 0801CS171045
#### 2) Sarvesh Gupta – 0801CS171068
We have contributed to this project equally. We implemented both the given
tasks (sift and resift) together.
