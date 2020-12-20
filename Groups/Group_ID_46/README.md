### FEATURE EXTRACTION FROM IMAGE (PIXEL AVERAGE-MEDIAN-MODE AND EDGE FEATURES)

The purpose of this study is to extract and analyse pixel and edge features of an image.

<b>Software and Libraries:</b>

- Python 2.7 or higher
- NumPy
- Matplotlib
- OpenCV2
- Pandas
- Scikit-image


<b>Project Structure</b>

This project has two major parts :

<b>pixel_features.py -</b> This code contains class for extracting features related to pixel of an image like Mean, Median, Mode of different channels (R,G,B) and also for entire image.

<b>Edge_Features.py -</b> This code contains class for extracting edge features of an image using Canny Edge Detection algorithm and Prewitt's horizontal and vertical edge detector.

- We showcase the usage of above mentioned classes with examples in different .ipynb files named as:
  - pixel_mean.ipynb
  - pixel_median.ipynb
  - pixel_mode.ipynb
  - Edge_Detection_Canny.ipynb
  - Edge_Features_Prewitt.ipynb
 
<b>Running the project:</b>

1.Ensure that feature extractor package and example images are stored in same directory in which you are working.

2.Run all .ipynb file one by one in Jupyter Notebook to see the application of feature extractor packages.