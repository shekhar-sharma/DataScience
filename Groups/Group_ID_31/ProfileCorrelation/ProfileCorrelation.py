import math
import sys
import timeit
from PIL import Image
import numpy

class ProfileCorrelation:
    def normalizeArray(self,a):
        """
        Normalize the given array to values between 0 and 1.
        Return a numpy array of floats (of the same shape as given)
        """
        w, h = a.shape
        minval = a.min()
        if minval < 0:  # shift to positive...
            a = a + abs(minval)
        maxval = a.max()  # THEN, get max value!
        new_a = numpy.zeros(a.shape, "d")
        for x in range(0, w):
            for y in range(0, h):
                new_a[x, y] = float(a[x, y]) / maxval
        return new_a


    def correlation(self,input, match):
        """
        Calculate the correlation coefficients between the given pixel arrays.

        input - an input (numpy) matrix representing an image
        match - the (numpy) matrix representing the image for which we are looking

        """
        t = timeit.Timer()
        assert match.shape < input.shape, "Match Template must be Smaller than the input"
        c = numpy.zeros(input.shape)  # store the coefficients...
        mfmean = match.mean()
        iw, ih = input.shape  # get input image width and height
        mw, mh = match.shape  # get match image width and height

        print("Computing Correleation Coefficients...")
        start_time = t.timer()

        for i in range(0, iw):
            for j in range(0, ih):

                # find the left, right, top
                # and bottom of the sub-image
                if i - mw / 2 <= 0:
                    left = 0
                elif iw - i < mw:
                    left = iw - mw
                else:
                    left = i

                right = left + mw

                if j - mh / 2 <= 0:
                    top = 0
                elif ih - j < mh:
                    top = ih - mh
                else:
                    top = j

                bottom = top + mh

                # take a slice of the input image as a sub image
                sub = input[left:right, top:bottom]
                assert sub.shape == match.shape, "SubImages must be same size!"
                localmean = sub.mean()
                temp = (sub - localmean) * (match - mfmean)
                s1 = temp.sum()
                temp = (sub - localmean) * (sub - localmean)
                s2 = temp.sum()
                temp = (match - mfmean) * (match - mfmean)
                s3 = temp.sum()
                denom = s2 * s3
                if denom == 0:
                    temp = 0
                else:
                    temp = s1 / math.sqrt(denom)

                c[i, j] = temp

        elapsed = round(t.timer() - start_time, 2)
        print(f"=> Correlation computed in: {elapsed} seconds")
        print(f"\tMax: {c.max()}\n\tMin: {c.min()}\n\tMean: {c.mean()}")
        return c


    def find_profilecorrelation(self,im1,im2,output_file="CORRELATION.jpg"):
        """Open the image files, and compute their correlation """
        

        # Convert from Image to Numpy array conversion
        f = numpy.asarray(im1)
        w = numpy.asarray(im2)
        corr = self.correlation(f, w)
        c = Image.fromarray(numpy.uint8(self.normalizeArray(corr) * 255))

        print(f"Saving as: {output_file}")
        c.save(output_file)
