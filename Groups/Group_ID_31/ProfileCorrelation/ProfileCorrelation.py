import math
import sys
import timeit
from PIL import Image
import numpy as n
from skimage.io import imread, imshow
from skimage import io, feature
from scipy import ndimage


class ProfileCorrelation:
    def normalizeArray(self, a):
        """
        Normalize the given array to values between 0 and 1.
        Return a numpy array of floats (of the same shape as given)
        """
        w, h = a.shape
        minval = a.min()
        if minval < 0:  # shift to positive...
            a = a + abs(minval)
        maxval = a.max()  # THEN, get max value!
        new_a = n.zeros(a.shape, "d")
        for x in range(0, w):
            for y in range(0, h):
                new_a[x, y] = float(a[x, y]) / maxval
        return new_a

    def correlationx(self, patch1, patch2):
        product = n.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product

    def find_profilecorrelation(self, im1, im2, output_file="CORRELATIONYASHIKA.jpg"):
        """Open the image files, and compute their correlation """

        # Convert from Image to Numpy array conversion
        # (3744, 5616)(645, 645)
        translation = feature.register_translation(im1, im2, upsample_factor=10)[0]
        im2_register = ndimage.shift(im2, translation)

        d = 1

        corr = n.zeros_like(im1)

        for i in range(d, sh_row - (d + 1)):
            for j in range(d, sh_col - (d + 1)):
                corr[i, j] = self.correlationx(im1[i - d: i + d + 1, j - d: j + d + 1],im2[i - d: i + d + 1, j - d: j + d + 1])

        c = Image.fromarray(n.uint8(self.normalizeArray(corr) * 255))

        print("Saving as: {output_file}")
        c.save(output_file)
        return corr


