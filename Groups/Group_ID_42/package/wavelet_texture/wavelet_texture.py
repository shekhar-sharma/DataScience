import numpy as np
from  skimage.feature import greycomatrix, greycoprops
from skimage import io,color, img_as_ubyte
def contrast_feature(x):
    contrast=greycoprops(x,'contrast')
    return "Contrast =",contrast
def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
	return "Dissimilarity = ", dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
	return "Homogeneity = ", homogeneity

def energy_feature(matrix_coocurrence):
	energy = greycoprops(matrix_coocurrence, 'energy')
	return "Energy = ", energy

def correlation_feature(matrix_coocurrence):
	correlation = greycoprops(matrix_coocurrence, 'correlation')
	return "Correlation = ", correlation

def asm_feature(matrix_coocurrence):
	asm = greycoprops(matrix_coocurrence, 'ASM')
	return "ASM = ", asm

# Load image

img=io.imread('/wavelet_texture/imgg.jpeg')
gray = color.rgb2gray(img)
image = img_as_ubyte(gray)
max_value = img.max()
matrix_coocurrence = greycomatrix(image, [1], [0, np.pi/4, np.pi/2], levels=max_value+1, normed=False, symmetric=False)

print(contrast_feature(matrix_coocurrence))
print(homogeneity_feature(matrix_coocurrence))
print(dissimilarity_feature(matrix_coocurrence))
print(energy_feature(matrix_coocurrence))
print(correlation_feature(matrix_coocurrence))
print(asm_feature(matrix_coocurrence))
