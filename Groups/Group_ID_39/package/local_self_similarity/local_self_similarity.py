import numpy as np
import cv2 as cv
from math import exp, sqrt


# Distance from center
def _distance(y0, x0):
	return sqrt(y0**2 + x0**2)


# Absolute
def _sub_abs(x0, x1):
	if x0 > x1:
		return x0-x1
	return x1-x0


# Calculate 'sum of squares difference'
def patch_ssd(img, yp, xp, yc, xc, patch_size):
	offset = (patch_size-1)//2
	diff = 0
	for y_off in range(-offset, offset, 1):
		for x_off in range(-offset, offset, 1):
			color_diff = 0
			for c in range(img.shape[2]):
				color_diff += _sub_abs(img[yp+y_off][xp+x_off][c], img[yc+y_off][xc+x_off][c])
			diff += color_diff**2
	return diff


# returns angle [0 .. 360] from center to pos_offset
def _get_theta(y_off, x_off):
	# III. Quadrant
	if x_off < 0 and y_off > 0:
		return 90. + _get_theta(-x_off, y_off)
	# II. Quadrant
	elif x_off <= 0 and y_off <= 0:
		return 180 + _get_theta(-y_off, -x_off)
	# I. Quadrant
	elif x_off > 0 and y_off < 0:
		return 270. + _get_theta(x_off, -y_off)

	if y_off >= x_off:
		# --> [0 .. 45]
		return float(x_off)/y_off * 45. 
	else:
		# --> ]45 .. 90]
		return 90. - (float(y_off)/x_off * 45.)


# calculate single self-similarity descriptor for a certain patch_center (yp, xp)
def self_similarity_descriptor(img, yp, xp, cor_radius, patch_size, radius=4, perimeter=20): 
	
	# noise --> (colour, illumination or due to noise)
	var_noise = 200000

	theta_step = (360./perimeter)
	rho_step = (float(cor_radius)/radius)

	circle_descriptor = []
	for rho_section in range(radius):
		temp = []
		for theta_section in range(perimeter):
			temp.append([])
		circle_descriptor.append(temp)

	correlation_img = np.zeros((cor_radius*2+1, cor_radius*2+1))

	for y_off in range(-cor_radius, cor_radius +1, 1):
		for x_off in range(-cor_radius, cor_radius +1, 1):

			# mask out /skip values with "radius to patch_center" > cor_radius
			if _distance(y_off, x_off) >= cor_radius or (y_off == 0 and x_off == 0):
				continue

			# calculate overall difference in pixels between patch and its offset-patch
			difference = patch_ssd(img, yp, xp, yp+y_off, xp+x_off, patch_size) 

			# similarity [0.0 .. 1.0] 
			# (with 0.0 --> not similar and 1.0 --> very similar)
			similarity = exp(-(difference/var_noise)) 

			correlation_img[y_off+cor_radius][x_off+cor_radius] = similarity

			# choose 2D-Descriptor section
			rho = _distance(y_off, x_off)
			theta = _get_theta(y_off, x_off)

			rho_section = int(rho // rho_step)
			theta_section = int(theta // theta_step)

			circle_descriptor[rho_section][theta_section].append(similarity)

	# create 1D-Descriptor from 2D-Circle Descriptor (circle_descriptor)
	descriptor = []

	for rho_section in range(radius):
		for theta_section in range(perimeter):
			count = len(circle_descriptor[rho_section][theta_section])
			avg = sum(circle_descriptor[rho_section][theta_section]) / count	
			descriptor.append(int(avg*255))

	return descriptor

# calculate many self-similarity descriptors for all patch_pos in the img
def local_self_similarity(img, cor_radius=40, patch_size=5, step=10):
	# patch_size needs to be odd (preferred = 5)
	# (yp, xp) == coordinate of patch (center)
	# offset: corner and edge (yp, xp)-values, which don't fit a complete correlation-circle into the image, have to be skipped
	# (ys, xs) == coord of correlation_square (center)

	height, width, c = img.shape

	offset = cor_radius+((patch_size-1)//2)

	descriptors = []

	for yp in range(offset, height-offset +1, step):
		for xp in range(offset, width-offset +1, step):
			desc = self_similarity_descriptor(img, yp, xp, cor_radius, patch_size)
			descriptors.append(desc)

	return descriptors
