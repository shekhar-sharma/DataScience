#!/usr/bin/env python
# coding: utf-8

# In[21]:


#importing required libs
from skimage.io import imshow
from skimage.transform import resize
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

class HOG:
    #preprocessing image
    def preprocessing_image(self,img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resized_image = resize(img_gray, (128,64)) 
        return resized_image
    
    def calculate_magnitude_orientation(self,img):
        gradient_y = np.empty(img.shape, dtype=np.double)
        gradient_y[0, :] = 0
        gradient_y[-1, :] = 0
        gradient_y[1:-1, :] = img[2:, :] - img[:-2, :]
        gradient_x = np.empty(img.shape, dtype=np.double)
        gradient_x[:, 0] = 0
        gradient_x[:, -1] = 0
        gradient_x[:, 1:-1] = img[:, 2:] - img[:, :-2]
        magnitude = np.hypot(gradient_x,gradient_y)
        orientation = np.rad2deg(np.arctan2(gradient_y, gradient_x)) % 180
        return magnitude,orientation
    
    def get_hist_cell(self,img,pixels_per_cell=(8,8),cells_per_block=(2,2),bins=9):
        no_of_cells_x = img.shape[1]//pixels_per_cell[1]
        no_of_cells_y = img.shape[0]//pixels_per_cell[0]
        
        # creating histogram matrix (shape : [no_of_cells_x][no_of_cells_y][bins])
        
        orientation_histogram = np.zeros((no_of_cells_y, no_of_cells_x, bins))
        
        magnitude,orientation = self.calculate_magnitude_orientation(img)
        
        for row in range(no_of_cells_y):
            for col in range(no_of_cells_x):
                for i in range(pixels_per_cell[0]):
                    for j in range(pixels_per_cell[1]):
                        angle = orientation[i+row*pixels_per_cell[0]][j+col*pixels_per_cell[1]].astype(int)
                        if(angle%20==0):
                            if(angle//20==9):
                                orientation_histogram[row][col][0] += magnitude[i+row*pixels_per_cell[0]][j+col*pixels_per_cell[1]]
                            else:
                                orientation_histogram[row][col][angle//20] += magnitude[i+row*pixels_per_cell[0]][j+col*pixels_per_cell[1]]
                        else:
                            prev_bin = (angle//20) 
                            next_bin = (angle//20 + 1)

                            prev_bin_mag = ((next_bin*20 - angle)/20)* magnitude[i+row*pixels_per_cell[0]][j+col*pixels_per_cell[1]]
                            next_bin_mag = ((angle - prev_bin*20)/20)* magnitude[i+row*pixels_per_cell[0]][j+col*pixels_per_cell[1]]
                            orientation_histogram[row][col][prev_bin] += prev_bin_mag
                            if(next_bin==9):
                                orientation_histogram[row][col][0] += next_bin_mag
                            else:
                                orientation_histogram[row][col][next_bin] += next_bin_mag
        return orientation_histogram
    
    
    def visualize_hog(self,im, hog, cell_size, block_size,bins):
        num_bins = bins
        max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
        im_h, im_w = im.shape
        num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
        num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
        histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
        histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
        angles = np.arange(0, np.pi, np.pi/num_bins)
        mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
        mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
        mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
        #plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        for i in range(num_bins):
            plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                       color="black", headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
        plt.show()
    
    
    def hog(self,image,pixels_per_cell=(8,8),cells_per_block=(2,2),bins=9,visualize=True):
        final_feature = []
        img = self.preprocessing_image(image)
        orientation_histogram = self.get_hist_cell(img,pixels_per_cell,cells_per_block,bins)
        
        for i in range(orientation_histogram.shape[0]-1):
            for j in range(orientation_histogram.shape[1]-1):
                norm = np.sqrt(np.sum(np.square(orientation_histogram[i:(i+cells_per_block[0]),j:(j+cells_per_block[1])])))
                copy_hog=orientation_histogram[i:(i+cells_per_block[0]),j:(j+cells_per_block[0])]
                if(norm!=0):
                    copy_hog/=norm
                final_feature.append(copy_hog.reshape(bins*cells_per_block[0]*cells_per_block[1]))
        feature_vector = np.array(list(np.concatenate(final_feature).flat))
        if(visualize):
            self.visualize_hog(img,feature_vector,pixels_per_cell[0],cells_per_block[0],bins)
        return feature_vector


# In[ ]:




