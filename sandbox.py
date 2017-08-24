# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:44:12 2017

@author: Yunshi_Zhao
"""
from functions import find_cars, add_heat, apply_threshold, draw_labeled_bboxes 
from scipy.ndimage.measurements import label

test_images = glob.glob('./test_images/*.jpg')
ystart = 400
ystop = 700
scale = 1

orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial = (32,32)
histbin = 16


bbox_lists = []
for file in test_images:
    img = mpimg.imread(file)
    bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                          pix_per_cell, cell_per_block, spatial, histbin)
    bbox_lists.append(bbox_list)

heat = np.zeros_like(img[:,:,0]).astype(np.float)
heat = add_heat(heat,bbox_list)
heat = apply_threshold(heat,1)
heatmap = np.clip(heat, 0, 255)
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(img), labels)

