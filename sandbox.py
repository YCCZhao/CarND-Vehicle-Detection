# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:44:12 2017

@author: Yunshi_Zhao
"""
from functions import find_cars, add_heat, apply_threshold, draw_labeled_bboxes 
from scipy.ndimage.measurements import label
import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import pickle

trained_clf = '../Data/vehicle-detection-basic/trained_clf.p'
with open(trained_clf, mode='rb') as f:
    clf = pickle.load(f)
    
svc = clf['svc']
X_scaler = clf['scaler'] 
colorspace = clf['colorspace'] 
hog_channel = clf['hog_channel']
orient = clf['orient']
pix_per_cell = clf['pix_per_cell']
cell_per_block = clf['cell_per_block'] 
spatial = clf['spatial_size']
histbin = clf['hist_bins']


test_images = glob.glob('./test_images/test*.jpg')
ystart = 400
ystop = 700
scale = 1.8


#bbox_lists = []
for file in test_images:
    img = mpimg.imread(file)
    bbox_list = find_cars(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                          pix_per_cell, cell_per_block, (spatial, spatial), histbin,
                         hog_channel, spatial_feat=True, hist_feat=True)
    #bbox_lists.append(bbox_list)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,bbox_list)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    out = './test_images/output22_'+file.split('\\')[1].split('.')[0]+'.jpg'
    mpimg.imsave(out, draw_img)

