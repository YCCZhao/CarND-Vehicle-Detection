# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:44:12 2017

@author: Yunshi_Zhao
"""
from functions import find_cars, add_heat, apply_threshold, draw_labeled_bboxes 
from scipy.ndimage.measurements import label
import numpy as np
import matplotlib.image as mpimg
import glob
import pickle
from moviepy.editor import VideoFileClip


trained_clf = '../Data/vehicle-detection-basic/trained_clf3.p'

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

#test_images = glob.glob('./test_images/test*.jpg')
search_boxes = [(1,400,530), (1.5,400,600), (2,400,700),
                (2.75,400,700), (2.5,400,700), (3,400,700)]

#fig = plt.figure(figsize=(10,10))
#for idx, file in enumerate(test_images):

def vehicle_detection(img):
    #img = mpimg.imread(file)
    total_heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for scale, ystart, ystop in search_boxes:
        bbox_list = find_cars(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                              pix_per_cell, cell_per_block, (spatial, spatial), histbin,
                             hog_channel, spatial_feat=True, hist_feat=True)
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat,bbox_list)
        heat = apply_threshold(heat, 1)
        total_heat = total_heat + heat

    heatmap = np.clip(total_heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    #a=fig.add_subplot(3,2,idx+1)
    #plt.imshow(heatmap,cmap='hot')
    
    return draw_img

#fig.savefig('./examples/heatmap.jpg')

input_video = './test_video.mp4'
output_video = './test_project_video.mp4'
clip = VideoFileClip(input_video)
output_clip = clip.fl_image(vehicle_detection) 
output_clip.write_videofile(output_video, audio=False) 
