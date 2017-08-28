# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:08:50 2017

@author: Yunshi_Zhao
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from functions import slide_window, draw_boxes, get_hog_features, convert_color,find_cars
from get_images import get_images


cars, notcars = get_images()

car_sample = mpimg.imread(cars[random.randint(0, len(cars))])
notcar_sample = mpimg.imread(notcars[random.randint(0, len(notcars))])


fig = plt.figure(figsize=(10,5))
a=fig.add_subplot(1,2,1)
a.set_title("car")
plt.imshow(car_sample)
b=fig.add_subplot(1,2,2)
b.set_title("not car")
plt.imshow(notcar_sample)
fig.savefig('./examples/car_notcar.jpg')

car_sample_YCrCb = convert_color(car_sample, cspace='YCrCb')
not_car_sample_YCrCb = convert_color(notcar_sample, cspace='YCrCb')
fig2 = plt.figure(figsize=(10,15))
a=fig2.add_subplot(3,2,1)
a.set_title("car-YCrCB ch1")
plt.imshow(cv2.resize(car_sample_YCrCb[:,:,0], (32,32)),cmap='gray')
b=fig2.add_subplot(3,2,2)
b.set_title("not car-YCrCB ch1")
plt.imshow(cv2.resize(not_car_sample_YCrCb[:,:,0], (32,32)),cmap='gray')
c=fig2.add_subplot(3,2,3)
c.set_title("car-YCrCB ch2")
plt.imshow(cv2.resize(car_sample_YCrCb[:,:,1], (32,32)),cmap='gray')
d=fig2.add_subplot(3,2,4)
d.set_title("not car-YCrCB ch2")
plt.imshow(cv2.resize(not_car_sample_YCrCb[:,:,1], (32,32)),cmap='gray')
e=fig2.add_subplot(3,2,5)
e.set_title("car-YCrCB ch3")
plt.imshow(cv2.resize(car_sample_YCrCb[:,:,2], (32,32)),cmap='gray')
f=fig2.add_subplot(3,2,6)
f.set_title("not car-YCrCB ch3")
plt.imshow(cv2.resize(not_car_sample_YCrCb[:,:,2], (32,32)),cmap='gray')
fig2.savefig('./examples/car_notcar_spatial.jpg')


car_hist1=np.histogram(car_sample_YCrCb[:,:,0], bins=16, range=(0,256))
notcar_hist1=np.histogram(not_car_sample_YCrCb[:,:,0], bins=16, range=(0,256))
car_hist2=np.histogram(car_sample_YCrCb[:,:,1], bins=16, range=(0,256))
notcar_hist2=np.histogram(not_car_sample_YCrCb[:,:,1], bins=16, range=(0,256))
car_hist3=np.histogram(car_sample_YCrCb[:,:,2], bins=16, range=(0,256))
notcar_hist3=np.histogram(not_car_sample_YCrCb[:,:,2], bins=16, range=(0,256))
bin_edges = a[1]
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

fig3 = plt.figure(figsize=(10,15))
a=fig3.add_subplot(3,2,1)
plt.bar(bin_centers, car_hist1[0])
a.set_title("car-YCrCB ch1 histogram")
b=fig3.add_subplot(3,2,2)
plt.bar(bin_centers, notcar_hist1[0])
b.set_title("not car-YCrCB ch1 histogram")
c=fig3.add_subplot(3,2,3)
plt.bar(bin_centers, car_hist2[0])
c.set_title("car-YCrCB ch2 histogram")
d=fig3.add_subplot(3,2,4)
plt.bar(bin_centers, notcar_hist2[0])
d.set_title("not car-YCrCB ch2 histogram")
e=fig3.add_subplot(3,2,5)
plt.bar(bin_centers, car_hist3[0])
e.set_title("car-YCrCB ch3 histogram")
f=fig3.add_subplot(3,2,6)
plt.bar(bin_centers, notcar_hist3[0])
f.set_title("not car-YCrCB ch3 histogram")
fig3.savefig('./examples/car_notcar_histogram.jpg')

fig4 = plt.figure(figsize=(10,15))
a=fig4.add_subplot(3,2,1)
hog, img = get_hog_features(car_sample_YCrCb[:,:,0], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
plt.imshow(img, cmap = 'gray')
a.set_title("car-YCrCB ch1 hog")
b=fig4.add_subplot(3,2,2)
hog, img = get_hog_features(not_car_sample_YCrCb[:,:,0], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
plt.imshow(img, cmap = 'gray')
b.set_title("not car-YCrCB ch1 hog")
c=fig4.add_subplot(3,2,3)
hog, img = get_hog_features(car_sample_YCrCb[:,:,1], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
plt.imshow(img, cmap = 'gray')
a.set_title("car-YCrCB ch2 hog")
d=fig4.add_subplot(3,2,4)
hog, img = get_hog_features(not_car_sample_YCrCb[:,:,1], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
plt.imshow(img, cmap = 'gray')
b.set_title("not car-YCrCB ch2 hog")
e=fig4.add_subplot(3,2,5)
hog, img = get_hog_features(car_sample_YCrCb[:,:,2], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
plt.imshow(img, cmap = 'gray')
a.set_title("car-YCrCB ch3 hog")
f=fig4.add_subplot(3,2,6)
hog, img = get_hog_features(not_car_sample_YCrCb[:,:,2], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
plt.imshow(img, cmap = 'gray')
b.set_title("not car-YCrCB ch3 hog")
fig4.savefig('./examples/car_notcar_hog.jpg')

image = mpimg.imread('./test_images/test1.jpg')
window_size = 64
search_boxes = [(1,0,199,400,530), (1.5,200,399,400,600), (2,400,599,400,700),
                (2.5,600,799,400,700), (2.75,800,999,400,700), (3,1000,1199,400,700)]
for scale, xstart, xstop, ystart, ystop in search_boxes:
    windows = slide_window(image, x_start_stop=[xstart, xstop], y_start_stop=[ystart,ystop], 
                    xy_window=(int(scale*window_size), int(scale*window_size)), xy_overlap=(0, 0))
                       
    image = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imsave('./examples/sliding_windows.jpg', image)

test_images = glob.glob('./test_images/test*.jpg')
total_heat = np.zeros_like(image[:,:,0]).astype(np.float)
for scale, ystart, ystop in search_boxes:
    bbox_list = find_cars(image, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                          pix_per_cell, cell_per_block, (spatial, spatial), histbin,
                         hog_channel, spatial_feat=True, hist_feat=True)
    image = draw_boxes(image, bbox_list, color=(0, 0, 255), thick=6)  
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,bbox_list)
    heat = apply_threshold(heat, 1)
    total_heat = total_heat + heat

heatmap = np.clip(total_heat, 0, 255)
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(img), labels)
#out = './test_images/output_'+file.split('\\')[1].split('.')[0]+'.jpg'
#mpimg.imsave(out, draw_img)
return draw_img