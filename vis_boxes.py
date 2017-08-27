# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:08:50 2017

@author: Yunshi_Zhao
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from functions import slide_window, draw_boxes


image = mpimg.imread('./test_images/test1.jpg')


window_size = 64
scale = 2.75
y_start_stop = [400, 700]
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(int(scale*window_size), int(scale*window_size)), xy_overlap=(0, 0))
                       
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)