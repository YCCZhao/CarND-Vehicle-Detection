# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:33:30 2017

@author: Yunshi_Zhao
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import glob
import pickle


# load images
non_vehicle_images = glob.glob('../Data/vehicle-detection-basic/non-vehicles/non-vehicles/*.png')
vehicle_images = []
vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Far/*.png'))
vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Left/*.png'))
vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_MiddleClose/*.png'))
vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_RIght/*.png'))
vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/KITTI_extracted/*.png'))

