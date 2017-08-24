# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:33:30 2017

@author: Yunshi_Zhao
"""

import glob

# load images
def get_images():  
  non_vehicle_images = glob.glob('../Data/vehicle-detection-basic/non-vehicles/non-vehicles/*.png')
  vehicle_images = []
  vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Far/*.png'))
  vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Left/*.png'))
  vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_MiddleClose/*.png'))
  vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_RIght/*.png'))
  vehicle_images.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/KITTI_extracted/*.png'))
  
  return vehicle_images, non_vehicle_images
