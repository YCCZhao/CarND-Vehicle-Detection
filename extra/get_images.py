# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:33:30 2017

@author: Yunshi_Zhao
"""

import glob
from sklearn.utils import shuffle

# load images
def get_images():  
  notcars = []
  notcars.extend(glob.glob('../Data/vehicle-detection-basic/non-vehicles/non-vehicles/*.png'))
  notcars.extend(glob.glob('../Data/vehicle-detection-basic/non-vehicles/aug/*.jpg'))
  cars = []
  cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Far/*.png'))
  cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Left/*.png'))
  cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_MiddleClose/*.png'))
  cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_RIght/*.png'))
  cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/KITTI_extracted/*.png'))
  cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/aug/*.jpg'))
  cars = shuffle(cars)
  notcars = shuffle(notcars)
  cars, notcars = cars[:40000], notcars[:40000]
  
  return cars, notcars


