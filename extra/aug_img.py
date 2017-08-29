# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 22:42:36 2017

@author: Yunshi_Zhao
"""
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.image as mpimg


def image_aug(image_set_file, image_aug_set = []):
    
    for image_file in image_set_file: 
        
        image = mpimg.imread(image_file)

        shear = iaa.Affine(shear=(-5,5)) 
        for i in range(2):
            image_aug_set.append(shear.augment_image(np.copy(image)))
        
        translater = iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)})
        for i in range(2):
            image_aug_set.append(translater.augment_image(np.copy(image)))
            
        scaler = iaa.Affine(scale={"y": (0.8, 1.2)}) 
        for i in range(2):
            image_aug_set.append(scaler.augment_image(np.copy(image)))
        
        rotate = iaa.Affine(rotate=(-15, 15)) 
        for i in range(2):
            image_aug_set.append(rotate.augment_image(np.copy(image)))
        
        flipper = iaa.Fliplr(1.0)
        image_aug_set.append(flipper.augment_image(np.copy(image)))
        
        blurer = iaa.GaussianBlur(2.0)
        image_aug_set.append(blurer.augment_image(np.copy(image)))
        
        #add = iaa.Add((-10, 10), per_channel=0.5)
        #image_aug_set.append(add.augment_image(np.copy(image)))
        print(len(image_aug_set))
        
    return image_aug_set

image_aug_set = image_aug(notcars)

for idx, image in enumerate(image_aug_set):
    out = '../Data/vehicle-detection-basic/non-vehicles/aug/aug_'+str(idx)+'.jpg'
    mpimg.imsave(out, image)
    
    